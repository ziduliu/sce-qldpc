"""Relay-BP with RelayDecoderF32 semantics on the GPU (Triton kernels), one
shot at a time. The relay sets run in order, since each set starts from the
previous set's posterior; each leg is captured as a CUDA graph, and a leg that
converges early is replayed eagerly for exactly that many iterations to recover
the error vector at the first converged iteration.
"""
import os

import numpy as np
import scipy.sparse as sp
import torch
import triton as tr
import triton.language as tl

FLT_MAX = 3.4028234663852886e38

# launch-shape defaults, overridable from the environment
_ENV_VAR_WARPS = int(os.environ.get("PLG_VAR_WARPS", "4"))
_ENV_VAR_VS = os.environ.get("PLG_VAR_VS", "")
_ENV_CHECK_BUCKETS = os.environ.get("PLG_CHECK_BUCKETS", "single")
_ENV_VAR_BUCKETS = os.environ.get("PLG_VAR_BUCKETS", "single")


@tr.jit
def _check(Q, CV, PTR, ROWS, SYN, DONE, NB,
           NR: tl.constexpr, K: tl.constexpr, BS: tl.constexpr):
    ri = tl.program_id(0)
    row = tl.load(ROWS + ri)
    b = tl.program_id(1) * BS + tl.arange(0, BS)
    live = (b < NB) & (tl.load(DONE + b, b < NB, other=1) == 0)
    start = tl.load(PTR + row); end = tl.load(PTR + row + 1)
    edge = start + tl.arange(0, K)
    valid = (edge[:, None] < end) & live[None, :]
    q = tl.load(Q + edge[:, None] * NB + b[None, :], valid, other=0.)
    a = tl.where(valid & (tl.abs(q) < 3.4028234663852886e38), tl.abs(q),
                 3.4028234663852886e38)
    m1 = tl.min(a, 0)
    count = tl.sum(((a == m1[None, :]) & valid).to(tl.int32), 0)
    m2 = tl.min(tl.where(a == m1[None, :], 3.4028234663852886e38, a), 0)
    m2 = tl.where(count > 1, m1, m2)
    negative = (q.to(tl.int32, bitcast=True) < 0) & valid
    parity = tl.sum(negative.to(tl.int32), 0) & 1
    syn = tl.load(SYN + row * NB + b, live, other=0)
    sign = (parity[None, :] ^ negative.to(tl.int32) ^ syn[None, :]) != 0
    mag = tl.where(a == m1[None, :], m2[None, :], m1[None, :])
    tl.store(CV + edge[:, None] * NB + b[None, :], tl.where(sign, -mag, mag), valid)


@tr.jit
def _variable_gam(Q, CV, POST, PRIOR, GAM, PTR, MAP, VARS, DONE, HARD,
                  NV: tl.constexpr, NB, D: tl.constexpr,
                  BS: tl.constexpr, VS: tl.constexpr):
    vi = tl.program_id(0) * VS + tl.arange(0, VS)
    v = tl.load(VARS + vi, vi < NV, other=0)
    b = tl.program_id(1) * BS + tl.arange(0, BS)
    live = (vi[:, None] < NV) & (b[None, :] < NB) & (
        tl.load(DONE + b, b < NB, other=1)[None, :] == 0)
    start = tl.load(PTR + v)
    end = tl.load(PTR + v + 1)
    prior = tl.load(PRIOR + v)[:, None]
    old = tl.load(POST + v[:, None] * NB + b[None, :], live, other=0.)
    # prior * (1 - g) + posterior * g with a per-variable relay gamma g, in the
    # CPU decoder's operation order (unfused f32)
    g = tl.load(GAM + v)[:, None]
    acc = prior * (1. - g) + old * g
    limit: tl.constexpr = 3.4028234663852886e38
    acc = tl.where(prior == limit, prior, acc)
    # Prefixes go into Q and are read back, so the compiled instruction count
    # does not grow with the degree bound D.
    for j in range(D):
        idx = start + j
        m = live & (idx[:, None] < end[:, None])
        edge = tl.load(MAP + idx, (vi < NV) & (idx < end), other=0)
        cv = tl.load(CV + edge[:, None] * NB + b[None, :], m, other=0.)
        tl.store(Q + edge[:, None] * NB + b[None, :], acc, m)
        acc = tl.where(m, acc + cv, acc)
    tl.store(POST + v[:, None] * NB + b[None, :], acc, live)
    tl.store(HARD + v[:, None] * NB + b[None, :], acc <= 0., live)
    suffix = tl.full((VS, BS), 0., tl.float32)
    for jj in range(D):
        idx = start + D - 1 - jj
        m = live & (idx[:, None] < end[:, None])
        edge = tl.load(MAP + idx, (vi < NV) & (idx < end), other=0)
        cv = tl.load(CV + edge[:, None] * NB + b[None, :], m, other=0.)
        prefix = tl.load(Q + edge[:, None] * NB + b[None, :], m, other=0.)
        tl.store(Q + edge[:, None] * NB + b[None, :], prefix + suffix, m)
        suffix = tl.where(m, suffix + cv, suffix)


@tr.jit
def _syndrome_count(HARD, PTR, IND, ROWS, SYN, NBAD, NB,
                    K: tl.constexpr, BS: tl.constexpr):
    """Syndrome check reduced to a device-side mismatch count in NBAD, so the
    host reads convergence once per leg instead of syncing every iteration.
    """
    row = tl.load(ROWS + tl.program_id(0))
    b = tl.program_id(1) * BS + tl.arange(0, BS)
    live = b < NB
    start = tl.load(PTR + row); end = tl.load(PTR + row + 1)
    edge = start + tl.arange(0, K)
    var = tl.load(IND + edge, edge < end, other=0)
    h = tl.load(HARD + var[:, None] * NB + b[None, :],
                (edge[:, None] < end) & live[None, :], other=0)
    parity = tl.sum(h.to(tl.int32), 0) & 1
    syn = tl.load(SYN + row * NB + b, live, other=0)
    bad = (parity != syn).to(tl.int32)
    tl.atomic_add(NBAD + b, bad, mask=live)


class GpuRelay:
    """One code/basis worth of GPU relay state.  Decodes ONE shot at a time.

    stream_offset is the number of uniform draws the decoder instance has
    already consumed; decode() returns how many this shot consumed so the
    caller can advance it, exactly as the Rust decoder's RNG advances.
    """

    def __init__(self, H, pri, cfg, device="cuda", variable_vs="env",
                 check_bs=1, check_warps=8, variable_warps=None,
                 syndrome_warps=8, capture=True,
                 check_buckets=None, variable_buckets=None):
        if variable_vs == "env":
            variable_vs = int(_ENV_VAR_VS) if _ENV_VAR_VS else None
        if variable_warps is None:
            variable_warps = _ENV_VAR_WARPS
        if check_buckets is None:
            check_buckets = _ENV_CHECK_BUCKETS
        if variable_buckets is None:
            variable_buckets = _ENV_VAR_BUCKETS
        from relay_bp import RelayDecoderF32
        if cfg.get("stop_nconv", 1) != 1:
            raise ValueError("only stop_nconv=1 is reproduced here")
        if cfg.get("alpha") is not None:
            raise ValueError("alpha must be None: the kernels assume alpha=1")
        self.cfg = dict(cfg)
        self.dev = device
        self.gamma0 = float(cfg["gamma0"])
        self.pre_iter = int(cfg["pre_iter"])
        self.num_sets = int(cfg["num_sets"])
        self.set_max_iter = int(cfg["set_max_iter"])
        self.check_warps, self.variable_warps = check_warps, variable_warps
        self.syndrome_warps = syndrome_warps
        self.check_bs = check_bs

        H = sp.csr_matrix(H, dtype=np.uint8)
        H.sum_duplicates()
        if not H.has_canonical_format or not np.all(H.data == 1):
            raise ValueError("Require a canonical binary CSR matrix")
        self.H = H
        self.C, self.N = H.shape
        self.nnz = int(H.nnz)

        # f32 log priors taken from the Rust decoder rather than recomputed in
        # numpy, so they match bit for bit
        d = RelayDecoderF32(check_matrix=H,
                            error_priors=np.asarray(pri, np.float64),
                            gamma0=self.gamma0, pre_iter=0, num_sets=0)
        p = d.decode_detailed(np.zeros(self.C, np.uint8)).posterior_ratios
        self.prior_np = np.asarray(p, np.float32)
        self.prior = torch.tensor(self.prior_np, device=device)

        def gpu(x):
            return torch.as_tensor(np.asarray(x, dtype=np.int32), device=device)

        self.ptr = gpu(H.indptr)
        self.ind = gpu(H.indices)
        numbered = sp.csr_matrix((np.arange(H.nnz, dtype=np.int32) + 1,
                                  H.indices, H.indptr), shape=H.shape).tocsc()
        self.vptr = gpu(numbered.indptr)
        self.mapping = gpu(numbered.data - 1)

        # "single" pads every row/variable to the maximum degree and launches
        # once (faster at one shot); otherwise rows/variables are bucketed by
        # power-of-two degree.
        def groups(degrees, mode):
            powers = np.array([tr.next_power_of_2(max(1, int(n)))
                               for n in degrees])
            if mode == "single":
                return [(int(powers.max()),
                         gpu(np.arange(len(degrees), dtype=np.int32)))]
            return [(int(k), gpu(np.flatnonzero(powers == k)))
                    for k in np.unique(powers)]

        self.rows = groups(np.diff(H.indptr), check_buckets)
        self.variables = groups(np.diff(numbered.indptr), variable_buckets)
        self.variable_vs = variable_vs

        # Fixed-address working set: a CUDA graph bakes in these pointers.
        z = lambda n, dt: torch.zeros(n, dtype=dt, device=device)
        self.q = z(self.nnz, torch.float32)
        self.cv = z(self.nnz, torch.float32)
        self.post = z(self.N, torch.float32)
        self.post_save = z(self.N, torch.float32)
        self.hard = z(self.N, torch.uint8)
        self.syn = z(self.C, torch.uint8)
        self.gam = z(self.N, torch.float32)
        self.done = z(1, torch.bool)
        self.nbad = z(max(self.pre_iter, self.set_max_iter), torch.int32)
        self.prior_edges = self.prior.index_select(0, self.ind.long()).contiguous()

        self._graphs = {}
        self._capture = capture
        self.stats = {}

    # ---- launch shapes ----------------------------------------------------
    # At BS=1 a tile has one element per row/variable; size it to the thread
    # count.
    def _check_plan(self, k):
        warps = min(8, max(1, k // 64))
        return warps

    def _variable_plan(self, degree, count):
        warps = self.variable_warps
        vs = self.variable_vs or (warps * 32)
        return vs, warps

    # ---- one BP iteration, as launches ------------------------------------
    def _iteration(self, nbad_slot):
        NB, bs = 1, self.check_bs
        for k, rows in self.rows:
            _check[(len(rows), 1)](
                self.q, self.cv, self.ptr, rows, self.syn, self.done, NB,
                len(rows), k, bs, num_warps=self._check_plan(k),
                enable_fp_fusion=False)
        for degree, variables in self.variables:
            vs, warps = self._variable_plan(degree, len(variables))
            _variable_gam[(tr.cdiv(len(variables), vs), 1)](
                self.q, self.cv, self.post, self.prior, self.gam, self.vptr,
                self.mapping, variables, self.done, self.hard, len(variables),
                NB, degree, bs, vs, num_warps=warps,
                enable_fp_fusion=False)
        for k, rows in self.rows:
            _syndrome_count[(len(rows), 1)](
                self.hard, self.ptr, self.ind, rows, self.syn, nbad_slot, NB,
                k, bs, num_warps=self._check_plan(k), enable_fp_fusion=False)

    def warmup(self):
        """JIT every kernel and capture both leg graphs, before any real state.

        Capture has to happen while the working buffers are scratch: the
        warm-up launch and the capture itself both write q, cv, post and hard.
        """
        if not self._capture or self._graphs:
            return
        for iters in (self.pre_iter, self.set_max_iter):
            if iters in self._graphs or iters <= 0:
                continue
            s = torch.cuda.Stream()
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):
                    self._iteration(self.nbad)
            torch.cuda.current_stream().wait_stream(s)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for it in range(iters):
                    self._iteration(self.nbad[it:it + 1])
            self._graphs[iters] = graph
        torch.cuda.synchronize()

    def _run_leg(self, iters, eager=False):
        """Run a leg from the current state.  Returns the first converged
        iteration (1-based) or None.  `eager` bypasses the graph, which is how
        a leg is replayed for a partial number of iterations.
        """
        self.nbad[:iters].zero_()
        g = None if eager else self._graphs.get(iters)
        if g is not None:
            g.replay()
        else:
            for it in range(iters):
                self._iteration(self.nbad[it:it + 1])
        nb = self.nbad[:iters].cpu().numpy()
        hit = np.flatnonzero(nb == 0)
        return (int(hit[0]) + 1) if hit.size else None

    def _enter_pre(self):
        """initialize_decoder(): memory := gamma0, c2v := 0, v2c := priors,
        posterior := priors."""
        self.gam.fill_(float(np.float32(self.gamma0)))
        self.q.copy_(self.prior_edges)
        self.cv.zero_()
        self.post.copy_(self.prior)

    def _enter_set(self, gam_row):
        """init_next_set + initialize_check_to_variable +
        initialize_variable_to_check.  posterior_ratios is deliberately left
        alone -- that carry-over is the relay."""
        self.gam.copy_(gam_row)
        self.q.copy_(self.prior_edges)
        self.cv.zero_()

    def _vector(self):
        return self.hard.cpu().numpy().astype(np.uint8)

    def decode(self, det, gammas=None, want_vector=True):
        """Decode one shot.

        gammas: (num_sets, N) float32, row s-1 holding relay set s's gammas --
        the draws the Rust RNG makes at this shot's stream position.

        Returns dict(decoding, iterations, converged, winner, draws), where
        `draws` is how far the RNG advanced and must be added to the caller's
        stream offset, and `winner` is 0 for the pre-leg, s for set s, None if
        nothing converged.
        """
        det = np.ascontiguousarray(np.asarray(det, np.uint8))
        if det.shape != (self.C,):
            raise ValueError("detectors must be one shot of length %d" % self.C)
        self.syn.copy_(torch.from_numpy(det))
        self.done.zero_()

        self._enter_pre()
        j = self._run_leg(self.pre_iter)
        if j is not None:
            if j != self.pre_iter:
                self._enter_pre()
                self._run_leg(j, eager=True)
            return dict(decoding=self._vector() if want_vector else None,
                        iterations=j, converged=True, winner=0, draws=0)
        # A relay where no set converges returns the PRE-LEG's decoding.
        pre_vector = self._vector() if want_vector else None

        if self.num_sets:
            if gammas is None:
                raise ValueError("gammas required when num_sets > 0")
            gam = torch.as_tensor(np.ascontiguousarray(
                np.asarray(gammas, np.float32)), device=self.dev)
            if gam.shape != (self.num_sets, self.N):
                raise ValueError("gammas must be (num_sets, N), got %s"
                                 % (tuple(gam.shape),))
        for s in range(1, self.num_sets + 1):
            self.post_save.copy_(self.post)
            self._enter_set(gam[s - 1])
            j = self._run_leg(self.set_max_iter)
            if j is not None:
                if j != self.set_max_iter:
                    self.post.copy_(self.post_save)
                    self._enter_set(gam[s - 1])
                    self._run_leg(j, eager=True)
                return dict(decoding=self._vector() if want_vector else None,
                            iterations=self.pre_iter
                            + (s - 1) * self.set_max_iter + j,
                            converged=True, winner=s, draws=s * self.N)
        return dict(decoding=pre_vector,
                    iterations=self.pre_iter
                    + self.num_sets * self.set_max_iter,
                    converged=False, winner=None,
                    draws=self.num_sets * self.N)
