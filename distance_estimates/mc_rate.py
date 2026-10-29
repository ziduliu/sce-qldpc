"""Batched Monte-Carlo failure oracle for one side of a CSS code.

SideDecoder.fail_mask decodes a batch of error patterns on the device:
syndromes by one matmul, segmented min-sum BP (bp_minsum_syn) with
convergence latching, and OSD (gpu_core_v2.batched_rref / osd_solutions) only
for the elements BP left unconverged. An element fails iff its residual
e XOR e_hat has odd overlap with some row of L.
"""
import numpy as np
import torch

import gpu_core_v2 as core
from gpu_core_v2 import LARGE, batched_rref, check_table, osd_solutions

_GRAPH_WARMUP = 3


# ---------------------------------------------------------------------------
# generalized batched min-sum BP (arbitrary per-element syndrome)
# ---------------------------------------------------------------------------

def bp_minsum_syn(Htab, prior, syn, n, iters=30, alpha=0.8, m_cv=None):
    """Batched min-sum syndrome BP against a check matrix H with an arbitrary
    per-element syndrome: gpu_core_v2.bp_minsum with the syndrome entering
    the check update as a per-(B, C) sign (+1 for bit 0, -1 for bit 1).

    Htab  : [C, dH] long, check_table(H), shared across the batch
    prior : [B, n+1] float LLRs; slot n is the dummy and must be +LARGE
    syn   : [B, C] 0/1 syndrome bits (any integer dtype)
    m_cv  : optional [B, C, dH] float check-to-variable messages from a
            previous call; passing them back resumes BP instead of
            restarting it.

    Returns (posterior [B, n+1], m_cv [B, C, dH]). No host syncs and no
    RNG, so the body can be captured in a CUDA graph (_GraphedSegBP).
    """
    B = prior.shape[0]
    C, dH = Htab.shape
    dev = prior.device
    Hflat = Htab.reshape(-1)                                    # [C*dH]
    padH = (Htab == n).view(1, C, dH)
    padB = padH.expand(B, -1, -1)
    # +1 for syndrome bit 0, -1 for bit 1; broadcast over the message slots
    sflip = (1.0 - 2.0 * syn.to(prior.dtype)).view(B, C, 1)

    if m_cv is None:
        m_cv = torch.zeros(B, C, dH, device=dev, dtype=prior.dtype)

    for _ in range(iters):
        tot = prior.clone()
        tot.index_add_(1, Hflat, m_cv.reshape(B, -1))
        m_vc = tot[:, Hflat].view(B, C, dH) - m_cv
        m_vc = torch.where(padH, torch.full_like(m_vc, LARGE), m_vc)
        m_cv = core._check_update(m_vc, padB, sflip, alpha)

    tot = prior.clone()
    tot.index_add_(1, Hflat, m_cv.reshape(B, -1))
    return tot, m_cv


# ---------------------------------------------------------------------------
# CUDA-graph wrapper for one BP segment
# ---------------------------------------------------------------------------

class _GraphedSegBP:
    """Shape-keyed CUDA-graph cache for one bp_minsum_syn segment.

    Key = (B, C, dH, n, seg); alpha is fixed per SideDecoder. Each graph has
    a private memory pool. The returned (post, m_cv) live in static buffers
    valid until the next replay with the same key; inputs are copied into
    the static buffers before each replay.
    """

    def __init__(self, warmup=_GRAPH_WARMUP):
        self.warmup = warmup
        self._cache = {}

    def _capture(self, statics, n, seg, alpha):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.warmup):
                bp_minsum_syn(statics[0], statics[1], statics[2], n,
                              iters=seg, alpha=alpha, m_cv=statics[3])
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, pool=None):            # private pool
            outs = bp_minsum_syn(statics[0], statics[1], statics[2], n,
                                 iters=seg, alpha=alpha, m_cv=statics[3])
        return g, statics, outs

    def __call__(self, Htab, prior, syn, m_cv, n, seg, alpha):
        if not prior.is_cuda:
            raise RuntimeError(
                "_GraphedSegBP requires CUDA tensors; SideDecoder forces "
                "use_graphs=False on cpu")
        C, dH = Htab.shape
        key = (prior.shape[0], C, dH, int(n), int(seg))
        ent = self._cache.get(key)
        if ent is None:
            statics = (Htab.clone(), prior.clone(), syn.clone(),
                       m_cv.clone())
            ent = self._capture(statics, int(n), int(seg), float(alpha))
            self._cache[key] = ent
        g, statics, outs = ent
        for dst, src in zip(statics, (Htab, prior, syn, m_cv)):
            dst.copy_(src)
        g.replay()
        return outs


# ---------------------------------------------------------------------------
# batched failure oracle
# ---------------------------------------------------------------------------

class SideDecoder:
    """One side (X or Z) of a CSS code as a batched failure oracle.

    H : [C, n] 0/1 check matrix of the decoded side
    L : [k, n] 0/1 logical rows tested against the residual

    device       : "auto" -> cuda when available, else cpu
    bp_iters_max : BP iteration budget, rounded up to whole segments of
                   `seg` iterations
    seg          : BP iterations per segment; one convergence check and
                   one host sync per segment
    lam          : OSD combination-sweep width (gpu_core_v2.osd_solutions)
    alpha        : min-sum normalization factor
    use_graphs   : "auto" -> CUDA graph for the BP segment on CUDA; always
                   off on cpu. The OSD fallback is always eager.
    p_prior      : physical error rate assumed by the BP prior; fixed,
                   independent of the error patterns passed in.

    After each fail_mask call:
      .stats         dict(bp_converged_frac, osd_count, segments_run)
      .last_residual [B, n] uint8 residuals r = e XOR e_hat on the
                     decoder's device (H r == 0 mod 2).
    """

    def __init__(self, H, L, device="auto", bp_iters_max=48, seg=8, lam=24,
                 alpha=0.8, use_graphs="auto", p_prior=0.05):
        H = np.asarray(H, dtype=np.uint8) % 2
        L = np.asarray(L, dtype=np.uint8) % 2
        if H.ndim != 2 or L.ndim != 2 or H.shape[1] != L.shape[1]:
            raise ValueError("H and L must be 2-D with equal column counts")
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dev = torch.device(device)
        self.C, self.n = H.shape
        self.seg = int(seg)
        self.n_seg = max(1, -(-int(bp_iters_max) // self.seg))  # ceil
        self.lam = int(lam)
        self.alpha = float(alpha)
        # log((1-p)/p): the constant per-variable LLR of an iid prior
        self.llr0 = float(np.log((1.0 - p_prior) / p_prior))

        self.Htab = torch.as_tensor(check_table(H), device=self.dev)
        self.dH = int(self.Htab.shape[1])
        self.Ht = torch.as_tensor(H, device=self.dev)               # uint8
        # transposed fp32 copies for the syndrome / overlap matmuls
        # (0/1 entries: fp32 sums are exact far beyond any code size here)
        self.HfT = self.Ht.to(torch.float32).t().contiguous()      # [n, C]
        self.LfT = torch.as_tensor(L.astype(np.float32),
                                   device=self.dev).t().contiguous()

        if self.dev.type != "cuda":
            self.graphed = False              # graphs cannot exist on cpu
        elif use_graphs == "auto":
            self.graphed = True
        else:
            self.graphed = bool(use_graphs)
        self._graph = _GraphedSegBP() if self.graphed else None

        self.stats = None
        self.last_residual = None

    def fail_mask(self, E):
        """E: [B, n] uint8 0/1 error patterns (torch, any device; moved to
        the decoder's device here). Returns a bool [B] tensor on the
        decoder's device: True where the decoded residual has odd overlap
        with some row of L. Deterministic: no RNG anywhere in the pipeline,
        so repeated calls on the same E return identical masks."""
        E = torch.as_tensor(E).to(self.dev).to(torch.uint8)
        if E.dim() != 2 or E.shape[1] != self.n:
            raise ValueError("E must be [B, %d], got %s"
                             % (self.n, tuple(E.shape)))
        B, n, C = E.shape[0], self.n, self.C
        dev = self.dev

        # a. syndromes, one batched matmul
        Ef = E.to(torch.float32)
        s_f = torch.remainder(Ef @ self.HfT, 2.0)                  # [B, C]
        syn = s_f.to(torch.uint8)

        pri = torch.full((B, n + 1), LARGE, device=dev,
                         dtype=torch.float32)
        pri[:, :n] = self.llr0
        m_cv = torch.zeros(B, C, self.dH, device=dev,
                           dtype=torch.float32)

        # b. segmented BP with convergence latching. m_cv threads through
        # the segments, so this is plain bp_iters_max-iteration BP with a
        # convergence check every `seg` iterations, not a restart.
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        ehat = torch.zeros(B, n, dtype=torch.uint8, device=dev)
        post = pri
        segments = 0
        for _ in range(self.n_seg):
            if self.graphed:
                post, m_cv = self._graph(self.Htab, pri, syn, m_cv, n,
                                         self.seg, self.alpha)
            else:
                post, m_cv = bp_minsum_syn(self.Htab, pri, syn, n,
                                           iters=self.seg,
                                           alpha=self.alpha, m_cv=m_cv)
            segments += 1
            hard = (post[:, :n] < 0).to(torch.uint8)
            hsyn = torch.remainder(hard.to(torch.float32) @ self.HfT, 2.0)
            ok = (hsyn == s_f).all(1)
            # latch: only elements converging for the FIRST time write
            # their hard decision; later oscillation cannot overwrite it
            newly = ok & ~done
            ehat = torch.where(newly.view(B, 1), hard, ehat)
            done = done | ok
            if bool(done.all()):           # the ONE host sync per segment
                break

        # c. OSD fallback, compacted to the unconverged remainder only.
        # Eager even under graphs: the sub-batch size is data-dependent.
        todo = ~done
        osd_count = int(todo.sum())
        if osd_count:
            idx = todo.nonzero(as_tuple=False).view(-1)
            # ascending posterior LLR = most-likely-error first, the
            # column priority batched_rref expects
            order = torch.argsort(post[idx][:, :n], dim=1)
            M = self.Ht.unsqueeze(0).expand(osd_count, C, n)
            packed, rank, pcol, ispiv = batched_rref(M, order, syn[idx], n)
            _, V = osd_solutions(packed, rank, pcol, ispiv, order, n, C,
                                 lam=self.lam)
            ehat[idx] = V

        self.stats = {
            "bp_converged_frac": 1.0 - osd_count / float(B),
            "osd_count": osd_count,
            "segments_run": segments,
        }

        # d. residual against the logical rows
        r = E ^ ehat
        self.last_residual = r
        ov = torch.remainder(r.to(torch.float32) @ self.LfT, 2.0)
        return (ov > 0.5).any(1)
