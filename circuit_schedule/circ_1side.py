"""Circuit-level pL of lifted-product codes: greedy layer schedule, memory circuit
and DEM matrices, with min-sum BP on the GPU and BP+OSD on the CPU only for the
shots BP leaves unconverged (or BP+OSD for every shot when CIRC_GPU is off).

Schedule: each q x q block joining a check block to a data block is split into
perfect matchings, each a layer of q CNOTs; layers are coloured greedily, X
checks before Z checks. Noise: two-qubit depolarising after every CNOT,
one-qubit depolarising after ancilla reset and before measurement, no idling
noise. main() runs the four n=600 codes listed there.
"""
import json
import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault("LAYOUT_SQFAST_PATH", "/u/zdliu/lp/dopt/agentA")
for p in ("/u/zdliu/lp/lps_graftgate", "/u/zdliu/lp/dopt/bdirect_v2",
          "/u/zdliu/lp/lifted_product_search"):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
import scipy.sparse as sp
import stim
import torch
import fast_gf2
from fast_gf2 import logicals
from ldpc import BpOsdDecoder
from gpu_core_v2 import LARGE
from mc_rate import bp_minsum_syn

P = float(os.environ.get("CIRC_P", "0.001"))
SHOTS = int(os.environ.get("CIRC_SHOTS", "20000"))
ROUNDS = int(os.environ.get("CIRC_ROUNDS", "0"))       # 0 -> d
BP_ITERS = int(os.environ.get("CIRC_BP_ITERS", "60"))
BP_SEG = int(os.environ.get("CIRC_BP_SEG", "10"))
OSD_ORDER = int(os.environ.get("CIRC_OSD", "0"))
NPROC = int(os.environ.get("CIRC_NPROC", "8"))
Q = int(os.environ.get("CIRC_Q", "120"))
GPU_MEM_FRAC = float(os.environ.get("CIRC_GPU_FRAC", "0.55"))
USE_GPU = os.environ.get("CIRC_GPU", "0") == "1"


# --------------------------------------------------------------------------
# schedule: layers from perfect matchings of the blocks
# --------------------------------------------------------------------------

def perfect_matchings(sub):
    """Split a q x q 0-1 block of constant row and column weight a into a matchings.

    Each matching is a layer of q disjoint CNOTs and the a matchings cover the
    block exactly once; they are not in general the permutation matrices of
    the cell's group elements.
    """
    q = sub.shape[0]
    rem = sub.copy()
    out = []
    while rem.sum():
        match_r = [-1] * q
        match_c = [-1] * q
        adj = [np.flatnonzero(rem[r]).tolist() for r in range(q)]
        for r in range(q):
            seen = [False] * q
            stack = [(r, iter(adj[r]))]
            path = []
            while stack:
                rr, it = stack[-1]
                adv = False
                for cc in it:
                    if seen[cc]:
                        continue
                    seen[cc] = True
                    path.append((rr, cc))
                    if match_c[cc] == -1:
                        for (a, b) in path:
                            match_r[a] = b
                            match_c[b] = a
                        stack = []
                        adv = True
                        break
                    stack.append((match_c[cc], iter(adj[match_c[cc]])))
                    adv = True
                    break
                if not adv:
                    stack.pop()
                    if path:
                        path.pop()
        if any(m == -1 for m in match_r):
            raise AssertionError("block is not regular: no perfect matching")
        out.append(list(match_r))
        for r, c in enumerate(match_r):
            rem[r, c] = 0
            adj[r].remove(c)
    return out


def group_layers(H, q):
    m, n = H.shape
    nrb, ncb = m // q, n // q
    units = []
    for rb in range(nrb):
        for cb in range(ncb):
            sub = H[rb * q:(rb + 1) * q, cb * q:(cb + 1) * q]
            if not sub.any():
                continue
            for mt in perfect_matchings(sub):
                units.append((rb, cb, mt))
    return units, nrb, ncb


def schedule(units, nrb, ncb):
    """Greedy edge colouring of the check-block/data-block multigraph."""
    busy_rb = [set() for _ in range(nrb)]
    busy_cb = [set() for _ in range(ncb)]
    slot = []
    for rb, cb, _ in units:
        t = 0
        while t in busy_rb[rb] or t in busy_cb[cb]:
            t += 1
        busy_rb[rb].add(t)
        busy_cb[cb].add(t)
        slot.append(t)
    return slot, (max(slot) + 1 if slot else 0)


def build_circuit(HX, HZ, LZ, rounds, p, q):
    n = HX.shape[1]
    ux, nrbx, ncb = group_layers(HX, q)
    uz, nrbz, _ = group_layers(HZ, q)
    sx, dx = schedule(ux, nrbx, ncb)
    sz, dz = schedule(uz, nrbz, ncb)
    nax, naz = HX.shape[0], HZ.shape[0]
    ax = [n + i for i in range(nax)]
    az = [n + nax + i for i in range(naz)]
    c = stim.Circuit()
    rec, tot = {}, [0]

    def sub(anc, units, slots, depth, kind, r):
        c.append("R", anc)
        c.append("DEPOLARIZE1", anc, p)
        if kind == "x":
            c.append("H", anc)
        for t in range(depth):
            pairs = []
            for (rb, cb, mt), s in zip(units, slots):
                if s != t:
                    continue
                for i in range(q):
                    a = anc[rb * q + i]
                    d = cb * q + mt[i]
                    pairs += ([a, d] if kind == "x" else [d, a])
            if pairs:
                c.append("CX", pairs)
                c.append("DEPOLARIZE2", pairs, p)
        if kind == "x":
            c.append("H", anc)
        c.append("DEPOLARIZE1", anc, p)
        c.append("M", anc)
        for i in range(len(anc)):
            rec[(r, kind, i)] = tot[0] + i
        tot[0] += len(anc)

    # Z-memory: both check types are extracted, but only the Z checks carry
    # detectors (the X outcomes are random on |0...0>).
    for r in range(rounds):
        sub(ax, ux, sx, dx, "x", r)
        sub(az, uz, sz, dz, "z", r)
        for i in range(naz):
            if r == 0:
                c.append("DETECTOR", [stim.target_rec(rec[(0, "z", i)] - tot[0])])
            else:
                c.append("DETECTOR",
                         [stim.target_rec(rec[(r, "z", i)] - tot[0]),
                          stim.target_rec(rec[(r - 1, "z", i)] - tot[0])])
    c.append("DEPOLARIZE1", list(range(n)), p)
    c.append("M", list(range(n)))
    dstart = tot[0]
    tot[0] += n
    for j in range(naz):
        t = [stim.target_rec(dstart + int(qq) - tot[0])
             for qq in np.flatnonzero(HZ[j]).tolist()]
        t.append(stim.target_rec(rec[(rounds - 1, "z", j)] - tot[0]))
        c.append("DETECTOR", t)
    for i, lz in enumerate(LZ):
        c.append("OBSERVABLE_INCLUDE",
                 [stim.target_rec(dstart + int(qq) - tot[0])
                  for qq in np.flatnonzero(lz).tolist()], i)
    return c, dx, dz


def dem_matrices(dem):
    # Expand repeat blocks and apply their detector shifts before reading the
    # instructions; the walk below handles only flat error instructions.
    dem = dem.flattened()
    rows, cols, obs_r, obs_c, pri = [], [], [], [], []
    j = [0]

    def walk(block, reps=1):
        for _ in range(reps):
            for inst in block:
                if inst.type == "error":
                    pr = inst.args_copy()[0]
                    for t in inst.targets_copy():
                        if t.is_relative_detector_id():
                            rows.append(t.val)
                            cols.append(j[0])
                        elif t.is_logical_observable_id():
                            obs_r.append(t.val)
                            obs_c.append(j[0])
                    pri.append(pr)
                    j[0] += 1
                elif inst.type == "repeat":
                    walk(inst.body_copy(), inst.repeat_count)

    walk(dem)
    H = sp.csr_matrix((np.ones(len(rows), np.uint8), (rows, cols)),
                      shape=(dem.num_detectors, j[0]))
    L = sp.csr_matrix((np.ones(len(obs_r), np.uint8), (obs_r, obs_c)),
                      shape=(dem.num_observables, j[0]))
    return H, L, np.array(pri)


def merge_columns(H, L, pri):
    """Merge DEM columns with identical detector and observable support into
    one column with p = p1(1-p2) + p2(1-p1)."""
    Hc, Lc = H.tocsc(), L.tocsc()
    seen = {}
    order = []
    probs = []
    for j in range(H.shape[1]):
        key = (Hc.indices[Hc.indptr[j]:Hc.indptr[j + 1]].tobytes(),
               Lc.indices[Lc.indptr[j]:Lc.indptr[j + 1]].tobytes())
        i = seen.get(key)
        if i is None:
            seen[key] = len(order)
            order.append(j)
            probs.append(pri[j])
        else:
            probs[i] = probs[i] * (1 - pri[j]) + pri[j] * (1 - probs[i])
    sel = np.asarray(order)
    return H[:, sel].tocsr(), L[:, sel].tocsr(), np.asarray(probs)


# --------------------------------------------------------------------------
# GPU BP
# --------------------------------------------------------------------------

def csr_table(M, n):
    """gpu_core_v2.check_table's [C, dH] index table, built from CSR without
    densifying M. Unused slots point at column index n, which the caller pins
    to +LARGE in the LLR vector and to 0 in the hard-decision vector.
    """
    M = M.tocsr()
    deg = np.diff(M.indptr)
    d = int(deg.max()) if len(deg) else 1
    T = np.full((M.shape[0], d), n, dtype=np.int64)
    for c in range(M.shape[0]):
        s, e = M.indptr[c], M.indptr[c + 1]
        if e > s:
            T[c, :e - s] = M.indices[s:e]
    return T


class DemBP:
    def __init__(self, H, pri, iters=BP_ITERS, seg=BP_SEG, alpha=0.8):
        self.dev = torch.device("cuda")
        self.C, self.n = H.shape
        self.seg = int(seg)
        self.n_seg = max(1, -(-int(iters) // self.seg))
        self.alpha = float(alpha)
        self.Htab = torch.as_tensor(csr_table(H, self.n), device=self.dev)
        self.dH = int(self.Htab.shape[1])
        self.Hflat = self.Htab.reshape(-1)
        pr = np.clip(np.asarray(pri, np.float64), 1e-12, 0.5 - 1e-12)
        self.llr = torch.as_tensor(np.log((1.0 - pr) / pr).astype(np.float32),
                                   device=self.dev)

    def batch_for(self, frac=GPU_MEM_FRAC):
        free, _ = torch.cuda.mem_get_info()
        # peak memory per batch element: about twelve float32-sized [C, dH]
        # temporaries (_check_update mixes float32 and int64)
        per = 4 * (12 * self.C * self.dH + 3 * (self.n + 1))
        return max(1, min(256, int(free * frac / per)))

    def decode(self, det):
        B = det.shape[0]
        n, C, dev = self.n, self.C, self.dev
        syn = torch.as_tensor(np.ascontiguousarray(det, np.uint8), device=dev)
        pri = torch.full((B, n + 1), LARGE, device=dev, dtype=torch.float32)
        pri[:, :n] = self.llr
        m_cv = torch.zeros(B, C, self.dH, device=dev, dtype=torch.float32)
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        ehat = torch.zeros(B, n + 1, dtype=torch.uint8, device=dev)
        for _ in range(self.n_seg):
            post, m_cv = bp_minsum_syn(self.Htab, pri, syn, n, iters=self.seg,
                                       alpha=self.alpha, m_cv=m_cv)
            hard = torch.zeros(B, n + 1, dtype=torch.uint8, device=dev)
            hard[:, :n] = (post[:, :n] < 0).to(torch.uint8)
            par = hard[:, self.Hflat].view(B, C, self.dH).sum(-1) % 2
            ok = (par == syn).all(1)
            fresh = ok & (~done)
            if bool(fresh.any()):
                ehat[fresh] = hard[fresh]
                done = done | ok
            if bool(done.all()):
                break
        return ehat[:, :n].cpu().numpy(), done.cpu().numpy()


# --------------------------------------------------------------------------
# CPU OSD, only for the BP stalls
# --------------------------------------------------------------------------

_W = {}


def _init(H, pri):
    _W["dec"] = BpOsdDecoder(H, error_rate=None, error_channel=list(pri),
                             max_iter=BP_ITERS, bp_method="ms",
                             osd_method="osd_0" if OSD_ORDER == 0 else "osd_cs",
                             osd_order=OSD_ORDER)


def _osd(block):
    return np.stack([_W["dec"].decode(s) for s in block]).astype(np.uint8)


def run(name, HX, HZ, d_known, eff_known, mult_known):
    HX = np.asarray(HX, np.uint8) % 2
    HZ = np.asarray(HZ, np.uint8) % 2
    n = HX.shape[1]
    k = int(n - fast_gf2.rank(HX) - fast_gf2.rank(HZ))
    w = int(max(HX.sum(1).max(), HZ.sum(1).max()))
    LZ = np.asarray(logicals(HX, HZ), np.uint8) % 2
    rounds = ROUNDS or d_known
    t0 = time.perf_counter()
    c0, dx, dz = build_circuit(HX, HZ, LZ, rounds, 0.0, Q)
    s0 = c0.compile_detector_sampler().sample(64, separate_observables=True)
    if s0[0].any() or s0[1].any():
        print("   %-26s fires at p=0: circuit construction error" % name, flush=True)
        return None
    c, _, _ = build_circuit(HX, HZ, LZ, rounds, P, Q)
    dem = c.detector_error_model(decompose_errors=False,
                                 approximate_disjoint_errors=True)
    H, L, pri = dem_matrices(dem)
    raw_mech = H.shape[1]
    H, L, pri = merge_columns(H, L, pri)
    Li = L.astype(np.int32)
    bp = DemBP(H, pri) if USE_GPU else None
    B = bp.batch_for() if USE_GPU else 200
    print("   %-26s [[%d,%d,%d]] w=%d mult=%-5d | rounds %d depth X%d+Z%d | "
          "detectors %d mechanisms %d->%d | %s (%.0fs)"
          % (name, n, k, d_known, w, mult_known, rounds, dx, dz,
             dem.num_detectors, raw_mech, H.shape[1],
             ("GPU batch %d" % B) if USE_GPU else ("CPU %d cores" % NPROC),
             time.perf_counter() - t0), flush=True)
    smp = c.compile_detector_sampler()
    t1 = time.perf_counter()
    fails = stalls = 0
    stall_syn, stall_obs = [], []
    done_n = 0
    while done_n < SHOTS:
        b = min(B, SHOTS - done_n)
        det, obs = smp.sample(b, separate_observables=True)
        det = det.astype(np.uint8)
        obs = obs.astype(np.uint8)
        if USE_GPU:
            eh, ok = bp.decode(det)
            # int32 on purpose: L rows have degree far above 255, so a uint8
            # product would wrap before the mod 2
            oh = (np.asarray(Li @ eh.T.astype(np.int32)) % 2).T.astype(np.uint8)
            fails += int(((oh[ok] != obs[ok]).any(1)).sum())
        else:
            ok = np.zeros(b, bool)      # every shot goes to BP+OSD
        if (~ok).any():
            stall_syn.append(det[~ok])
            stall_obs.append(obs[~ok])
            stalls += int((~ok).sum())
        done_n += b
    t2 = time.perf_counter()
    if USE_GPU:
        print("        GPU BP %d shots in %.0fs, %d stalled shots handed to CPU OSD"
              % (SHOTS, t2 - t1, stalls), flush=True)
    if stalls:
        S = np.concatenate(stall_syn)
        O = np.concatenate(stall_obs)
        nb = max(1, len(S) // NPROC + 1)
        blocks = [S[i:i + nb] for i in range(0, len(S), nb)]
        with mp.Pool(min(NPROC, len(blocks)), initializer=_init,
                     initargs=(H, pri)) as pool:
            E = np.concatenate(pool.map(_osd, blocks))
        OH = (np.asarray(Li @ E.T.astype(np.int32)) % 2).T.astype(np.uint8)
        fails += int(((OH != O).any(1)).sum())
        print("        BP+OSD finished %d shots in %.0fs" % (stalls, time.perf_counter() - t2),
              flush=True)
    pl = fails / float(SHOTS)
    se = np.sqrt(max(pl, 1e-9) * (1 - pl) / SHOTS)
    print("      pL = %.4g  +/- %.2g   (%d/%d failed, total %.0fs)"
          % (pl, 1.96 * se, fails, SHOTS, time.perf_counter() - t1), flush=True)
    return dict(name=name, n=n, k=k, d=d_known, w=w, eff=eff_known,
                mult=mult_known, rounds=rounds, depth=dx + dz,
                detectors=int(dem.num_detectors), mech=int(H.shape[1]),
                stalls=stalls, pL=pl, err=1.96 * se, fails=fails, shots=SHOTS)


def main():
    print("circuit level pL,  p=%g,  %d shots,  R=d,  GPU BP %d iterations + CPU OSD-%d fallback"
          % (P, SHOTS, BP_ITERS, OSD_ORDER), flush=True)
    print("layer schedule from greedy matchings, no idling noise, separate X/Z sub-rounds -- convention of 2607.28795", flush=True)
    print("GPU: %s\n" % (torch.cuda.get_device_name(0)
                         if torch.cuda.is_available() else "none!"), flush=True)
    print("   Four codes with the same n, w and d; only k and mult differ. "
          "The two [[600,124,24]] codes both have proxy score 119.04; their mult differs 3.2x\n", flush=True)
    T = []
    for f, nm, d, e, mu in (
            ("/u/zdliu/lp/jobs/v126_best.npz", "[[600,126,24]] 120.96", 24, 120.96, 1759),
            ("/u/zdliu/lp/jobs/g120_pin_best_id6.npz", "[[600,125,24]] 120.00", 24, 120.00, 1055),
            ("/u/zdliu/lp/jobs/g120_pin_best_id7.npz", "[[600,124,24]] 119.04 A", 24, 119.04, 2887),
            ("/u/zdliu/lp/jobs/g120_pin_best_id1.npz", "[[600,124,24]] 119.04 B", 24, 119.04, 906)):
        if not os.path.exists(f):
            continue
        z = np.load(f)
        try:
            r = run(nm, z["HX"], z["HZ"], d, e, mu)
        except Exception as ex:
            import traceback
            traceback.print_exc()
            print("   %-26s failed: %s" % (nm, str(ex)[:110]), flush=True)
            continue
        if r:
            T.append(r)
            json.dump(T, open("/u/zdliu/lp/jobs/circ.json", "w"), indent=1)
    if len(T) < 2:
        return
    print("\n===== results =====", flush=True)
    by_pl = sorted(T, key=lambda r: r["pL"])
    print("   %-26s %-9s %-7s %-13s %s" % ("code", "k*d^2/n", "mult", "circuit pL", "pL rank"),
          flush=True)
    for r in sorted(T, key=lambda r: -r["eff"]):
        print("   %-26s %-9.2f %-7d %-13.4g #%d"
              % (r["name"], r["eff"], r["mult"], r["pL"], by_pl.index(r) + 1), flush=True)
    a = [r for r in T if r["eff"] == 119.04]
    if len(a) == 2:
        x, y = a
        sep = abs(x["pL"] - y["pL"]) > (x["err"] + y["err"])
        print("\n   the two codes with equal proxy score: pL = %.4g vs %.4g   %s"
              % (x["pL"], y["pL"],
                 "statistically separable -> the proxy misses the mult dimension"
                 if sep else "not separable -> the proxy suffices at this precision"), flush=True)
    same = [r["name"] for r in sorted(T, key=lambda r: -r["eff"])] == \
           [r["name"] for r in by_pl]
    print("   overall ranking identical: %s" % ("yes" if same else "no"), flush=True)


if __name__ == "__main__":
    main()
