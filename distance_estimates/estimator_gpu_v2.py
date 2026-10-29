"""GPU directed-decoder distance estimator for CSS codes.

For logical class i, BP+OSD decodes the syndrome (0,...,0,1) against the
stacked matrix [H ; L_i], so every solution is a logical representative of
class i. Batches run over (class, restart) pairs; solutions are polished by
stabilizer-coset descent and re-verified on the host with numpy.

    estimate_d_gpu(Hx, Hz, Lx, Lz, budget_seconds=..., seed=..., device='cuda')
      -> dict(d, dx, dz, multiplicity, method, witnesses, spectrum, time)

RREF_FN is the RREF used on the eager path. use_graphs ("auto" | True |
False) replays bp_minsum, batched_rref and osd_solutions as CUDA graphs on
CUDA (never on cpu). shape_bucket (env CSSDIST_SHAPE_BUCKET, default 16; <= 0
disables) rounds the BP table widths up to a multiple of the bucket and pads
the last phase-1 batch, so that codes of similar shape share one graph
capture. target_d stops the search once d <= target_d.
"""
import os
import time

import numpy as np
import torch

from gpu_core_v2 import (check_table, logical_table, bp_minsum, batched_rref,
                         osd_solutions, gpu_descent, gpu_descent_pairs, LARGE)
from gpu_graphs import GraphedBP, GraphedOSD, GraphedRREF
from rref_panel import batched_rref_panel

# Swappable RREF entry point (eager path); run_batch resolves this name at
# call time. batched_rref gives identical outputs, one column at a time.
RREF_FN = batched_rref_panel


def _verify_np(v, H, L):
    v = np.asarray(v, dtype=np.uint8) % 2
    if v.sum() == 0:
        return False
    if np.any((H @ v) % 2):
        return False
    return bool(np.any((L @ v) % 2))


# Module-level graph wrappers: their caches are shape-keyed and persist across
# estimate_d_gpu calls, so each shape is captured once. Each graph uses a
# private memory pool (rref replays before osd is captured, which a shared
# pool does not allow). Constructing them does not touch CUDA.
_G_BP, _G_RREF, _G_OSD = GraphedBP(), GraphedRREF(), GraphedOSD()


def clear_graph_cache():
    """Drop all captured graphs (frees their pools on next GC)."""
    for g in (_G_BP, _G_RREF, _G_OSD):
        g._cache.clear()


def _resolve_shape_bucket(shape_bucket):
    """None -> env CSSDIST_SHAPE_BUCKET -> default 16. A value <= 0
    disables bucketing (natural table widths, no batch padding)."""
    if shape_bucket is None:
        try:
            shape_bucket = int(os.environ.get("CSSDIST_SHAPE_BUCKET", "16"))
        except ValueError:
            shape_bucket = 16
    return max(0, int(shape_bucket))


def _resolve_use_graphs(use_graphs, dev):
    """"auto" -> CUDA yes, CPU no. Explicit True is still forced off on CPU
    (CUDA graphs cannot exist there); False stays False everywhere."""
    if dev.type != "cuda":
        return False
    if use_graphs == "auto":
        return True
    return bool(use_graphs)


def search_side_gpu(H, L, S, budget_s, rng, device, chunk=128, bp_iters=24,
                    lam=48, pair_descent=True, verbose=False, target_d=None,
                    use_graphs="auto", phase1_only=False, shape_bucket=None):
    """Min-weight vector in ker(H) anticommuting with some row of L.

    S: stabilizer rows (rows of the complementary CSS matrix) used for the
    coset descent polish. Returns (best_weight, {bytes: vec}).
    target_d: when not None, return as soon as best_weight <= target_d.
    use_graphs: see the module docstring; resolved via _resolve_use_graphs.
    shape_bucket: see the module docstring; resolved via
    _resolve_shape_bucket (None -> env -> 16; <= 0 disables).
    """
    t0 = time.time()
    m, n = H.shape
    k = L.shape[0]
    dev = torch.device(device)

    graphed = _resolve_use_graphs(use_graphs, dev)
    if graphed:
        g_bp, g_rref, g_osd = _G_BP, _G_RREF, _G_OSD

    bucket = _resolve_shape_bucket(shape_bucket)
    if bucket:
        # Round the dense-table widths UP to the next multiple of `bucket`
        # so candidates whose max row weights differ (but land in the same
        # bucket) produce identically shaped Htab/Ltab and hit the same
        # CUDA-graph cache entry. Padded slots point at the inert dummy
        # column n; see check_table's docstring for why that is bit-inert.
        padH = -(-int(H.sum(1).max()) // bucket) * bucket
        padL = -(-int(L.sum(1).max()) // bucket) * bucket
    else:
        padH = padL = None
    Htab = torch.as_tensor(check_table(H, pad_to=padH), device=dev)
    Ltab = torch.as_tensor(logical_table(L, pad_to=padL), device=dev)
    Ht = torch.as_tensor(H.astype(np.uint8), device=dev)
    St = torch.as_tensor(S.astype(np.float32), device=dev)
    Gs = St @ St.t()
    Hf = Ht.to(torch.float32)
    Lf = torch.as_tensor(L.astype(np.float32), device=dev)

    best_w, winners = n + 1, {}
    class_best = np.full(k, n + 1, dtype=np.int64)

    def harvest(V, cls):
        """V: [B, n] uint8 on device. Keep verified logicals, update bests."""
        nonlocal best_w
        Vf = V.to(torch.float32)
        ok_ker = (torch.remainder(Vf @ Hf.t(), 2.0).abs().sum(1) == 0)
        ok_log = (torch.remainder(Vf @ Lf.t(), 2.0).abs().sum(1) > 0)
        ok = ok_ker & ok_log
        w = V.sum(1).to(torch.long)
        w = torch.where(ok, w, torch.full_like(w, n + 1))
        wc = w.cpu().numpy()
        for j, ci in enumerate(cls):
            if wc[j] <= n:
                class_best[ci] = min(class_best[ci], int(wc[j]))
        mn = int(wc.min()) if len(wc) else n + 1
        if mn > n:
            return
        if mn < best_w:
            best_w = mn
            winners.clear()
        if mn == best_w:
            Vc = V.cpu().numpy()
            for j in np.flatnonzero(wc == mn):
                v = Vc[j].astype(np.uint8)
                winners[v.tobytes()] = v
        return

    def run_batch(cls, priors):
        """cls: list of class indices; priors: [B, n] numpy float p-values."""
        B = len(cls)
        pri = torch.full((B, n + 1), LARGE, device=dev, dtype=torch.float32)
        p = torch.as_tensor(priors, device=dev, dtype=torch.float32)
        pri[:, :n] = torch.log((1 - p) / p)
        Ltab_b = Ltab[torch.as_tensor(cls, device=dev)]

        # Graphed outputs live in static buffers valid until the next
        # replay; each is consumed immediately below (argsort / osd input
        # copies / descent), before any wrapper replays again.
        post = (g_bp(Htab, Ltab_b, pri, n, iters=bp_iters) if graphed
                else bp_minsum(Htab, Ltab_b, pri, n, iters=bp_iters))
        order = torch.argsort(post[:, :n], dim=1)          # ascending LLR

        M = torch.empty(B, m + 1, n, dtype=torch.uint8, device=dev)
        M[:, :m, :] = Ht.unsqueeze(0)
        M[:, m, :] = torch.as_tensor(L[cls].astype(np.uint8), device=dev)
        syn = torch.zeros(B, m + 1, dtype=torch.uint8, device=dev)
        syn[:, m] = 1

        if graphed:
            packed, rank, pcol, ispiv = g_rref(M, order, syn, n)
            _, V = g_osd(packed, rank, pcol, ispiv, order, n, m + 1,
                         lam=lam)
        else:
            packed, rank, pcol, ispiv = RREF_FN(M, order, syn, n)
            _, V = osd_solutions(packed, rank, pcol, ispiv, order, n, m + 1,
                                 lam=lam)
        V = (gpu_descent_pairs(V, St, G=Gs) if pair_descent
             else gpu_descent(V, St))
        harvest(V, cls)

    def hit_target():
        return target_d is not None and best_w <= target_d

    # phase 1: one full scan over all classes with uniform priors
    for a in range(0, k, chunk):
        cls = list(range(a, min(a + chunk, k)))
        if bucket and len(cls) < chunk:
            # Pad the last partial batch to a full chunk by repeating class 0,
            # so its batch size (part of every graph cache key) does not
            # depend on k. Repeated rows do not change the result: harvest
            # dedups winners and class_best takes a minimum.
            cls = cls + [0] * (chunk - len(cls))
        run_batch(cls, np.full((len(cls), n), 0.05, dtype=np.float32))
        if hit_target():
            return best_w, winners
        if time.time() - t0 > 0.5 * budget_s:
            break
    if phase1_only:
        # scan mode (race scheduling): one cheap look at this side, no
        # restart phase -- the caller decides where the budget goes next.
        return best_w, winners

    # phase 2: randomized-prior restarts, weighted to the lightest classes
    while time.time() - t0 < budget_s:
        focus = np.argsort(class_best)[:max(8, k // 6)]
        cls = [int(focus[rng.integers(len(focus))]) if rng.random() < 0.7
               else int(rng.integers(k)) for _ in range(chunk)]
        pr = np.exp(rng.uniform(np.log(0.003), np.log(0.35),
                                size=(chunk, n))).astype(np.float32)
        run_batch(cls, pr)
        if hit_target():
            break

    return best_w, winners


def estimate_d_gpu(Hx, Hz, Lx, Lz, budget_seconds=30, seed=0, device="cuda",
                   chunk=128, bp_iters=24, lam=48, pair_descent=True,
                   target_d=None, use_graphs="auto", shape_bucket=None):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    Hx = np.asarray(Hx, dtype=np.uint8) % 2
    Hz = np.asarray(Hz, dtype=np.uint8) % 2
    Lx = np.asarray(Lx, dtype=np.uint8) % 2
    Lz = np.asarray(Lz, dtype=np.uint8) % 2

    kw = dict(chunk=chunk, bp_iters=bp_iters, lam=lam,
              pair_descent=pair_descent, target_d=target_d,
              use_graphs=use_graphs, shape_bucket=shape_bucket)
    sent = Hx.shape[1] + 1        # "nothing found" sentinel for either side
    dz, wz = sent, {}
    dx, wx = sent, {}

    if target_d is not None:
        # target_d stops on d = min(dx, dz), and either side may reach it:
        # first run one phase-1 pass on each side and stop at the first
        # verified hit; the full-budget round below runs only if neither
        # scan reaches the target.
        scan = max(1.0, min(3.0, budget_seconds / 8))
        dz, wz = search_side_gpu(Hx, Lx, Hz, scan, rng, device,
                                 phase1_only=True, **kw)
        if dz > target_d:
            dx, wx = search_side_gpu(Hz, Lz, Hx, scan, rng, device,
                                     phase1_only=True, **kw)
    # Floor each side's remaining budget at min(2 s, budget_seconds / 2).
    floor = min(2.0, budget_seconds / 2)
    if target_d is None or min(dx, dz) > target_d:
        rem = max(floor, budget_seconds - (time.time() - t0))
        dz2, wz2 = search_side_gpu(Hx, Lx, Hz, rem / 2, rng, device, **kw)
        if dz2 < dz:
            dz, wz = dz2, wz2
        elif dz2 == dz:
            wz.update(wz2)
        if target_d is None or min(dx, dz) > target_d:
            rem = max(floor, budget_seconds - (time.time() - t0))
            dx2, wx2 = search_side_gpu(Hz, Lz, Hx, rem, rng, device, **kw)
            if dx2 < dx:
                dx, wx = dx2, wx2
            elif dx2 == dx:
                wx.update(wx2)

    # host-side check: drop any witness that fails the membership test
    wz = {kk: v for kk, v in wz.items() if _verify_np(v, Hx, Lx)}
    wx = {kk: v for kk, v in wx.items() if _verify_np(v, Hz, Lz)}
    dz = min([int(v.sum()) for v in wz.values()], default=Hx.shape[1] + 1)
    dx = min([int(v.sum()) for v in wx.values()], default=Hz.shape[1] + 1)
    # Witness counts per weight, taken before the minimum-weight filter below
    # (each side holds only the witnesses at its lightest weight found).
    from collections import Counter as _Counter
    spec_z = _Counter(int(v.sum()) for v in wz.values())
    spec_x = _Counter(int(v.sum()) for v in wx.values())

    wz = {kk: v for kk, v in wz.items() if int(v.sum()) == dz}
    wx = {kk: v for kk, v in wx.items() if int(v.sum()) == dx}

    d = min(dx, dz)
    mult = (len(wz) if dz == d else 0) + (len(wx) if dx == d else 0)
    spectrum = {}
    for c in (spec_z, spec_x):
        for w_, cnt in c.items():
            spectrum[w_] = spectrum.get(w_, 0) + cnt
    return {
        "d": int(d), "dx": int(dx), "dz": int(dz),
        "multiplicity": int(mult), "method": "gpu-directed-bp-osd-v2",
        "witnesses": {"z": list(wz.values()), "x": list(wx.values())},
        # weight -> number of verified witnesses found at that weight
        "spectrum": {int(k_): int(v_) for k_, v_ in sorted(spectrum.items())},
        "time": round(time.time() - t0, 2),
    }
