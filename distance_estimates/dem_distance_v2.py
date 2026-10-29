"""Circuit-level distance d_circ from the detector error model (DEM).

d_circ = min over the two memory bases of the minimum weight of a DEM error
set e with H_dem e = 0 and L_dem e != 0, for the greedy extraction schedule of
circ_1side.build_circuit. Each basis takes the minimum over three tiers:
  1. exact w <= 2 column scan;
  2. constructive embedding: the code-capacity minimum-weight witness is
     mapped onto final-window DEM columns and re-verified there;
  3. optional single-side GPU search on (H_dem, L_dem); a budget <= 0
     disables only this tier.
None means no tier produced a verified value. Env: CM_ROUNDS (DEM rounds,
default 2), CM_P (default 0.004), CM_DCIRC_BUDGET, CM_WIT_BUDGET,
CM_WIT_ALL_CAP.
"""
import multiprocessing
import time
import os

import numpy as np

DEM_ROUNDS = int(os.environ.get("CM_ROUNDS", "2"))
DEM_P = float(os.environ.get("CM_P", "0.004"))
DCIRC_BUDGET = float(os.environ.get("CM_DCIRC_BUDGET", "60"))
WIT_BUDGET = float(os.environ.get("CM_WIT_BUDGET", "2"))
# maximum number of minimum-weight witness supports returned per side (wx_all / wz_all)
WIT_ALL_CAP = int(os.environ.get("CM_WIT_ALL_CAP", "300"))


def _d2s_wit_proc(q, payload):
    """Code-capacity d plus ONE min-weight witness support per side
    (data-qubit index lists), for constructive embedding. Child body."""
    try:
        import numpy as _np
        from estimator_gpu_v2 import estimate_d_gpu
        HX, HZ, LX, LZ, budget = payload
        r = estimate_d_gpu(HX, HZ, LX, LZ, budget_seconds=budget, seed=0)
        if int(r["d"]) > HX.shape[1]:
            q.put("ERR:sentinel:no witness found")
            return

        def sup(side):
            ws = r["witnesses"][side]
            if not ws:
                return None
            v = min(ws, key=lambda w: int(w.sum()))
            return [int(i) for i in _np.flatnonzero(_np.asarray(v) % 2)]

        def sup_all(side):
            # every verified operator the search kept at that side's minimum
            # weight (the estimator already filtered to weight == d_side)
            ws = r["witnesses"][side]
            return [[int(i) for i in _np.flatnonzero(_np.asarray(v) % 2)]
                    for v in ws[:WIT_ALL_CAP]]
        q.put({"d": int(r["d"]), "dx": int(r["dx"]), "dz": int(r["dz"]),
               "wx": sup("x"), "wz": sup("z"),
               "wx_all": sup_all("x"), "wz_all": sup_all("z"),
               "spectrum": r.get("spectrum"), "multiplicity": r.get("multiplicity")})
    except Exception as e:
        q.put("ERR:%s:%s" % (type(e).__name__, str(e)[:150]))


def _d2s_batch_proc(q, payload):
    """Code-capacity screen for SEVERAL codes in one child (one torch
    import for the whole batch). Each item -> the _d2s_wit_proc-style
    dict, or "ERR:..." in its slot. Child body."""
    out = []
    try:
        import numpy as _np
        from estimator_gpu_v2 import estimate_d_gpu
        items, budget = payload
        for HX, HZ, LX, LZ in items:
            try:
                r = estimate_d_gpu(HX, HZ, LX, LZ,
                                   budget_seconds=budget, seed=0)
                if int(r["d"]) > HX.shape[1]:
                    out.append("ERR:sentinel:no witness found")
                    continue

                def sup(side, r=r):
                    ws = r["witnesses"][side]
                    if not ws:
                        return None
                    v = min(ws, key=lambda w: int(w.sum()))
                    return [int(i) for i in
                            _np.flatnonzero(_np.asarray(v) % 2)]

                def sup_all(side, r=r):
                    ws = r["witnesses"][side]
                    return [[int(i) for i in
                             _np.flatnonzero(_np.asarray(v) % 2)]
                            for v in ws[:WIT_ALL_CAP]]
                out.append({"d": int(r["d"]), "dx": int(r["dx"]),
                            "dz": int(r["dz"]),
                            "wx": sup("x"), "wz": sup("z"),
                            "wx_all": sup_all("x"), "wz_all": sup_all("z"),
                            "spectrum": r.get("spectrum"),
                            "multiplicity": r.get("multiplicity")})
            except Exception as e:
                out.append("ERR:%s:%s" % (type(e).__name__, str(e)[:150]))
        q.put(out)
    except Exception as e:
        q.put("ERR:%s:%s" % (type(e).__name__, str(e)[:150]))


def d2s_batch(items, budget=None, timeout_extra=240):
    """[(HX, HZ, LX, LZ), ...] -> list of wit-dicts / "ERR" / None."""
    budget = WIT_BUDGET if budget is None else budget
    res = _run_child(_d2s_batch_proc, (items, budget),
                     len(items) * (budget + 3) + timeout_extra)
    if not isinstance(res, list):
        return [None] * len(items)
    return res


# DEM search (tier 3): one search_side_gpu call per DEM with a single all-zero
# stabilizer row, so the coset descent is a no-op, and batches of 8 classes on
# DEMs wider than 12000 columns, since the budget is checked only between
# batches.
DEM_LAM = int(os.environ.get("CM_DEM_LAM", "48"))
DEM_K_MARGIN = int(os.environ.get("CM_DEM_K_MARGIN", "256"))


def _dcirc_proc(q, payload):
    """DEM distance, SEVERAL (H, L) items in one child (one torch import).
    use_graphs=False is mandatory on DEM-width matrices. Child body."""
    out = []
    try:
        import numpy as _np
        import estimator_gpu_v2 as E
        from rref_panel import batched_rref_panel
        items, budget = payload
        import fast_gf2 as _fg
        for H, L in items:
            try:
                rows, cols = H.shape[0] + 1, H.shape[1]
                # full-width RREF; small batches keep wide DEMs within the budget
                E.RREF_FN = batched_rref_panel
                chunk = 128 if cols <= 12000 else 8
                S0 = _np.zeros((1, cols), _np.uint8)
                d, _w = E.search_side_gpu(
                    H, L, S0, float(budget), _np.random.default_rng(0), "cuda",
                    chunk=chunk, bp_iters=24, lam=DEM_LAM, pair_descent=False,
                    target_d=None, use_graphs=False, shape_bucket=None)
                out.append(int(d))
            except Exception as e:
                out.append("ERR:%s:%s" % (type(e).__name__, str(e)[:200]))
        q.put(out)
    except Exception as e:
        q.put("ERR:%s:%s" % (type(e).__name__, str(e)[:200]))


def _run_child(target, payload, timeout):
    ctx = multiprocessing.get_context("spawn")
    qch = ctx.Queue()
    pr = ctx.Process(target=target, args=(qch, payload))
    pr.start()
    # Read the result before joining: a child whose result exceeds the pipe
    # buffer cannot exit until the parent drains the queue.
    res = None
    try:
        res = qch.get(timeout=timeout)
    except Exception:
        res = None
    pr.join(timeout=30 if res is not None else 1)
    if pr.is_alive():
        pr.terminate()
        pr.join(5)
        if pr.is_alive():
            pr.kill()
            pr.join()
    return res


def _strip(H, L):
    H = np.asarray(H, np.uint8) % 2
    L = np.asarray(L, np.uint8) % 2
    keep = (H.sum(0) + L.sum(0)) > 0
    return H[:, keep], L[:, keep]


def w2_exact(H, L):
    """Exact d_circ<=2 detector: weight-1 = zero H-column with nonzero
    L-column; weight-2 = two identical H-columns with different L."""
    zero_h = np.flatnonzero(H.sum(0) == 0)
    if any(L[:, j].any() for j in zero_h):
        return 1
    Hb = np.packbits(H.T, axis=1)
    Lb = np.packbits(L.T, axis=1)
    seen = {}
    for j in range(H.shape[1]):
        kk = Hb[j].tobytes()
        for i in seen.get(kk, []):
            if (Lb[i] != Lb[j]).any():
                return 2
        seen.setdefault(kk, []).append(j)
    return None


def basis_dem(HX, HZ, LX, LZ, basis, q, rounds, p):
    """Greedy-schedule memory circuit for one basis -> stripped (H, L)."""
    import circ_1side as C1
    if basis == "z":
        Am, Bm, Lg = HX, HZ, LZ
    else:
        Am, Bm, Lg = HZ, HX, LX
    o_ = C1.build_circuit(np.asarray(Am, np.uint8), np.asarray(Bm, np.uint8),
                          np.asarray(Lg, np.uint8) % 2, rounds, p, q)
    circ = o_[0] if isinstance(o_, tuple) else o_
    dem = circ.detector_error_model(decompose_errors=False,
                                    approximate_disjoint_errors=True)
    H, L, _pri = C1.dem_matrices(dem)
    if hasattr(H, "toarray"):
        H = H.toarray()
    if hasattr(L, "toarray"):
        L = L.toarray()
    return _strip(H, L)


def embed_weight(basis, H, L, HX, HZ, LX, LZ, wits):
    """Constructive cap: map the relevant code-level witness support
    (X-type logicals govern the Z-basis experiment and vice versa) onto
    final-window DEM columns by column signature, then re-verify the sum
    as a real circuit-level witness. Returns its weight, or None."""
    wit = (wits or {}).get("wx" if basis == "z" else "wz")
    if not wit:
        return None
    Bm = HZ if basis == "z" else HX
    Lg = LZ if basis == "z" else LX
    Bm = np.asarray(Bm, np.uint8)
    Lg = np.asarray(Lg, np.uint8) % 2
    mB = Bm.shape[0]
    if H.shape[0] < mB:
        return None
    sig = {}
    Hb = np.packbits(H.T, axis=1)
    Lb = np.packbits(L.T, axis=1)
    for j in range(H.shape[1]):
        sig.setdefault((Hb[j].tobytes(), Lb[j].tobytes()), j)
    cols = []
    for i in wit:
        want_h = np.zeros(H.shape[0], np.uint8)
        want_h[-mB:] = Bm[:, i]
        want_l = np.asarray(Lg[:, i], np.uint8)
        key = (np.packbits(want_h).tobytes(), np.packbits(want_l).tobytes())
        j = sig.get(key)
        if j is None:
            return None
        cols.append(j)
    e = np.zeros(H.shape[1], np.uint8)
    e[cols] = 1
    if ((H @ e.astype(np.int64)) % 2).any() \
            or not ((L @ e.astype(np.int64)) % 2).any():
        return None
    return len(wit)


def dcirc_batch(cands, q, budget=None, rounds=None, p=None, wits_list=None):
    """dcirc() over several candidates with one estimator child for all
    their DEMs; per-item budgets and tiers are those of dcirc().
    cands = [(HX, HZ, LX, LZ), ...]; wits_list aligns with cands (entries
    may be None). Returns a list of dcirc()-style dicts."""
    budget = DCIRC_BUDGET if budget is None else budget
    rounds = DEM_ROUNDS if rounds is None else rounds
    p = DEM_P if p is None else p
    wits_list = wits_list or [None] * len(cands)
    outs = []
    todo = []                      # (cand_index, basis)
    dems = {}
    for ci, ((HX, HZ, LX, LZ), wits) in enumerate(zip(cands, wits_list)):
        if not isinstance(wits, dict):
            wits = _run_child(_d2s_wit_proc, (HX, HZ, LX, LZ, WIT_BUDGET),
                              200)
            if not isinstance(wits, dict):
                wits = None
        out = {"code_d": (wits or {}).get("d"), "z": {}, "x": {},
               "d_circ": None}
        for basis in ("z", "x"):
            H, L = basis_dem(HX, HZ, LX, LZ, basis, q, rounds, p)
            dems[(ci, basis)] = (H, L)
            w2 = w2_exact(H, L)
            emb = embed_weight(basis, H, L, HX, HZ, LX, LZ, wits)
            out[basis] = {"w2": w2, "emb": emb, "reading": None}
            if w2 is None:
                todo.append((ci, basis))
        outs.append(out)
    est_status = {}
    if todo and budget > 0:
        t_est = time.time()
        res = _run_child(
            _dcirc_proc,
            ([dems[key] for key in todo], budget),
            len(todo) * budget + 300)
        secs = round(time.time() - t_est)
        if isinstance(res, list):
            for (ci, basis), r in zip(todo, res):
                if isinstance(r, int) and r <= dems[(ci, basis)][0].shape[1]:
                    outs[ci][basis]["reading"] = r
                    est_status[ci] = "ok"
                elif isinstance(r, int):
                    est_status.setdefault(ci, "no_witness")
                else:
                    est_status[ci] = "error:%s" % str(r)[:80]
        elif res is None:
            for ci, _b in todo:
                est_status[ci] = "timeout(%ds child limit)" % (len(todo) * budget + 300)
        else:
            for ci, _b in todo:
                est_status[ci] = "error:%s" % str(res)[:80]
        for ci in est_status:
            est_status[ci] += " %ds" % secs
    for ci, out in enumerate(outs):
        out["est_status"] = ("disabled:embedding_w2" if budget <= 0 else
                             est_status.get(ci, "skipped"))
    for out in outs:
        vals = []
        for basis in ("z", "x"):
            tier = [v for v in (out[basis]["w2"], out[basis]["emb"],
                                out[basis]["reading"]) if v is not None]
            out[basis]["value"] = min(tier) if tier else None
            if out[basis]["value"] is not None:
                vals.append(out[basis]["value"])
        out["d_circ"] = min(vals) if vals else None
    return outs


def dcirc(HX, HZ, LX, LZ, q, budget=None, rounds=None, p=None, wits=None):
    """Full measurement. Returns a dict:
      d_circ  min over bases, or None if no tier produced a value
      z / x   per-basis dicts {value, w2, emb, reading}
      code_d  the 2 s code-capacity d used for the embedding witness
    Pass a d2s_batch()/_d2s_wit_proc result as `wits` to reuse an
    already-measured code-capacity witness (skips one child spawn)."""
    budget = DCIRC_BUDGET if budget is None else budget
    rounds = DEM_ROUNDS if rounds is None else rounds
    p = DEM_P if p is None else p
    if not isinstance(wits, dict):
        wits = _run_child(_d2s_wit_proc, (HX, HZ, LX, LZ, WIT_BUDGET), 200)
        if not isinstance(wits, dict):
            wits = None
    out = {"code_d": (wits or {}).get("d"), "z": {}, "x": {},
           "d_circ": None}
    todo = []
    dems = {}
    for basis in ("z", "x"):
        H, L = basis_dem(HX, HZ, LX, LZ, basis, q, rounds, p)
        dems[basis] = (H, L)
        w2 = w2_exact(H, L)
        emb = embed_weight(basis, H, L, HX, HZ, LX, LZ, wits)
        out[basis] = {"w2": w2, "emb": emb, "reading": None}
        if w2 is None:
            todo.append(basis)
    if todo and budget > 0:
        res = _run_child(
            _dcirc_proc,
            ([dems[b] for b in todo], budget),
            len(todo) * budget + 240)
        if isinstance(res, list):
            for b, r in zip(todo, res):
                if isinstance(r, int) and r <= dems[b][0].shape[1]:
                    out[b]["reading"] = r
    if budget <= 0:
        out["est_status"] = "disabled:embedding_w2"
    vals = []
    for basis in ("z", "x"):
        tier = [v for v in (out[basis]["w2"], out[basis]["emb"],
                            out[basis]["reading"]) if v is not None]
        out[basis]["value"] = min(tier) if tier else None
        if out[basis]["value"] is not None:
            vals.append(out[basis]["value"])
    out["d_circ"] = min(vals) if vals else None
    return out
