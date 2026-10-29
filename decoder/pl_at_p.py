"""Logical error rate of a few codes at one physical error rate, in both memory
bases, with the one-sided memory circuit of circ_1side.build_circuit (X memory:
same builder with HX and HZ swapped) and a relay-BP decoder cascade. Reports
shot and per-logical failure counts with exact Clopper-Pearson 95% intervals;
partial results are written after every chunk, so rerunning resumes.

  PL_P=0.003 PL_ROUNDS=6 PL_SHOTS=50000 PL_BASES=Z,X python pl_at_p.py codes.json out.json
codes.json = [{"name":..., "npz":..., "q":..., "d":...}, ...]
"""
import json
import math
import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault("LAYOUT_SQFAST_PATH", "/u/zdliu/lp/dopt/agentA")
for p in ("/u/zdliu/lp/lps_graftgate", "/u/zdliu/lp/dopt/bdirect_v2"):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

P = float(os.environ.get("PL_P", "0.003"))
ROUNDS = int(os.environ.get("PL_ROUNDS", "6"))
SHOTS = int(os.environ.get("PL_SHOTS", "50000"))
WORKERS = int(os.environ.get("PL_WORKERS", str(mp.cpu_count())))
BASES = os.environ.get("PL_BASES", "Z,X").split(",")
CHUNK = int(os.environ.get("PL_CHUNK", "500"))
# Three-stage cascade: relay-BP (gamma0, Tm, R, T'm, S) = (0.1, 200, 20, 100, 1),
# a second relay-BP (0.1, 500, 200, 200, 1) on the shots the first leaves
# unsatisfied, and BP-OSD (min-sum, OSD-CS) on what is still unsatisfied.
# ms_scaling_factor=0 selects ldpc's adaptive scaling 1 - 2^-it.
GAMMA_INT = (-0.24, 0.6)
STAGE1 = dict(gamma0=0.1, pre_iter=200, num_sets=20, set_max_iter=100, stop_nconv=1, gamma_dist_interval=GAMMA_INT)
STAGE2 = dict(gamma0=0.1, pre_iter=500, num_sets=200, set_max_iter=200, stop_nconv=1, gamma_dist_interval=GAMMA_INT)
OSD_ITERS = int(os.environ.get("PL_OSD_ITERS", "300"))
OSD_ORDER = int(os.environ.get("PL_OSD_ORDER", "7"))
CASCADE = os.environ.get("PL_CASCADE", "1") == "1"


def clopper_pearson(k, n, alpha=0.05):
    """Exact two-sided binomial interval."""
    from scipy.stats import beta
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.isf(alpha / 2, k + 1, n - k))
    return lo, hi


def build(npz, q, basis, cornucopia=None):
    import circ_1side as C1
    from fast_gf2 import logicals
    z = np.load(npz)
    hx = np.asarray(z["HX"], np.uint8) % 2
    hz = np.asarray(z["HZ"], np.uint8) % 2
    n = hx.shape[1]
    if basis == "Z":
        lobs = np.asarray(logicals(hx, hz), np.uint8) % 2      # logical Z
    else:
        lobs = np.asarray(logicals(hz, hx), np.uint8) % 2      # logical X
    if cornucopia:
        # schedule and noise channels of cornucopia_circuit.py
        import cornucopia_circuit as CC
        kw = dict(det_mode=os.environ.get("PL_DET", "all"), noise=os.environ.get("PL_NOISE", "paper"))
        c0 = CC.build_circuit(cornucopia, basis, ROUNDS, 0.0, hx, hz, lobs, **kw)
        s0 = c0.compile_detector_sampler().sample(64, separate_observables=True)
        assert not (s0[0].any() or s0[1].any()), "p=0 gate failed"
        ncx = sum(len(inst.targets_copy()) // 2 for inst in c0.flattened() if inst.name == "CX")
        assert ncx == int(hx.sum() + hz.sum()) * ROUNDS, "CNOT count mismatch"
        return CC.build_circuit(cornucopia, basis, ROUNDS, P, hx, hz, lobs, **kw), n, lobs.shape[0]
    assert n % q == 0 and hx.shape[0] % q == 0 and hz.shape[0] % q == 0, "n or check count not divisible by q=%d" % q
    h1, h2 = (hx, hz) if basis == "Z" else (hz, hx)
    c0, _, _ = C1.build_circuit(h1, h2, lobs, ROUNDS, 0.0, q)
    s0 = c0.compile_detector_sampler().sample(64, separate_observables=True)
    assert not (s0[0].any() or s0[1].any()), "p=0 gate failed"
    ncx = sum(len(inst.targets_copy()) // 2 for inst in c0.flattened() if inst.name == "CX")
    expect = int(hx.sum() + hz.sum()) * ROUNDS
    assert ncx == expect, "CNOT count %d != nnz*rounds %d (wrong q?)" % (ncx, expect)
    c, _, _ = C1.build_circuit(h1, h2, lobs, ROUNDS, P, q)
    return c, n, lobs.shape[0]


def pending_chunks(st, shots=None, chunk=None):
    """Chunks (index, seed, size) of a unit not yet completed; seed = 1000 + index.

    The completed chunks are recorded by seed in st["seeds_done"]: results arrive
    out of order, so a count of finished shots does not say which seeds ran."""
    shots = SHOTS if shots is None else shots
    chunk = CHUNK if chunk is None else chunk
    done = set(st.get("seeds_done") or [])
    out = []
    for i in range(int(math.ceil(shots / float(chunk)))):
        if 1000 + i not in done:
            out.append((i, 1000 + i, min(chunk, shots - i * chunk)))
    return out


def _worker_seeded(args):
    """_worker plus the chunk's seed, so an unordered pool can record what finished."""
    return args[4], _worker(args)


def _worker(args):
    """Decode one chunk of shots with the cascade;
    -> (shot_fail, per_obs_fail[k], unsat_after_all, stage_counts[3], shots)."""
    circ_text, H_data, pri, L_data, seed, nshots = args
    import stim
    import scipy.sparse as sp
    from relay_bp import RelayDecoderF32
    c = stim.Circuit(circ_text)
    Hs = sp.csr_matrix((H_data[0], H_data[1], H_data[2]), shape=H_data[3])
    Ls = sp.csr_matrix((L_data[0], L_data[1], L_data[2]), shape=L_data[3])
    pri64 = np.asarray(pri, np.float64)
    dec1 = RelayDecoderF32(check_matrix=Hs, error_priors=pri64, **STAGE1)
    dec2 = dec3 = None
    det, obs = c.compile_detector_sampler(seed=seed).sample(nshots, separate_observables=True)
    k = Ls.shape[0]
    per_obs = np.zeros(k, np.int64)
    shot_fail = 0
    unsat = 0
    stages = [0, 0, 0]

    def ok(e, d):
        return not ((Hs @ e) % 2 != d).any()

    for i in range(nshots):
        d = det[i].astype(np.uint8)
        d64 = d.astype(np.int64)
        e = np.asarray(dec1.decode(d), np.int64)
        if ok(e, d64):
            stages[0] += 1
        elif CASCADE:
            if dec2 is None:
                dec2 = RelayDecoderF32(check_matrix=Hs, error_priors=pri64, **STAGE2)
            e = np.asarray(dec2.decode(d), np.int64)
            if ok(e, d64):
                stages[1] += 1
            else:
                if dec3 is None:
                    from ldpc import BpOsdDecoder
                    dec3 = BpOsdDecoder(Hs, error_channel=list(pri64), max_iter=OSD_ITERS, bp_method="minimum_sum",
                                        ms_scaling_factor=0.0, osd_method="osd_cs", osd_order=OSD_ORDER)
                e = np.asarray(dec3.decode(d), np.int64)
                if ok(e, d64):
                    stages[2] += 1
                else:
                    unsat += 1
        else:
            unsat += 1
        pred = (Ls @ e) % 2
        mism = pred.astype(np.uint8) != obs[i].astype(np.uint8)
        if mism.any():
            shot_fail += 1
            per_obs += mism
    return shot_fail, per_obs, unsat, stages, nshots


def run_code(code, basis, out_path, partial):
    import circ_1side as C1
    circ, n, k = build(code["npz"], int(code.get("q", 1)), basis, cornucopia=code.get("cornucopia"))
    dem = circ.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
    H, L, pri = C1.dem_matrices(dem)
    H = H.tocsr(); L = L.tocsr()
    H_data = (H.data, H.indices, H.indptr, H.shape)
    L_data = (L.data, L.indices, L.indptr, L.shape)
    key = "%s|%s" % (code["name"], basis)
    st = partial.get(key, {"shots": 0, "shot_fail": 0, "per_obs": [0] * k, "unsat": 0, "stages": [0, 0, 0]})
    st.setdefault("stages", [0, 0, 0])
    st.setdefault("seeds_done", [])
    done = st["shots"]
    pend = pending_chunks(st)
    print("%s basis=%s [[%d,%d,%s]] DEM %dx%d, %d shots done, %d to go" % (
        code["name"], basis, n, k, code.get("d"), H.shape[0], H.shape[1], done, sum(m for _, _, m in pend)), flush=True)
    circ_text = str(circ)
    chunks = [(circ_text, H_data, pri, L_data, seed, m) for _, seed, m in pend]
    t0 = time.time()
    with mp.get_context("spawn").Pool(WORKERS) as pool:
        for seed, (shot_fail, per_obs, unsat, stages, nshots) in pool.imap_unordered(_worker_seeded, chunks):
            st["shots"] += nshots
            st["shot_fail"] += int(shot_fail)
            st["unsat"] += int(unsat)
            st["stages"] = [int(a + b) for a, b in zip(st["stages"], stages)]
            st["per_obs"] = [int(a + b) for a, b in zip(st["per_obs"], per_obs.tolist())]
            st["seeds_done"].append(int(seed))
            partial[key] = st
            json.dump(partial, open(out_path + ".partial", "w"))
    secs = time.time() - t0
    return summarize(code, basis, n, k, st, secs)


def summarize(code, basis, n, k, st, secs):
    N = st["shots"]
    f = st["shot_fail"]
    lo, hi = clopper_pearson(f, N)
    per = np.array(st["per_obs"], float)
    tot_obs = int(per.sum())
    p_obs_mean = tot_obs / float(N * k)              # per shot per logical, measured
    lo_o, hi_o = clopper_pearson(tot_obs, N * k)
    rec = dict(name=code["name"], basis=basis, n=n, k=k, d=code.get("d"), p=P, rounds=ROUNDS, shots=N,
               shot_failures=f, pl_shot=f / N, pl_shot_ci95=[lo, hi],
               logical_failures_total=tot_obs, pl_shot_per_logical=p_obs_mean, pl_shot_per_logical_ci95=[lo_o, hi_o],
               pl_shot_per_logical_max=float(per.max() / N), worst_logical=int(per.argmax()),
               pl_round_per_logical_approx=(1.0 - (1.0 - p_obs_mean) ** (1.0 / ROUNDS)) if p_obs_mean < 1 else 1.0,
               unsatisfied_syndromes=st["unsat"], resolved_by_stage=st.get("stages"), decoder="relay-BP x2 + BP-OSD" if CASCADE else "relay-BP",
               circuit=("paper schedule, noise=%s, detectors=%s" % (os.environ.get("PL_NOISE", "paper"), os.environ.get("PL_DET", "all"))) if code.get("cornucopia") else "circ_1side group layers + depolarising", secs=secs)
    print("%-20s %s [[%d,%d,%s]] shots %d: any-logical fail %d (pL/shot %.3g, 95%% [%.2g, %.2g]); "
          "per-logical per shot %.3g [%.2g, %.2g], worst logical %.3g; per round per logical ~%.3g; stages %s unsat %d; %.0f s" % (
              code["name"], basis, n, k, code.get("d"), N, f, rec["pl_shot"], lo, hi, p_obs_mean, lo_o, hi_o,
              rec["pl_shot_per_logical_max"], rec["pl_round_per_logical_approx"], st.get("stages"), st["unsat"], secs), flush=True)
    return rec


def main(codes_path, out_path):
    codes = json.load(open(codes_path))
    partial = json.load(open(out_path + ".partial")) if os.path.exists(out_path + ".partial") else {}
    out = []
    for basis in BASES:
        for c in codes:
            out.append(run_code(c, basis, out_path, partial))
            json.dump(out, open(out_path, "w"), indent=1)
    print("wall done; results in", out_path, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
