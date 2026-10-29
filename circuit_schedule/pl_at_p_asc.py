"""Circuit-level pL on circuits with the depth-optimal ASC (Auto-Stabilizer-Check) schedule.

Same protocol as decoder/pl_at_p.py (noise channels, Z-only detectors, p=0 check,
CNOT-count assert, both bases, per-logical counts, Clopper-Pearson intervals,
output schema, decoder cascade); only the syndrome extraction circuit differs:
asc_schedule.build_circuit with the schedule read from the JSON file ASC_SCHED
(written offline by asc_schedule.solve_schedule and re-checked here) instead of
circ_1side.build_circuit.

  PL_P=0.001 PL_ROUNDS=6 PL_SHOTS=20000 ASC_SCHED=asc_schedules.json \
      python pl_at_p_asc.py codes.json out.json
"""
import json
import multiprocessing as mp
import os
import sys
import time

os.environ.setdefault("LAYOUT_SQFAST_PATH", "/u/zdliu/lp/dopt/agentA")
for _p in ("/u/zdliu/lp/lps_graftgate", "/u/zdliu/lp/dopt/bdirect_v2",
           "/u/zdliu/lp/circmean"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

P = float(os.environ.get("PL_P", "0.001"))
ROUNDS = int(os.environ.get("PL_ROUNDS", "6"))
SHOTS = int(os.environ.get("PL_SHOTS", "20000"))
WORKERS = int(os.environ.get("PL_WORKERS", str(mp.cpu_count())))
BASES = os.environ.get("PL_BASES", "Z,X").split(",")
CHUNK = int(os.environ.get("PL_CHUNK", "250"))
SCHED = os.environ.get("ASC_SCHED", "/u/zdliu/lp/flagship_pl_20260910/asc_schedules.json")
IDLE_P = os.environ.get("PL_IDLE_P")
IDLE_P = float(IDLE_P) if IDLE_P else None

# PL_NOISE_MODEL: "default" keeps the circuit's DEPOLARIZE1(p) after ancilla
# resets and before measurements; "std" turns each of those into X_ERROR(p) and
# adds X_ERROR(p) on the data qubits at preparation (no idling noise). Gates,
# schedule, detectors and observables are the same in both.
NOISE_MODEL = os.environ.get("PL_NOISE_MODEL") or "default"
if NOISE_MODEL not in ("default", "std"):
    raise ValueError("PL_NOISE_MODEL must be 'default' or 'std', got %r" % NOISE_MODEL)
if NOISE_MODEL == "std" and IDLE_P:
    raise ValueError("PL_NOISE_MODEL=std assumes no idling noise; PL_IDLE_P is set")

# decoder settings, as in decoder/pl_at_p.py
GAMMA_INT = (-0.24, 0.6)
STAGE1 = dict(gamma0=0.1, pre_iter=200, num_sets=20, set_max_iter=100,
              stop_nconv=1, gamma_dist_interval=GAMMA_INT)
STAGE2 = dict(gamma0=0.1, pre_iter=500, num_sets=200, set_max_iter=200,
              stop_nconv=1, gamma_dist_interval=GAMMA_INT)
OSD_ITERS = int(os.environ.get("PL_OSD_ITERS", "300"))
OSD_ORDER = int(os.environ.get("PL_OSD_ORDER", "7"))
CASCADE = os.environ.get("PL_CASCADE", "1") == "1"


def clopper_pearson(k, n, alpha=0.05):
    from scipy.stats import beta
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.isf(alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def standard_noise(c, n):
    """The default circuit rewritten into the "std" noise model (see NOISE_MODEL)."""
    import stim
    ps = set()
    for inst in c:
        if isinstance(inst, stim.CircuitRepeatBlock):
            raise ValueError("repeat blocks are not expected in the memory circuit")
        if inst.name == "DEPOLARIZE1":
            ps.add(inst.gate_args_copy()[0])
    if len(ps) != 1:
        raise ValueError("expected one one-qubit noise strength, got %s" % sorted(ps))
    p = ps.pop()
    out = stim.Circuit()
    out.append("X_ERROR", list(range(n)), p)
    for inst in c:
        if inst.name == "DEPOLARIZE1":
            out.append("X_ERROR", inst.targets_copy(), p)
        else:
            out.append(inst)
    return out


def build(npz, q, basis, name, sched):
    """Same gates and same detectors as pl_at_p.build; different gate ORDER."""
    import asc_schedule as A
    from fast_gf2 import logicals
    z = np.load(npz)
    hx = np.asarray(z["HX"], np.uint8) % 2
    hz = np.asarray(z["HZ"], np.uint8) % 2
    n = hx.shape[1]
    lobs = (np.asarray(logicals(hx, hz), np.uint8) % 2 if basis == "Z"
            else np.asarray(logicals(hz, hx), np.uint8) % 2)
    assert n % q == 0 and hx.shape[0] % q == 0 and hz.shape[0] % q == 0, \
        "n or check count not divisible by q=%d" % q
    h1, h2 = (hx, hz) if basis == "Z" else (hz, hx)

    key = "%s|%s" % (name, basis)
    rec = sched.get(key) or sched.get(name)
    assert rec is not None, "no ASC schedule stored for %s" % key
    units = A.build_units(h1, h2, q)
    ticks, depth = rec["ticks"], rec["depth"]
    assert len(ticks) == len(units), \
        "%s: stored schedule has %d ticks, this matrix gives %d units" \
        % (key, len(ticks), len(units))

    # the schedule must still be a proper colouring for THIS matrix
    checks, datas = A.block_members(units)
    for idxs in list(checks.values()) + list(datas.values()):
        v = [ticks[i] for i in idxs]
        assert len(set(v)) == len(v), "%s: stored schedule double-books a block" % key

    c0 = A.build_circuit(h1, h2, lobs, ROUNDS, 0.0, q, units, ticks, depth,
                         idle_p=None)
    s0 = c0.compile_detector_sampler().sample(64, separate_observables=True)
    assert not (s0[0].any() or s0[1].any()), \
        "p=0 gate failed -- schedule violates the X-Z commutativity condition"
    ncx = sum(len(i.targets_copy()) // 2 for i in c0.flattened() if i.name == "CX")
    expect = int(hx.sum() + hz.sum()) * ROUNDS
    assert ncx == expect, "CNOT count %d != nnz*rounds %d" % (ncx, expect)
    c = A.build_circuit(h1, h2, lobs, ROUNDS, P, q, units, ticks, depth,
                        idle_p=IDLE_P)
    if NOISE_MODEL == "std":
        c = standard_noise(c, n)
    return c, n, lobs.shape[0], depth


def _worker(args):
    """Decode one chunk of shots with the relay-BP / BP-OSD cascade; returns the failure counts."""
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
    det, obs = c.compile_detector_sampler(seed=seed).sample(
        nshots, separate_observables=True)
    k = Ls.shape[0]
    per_obs = np.zeros(k, np.int64)
    shot_fail = unsat = 0
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
                    dec3 = BpOsdDecoder(Hs, error_channel=list(pri64),
                                        max_iter=OSD_ITERS, bp_method="minimum_sum",
                                        ms_scaling_factor=0.0, osd_method="osd_cs",
                                        osd_order=OSD_ORDER)
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


def run_code(code, basis, out_path, partial, sched):
    import circ_1side as C1
    key = "%s|%s" % (code["name"], basis)
    circ, n, k, depth = build(code["npz"], int(code["q"]), basis,
                              code["name"], sched)
    dem = circ.detector_error_model(decompose_errors=False,
                                    approximate_disjoint_errors=True)
    H, L, pri = C1.dem_matrices(dem)
    import scipy.sparse as sp
    Hs, Ls = sp.csr_matrix(H), sp.csr_matrix(L)
    prev = partial.get(key) or {}
    # The partial file holds finished records; their accumulator is kept under
    # "state" so a rerun can resume from it instead of reading record fields.
    st = dict(prev.get("state") or {"shots": 0, "shot_fail": 0, "per_obs": [0] * k,
                                    "unsat": 0, "stages": [0, 0, 0]})
    print("%s basis=%s [[%d,%d,%s]] ASC depth %d | DEM %dx%d nnz %d | %d to go"
          % (code["name"], basis, n, k, code.get("d"), depth, Hs.shape[0],
             Hs.shape[1], Hs.nnz, max(0, SHOTS - st["shots"])), flush=True)
    todo = max(0, SHOTS - st["shots"])
    if todo:
        txt = str(circ)
        Hd = (Hs.data, Hs.indices, Hs.indptr, Hs.shape)
        Ld = (Ls.data, Ls.indices, Ls.indptr, Ls.shape)
        chunks = []
        s = st["shots"]
        while s < SHOTS:
            b = min(CHUNK, SHOTS - s)
            chunks.append((txt, Hd, pri, Ld, 1000 + s, b))
            s += b
        t0 = time.perf_counter()
        with mp.Pool(WORKERS) as pool:
            for sf, po, un, stg, ns in pool.imap_unordered(_worker, chunks):
                st["shots"] += ns
                st["shot_fail"] += sf
                st["per_obs"] = [a + int(b) for a, b in zip(st["per_obs"], po)]
                st["unsat"] += un
                st["stages"] = [a + b for a, b in zip(st["stages"], stg)]
        secs = time.perf_counter() - t0
    else:
        secs = 0.0

    N, f = st["shots"], st["shot_fail"]
    lo, hi = clopper_pearson(f, N)
    per = np.array(st["per_obs"], float)
    tot_obs = int(per.sum())
    p_obs = tot_obs / float(N * k)
    lo_o, hi_o = clopper_pearson(tot_obs, N * k)
    rec = dict(name=code["name"], basis=basis, n=n, k=k, d=code.get("d"), p=P,
               rounds=ROUNDS, shots=N, shot_failures=f, pl_shot=f / N,
               pl_shot_ci95=[lo, hi], logical_failures_total=tot_obs,
               pl_shot_per_logical=p_obs, pl_shot_per_logical_ci95=[lo_o, hi_o],
               pl_shot_per_logical_max=float(per.max() / N),
               worst_logical=int(per.argmax()),
               pl_round_per_logical_approx=(1.0 - (1.0 - p_obs) ** (1.0 / ROUNDS))
               if p_obs < 1 else 1.0,
               unsatisfied_syndromes=st["unsat"], resolved_by_stage=st["stages"],
               decoder="relay-BP x2 + BP-OSD", schedule="ASC depth-optimal",
               asc_depth=depth, idling=IDLE_P,
               circuit="asc_schedule interleaved + depolarising", secs=secs,
               state=dict(st))
    print("%-24s %s [[%d,%d,%s]] shots %d: fail %d (pL %.3g, 95%% [%.2g, %.2g]); "
          "per-logical %.3g; stages %s unsat %d; %.0f s"
          % (code["name"], basis, n, k, code.get("d"), N, f, rec["pl_shot"],
             lo, hi, p_obs, st["stages"], st["unsat"], secs), flush=True)
    partial[key] = rec
    json.dump(partial, open(out_path + ".partial", "w"))
    return rec


def main(codes_path, out_path):
    codes = json.load(open(codes_path))
    sched = json.load(open(SCHED))
    partial = (json.load(open(out_path + ".partial"))
               if os.path.exists(out_path + ".partial") else {})
    out = []
    for basis in BASES:
        for c in codes:
            out.append(run_code(c, basis, out_path, partial, sched))
    json.dump(out, open(out_path, "w"), indent=1)
    print("wrote %s" % out_path, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
