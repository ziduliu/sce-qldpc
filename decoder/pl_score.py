"""Circuit-level pL score of a list of codes:

    s = -log10( upper end of the 95% Clopper-Pearson CI on the per-round per-logical rate )

The rate is the fraction of shots with any logical error, converted per round
and divided by k; the score of a code is that of its worse basis, and with zero
failures it saturates at score_ceiling. The sampler (pl_sample.py) runs as a
subprocess; schedules are read from a JSON file or solved on demand, a code
without a schedule scores 0, and sampler failures are printed and recorded in
report["error"].

  from pl_score import score_codes
  report = {}
  scores = score_codes(codes, sched_path, workdir, report=report)
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

CIRCMEAN = "/u/zdliu/lp/circmean"
PY = os.environ.get("PL_SCORE_PY", "/u/zdliu/lpenv/bin/python")
DRIVER = os.environ.get("PL_SCORE_DRIVER",
                        "/u/zdliu/lp/plevolve_20260912/pl_sample.py")

P = os.environ.get("PL_SCORE_P", "0.002")
ROUNDS = int(os.environ.get("PL_SCORE_ROUNDS", "6"))
# stopping rule: PL_SCORE_SHOTS shots or PL_SCORE_MAX_ERRORS failed shots
SHOTS = int(os.environ.get("PL_SCORE_SHOTS", "20000"))
MAX_ERRORS = int(os.environ.get("PL_SCORE_MAX_ERRORS", "100"))
SCHED_TIME = float(os.environ.get("PL_SCORE_SCHED_TIME", "120"))
# sampler timeout; kept below the caller's own timeout so that a hung sampler
# is reported here
TIMEOUT = float(os.environ.get("PL_SCORE_TIMEOUT", "4800"))
KEEP = os.environ.get("PL_SCORE_KEEP", "0") == "1"
LOG_TAIL = int(os.environ.get("PL_SCORE_LOG_TAIL", "40"))
FLOOR = 1e-12          # guards log10 if a bound ever comes back as 0
# bases to measure, default both; the one-circuit runs (records/pl_sweep_20261030_onecircuit) measure one per run
BASES = tuple(b for b in os.environ.get("PL_SCORE_BASES", "Z,X").split(",") if b)
ORTOOLS = os.environ.get("PL_SCORE_ORTOOLS", "/ptmp/zdliu/ortools_local")

# Optional: share the first code's schedules with the other codes (sched_share);
# a code that fails the circuit checks solves its own. This changes the circuit,
# hence the pL, of the other codes.
SHARE_SCHED = os.environ.get("PL_SCORE_SHARE_SCHED", "0") == "1"
# Optional: run the (code, basis) units in parallel under srun, one rank each;
# counts do not depend on the distribution. Needs at least one task per unit.
SRUN = os.environ.get("PL_SCORE_SRUN", "0") == "1"
# Optional: spread every unit over SHARDS ranks (chunk ids split round-robin,
# counts summed by --collect). Needs SHARDS x units tasks and
# PL_SCORE_MAX_ERRORS=0, since each shard would stop on its own count.
SHARDS = max(1, int(os.environ.get("PL_SCORE_SHARDS", "1")))
# hands each rank one of its node's GPUs by local rank, so several ranks can
# share a node without all of them landing on cuda:0
SRUN_SHIM = os.environ.get("PL_SCORE_SRUN_SHIM",
                           "/u/zdliu/lp/circmean/rankgpu.sh")


def _say(msg):
    """Failures go to both streams: the worker log keeps stdout, slurm stderr."""
    print(msg, flush=True)
    print(msg, file=sys.stderr, flush=True)


def ensure_schedule(name, HX, HZ, q, sched_path):
    """Return (ok, note) -- True when both bases have a stored schedule.

    Solves and appends the missing ones with asc_schedule.solve_schedule; a
    timeout is reported in the note.
    """
    sys.path.insert(0, CIRCMEAN)
    if os.path.isdir(ORTOOLS) and ORTOOLS not in sys.path:
        sys.path.insert(0, ORTOOLS)
    import asc_schedule as A

    sched = {}
    if os.path.exists(sched_path):
        try:
            sched = json.load(open(sched_path))
        except Exception:
            sched = {}
    need = [b for b in BASES if ("%s|%s" % (name, b)) not in sched]
    if not need:
        return True, "cached"

    notes = []
    for basis in need:
        h1, h2 = (HX, HZ) if basis == "Z" else (HZ, HX)
        t0 = time.time()
        try:
            res = A.solve_schedule(h1, h2, q, time_limit=SCHED_TIME,
                                   verbose=False)
        except Exception as exc:
            notes.append("%s:solver-%s" % (basis, type(exc).__name__))
            continue
        if res.get("status") != "sat":
            # no schedule within the time limit, so no circuit
            notes.append("%s:%s after %.0fs" % (basis, res.get("status"),
                                                time.time() - t0))
            continue
        units = A.build_units(h1, h2, q)
        sched["%s|%s" % (name, basis)] = {
            "ticks": [int(t) for t in res["ticks"]],
            "depth": int(res["depth"]),
            "lower_bound": int(res["lower_bound"]),
            "q": int(q),
            "units": len(units),
        }
        notes.append("%s:solved depth %d in %.0fs"
                     % (basis, res["depth"], time.time() - t0))

    ok = all(("%s|%s" % (name, b)) in sched for b in BASES)
    if ok:
        tmp = sched_path + ".%d.tmp" % os.getpid()
        json.dump(sched, open(tmp, "w"))
        os.replace(tmp, sched_path)
    return ok, ",".join(notes)


def _per_basis(r):
    """The fields the feedback table needs out of one sampler record."""
    per = list(r.get("per_obs") or [])
    worst = int(r.get("worst_logical") or 0)
    # heaviest first, ties by index, so the first entry IS worst_logical
    top = sorted(((int(v), i) for i, v in enumerate(per) if v),
                 key=lambda t: (-t[0], t[1]))[:6]
    # score: upper CI end of the shot failure rate, per round, divided by k;
    # recomputed from pl_shot_ci95 so records without the *_std fields score
    # on the same definition
    ci = r.get("pl_shot_ci95") or [None, None]
    hi = ci[1]
    k = int(r.get("k") or 0)
    s = None
    per_round_hi = None
    if hi is not None and k > 0:
        per_round_hi = (1.0 - (1.0 - min(hi, 1.0)) ** (1.0 / ROUNDS)) / k
        s = round(-math.log10(max(per_round_hi, FLOOR)), 4)
    ceil = r.get("score_ceiling")
    if ceil is None or "pl_round_per_logical_std_ci95_hi" not in r:
        # record without the standard ceiling: rebuild it
        N = int(r.get("shots") or 0)
        if N > 0 and k > 0:
            from scipy.stats import beta as _beta
            hi0 = float(_beta.isf(0.025, 1, N))
            ceil = -math.log10(max((1.0 - (1.0 - hi0) ** (1.0 / ROUNDS)) / k, FLOOR))
    return dict(
        basis=r.get("basis"), score=s,
        score_ceiling=round(float(ceil or 0.0), 4),
        score_perobs=r.get("score_perobs"),
        shots=r.get("shots"), stopped_on=r.get("stopped_on"),
        max_errors=r.get("max_errors"), max_shots=r.get("max_shots"),
        shot_failures=r.get("shot_failures"),
        logical_failures=r.get("logical_failures_total"),
        logicals_failed=r.get("logicals_failed"),
        worst_logical=worst,
        worst_logical_failures=int(per[worst]) if per else None,
        # the per-logical-qubit failure counts, heaviest first (all of them if
        # few failed); the full vector is k long and k reaches 200 here
        top_logicals=[[i, v] for v, i in top],
        # standard convention (the score), and the per-observable mean for reference
        pl_round_per_logical=(r.get("pl_round_per_logical_std")
                              if "pl_round_per_logical_std" in r else
                              ((1.0 - (1.0 - (r.get("shot_failures") or 0) / float(r.get("shots") or 1))
                                ** (1.0 / ROUNDS)) / k if k > 0 else None)),
        pl_round_per_logical_ci_hi=per_round_hi,
        pl_round_per_obs_mean=r.get("pl_round_per_logical_approx"),
        pl_round_per_obs_ci_hi=r.get("pl_round_per_logical_ci95_hi"),
        resolved_by_stage=r.get("resolved_by_stage"),
        unsatisfied=r.get("unsatisfied_syndromes"),
        depth=r.get("depth"), k=r.get("k"), n=r.get("n"),
        attrib=r.get("attrib"),
        secs=r.get("secs"))


def score_codes(codes, sched_path, workdir=None, report=None):
    """codes: list of dicts with name, HX, HZ, q, n, k (d optional).

    Returns {name: score}; a code that could not be scheduled or measured maps
    to 0.0 so it contributes nothing rather than silently inflating the mean.
    `report` (a dict, optional) is filled with the schedule notes, the per-basis
    measurement of every code, and any sampler failure text.
    """
    rep = report if report is not None else {}
    out = {c["name"]: 0.0 for c in codes}
    rep.update(driver=DRIVER, p=P, rounds=ROUNDS, max_shots=SHOTS,
               max_errors=MAX_ERRORS, sched={}, codes={}, error=None, log=None,
               sampler_secs=0.0)
    if not codes:
        return out
    tmp = workdir or tempfile.mkdtemp(prefix="plscore_")
    os.makedirs(tmp, exist_ok=True)

    manifest = []
    for c in codes:
        HX = np.asarray(c["HX"], np.uint8) % 2
        HZ = np.asarray(c["HZ"], np.uint8) % 2
        # the npz is written before schedules are decided: the shared-schedule
        # path builds the circuit from it. Unscheduled codes have theirs
        # removed below.
        npz = os.path.join(tmp, c["name"] + ".npz")
        np.savez_compressed(npz, HX=HX, HZ=HZ)
        # the CNOT count of the circuit is exactly nnz(HX)+nnz(HZ) per round --
        # the same number pl_at_p_asc asserts the built circuit against
        rep["codes"][c["name"]] = dict(
            scheduled=False, sched_note=None, sched_secs=0.0,
            cnots=int(HX.sum() + HZ.sum()) * ROUNDS, bases={})
        manifest.append({"name": c["name"], "npz": npz, "q": int(c["q"]),
                         "d": int(c.get("d") or 0), "n": int(c["n"]),
                         "k": int(c["k"]), "HX": HX, "HZ": HZ,
                         "shapes": c.get("shapes")})

    if SHARE_SCHED and manifest:
        import sched_share
        t0 = time.time()
        _all_ok, snotes = sched_share.ensure_schedules(
            manifest, sched_path, solve_time=SCHED_TIME, rounds=ROUNDS)
        for name, info in snotes.items():
            rep["sched"][name] = info["note"]
            rep["codes"][name].update(scheduled=bool(info["ok"]),
                                      sched_note=info["note"],
                                      sched_secs=info["secs"],
                                      sched_source=info["sched_source"],
                                      sched_detail=info["bases"])
        rep["sched_shared"] = True
        rep["sched_shared_secs"] = round(time.time() - t0, 1)
    else:
        for c in manifest:
            t0 = time.time()
            ok, note = ensure_schedule(c["name"], c["HX"], c["HZ"], c["q"],
                                       sched_path)
            rep["sched"][c["name"]] = note
            rep["codes"][c["name"]].update(
                scheduled=bool(ok), sched_note=note,
                sched_secs=round(time.time() - t0, 1),
                sched_source=("Z:own,X:own" if ok else "none"))

    kept = []
    for c in manifest:
        info = rep["codes"][c["name"]]
        if info["scheduled"]:
            kept.append({kk: c[kk] for kk in
                         ("name", "npz", "q", "d", "n", "k", "shapes")})
        else:
            _say("pl_score: %s has no ASC schedule (%s) -- it scores 0"
                 % (c["name"], info["sched_note"]))
            try:
                os.remove(c["npz"])       # inode hygiene: it will not be used
            except OSError:
                pass
    manifest = kept
    if not manifest:
        rep["error"] = "no code could be scheduled"
        return out

    cj = os.path.join(tmp, "codes.json")
    oj = os.path.join(tmp, "pl.json")
    logp = os.path.join(tmp, "pl_sampler.log")
    rep["log"] = logp
    json.dump(manifest, open(cj, "w"))

    env = dict(os.environ)
    env.update(PL_P=P, PL_ROUNDS=str(ROUNDS), PL_SHOTS=str(SHOTS),
               PL_MAX_ERRORS=str(MAX_ERRORS), PL_BASES=",".join(BASES),
               PL_CHUNK=os.environ.get("PL_CHUNK", "250"),
               ASC_SCHED=sched_path,
               # GPU sampler configuration
               PLC_MODE=os.environ.get("PLC_MODE", "gpu2"),
               PLG_STAGE1=os.environ.get("PLG_STAGE1", "gpu"),
               PLC_STAGE3=os.environ.get("PLC_STAGE3", "lsd"),
               PLF_WORKERS=os.environ.get("PLF_WORKERS", "0"))
    # asc_schedule prefers CP-SAT and silently falls back to a DFS that times out
    # on every real code; ortools lives outside the venv, so put it on the path
    if os.path.isdir(ORTOOLS):
        env["PYTHONPATH"] = ORTOOLS + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.time()
    rc, failure = 0, None
    cmd = [PY, "-u", DRIVER, cj, oj]
    nunits = len(manifest) * len(BASES)
    alloc = int(os.environ.get("SLURM_NTASKS") or 0)
    ranks = nunits * SHARDS
    if SHARDS > 1 and MAX_ERRORS:
        raise ValueError("PL_SCORE_SHARDS=%d needs PL_SCORE_MAX_ERRORS=0: "
                         "shards cannot share a stopping rule" % SHARDS)
    if SRUN and 0 < ranks <= alloc:
        # one rank per (code, basis) unit, or SHARDS ranks per unit
        cmd = ["srun", "--ntasks=%d" % ranks, "--cpu-bind=cores",
               SRUN_SHIM] + cmd
        rep["parallel"] = dict(ranks=ranks, units=nunits, shards=SHARDS,
                               alloc_tasks=alloc)
    elif SRUN:
        rep["parallel"] = dict(ranks=1, units=nunits, alloc_tasks=alloc)
        _say("pl_score: PL_SCORE_SRUN=1 but this allocation has %d tasks for "
             "%d units -- running the units serially" % (alloc, nunits))
    if cmd[0] != "srun":
        # A serial launch inherits the batch step's SLURM_PROCID/SLURM_NTASKS;
        # with NTASKS >= 2 the sampler would take itself for rank 0 of a sharded
        # run and leave every other unit unmeasured.
        for var in ("SLURM_PROCID", "SLURM_NTASKS", "SLURM_NPROCS"):
            env.pop(var, None)
    rep["command"] = " ".join(cmd)
    with open(logp, "w") as lg:
        try:
            subprocess.run(cmd, cwd=CIRCMEAN, env=env,
                           timeout=TIMEOUT, stdout=lg, stderr=subprocess.STDOUT,
                           check=True)
            if rep.get("parallel", {}).get("ranks", 1) > 1:
                # each rank wrote its own part file; --collect adds the per-unit
                # integer counts into the one result file the reader expects
                subprocess.run([PY, "-u", DRIVER, "--collect", oj, cj],
                               cwd=CIRCMEAN, env=env, timeout=900,
                               stdout=lg, stderr=subprocess.STDOUT, check=True)
        except subprocess.TimeoutExpired:
            failure = "sampler exceeded PL_SCORE_TIMEOUT=%.0fs" % TIMEOUT
        except subprocess.CalledProcessError as exc:
            rc = exc.returncode
            failure = "sampler exited %s" % rc
        except Exception as exc:
            failure = "sampler could not be started: %s: %s" % (
                type(exc).__name__, str(exc)[:200])
    rep["sampler_secs"] = round(time.time() - t0, 1)

    tail = ""
    try:
        tail = "".join(open(logp).read().splitlines(True)[-LOG_TAIL:])
    except Exception:
        pass
    if failure:
        rep["error"] = failure
        rep["log_tail"] = tail
        _say("=" * 72)
        _say("pl_score: SAMPLER FAILED -- %s. Every code scores 0." % failure)
        _say("pl_score: command: %s" % rep.get("command", ""))
        _say("pl_score: last %d lines of %s:\n%s" % (LOG_TAIL, logp, tail))
        _say("=" * 72)
        return out

    try:
        recs = json.load(open(oj))
    except Exception as exc:
        rep["error"] = "sampler wrote no readable result: %s" % str(exc)[:200]
        rep["log_tail"] = tail
        _say("=" * 72)
        _say("pl_score: %s. Every code scores 0." % rep["error"])
        _say("pl_score: last %d lines of %s:\n%s" % (LOG_TAIL, logp, tail))
        _say("=" * 72)
        return out

    # worse basis wins: a memory is only as good as its weaker basis
    worst = {}
    for r in recs:
        name = r.get("name")
        if name not in rep["codes"]:
            continue
        pb = _per_basis(r)
        rep["codes"][name]["bases"][r.get("basis")] = pb
        if pb["score"] is None:
            continue
        if name not in worst or pb["score"] < worst[name][0]:
            worst[name] = (pb["score"], r.get("basis"))
    for name, (s, basis) in worst.items():
        got = set(rep["codes"][name]["bases"])
        if not set(BASES) <= got:
            # a half-measured code is not comparable with a fully measured one
            _say("pl_score: %s measured only %s -- it scores 0"
                 % (name, ",".join(sorted(got)) or "nothing"))
            rep["codes"][name]["incomplete"] = sorted(got)
            continue
        out[name] = round(float(s), 4)
        rep["codes"][name]["score"] = out[name]
        rep["codes"][name]["worse_basis"] = basis
        rep["codes"][name]["score_ceiling"] = min(
            rep["codes"][name]["bases"][b]["score_ceiling"] for b in BASES)
    for name in out:
        if name in rep["codes"] and rep["codes"][name]["scheduled"] \
                and "score" not in rep["codes"][name]:
            _say("pl_score: no measurement came back for %s -- it scores 0"
                 % name)
    if not KEEP:
        for m in manifest:
            try:
                os.remove(m["npz"])       # inode hygiene: one npz per code
            except OSError:
                pass
    return out


if __name__ == "__main__":
    # self-test on a reference code with a stored schedule and a known pL
    sched = os.environ.get(
        "PL_SELFTEST_SCHED",
        "/u/zdliu/lp/flagship_pl_20260910/asc_schedules.json")
    src = os.environ.get("PL_SELFTEST_CODES",
                         "/u/zdliu/lp/flagship_pl_20260910/codes.json")
    want = os.environ.get("PL_SELFTEST_NAME", "v1_1x2_585_117_26")
    cj = json.load(open(src))
    pick = [c for c in cj if c["name"] == want]
    if not pick:
        print("reference code %s not in %s" % (want, src))
        sys.exit(1)
    z = np.load(pick[0]["npz"])
    codes = [dict(name=pick[0]["name"], HX=z["HX"], HZ=z["HZ"], q=pick[0]["q"],
                  n=pick[0]["n"], k=pick[0]["k"], d=pick[0]["d"])]
    print("scoring %s at p=%s, <=%d shots / <=%d errors ..."
          % (codes[0]["name"], P, SHOTS, MAX_ERRORS))
    rep = {}
    print(score_codes(codes, sched, report=rep))
    print(json.dumps(rep, indent=1)[:4000])
