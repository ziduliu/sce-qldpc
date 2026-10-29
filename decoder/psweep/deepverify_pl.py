"""Circuit-level re-measurement of selected codes with pl_score.score_codes, at
a larger sample size and a given physical error rate. Codes are read from a
run's code archive (--pick, --hash) or from npz files (--npz).

  python deepverify_pl.py --archive .../archive --out deep.json \\
      --pick 'z29_o29_24x24_affine_floor2:6,7,8,3,2' \\
      --pick 'd22_o22_24x35_cycle_screen:15,35,4,30,39'

Sample size and p come from pl_score's environment variables
(PL_SCORE_P / PL_SCORE_SHOTS / PL_SCORE_MAX_ERRORS); one invocation is one leg.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "/u/zdliu/lp/circmean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", required=True)
    ap.add_argument("--pick", action="append", default=[],
                    help="family:q:idx,idx,... (repeatable). q is given here "
                         "because the archive metadata does not carry it and "
                         "the ASC schedule builder needs the lift order.")
    ap.add_argument("--npz", action="append", default=[],
                    help="name:q:path (repeatable). A code given directly as an "
                         "npz of HX/HZ rather than looked up in the archive - "
                         "used for published reference codes (the BB family), "
                         "which the run's archive does not contain.")
    ap.add_argument("--hash", action="append", default=[],
                    help="name:q:sha (repeatable). Addresses a code in the "
                         "archive by its content hash, which is unique; "
                         "(family, idx) can name more than one code because "
                         "FAMILY_NAME is reused down a lineage.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", default=None)
    a = ap.parse_args()
    if not a.pick:
        a.pick = []

    import pl_score

    want, qof = [], {}
    for spec in a.pick:
        fam, q, idxs = spec.split(":")
        qof[fam] = int(q)
        # "+" as well as ",": sbatch's --export=ALL,VAR=value treats a comma as
        # a variable separator
        for i in __import__("re").split(r"[+,]", idxs):
            want.append((fam, int(i)))
    print("looking for %d codes in %s" % (len(want), a.archive), flush=True)

    found = {}
    t0 = time.time()
    for f in glob.glob(os.path.join(a.archive, "codes", "*.json")):
        try:
            m = json.load(open(f))
        except Exception:
            continue
        key = (m.get("family"), m.get("idx"))
        if key in want and key not in found:
            found[key] = (os.path.basename(f)[:-5], m)
    print("archive scan: %.1fs, matched %d/%d"
          % (time.time() - t0, len(found), len(want)), flush=True)
    missing = [k for k in want if k not in found]
    if missing:
        # loud: a silently short list would be reported as a complete result
        print("FATAL: not in the archive: %r" % (missing,))
        return 2

    codes = []
    for key in want:
        h, m = found[key]
        z = np.load(os.path.join(a.archive, "codes", h + ".npz"))
        n = int(m["n"])
        HX = np.unpackbits(z["HX"], axis=1)[:, :n].astype(np.uint8)
        HZ = np.unpackbits(z["HZ"], axis=1)[:, :n].astype(np.uint8)
        # the archive is the authority on n/k/w; a mismatch here means the
        # packed matrices and the metadata have drifted apart
        assert HX.shape[1] == n and HZ.shape[1] == n, (key, HX.shape, n)
        assert not ((HX @ HZ.T) % 2).any(), "%r is not CSS" % (key,)
        codes.append(dict(name="%s#%d" % key, HX=HX, HZ=HZ, q=qof[key[0]], n=n,
                          k=int(m["k"]), d=m.get("d2s"),
                          family=key[0], idx=key[1], w=m.get("w")))
    stored_sched = {}
    for spec in a.hash:
        name, q, sha = spec.split(":", 2)
        m = json.load(open(os.path.join(a.archive, "codes", sha + ".json")))
        # reuse the schedule stored with the code in the archive, if any (same
        # circuit); pl_at_p_asc.build re-verifies it against the matrix
        for b, e in (m.get("asc_sched") or {}).items():
            if isinstance(e, dict) and "ticks" in e:
                stored_sched["%s|%s" % (name, b)] = {"ticks": e["ticks"],
                                                     "depth": e.get("depth")}
        z = np.load(os.path.join(a.archive, "codes", sha + ".npz"))
        n = int(m["n"])
        HX = np.unpackbits(z["HX"], axis=1)[:, :n].astype(np.uint8)
        HZ = np.unpackbits(z["HZ"], axis=1)[:, :n].astype(np.uint8)
        assert not ((HX @ HZ.T) % 2).any(), "%s is not CSS" % name
        codes.append(dict(name=name, HX=HX, HZ=HZ, q=int(q), n=n,
                          k=int(m["k"]), d=m.get("d2s"), family=m.get("family"),
                          idx=m.get("idx"), w=m.get("w")))

    for spec in a.npz:
        name, q, path = spec.split(":", 2)
        z = np.load(path)
        HX = np.asarray(z["HX"], np.uint8) % 2
        HZ = np.asarray(z["HZ"], np.uint8) % 2
        assert HX.shape[1] == HZ.shape[1], (name, HX.shape, HZ.shape)
        assert not ((HX.astype(np.int64) @ HZ.astype(np.int64).T) % 2).any(), \
            "%s is not CSS" % name
        n = int(HX.shape[1])
        # k from the ranks, not from the file name: a stored npz carries no k
        # and a wrong k rescales every per-logical number silently.
        k = n - _rank2(HX) - _rank2(HZ)
        codes.append(dict(name=name, HX=HX, HZ=HZ, q=int(q), n=n, k=int(k),
                          d=None, family=name, idx=-1,
                          w=int(max(HX.sum(1).max(), HZ.sum(1).max()))))

    if not codes:
        print("FATAL: no codes given (--pick and/or --npz)")
        return 2
    for c in codes:
        print("  %-44s [[%d,%d]] w%s d2s=%s q=%s"
              % (c["name"], c["n"], c["k"], c["w"], c["d"], c["q"]), flush=True)

    work = a.workdir or os.path.join(os.environ.get("TMPDIR", "/tmp"),
                                     "deepverify_%d" % os.getpid())
    os.makedirs(work, exist_ok=True)
    sp = os.path.join(work, "asc_schedules.json")
    if stored_sched:
        # seed the schedule file so ensure_schedule finds these and does not
        # re-solve; anything missing is still solved as before
        try:
            cur = json.load(open(sp)) if os.path.exists(sp) else {}
        except Exception:
            cur = {}
        for kk, v in stored_sched.items():
            cur.setdefault(kk, v)
        json.dump(cur, open(sp, "w"))
        print("schedules reused from the archive: %d" % len(stored_sched), flush=True)
    rep = {}
    t0 = time.time()
    scores = pl_score.score_codes(
        [dict(name=c["name"], HX=c["HX"], HZ=c["HZ"], q=c["q"], n=c["n"],
              k=c["k"], d=c["d"]) for c in codes],
        os.path.join(work, "asc_schedules.json"), work, report=rep)
    secs = time.time() - t0

    out = dict(p=rep.get("p"), rounds=rep.get("rounds"),
               max_shots=rep.get("max_shots"), max_errors=rep.get("max_errors"),
               seconds=round(secs, 1), error=rep.get("error"),
               codes=[])
    for c in codes:
        info = (rep.get("codes") or {}).get(c["name"], {})
        out["codes"].append(dict(name=c["name"], family=c["family"], idx=c["idx"],
                                 n=c["n"], k=c["k"], w=c["w"], d2s=c["d"],
                                 score=scores.get(c["name"]),
                                 detail=info))
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print("\np=%s shots<=%s errors<=%s  %.0fs  ->  %s"
          % (out["p"], out["max_shots"], out["max_errors"], secs, a.out))
    for r in out["codes"]:
        d = r["detail"] or {}
        print("  %-44s s=%-8s ceiling=%-8s %s"
              % (r["name"], _f(r["score"]), _f(d.get("score_ceiling")),
                 _bases(d)))
    if rep.get("error"):
        print("\nSAMPLER ERROR: %s" % rep["error"])
        print(str(rep.get("log_tail", ""))[-2000:])
        return 1
    return 0


def _q_from(m, n):
    return 0


def _rank2(H):
    """GF(2) rank by Gaussian elimination on a bit-packed copy."""
    M = np.array(H, dtype=np.uint8) % 2
    rows, cols = M.shape
    r = 0
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c]:
                piv = i
                break
        if piv is None:
            continue
        if piv != r:
            M[[r, piv]] = M[[piv, r]]
        hit = np.nonzero(M[:, c])[0]
        hit = hit[hit != r]
        if hit.size:
            M[hit] ^= M[r]
        r += 1
        if r == rows:
            break
    return int(r)


def _f(x):
    try:
        return "%.3f" % float(x)
    except (TypeError, ValueError):
        return str(x)


def _bases(d):
    bits = []
    for b in ("X", "Z"):
        e = (d.get("bases") or {}).get(b) or d.get(b) or {}
        if not isinstance(e, dict):
            continue
        bits.append("%s:%s/%s%s" % (b, e.get("logical_failures"), e.get("shots"),
                                    "" if e.get("stopped_on") != "errors" else "*"))
    return " ".join(bits)


if __name__ == "__main__":
    sys.exit(main())
