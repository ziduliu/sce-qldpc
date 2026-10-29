"""Select the per-cell champion codes of a distance (code-capacity) run and write
the codes file that deepverify_cc.sbatch consumes.

Per rate cell the champion program is the ledger row with the highest fitness; its
champion code is the measured slot with the largest k d30s^2/n, looked up in the
run archive by (family, idx, n, k).

  python select_cc_champions.py --run /u/zdliu/lp/oe_run/<run> --arm lunahigh --out codes.json
"""
import argparse
import glob
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(os.path.join(a.run, "eval_ledger.jsonl")) if l.strip()]
    best = {}
    for r in rows:
        c = r.get("rate_bin")
        f = float(r.get("fitness") or 0)
        if c is not None and f > 0 and (c not in best or f > best[c][0]):
            best[c] = (f, r)
    index = {}
    for f in glob.glob(os.path.join(a.run, "archive", "codes", "*.json")):
        try:
            m = json.load(open(f))
        except ValueError:
            continue
        key = (m.get("family"), int(m.get("idx", -1)), int(m.get("n", -1)), int(m.get("k", -1)))
        index.setdefault(key, []).append((m.get("ts") or 0, f, m))
    entries = []
    for c in sorted(best):
        fit, r = best[c]
        ms = [m for m in r.get("measured", []) if isinstance(m, list) and m[5] is not None]
        if not ms:
            print("cell %d: champion %s has no measured slots" % (c, r["family"])); continue
        m = max(ms, key=lambda m: m[2] * m[5] ** 2 / m[1])
        idx, n, k, w, d = int(m[0]), int(m[1]), int(m[2]), m[3], int(m[5])
        cands = sorted(index.get((r["family"], idx, n, k), []), reverse=True)
        if not cands:
            print("cell %d: no archive entry for %s idx %d [[%d,%d]]" % (c, r["family"], idx, n, k)); continue
        # several archive entries can share (family, idx, n, k): keep those whose
        # row weight and 30 s reading match the ledger's measured slot, newest first
        def _i(x):
            try:
                return int(x)
            except (TypeError, ValueError):
                return -1
        exact = [t for t in cands if _i(t[2].get("w")) == _i(w) and _i(t[2].get("d_circ")) == int(d)]
        if not exact:
            print("cell %d: WARNING no archive entry matches w=%s d30s=%d among %d siblings of %s idx %d; taking newest" % (
                c, w, d, len(cands), r["family"], idx))
            exact = cands
        _, f, meta = exact[0]
        sha = os.path.basename(f)[:-5]
        print("cell %d: %-50s fitness %7.2f slot %3d [[%d,%d,%d]] w%s  %.2f  sha %s (%d archive entries)" % (
            c, r["family"][:50], fit, idx, n, k, d, w, k * d * d / n, sha, len(cands)))
        entries.append(dict(arm=a.arm, cell=int(c), family=r["family"], idx=idx, n=n, k=k, w=w, d30s=d, fitness=fit,
                            sha=sha, npz=f[:-5] + ".npz"))
    json.dump(entries, open(a.out, "w"), indent=1)
    print("written %d entries -> %s" % (len(entries), a.out))


if __name__ == "__main__":
    main()
