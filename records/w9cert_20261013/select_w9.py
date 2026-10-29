#!/usr/bin/env python3
"""Pick the five measured codes of the best program of every rate cell of the weight-9
gpt-6.1-sol run (ledger rows of the weight-9 table), find their matrices in the run's code archive,
check n, k, w and CSS, and write manifest.json for the certification waves (2026-10-13)."""
import json, glob, os, shutil, sys
import numpy as np
RUN = "/u/zdliu/lp/oe_run/cc_v7_gpt61sol500_w9_10c_out"
ARCH = "/ptmp/zdliu/archives_live/cc_v7_gpt61sol500_w9_10c_out/archive/codes"
OUT = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT + "/npz", exist_ok=True)
rows = [json.loads(l) for l in open(RUN + "/eval_ledger.jsonl") if l.strip()][:516]
best = {}
for i, r in enumerate(rows[:508]):                       # first 500 children
    b = r.get("rate_bin"); f = float(r.get("fitness") or 0)
    if b is not None and (b not in best or f > best[b][0]): best[b] = (f, i, r)
fams = {r["family"] for _, _, r in best.values()}
arch = {}
for f in glob.glob(ARCH + "/*.json"):
    try: d = json.load(open(f))
    except Exception: continue
    if d.get("family") in fams: arch.setdefault((d["family"], d["idx"]), []).append((d.get("ts", 0), os.path.basename(f)[:-5], d))
def gf2_rank(M):
    M = (np.asarray(M, np.uint8) % 2).copy(); r = 0; m, n = M.shape
    for c in range(n):
        p = np.flatnonzero(M[r:, c])
        if p.size == 0: continue
        p = r + p[0]
        if p != r: M[[r, p]] = M[[p, r]]
        o = np.flatnonzero(M[:, c]); o = o[o != r]
        if o.size: M[o] ^= M[r]
        r += 1
        if r == m: break
    return r
man = []
for b in sorted(best):
    f, i, r = best[b]
    for (slot, n, k, w, d2s, d30) in r["measured"]:
        cands = [x for x in arch.get((r["family"], slot), []) if x[2].get("n") == n and x[2].get("k") == k and x[0] <= r["ts"] + 1]
        cands.sort(key=lambda x: x[0])
        if not cands: print("MISSING cell", b, "slot", slot); continue
        ts, sha, d = cands[-1]
        z = np.load(ARCH + "/" + sha + ".npz")
        hx = np.unpackbits(z["HX"], axis=1)[:, :n]; hz = np.unpackbits(z["HZ"], axis=1)[:, :n]   # archive stores packed bits
        css = not np.any((hx.astype(np.int64) @ hz.T.astype(np.int64)) % 2)
        kk = hx.shape[1] - gf2_rank(hx) - gf2_rank(hz); ww = int(max(hx.sum(1).max(), hz.sum(1).max()))
        tag = "w9c%ds%d" % (b, slot)
        shutil.copy(ARCH + "/" + sha + ".npz", OUT + "/npz/" + tag + ".npz")
        man.append(dict(tag=tag, cell=b, ledger_row=i, family=r["family"], group=r.get("glabel"), q=r.get("q"), slot=slot, n=n, k=k, w=w, d30s=d30, sha=sha,
                        check=dict(n=int(hx.shape[1]), k=int(kk), w=ww, css=bool(css), n_siblings=len(cands))))
        print(tag, "[[%d,%d,<=%d]] w%d" % (n, k, d30, w), "sha", sha, "check", man[-1]["check"])
json.dump(man, open(OUT + "/manifest.json", "w"), indent=1)
bad = [m for m in man if not (m["check"]["css"] and m["check"]["k"] == m["k"] and m["check"]["n"] == m["n"] and m["check"]["w"] <= 9)]
print("codes", len(man), "failed checks", len(bad), [m["tag"] for m in bad])
