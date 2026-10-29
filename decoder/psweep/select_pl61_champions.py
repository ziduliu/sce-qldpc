"""Archive entries of the ten cell-champion codes of the gpt-6.1-sol circuit-level
run (pl10_gpt61sol_out): the best-scoring code of the champion program of every
rate cell. Writes one record per code with the HASH spec deepverify_pl expects
(cell:q:sha)."""
import glob
import json
import os

RUN = "/u/zdliu/lp/oe_run/pl10_gpt61sol_out"
OUT = "/u/zdliu/lp/jobs/codes_pl61_champions_20261006.json"

rows = [json.loads(l) for l in open(os.path.join(RUN, "eval_ledger.jsonl")) if l.strip()]
rows.sort(key=lambda r: r["ts"])
best, champ = {}, {}
for i, r in enumerate(rows):
    b, f = r.get("rate_bin"), r.get("fitness") or 0
    if f > best.get(b, 0):
        best[b], champ[b] = f, (i - 7, r)
index = {}
for f in glob.glob(os.path.join(RUN, "archive", "codes", "*.json")):
    try:
        m = json.load(open(f))
    except ValueError:
        continue
    index.setdefault((m.get("family"), int(m.get("idx", -1)), int(m.get("n", -1)), int(m.get("k", -1))), []).append((f, m))
print("archive entries indexed:", sum(len(v) for v in index.values()))
out = []
for b in sorted(best):
    child, r = champ[b]
    pls = r.get("pl") or []
    top = max(pls, key=lambda p: p[1])          # [idx, score, basis, ...]
    idx = top[0]
    hit = [m for m in r["measured"] if m[0] == idx]
    assert hit, ("champion pl index not in measured", b, idx)
    _, n, k, w, d2, d30 = hit[0]
    ents = index.get((r["family"], idx, n, k), [])
    assert len(ents) == 1, ("ambiguous or missing archive entry", b, r["family"], idx, len(ents))
    f, m = ents[0]
    sha = os.path.basename(f)[:-5]
    out.append(dict(arm="plc61", cell=b, role="circuit-champion", child=child, family=r["family"], idx=idx, n=n, k=k, w=w,
                    d2s=d2, score=top[1], fitness=r.get("fitness"), sha=sha, npz=f[:-5] + ".npz", q=r["q"],
                    glabel=r.get("glabel"), a_shape=r["a_shape"], b_shape=r["b_shape"], hash_spec="c%d:%d:%s" % (b, r["q"], sha)))
    print("cell %d child %3d idx %2d [[%d,%d]] w%d d2s=%d s=%.3f %s q=%s sha %s" % (b, child, idx, n, k, w, d2, top[1], (r.get("glabel") or "")[:24], r["q"], sha))
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT, len(out), "codes")
