"""Re-measure the code-capacity distance of archived codes with
estimator_gpu_v2.estimate_d_gpu: --reps runs of --budget seconds with different
seeds. Each reading is an upper bound on d; d_verified is the minimum over the
runs and `agree` records whether all runs gave the same value.

  python deepverify_cc.py --codes codes.json --out results.json \
      [--budget 600] [--reps 2] [--part i/N]

--part i/N takes every N-th code starting at i; results are appended per code.
"""
import argparse
import json
import os
import sys
import time

for _p in ("/u/zdliu/lp/circmean", "/u/zdliu/lp/lps_graftgate",
           "/u/zdliu/lp/dopt/bdirect_v2", "/u/zdliu/lp/lifted_product_search"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import fast_gf2
from fast_gf2 import logicals
from estimator_gpu_v2 import estimate_d_gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--budget", type=float, default=600)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument("--part", default="0/1")
    a = ap.parse_args()
    i, nparts = (int(x) for x in a.part.split("/"))
    codes = [c for j, c in enumerate(json.load(open(a.codes))) if j % nparts == i and c.get("npz")]
    done = set()
    if os.path.exists(a.out):
        for line in open(a.out):
            try:
                done.add(json.loads(line)["npz"])
            except (ValueError, KeyError):
                pass
    print("part %s: %d codes, %d already done" % (a.part, len(codes), len([c for c in codes if c["npz"] in done])), flush=True)
    for c in codes:
        if c["npz"] in done:
            continue
        if not os.path.exists(c["npz"]):
            print("  MISSING npz, skipped: %s cell %s %s" % (c["arm"], c["cell"], c["npz"]), flush=True)
            continue
        z = np.load(c["npz"])
        # the archive stores HX/HZ bit-packed along axis 1; the true column
        # count is the recorded n
        n = int(c["n"])
        HX = np.ascontiguousarray(np.unpackbits(z["HX"], axis=1)[:, :n].astype(np.uint8))
        HZ = np.ascontiguousarray(np.unpackbits(z["HZ"], axis=1)[:, :n].astype(np.uint8))
        assert not ((HX.astype(np.int64) @ HZ.T.astype(np.int64)) % 2).any(), "%s is not CSS" % c["npz"]
        k = int(n - fast_gf2.rank(HX) - fast_gf2.rank(HZ))
        w = int(max(HX.sum(1).max(), HZ.sum(1).max()))
        assert k == int(c["k"]) and w == int(c["w"]), ("archive/ledger mismatch", c["npz"], k, c["k"], w, c["w"])
        LZ = np.asarray(logicals(HX, HZ), np.uint8) % 2
        LX = np.asarray(logicals(HZ, HX), np.uint8) % 2
        reads = []
        for rep in range(a.reps):
            t0 = time.time()
            r = estimate_d_gpu(HX, HZ, LX, LZ, budget_seconds=a.budget, seed=rep)
            d = r.get("d") if isinstance(r, dict) else r
            reads.append(dict(rep=rep, d=d, secs=round(time.time() - t0, 1),
                              dx=r.get("dx") if isinstance(r, dict) else None,
                              dz=r.get("dz") if isinstance(r, dict) else None))
            print("  %-8s cell %-8s %-34s idx %2s [[%d,%d]] w%d  30s-reading %s  rep %d: d<=%s (x%s/z%s) %.0fs" % (
                c["arm"], c["cell"], c["family"][:34], c["idx"], n, k, w, c.get("d30s"), rep, d,
                reads[-1]["dx"], reads[-1]["dz"], reads[-1]["secs"]), flush=True)
        ds = [x["d"] for x in reads if x["d"] is not None]
        rec = dict(c, n_check=n, k_check=k, w_check=w, reads=reads,
                   d_verified=min(ds) if ds else None, agree=(len(set(ds)) == 1) if ds else False,
                   budget=a.budget)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
    print("DONE part %s" % a.part, flush=True)


if __name__ == "__main__":
    main()
