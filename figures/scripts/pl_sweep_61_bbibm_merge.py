#!/usr/bin/env python3
"""Figure 2 with the [[144,12,12]] reference on its depth-7 syndrome schedule: merges
records/pl_sweep_20261009_bbibm/ into figures/data/pl_sweep_20261009_61_fig_bbibm.json and
draws pl_sweep_61_ab_v_bbibm.{pdf,png}."""
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "pl_sweep_61_build.py")).read().split("old = scan(OLD")[0]
B = {"__file__": os.path.join(HERE, "pl_sweep_61_build.py")}
exec(compile(src, "pl_sweep_61_build.py", "exec"), B)

rows = [r for r in json.load(open(os.path.join(HERE, "..", "data", "pl_sweep_20261007_61_fig.json"))) if r["arm"] != "bb144"]
bb = {}
pat = re.compile(r"^bb144_ibm\w*?_p([0-9.]+)_")
for f in sorted(glob.glob(os.path.join(HERE, "..", "..", "records", "pl_sweep_20261009_bbibm", "*.json"))):
    m = pat.match(os.path.basename(f))
    if not m:
        continue
    p = float(m.group(1))
    if p not in B["PS"]:
        continue
    B["keep"](bb, p, B["row_from_file"](f, "bb144", None, None, "bbibm_depth7"))
# reference points drawn although they have fewer than ten failures
FORCE_DRAW = (0.002,)
for p in sorted(bb):
    r = bb[p]
    if p in FORCE_DRAW:
        r["drawn"] = True
    assert r["X_depth"] == 7 and r["Z_depth"] == 7, "not the depth-7 schedule"
    print("bb144 IBM p=%g  worse %s %d/%d  drawn=%s" % (p, r["worse_basis"], r["worse_failures"], B["shots_of"](r), r["drawn"]))
out = "pl_sweep_20261009_61_fig_bbibm.json"
json.dump(list(bb.values()) + rows, open(os.path.join(HERE, "..", "data", out), "w"), indent=1)
print("written", out)
