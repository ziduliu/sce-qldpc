#!/usr/bin/env python3
"""Build figures/data/pl_sweep_20261007_61_fig.json, the row table behind the circuit-level
sweep figures and the code table, from records/pl_sweep_20261006_61/ (sweep),
records/pl_sweep_20261007_topup/ (top-up) and the EXTRA directories; EXTRA wins over the
top-up, which wins over the sweep. Within one directory the file with more shots wins; files
that carry an "error" are treated as missing. A point is drawn only if its worse basis has at
least MIN_FAIL failures. [[144,12,12]] rows missing from these sources are taken from the
previous build (OUT + ".prev").
"""
import glob
import json
import os
import re
import shutil

HERE = os.path.dirname(os.path.abspath(__file__))
OLD = os.path.join(HERE, "..", "..", "records", "pl_sweep_20261006_61")
NEW = os.path.join(HERE, "..", "..", "records", "pl_sweep_20261007_topup")
EXTRA = [os.path.join(HERE, "..", "..", "records", d) for d in ("pl_sweep_20261009_c0dgx",)]
OUT = os.path.join(HERE, "..", "data", "pl_sweep_20261007_61_fig.json")
PS = (0.001, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005)
MIN_FAIL = 10
# codes in the figures and the table: (arm, cell) -> matrix hash
CODES = {("dist", 0): "fd925ff20fe235cc", ("dist", 1): "3c7722431bc2662e", ("dist", 2): "dc33333253c79e16",
         ("dist", 3): "d233cc7273de92c2", ("dist", 4): "b51b14afe214768f", ("dist", 5): "15d39679803f52d4",
         ("dist", 6): "db18259eb1856d92", ("dist", 7): "b7bb3363de23fb63", ("dist", 8): "612dfb68fc96d396",
         ("dist", 9): "cf5015a213992503",
         # the second two-block code of cell 0, drawn next to the champion
         ("dist2", 0): "6612ba96961ef2c1",
         # cells 0 and 4: the best programs of those cells whose program text is available
         ("plc", 0): "82b2bf20fa54f6db", ("plc", 1): "30b645d7a9a46846", ("plc", 2): "ba427de15c84301e",
         ("plc", 3): "239bca5f913aa8b1", ("plc", 4): "7084834cc57bd76f", ("plc", 5): "8031fd1f7b90d19b",
         ("plc", 6): "d8d92f24197070fc", ("plc", 7): "ccf067d3f16c1b1e", ("plc", 8): "f4eda3e53eb8fdb2",
         ("plc", 9): "217b1dab11f6cb77"}
SHA8 = {v[:8]: k for k, v in CODES.items()}
PAT = re.compile(r"^(dist|plc)61_c(\d)_([0-9a-f]{8})_p([0-9.]+)_p[0-9.]+_s(\d+)\.json$")
PATBB = re.compile(r"^bb144_p([0-9.]+)_p[0-9.]+_s(\d+)\.json$")


def row_from_file(path, arm, cell, sha, source):
    d = json.load(open(path))
    if d.get("error") or not d.get("codes"):
        print("skipped (error, treated as missing): %s: %s" % (os.path.basename(path), str(d.get("error"))[:60]))
        return None
    c = d["codes"][0]
    det = c["detail"]
    r = dict(arm=arm, cell=cell, sha=sha, source=source, p=float(d["p"]), rounds=d["rounds"],
             max_shots=d["max_shots"], max_errors=d["max_errors"], seconds=d.get("seconds"),
             family=c.get("family"), idx=c.get("idx"), n=c["n"], k=c["k"], w=c.get("w"), d2s=c.get("d2s"),
             score=det.get("score"), score_ceiling=det.get("score_ceiling"), worse_basis=det.get("worse_basis"),
             sched_note=det.get("sched_note"), cnots=det.get("cnots"))
    for b in ("X", "Z"):
        pb = det["bases"].get(b, {})
        for key in ("score", "shots", "shot_failures", "stopped_on", "logical_failures", "worst_logical_failures",
                    "pl_round_per_logical", "pl_round_per_logical_ci_hi", "depth", "resolved_by_stage", "unsatisfied"):
            r["%s_%s" % (b, key)] = pb.get(key)
    wb = r["worse_basis"] or "Z"
    r["worse_failures"] = r.get(wb + "_shot_failures") or 0
    r["drawn"] = bool(r["worse_failures"] >= MIN_FAIL)
    return r


def scan(dirpath, source):
    rows = {}
    for f in sorted(glob.glob(os.path.join(dirpath, "*.json"))):
        base = os.path.basename(f)
        m = PAT.match(base)
        if m:
            arm, cell, sha8, p = m.group(1), int(m.group(2)), m.group(3), float(m.group(4))
            if sha8 not in SHA8 or SHA8[sha8][1] != cell:
                continue                    # a code of the sweep that is not in the figure
            arm = SHA8[sha8][0]             # "dist2" for the second cell-0 code
            key = (arm, cell, p)
            keep(rows, key, row_from_file(f, arm, cell, CODES[(arm, cell)], source))
            continue
        m = PATBB.match(base)
        if m:
            p = float(m.group(1))
            keep(rows, ("bb144", None, p), row_from_file(f, "bb144", None, None, source))
    return rows


def shots_of(r):
    wb = r.get("worse_basis") or "Z"
    return r.get(wb + "_shots") or 0


def keep(rows, key, r):
    # a failed file never enters; of two valid files for one leg, the larger sample wins
    if r is None:
        return
    if key not in rows or shots_of(r) > shots_of(rows[key]):
        rows[key] = r


old = scan(OLD, "sweep") if os.path.isdir(OLD) else {}
new = scan(NEW, "topup") if os.path.isdir(NEW) else {}
extra = {}
for d in EXTRA:
    if os.path.isdir(d):
        for k, r in scan(d, os.path.basename(d)).items():
            keep(extra, k, r)
prev_path = OUT + ".prev"
if os.path.exists(OUT) and not os.path.exists(prev_path):
    shutil.copy(OUT, prev_path)
bb_prev = {}
if os.path.exists(prev_path):
    for r in json.load(open(prev_path)):
        if r["arm"] == "bb144":
            r = dict(r, source=r.get("source") or "sweep_20260928")
            wb = r.get("worse_basis") or "Z"
            r["worse_failures"] = r.get(wb + "_shot_failures") or 0
            r["drawn"] = bool(r["worse_failures"] >= MIN_FAIL)
            bb_prev[("bb144", None, float(r["p"]))] = r

merged = {}
for key in set(old) | set(new) | set(extra) | set(bb_prev):
    if key[2] not in PS:
        continue
    merged[key] = extra.get(key) or new.get(key) or old.get(key) or bb_prev.get(key)
rows = [merged[k] for k in sorted(merged, key=lambda k: (k[0] != "bb144", k[0], k[1] if k[1] is not None else -1, k[2]))]
json.dump(rows, open(OUT, "w"), indent=1)
n_new = sum(r["source"] == "topup" for r in rows)
n_draw = sum(r["drawn"] for r in rows)
print("wrote %s: %d rows (%d from the top-up), %d drawn (>=%d failures in the worse basis); not drawn:"
      % (os.path.basename(OUT), len(rows), n_new, n_draw, MIN_FAIL))
for r in rows:
    if not r["drawn"]:
        print("   %-6s cell %-4s p=%-6g fX=%s fZ=%s" % (r["arm"], r["cell"], r["p"], r["X_shot_failures"], r["Z_shot_failures"]))
