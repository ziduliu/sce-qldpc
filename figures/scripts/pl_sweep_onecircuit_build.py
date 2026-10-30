#!/usr/bin/env python3
"""Build figures/data/pl_sweep_20261030_onecircuit_fig.json, the rows behind Figure 2 (one row per code and
physical error rate p, with the X- and Z-basis counts), from the circuit-level records.

Every code of the figure is measured on one fixed syndrome-extraction schedule,
records/pl_sweep_20261030_onecircuit/sched/<code>.json, in both bases and at every p; the X-basis and Z-basis
experiments are the same physical circuit. The counts of a basis come either from a line of
records/pl_sweep_20261030_onecircuit/legs.tsv (result in out/<file>; row field one_circuit) or from the sweep
record of that point (records/pl_sweep_20261006_61/, records/pl_sweep_20261007_topup/,
records/pl_sweep_20261009_c0dgx/; row field kept_basis), whose own work directory must hold the same schedule
for that basis. A row with a basis that cannot be traced to the fixed schedule stops the build if the row is
drawn and is left out otherwise. The [[144,12,12]] rows come from records/pl_sweep_20261009_bbibm/ (the
published depth-7 schedule). The plotted basis of a point is the one with the larger per-round per-logical rate;
a point is drawn if that basis has at least MIN_FAIL failures."""
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.normpath(os.path.join(HERE, "..", ".."))
REC = os.path.join(ROOT, "records")
SWEEPS = [("sweep", "pl_sweep_20261006_61"), ("topup", "pl_sweep_20261007_topup"),
          ("pl_sweep_20261009_c0dgx", "pl_sweep_20261009_c0dgx")]          # later entries win
ONE = os.path.join(REC, "pl_sweep_20261030_onecircuit")
OUT = os.path.join(ROOT, "figures", "data", "pl_sweep_20261030_onecircuit_fig.json")
PS = (0.001, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005)
MIN_FAIL = 10
FIELDS = ("score", "shots", "shot_failures", "stopped_on", "logical_failures", "worst_logical_failures",
          "pl_round_per_logical", "pl_round_per_logical_ci_hi", "depth", "resolved_by_stage", "unsatisfied")
# codes in the figure: (arm, cell) -> matrix hash; "dist2" is the second two-block code of cell 0
CODES = {("dist", 0): "fd925ff20fe235cc", ("dist", 1): "3c7722431bc2662e", ("dist", 2): "dc33333253c79e16",
         ("dist", 3): "d233cc7273de92c2", ("dist", 4): "b51b14afe214768f", ("dist", 5): "15d39679803f52d4",
         ("dist", 6): "db18259eb1856d92", ("dist", 7): "b7bb3363de23fb63", ("dist", 8): "612dfb68fc96d396",
         ("dist", 9): "cf5015a213992503", ("dist2", 0): "6612ba96961ef2c1",
         ("plc", 0): "82b2bf20fa54f6db", ("plc", 1): "30b645d7a9a46846", ("plc", 2): "ba427de15c84301e",
         ("plc", 3): "239bca5f913aa8b1", ("plc", 4): "7084834cc57bd76f", ("plc", 5): "8031fd1f7b90d19b",
         ("plc", 6): "d8d92f24197070fc", ("plc", 7): "ccf067d3f16c1b1e", ("plc", 8): "f4eda3e53eb8fdb2",
         ("plc", 9): "217b1dab11f6cb77"}
SHA8 = {v[:8]: k for k, v in CODES.items()}
PAT = re.compile(r"^(dist|plc)61_c(\d)_([0-9a-f]{8})_p([0-9.]+)_p[0-9.]+_s(\d+)\.json$")
PATBB = re.compile(r"^bb144_ibm\w*?_p([0-9.]+)_")


def row_from_file(path, arm, cell, sha, source):
    d = json.load(open(path))
    if d.get("error") or not d.get("codes"):
        return None
    c = d["codes"][0]
    det = c["detail"]
    r = dict(arm=arm, cell=cell, sha=sha, source=source, p=float(d["p"]), rounds=d["rounds"],
             family=c.get("family"), idx=c.get("idx"), n=c["n"], k=c["k"], w=c.get("w"), d2s=c.get("d2s"),
             score=det.get("score"), worse_basis=det.get("worse_basis"), cnots=det.get("cnots"))
    for b in ("X", "Z"):
        pb = det["bases"].get(b, {})
        for key in FIELDS:
            r["%s_%s" % (b, key)] = pb.get(key)
        r[b + "_max_shots"], r[b + "_max_errors"] = d["max_shots"], d["max_errors"]
    r["_record"] = os.path.relpath(path, ROOT)
    return r


def shots_of(r):
    return r.get((r.get("worse_basis") or "Z") + "_shots") or 0


def keep(rows, key, r):
    # a failed file never enters; of two valid files for one point, the larger sample wins
    if r is not None and (key not in rows or shots_of(r) > shots_of(rows[key])):
        rows[key] = r


def fail(msg):
    raise SystemExit("build stopped: " + msg)


# rows of the sweep records
base = {}
for source, d in SWEEPS:
    found = {}
    for f in sorted(glob.glob(os.path.join(REC, d, "*.json"))):
        m = PAT.match(os.path.basename(f))
        if not m or m.group(3) not in SHA8 or SHA8[m.group(3)][1] != int(m.group(2)) or float(m.group(4)) not in PS:
            continue
        arm, cell = SHA8[m.group(3)]
        r = row_from_file(f, arm, cell, CODES[(arm, cell)], source)
        if r is not None:
            r["_work"] = os.path.join(os.path.dirname(f), "work_%s61_c%s_%s_p%s_p%s" % (
                m.group(1), m.group(2), m.group(3), m.group(4), m.group(4)), "asc_schedules.json")
        keep(found, (arm, cell, float(m.group(4))), r)
    base.update(found)

# the one-circuit measurements
codes = json.load(open(os.path.join(ONE, "codes.json")))
lines = [l.rstrip("\n").split("\t") for l in open(os.path.join(ONE, "legs.tsv")) if l.strip()][1:]
seen = set()
for fname, code, cell, q, p, basis, cap, maxerr, shards in lines:
    c = codes[code]
    key = (c["arm"], int(c["cell"]), float(p))
    if (key, basis) in seen: fail("two lines for %s p=%s %s" % (code, p, basis))
    seen.add((key, basis))
    if key not in base: fail("no sweep row for %s p=%s" % (code, p))
    d = json.load(open(os.path.join(ONE, "out", fname)))
    rc = d["codes"][0]; det = rc["detail"]; bases = det["bases"]
    pr = []
    if rc.get("name") != "c%s" % cell or (rc["n"], rc["k"]) != (c["n"], c["k"]): pr.append("code")
    if det.get("sched_note") != "cached": pr.append("schedule %s" % det.get("sched_note"))
    if abs(float(d["p"]) - float(p)) > 1e-12 or sorted(bases) != [basis]: pr.append("p or basis")
    if (int(d["max_shots"]), int(d["max_errors"])) != (int(cap), int(maxerr)): pr.append("caps")
    v = bases.get(basis, {})
    if int(v.get("depth", -1)) != int(c["schedule_depth"]): pr.append("depth")
    s, nf = int(v.get("shots", 0)), int(v.get("shot_failures", 0))
    if not 0 < s <= int(cap) or (int(maxerr) == 0 and s != int(cap)) or (int(maxerr) > 0 and nf < int(maxerr) and s != int(cap)):
        pr.append("shots %d of %s" % (s, cap))
    if pr: fail("%s: %s" % (fname, ", ".join(pr)))
    r = base[key]
    for k in FIELDS:
        r["%s_%s" % (basis, k)] = v.get(k)
    r[basis + "_max_shots"], r[basis + "_max_errors"] = int(cap), int(maxerr)
    r.setdefault("one_circuit", {})[basis] = fname

rows, dropped, kept = [], [], 0
for key in sorted(base, key=lambda k: (k[0], k[1], k[2])):
    r = base[key]
    code = "%s_c%d_%s" % (r["arm"], r["cell"], r["sha"][:8])
    pinned = json.load(open(os.path.join(ONE, "sched", code + ".json")))
    oc = r.get("one_circuit", {})
    traced = True
    for b in ("X", "Z"):
        if b in oc: continue
        k = "c%d|%s" % (r["cell"], b)
        w = json.load(open(r["_work"])) if os.path.exists(r["_work"]) else {}
        if k not in w or w[k]["ticks"] != pinned[k]["ticks"]:
            traced = False; continue
        r.setdefault("kept_basis", {})[b] = r["_record"]
    rx, rz = float(r["X_pl_round_per_logical"]), float(r["Z_pl_round_per_logical"])
    wb = ("X" if rx > rz else "Z") if rx != rz else ("X" if float(r["X_pl_round_per_logical_ci_hi"]) >= float(r["Z_pl_round_per_logical_ci_hi"]) else "Z")
    r["worse_basis"], r["score"] = wb, r[wb + "_score"]
    r["worse_failures"] = r.get(wb + "_shot_failures") or 0
    r["drawn"] = bool(r["worse_failures"] >= MIN_FAIL)
    if not traced:
        if r["drawn"] or (r["arm"], r["cell"], r["p"]) == ("dist2", 0, 0.002):
            fail("%s p=%g has a basis not measured on the fixed schedule" % (code, r["p"]))
        dropped.append((code, r["p"])); continue
    r["source"] = "pl_sweep_20261030_onecircuit" if oc else r["source"]
    r["sched_note"] = "records/pl_sweep_20261030_onecircuit/sched/%s.json" % code
    for k in ("_record", "_work"): r.pop(k, None)
    kept += len(r.get("kept_basis", {}))
    rows.append(r)

bb = {}
for f in sorted(glob.glob(os.path.join(REC, "pl_sweep_20261009_bbibm", "*.json"))):
    m = PATBB.match(os.path.basename(f))
    if m and float(m.group(1)) in PS:
        keep(bb, float(m.group(1)), row_from_file(f, "bb144", None, None, "bbibm_depth7"))
for p in sorted(bb):
    r = bb[p]
    if r["X_depth"] != 7 or r["Z_depth"] != 7: fail("[[144,12,12]] p=%g is not on the depth-7 schedule" % p)
    r["worse_failures"] = r.get(r["worse_basis"] + "_shot_failures") or 0
    r["drawn"] = bool(r["worse_failures"] >= MIN_FAIL) or p == 0.002   # the reference at p = 0.002 is drawn
    r.pop("_record", None)
json.dump(list(bb.values()) + rows, open(OUT, "w"), indent=1)
print("wrote %s: %d rows (%d drawn) and %d [[144,12,12]] rows; %d bases from %s, %d kept from sweep records; "
      "left out (not drawn, not traced to the fixed schedule): %s"
      % (os.path.relpath(OUT, ROOT), len(rows), sum(r["drawn"] for r in rows), len(bb), len(lines),
         os.path.relpath(ONE, ROOT), kept, ", ".join("%s p=%g" % x for x in dropped) or "none"))
