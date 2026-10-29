#!/usr/bin/env python3
"""Classify the certificate runs per (matrix, type) from their verdict files.

  lower = largest W with a CERTIFIED level (d > W), upper = smallest verified FOUND weight (d <= upper)
  exact            lower >= upper - 1
  lower only       certified level, no stored operator
  upper only       operator, no certified level
  open             both, lower < upper - 1
  conflict         lower >= upper (a certified level contradicts a verified operator)

  python summarize.py /u/zdliu/lp/oe_run/exact_v5/runs"""
import glob, json, os, sys
from collections import defaultdict
root = sys.argv[1]
info = defaultdict(lambda: dict(lower=0, upper=None, runs=0, incomplete=0))
names = {}
mismatches = 0
for v in glob.glob(os.path.join(root, "*", "verdict.json")):
    m = json.load(open(os.path.join(os.path.dirname(v), "manifest.json")))
    r = json.load(open(v))
    bad = [k for k in ("run_id", "matrix_sha256", "problem_sha256", "type", "W", "slices") if r.get(k) != m.get(k)]
    if r.get("result") == "CERTIFIED" and r.get("lower") != m.get("W"):
        bad.append("lower")
    if bad:
        print("verdict does not match its manifest (%s):" % ", ".join(bad), v); mismatches += 1; continue
    k = (m["matrix_sha256"], m["type"]); names[m["matrix_sha256"]] = (m["npz"], m["n"], m.get("archive_name"))
    s = info[k]; s["runs"] += 1
    if r["result"] == "CERTIFIED": s["lower"] = max(s["lower"], r["lower"])
    elif r["result"] == "FOUND": s["upper"] = r["upper"] if s["upper"] is None else min(s["upper"], r["upper"])
    else: s["incomplete"] += 1
def cls(s):
    lo, up = s["lower"], s["upper"]
    if up is not None and lo >= up: return "CONFLICT"
    if up is not None and lo >= up - 1: return "exact %d" % up
    if up is None and lo > 0: return "lower only: d > %d" % lo
    if up is not None and lo == 0: return "upper only: d <= %d" % up
    if up is not None: return "open: %d < d <= %d" % (lo, up)
    return "nothing proved"
conflicts = 0
for fp in sorted(names):
    npz, n, arch = names[fp]
    parts = []
    for T in "XZ":
        s = info.get((fp, T))
        parts.append("%s: %s%s" % (T, cls(s) if s else "no runs", (" (%d incomplete levels)" % s["incomplete"]) if s and s["incomplete"] else ""))
    conflicts += sum(1 for q in parts if "CONFLICT" in q)
    print("%s n=%s %s | %s" % (arch or fp[:16], n, os.path.basename(npz), " | ".join(parts)))
if mismatches or conflicts:
    print("%d verdict/manifest mismatches, %d conflicts" % (mismatches, conflicts)); sys.exit(1)
