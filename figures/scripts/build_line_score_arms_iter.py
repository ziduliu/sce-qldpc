#!/usr/bin/env python3
"""Builds figures/data/line_score_arms_iter.json, the data of Fig. 3, from the evaluation ledgers in ledgers/.

The run score after a step is the sum over the ten rate cells of the best program score in the cell; step 0 is
the eight seed programs, then steps 1..500. Language-model runs: one model call per step, and a step whose
response is rejected by the engine produces no child, so every child of the ledger is matched to the engine
iteration of its model call by the completion time in the controller log (figures/data/iteration_completions.txt:
run, log file, time stamp, iteration, program id, score of every "Program ... completed" line, extracted from the
cluster logs on 2026-10-19); a child without a completion line within two minutes takes the iteration of its
nearest matched neighbour in ledger order. Random-mutation and hill-climbing runs: one proposal per step, so the
step of a child is its position in the ledger.
Usage: python build_line_score_arms_iter.py [OUT]   (default ../data/line_score_arms_iter.json)"""
import datetime
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
LEDGERS = os.path.join(HERE, "..", "..", "ledgers")
N = 500
# two children of the gpt-6.1-sol CLER run whose program text is in no checkpoint of the run (they were
# evaluated after its last checkpoint); by the convention of Fig. 3 and of the tables they are left out of this run,
# which lowers its score
LOST = {"bed8db4fa554c55e", "e4a316a7c7bcb268"}

LLM = {
    "distance": {"luna_low": "cc_v7_gpt6lunalow500_w12_10c_prod_out", "luna_high": "cc_v7_gpt6lunahigh500_w12_10c_prod_out",
                 "luna_xhigh": "cc_v7_gpt6luna500_w12_10c_rep2_out", "gpt6_sol_high": "cc_v7_gpt6sol50_w12_10c_cmp_out",
                 "gpt61_sol_high": "cc_v7_gpt61sol500_w12_10c_out"},
    "circuit": {"luna_low": "pl10_gpt6lunalow_out", "luna_high": "pl10_gpt6lunahigh_out", "luna_xhigh": "pl10_gpt6lunaxhigh_out",
                "gpt6_sol_high": "pl10_gpt6sol50_out", "gpt61_sol_high": "pl10_gpt61sol_out"},
}
# the CLER random-mutation repeat 4 of the paper is the run folder pl10_rnd500v5_rep6_out
RND = {"distance": [("random_v5_rep%d" % i, "cc_v7_rnd500v5_w12_10c_rep%d_out" % i) for i in range(1, 6)]
       + [("local_v5_rep%d" % i, "cc_v7_rnd520locv5_w12_10c_rep%d_out" % i) for i in range(1, 6)],
       "circuit": [("random_v5_rep%d" % i, "pl10_rnd500v5_rep%d_out" % f) for i, f in ((1, 1), (2, 2), (3, 3), (4, 6), (5, 5))]
       + [("local_v5_rep%d" % i, "pl10_rnd520locv5_rep%d_out" % i) for i in range(1, 6)]}
NOTE = ("Run score after each step (0 = the eight seeds, then 500 steps) for every run of Fig. 3, per objective "
        "(distance = FOM, circuit = CLER). Language-model runs: the step of a child is the engine iteration of its model "
        "call. random_v5_rep1..5: the five random-mutation repeats; local_v5_rep1..5: the five hill-climbing repeats.")


def ledger(run, lost=()):
    rows = [json.loads(l) for l in open(os.path.join(LEDGERS, run, "eval_ledger.jsonl")) if l.strip()]
    rows.sort(key=lambda r: r["ts"])
    rows = [r for r in rows if r.get("code_sha1") not in lost]
    return rows[:8], rows[8:]


def completions(run):
    """(time stamp, iteration) of every completion line of the run's controller logs; the logs are in the cluster's
    local time (UTC+2 for all runs) and the ledger holds UTC epochs."""
    out = []
    for l in open(os.path.join(DATA, "iteration_completions.txt")):
        parts = l.rstrip("\n").split("|")
        if len(parts) < 6 or parts[0] != run:
            continue
        ts = datetime.datetime.strptime(parts[2][:19], "%Y-%m-%d %H:%M:%S").replace(tzinfo=datetime.timezone.utc).timestamp() - 7200
        out.append((ts, int(parts[3])))
    return out


def assign_iterations(children, comps):
    """Iteration of every child: the completion line whose time stamp is nearest to the ledger row's time stamp (the
    ledger row is written seconds before the engine logs the completion), matched one to one by increasing time
    difference; children without a completion within two minutes get the iteration of the nearest matched child in
    ledger order."""
    pairs = []
    for i, r in enumerate(children):
        for j, (ts, it) in enumerate(comps):
            d = ts - r["ts"]
            if -5 <= d <= 120:
                pairs.append((abs(d), i, j))
    pairs.sort()
    its = [None] * len(children)
    used = set()
    for _, i, j in pairs:
        if its[i] is None and j not in used:
            its[i] = comps[j][1]
            used.add(j)
    exact = sum(1 for x in its if x is not None)
    for i in range(len(its)):
        if its[i] is None:
            left = next((its[k] for k in range(i - 1, -1, -1) if its[k] is not None), None)
            right = next((its[k] for k in range(i + 1, len(its)) if its[k] is not None), None)
            its[i] = left if left is not None else (right if right is not None else 0)
    return its, exact


def curve_by_iteration(seeds, children, its, n=N):
    best = {}
    for r in seeds:
        b, f = r.get("rate_bin"), float(r.get("fitness") or 0)
        if b is not None and f > best.get(b, 0):
            best[b] = f
    tot = [sum(best.values())]
    ev = sorted(zip(its, children), key=lambda t: t[0])
    j = 0
    for it in range(1, n + 1):
        while j < len(ev) and ev[j][0] <= it:
            r = ev[j][1]
            b, f = r.get("rate_bin"), float(r.get("fitness") or 0)
            if b is not None and f > best.get(b, 0):
                best[b] = f
            j += 1
        tot.append(sum(best.values()))
    beyond = sum(1 for it, _ in ev if it > n)
    return tot, beyond


if __name__ == "__main__":
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(DATA, "line_score_arms_iter.json")
    out = {}
    for block in ("distance", "circuit"):
        out[block] = {}
        for key, run in LLM[block].items():
            seeds, ch = ledger(run, LOST if (block == "circuit" and key == "gpt61_sol_high") else ())
            its, exact = assign_iterations(ch, completions(run))
            tot, beyond = curve_by_iteration(seeds, ch, its)
            out[block][key] = dict(run=run, total=tot, children=len(ch), exact=exact, beyond_N=beyond,
                                   final_by_children=None, max_iteration=max(its) if its else 0)
            print("%-8s %-15s children %3d  matched %3d  last iteration %d  beyond %d: %d  run score at %d steps %.2f"
                  % (block, key, len(ch), exact, max(its), N, beyond, N, tot[-1]))
        for key, run in RND[block]:
            seeds, ch = ledger(run)
            its = list(range(1, len(ch) + 1))
            tot, beyond = curve_by_iteration(seeds, ch, its)
            out[block][key] = dict(run=run, total=tot, children=len(ch), exact=len(ch), beyond_N=beyond)
            print("%-8s %-15s children %3d  run score at %d steps %.2f" % (block, key, len(ch), N, tot[-1]))
    out["_note"] = NOTE
    json.dump(out, open(out_path, "w"))
    print("wrote", out_path)
