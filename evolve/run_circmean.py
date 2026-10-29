"""Multi-seed launcher for an OpenEvolve run.

openevolve.cli takes one initial program; this launcher uses the library
API instead: construct the controller with the first seed, evaluate every
seed through the normal evaluator, add them to the database round-robin
over islands, then run.

Usage:
  python3 run_circmean.py --config oe_circmean.yaml \
      --seeds seeds_rates --evaluator oe_evaluator_circmean.py \
      --iterations 60 --output /u/zdliu/lp/oe_run/cmean2_out
"""
import argparse
import asyncio
import glob
import os
import sys
import uuid

from openevolve.controller import OpenEvolve
from openevolve.database import Program


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seeds", required=True,
                    help="directory of seed_*.py programs")
    ap.add_argument("--evaluator", default="oe_evaluator_circmean.py")
    ap.add_argument("--iterations", type=int, default=60,
                    help="fresh: total iterations; with --checkpoint: "
                         "ADDITIONAL iterations after the checkpoint")
    ap.add_argument("--output", required=True)
    ap.add_argument("--checkpoint", default="",
                    help="resume from this checkpoint dir (seeds are NOT "
                         "re-evaluated; the MAP-Elites grid is rebuilt from "
                         "the stored programs' metrics)")
    args = ap.parse_args()

    seeds = sorted(glob.glob(os.path.join(args.seeds, "seed_*.py")))
    if not seeds:
        print("no seeds in %s" % args.seeds)
        return 1
    print("seeds (%d): %s" % (len(seeds),
                              ", ".join(os.path.basename(s) for s in seeds)))

    evolve = OpenEvolve(
        initial_program_path=seeds[0],
        evaluation_file=args.evaluator,
        config=None,
        config_path=args.config,
        output_dir=args.output,
    )
    islands = evolve.config.database.num_islands

    if args.checkpoint:
        db = evolve.database
        db.load(args.checkpoint)
        # When feature_dimensions differ from the checkpoint's, load() purges
        # the grid and rebuilds it only if a LAYOUT_* protocol stamp is present.
        # No stamp is written here, so rebuild from the stored metrics (same
        # evaluator and budgets; only the cell coordinates changed).
        if not db.feature_map and db.programs:
            db._rebuild_grid_from_programs()
        print("resumed %s: %d programs, last_iteration=%d, grid cells=%d"
              % (args.checkpoint, len(db.programs), db.last_iteration,
                 len(db.feature_map)), flush=True)
        for key, pid in sorted(db.feature_map.items()):
            p = db.programs.get(pid)
            print("  cell %-5s %s score=%s" % (
                key, pid[:8], p.metrics.get("combined_score") if p else "?"),
                flush=True)
        best = await evolve.run(iterations=args.iterations)
        if best is not None:
            print("best: %s" % best.metrics.get("combined_score"))
        return 0

    # Add every seed here, including seeds[0]: the controller only adds its
    # initial program when the database is empty. Seeds are evaluated
    # concurrently (one spool request each), then added in seed order with
    # round-robin island assignment.
    async def _eval_seed(i, path):
        code = open(path).read()
        pid = str(uuid.uuid4())
        print("evaluating seed %s ..." % os.path.basename(path), flush=True)
        metrics = await evolve.evaluator.evaluate_program(code, pid)
        return i, path, code, pid, metrics

    results = await asyncio.gather(*[_eval_seed(i, p) for i, p in enumerate(seeds)])
    for i, path, code, pid, metrics in sorted(results, key=lambda t: t[0]):
        prog = Program(id=pid, code=code,
                       language=evolve.config.language,
                       metrics=metrics, iteration_found=0)
        evolve.database.add(prog, target_island=i % islands)
        print("  added %s: combined_score=%s island=%d"
              % (os.path.basename(path),
                 metrics.get("combined_score"), i % islands), flush=True)

    best = await evolve.run(iterations=args.iterations)
    if best is not None:
        print("best: %s" % best.metrics.get("combined_score"))
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
