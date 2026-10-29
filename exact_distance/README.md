# Exact-distance certifier, v5.1 (2026-10-14)

Certified bounds on the minimum distance of CSS codes by the irreducible-cluster branch and bound of
Dumer, Kovalev and Pryadko (IEEE TIT 2017, arXiv:1611.07164), implemented from the description in
Appendix B of arXiv:2609.30069. The authors' own implementation of the method is dist-m4ri
(github.com/qec-pages/dist-m4ri).

It agrees with exhaustive enumeration in 2,245 comparisons on 206 small matrices.

## What a result means

For one type (X: H = HZ, G = HX; Z: H = HX, G = HZ) and a weight limit W the search returns
NONE (no logical operator of weight <= W: d > W), FOUND (an operator of weight <= W; the search stops
at the first one, so this is an upper bound only) or TIMEOUT / ERROR (nothing proved). The driver reports

- `lower`: the largest W of a completed NONE search, 0 if none (the start of a climb is never a bound),
- `upper`: the smallest weight of a FOUND operator re-verified by the driver,
- `exact`: `upper`, only when `lower >= upper - 1`.

The quantum distance min(d_X, d_Z) is exact only when both types are bounded accordingly.

## Checks

- The lower bound starts at 0 and rises only on a completed NONE search; `wmin > wmax` is rejected; the
  exit status is 3 when a requested search did not complete and 2 on input errors.
- FOUND is an upper bound; a value is exact only when a certified level W = upper - 1 exists.
- A climb of one type (`--type X` or `--type Z`) gives no bound on the quantum distance: the type that
  was not searched counts as lower bound 0, and `d_exact` needs both types.
- Sliced levels: a manifest is written before submission (run id, matrix, orbit file, binary and
  problem fingerprints, slice count). Every slice and the aggregator check it; a level is certified
  only with exactly one completed record per slice 0..b-1 and equal frontier size and depth. Each
  slice record is checked against the binary's own `INFO` line (W, slice index and count, size of the
  frontier share, problem fingerprint, status) and carries only status and fingerprints, no bounds.
  A FOUND verdict keeps its valid upper bound and records whether the level was complete.
- Orbit files are re-validated on every run: same matrix fingerprint, each generator a permutation
  preserving both row spaces, orbits regenerated from the generators.
- Packed input requires n (argument or stored in the npz); shapes, values and padding bits are checked.
- Each submission has its own run directory; outputs are written atomically; the aggregator reads only
  that directory, deletes an earlier verdict before aggregating and never reports a level as
  certified when the aggregation itself failed.
- `dkp_bb` range-checks every index, count and root (malformed input exits 2); the time limit also
  covers the frontier build and the clock is checked every 65,536 nodes; the binary prints an
  explicit status, and the driver adds an external process guard.
- `summarize.py` checks run id, matrix and problem fingerprints, type, W and slice count of every
  verdict against its manifest and exits 1 on a mismatch or a conflict.

## Files

- `src/dkp_bb.cpp` search (C++, OpenMP); prints `INFO` (problem and frontier description), then
  `FOUND w v...` / `NONE W` / `TIMEOUT done/total`, then `STATS nodes secs`.
- `src/exact_bb.py` driver: climbs, manifests, slices, aggregation, self-test.
- `src/code_orbits.py` automorphism generators with nauty (pynauty), checked on the matrices.
- `src/summarize.py` classifies runs per (matrix, type): exact / lower only / upper only / open / conflict.
- `cluster_scripts/bb_climb_v5.sh`, `bb_slice_v5.sbatch`, `bb_agg_v5.sbatch` Slurm submission.
- `tests/test_regression.py` 31 regression cases, one per check above, and an end-to-end manifest/slice/aggregate flow.

## Build and test

```
g++ -O3 -march=x86-64-v3 -fopenmp -o src/dkp_bb src/dkp_bb.cpp
cd tests && DKP_BIN=../src/dkp_bb python test_regression.py      # 31 cases, 0 failed (macOS clang without OpenMP, and the cluster build)
```

## Usage

```
python src/exact_bb.py --npz code.npz --n 585 --type both --wmin 1 --wmax 24 --threads 8 --time 3600 --bin src/dkp_bb --out r.json
python src/code_orbits.py --npz code.npz --n 585 --out orbits.json
cluster_scripts/bb_climb_v5.sh TAG code.npz 585 X 23 orbits.json 23 "99:60"
python src/summarize.py /u/zdliu/lp/oe_run/exact_v5/runs
```

Limits: n and the number of checks at most 1024 (`NW = 16` in `dkp_bb.cpp`).

## Files and server paths

The code ran on the cluster from the original server paths below (hard-coded in the job scripts; this
folder does not change them). To run it elsewhere, set `CERT` and `OUTDIR` or edit the paths.

| file | what it does | server path |
|---|---|---|
| `src/dkp_bb.cpp` | branch-and-bound search of one type and weight level (compiled to `src/dkp_bb`) | `/u/zdliu/lp/cert_v5/src/dkp_bb.cpp` |
| `src/exact_bb.py` | driver: climbs, manifests, slices, aggregation, self-test | `/u/zdliu/lp/cert_v5/src/exact_bb.py` |
| `src/code_orbits.py` | automorphism generators and qubit orbits with nauty, checked on the matrices | `/u/zdliu/lp/cert_v5/src/code_orbits.py` |
| `src/summarize.py` | per (matrix, type) classification of the run directories | `/u/zdliu/lp/cert_v5/src/summarize.py` |
| `build_and_test.sbatch` | builds `dkp_bb` on the cluster and runs the regression tests | `/u/zdliu/lp/cert_v5/build_and_test.sbatch` |
| `cluster_scripts/bb_climb_v5.sh` | one weight level of a climb: manifest, slice jobs, aggregator job | `/u/zdliu/lp/cert_v5/cluster_scripts/bb_climb_v5.sh` |
| `cluster_scripts/bb_slice_v5.sbatch` | one slice of a level | `/u/zdliu/lp/cert_v5/cluster_scripts/bb_slice_v5.sbatch` |
| `cluster_scripts/bb_agg_v5.sbatch` | aggregates the slices of a level; a certified level starts the next one, up to `WMAX` | `/u/zdliu/lp/cert_v5/cluster_scripts/bb_agg_v5.sbatch` |
| `tests/test_regression.py` | regression cases | `/u/zdliu/lp/cert_v5/tests/` |

## How it was invoked for the paper

The codes come from the run archives as `.npz` files with `HX`, `HZ` (and `n`). For each code and type
(`X`, `Z`):

```
python src/code_orbits.py --npz code.npz --n <n> --out orbits.json         # automorphism orbits (lifted products: centre translations)
cluster_scripts/bb_climb_v5.sh <TAG> code.npz <n> <X|Z> <W start> orbits.json <WMAX> "<W>:<slices>,...,99:<slices>"
python src/summarize.py /u/zdliu/lp/oe_run/exact_v5/runs                  # exact / lower only / upper only / open
```

The aggregator starts the next level after a certified level and stops at the first FOUND operator or
incomplete level. `src/exact_bb.py` alone runs a whole climb in one process (see Usage above).
A distance is reported as exact only when a FOUND operator of weight `upper` is verified and the level
`upper - 1` is certified NONE for both types.

The records in `../records/recert_v5_20261014/`, `../records/exact_v5_20261017_w9cell0/` and
`../records/exact_v5_20261022_refinement/` were produced by this version (manifest, slice records and
verdict per level). The records in `../records/exact_bb_20261006_61/` and `../records/w9cert_20261013/`
were produced by an earlier version of the same search (per-level slice files and an aggregate,
`<label>_<type>_W<w>_agg.json` / `.txt`, with the climb log `<label>_<type>_climb.log`); `wit_<label>`
records are the searches that found the operator of weight `upper`.

## Dependencies

C++17 compiler with OpenMP (g++ from the `gcc/11` module on the cluster, `-march=x86-64-v3`), Python 3.12
with numpy, pynauty 2.8.8.1 (nauty) for `code_orbits.py`, Slurm for the cluster scripts.
