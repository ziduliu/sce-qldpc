# distance_estimates

GPU distance estimator (decoder-based search for light logical operators, Methods) and the
drivers of the re-measurements. The same function, `estimator_gpu_v2.estimate_d_gpu(HX, HZ, LX, LZ,
budget_seconds, seed)`, gives every distance number of the runs; only the time budget differs:

- 2 s screen of every distinct candidate of a child (both objectives),
- 30 s measurement of the five best candidates of a child (FOM objective),
- 600 s re-measurement of elite codes, run twice with different seeds; the smaller reading is reported.

Every returned weight belongs to an explicit logical operator that is re-verified on the host, so each
reading is an upper bound on the distance.

The code ran on the cluster from the original server paths listed below (hard-coded in the scripts; this
folder does not change them). To run it elsewhere, place each file at its server path or edit the paths.

## Files

| file | what it does | server path |
|---|---|---|
| `estimator_gpu_v2.py` | the estimator: for each logical row L_i decode the syndrome (0,...,0,1) of [H; L_i] by scaled min-sum BP, OSD-0 with single and double flips of the 48 least reliable free columns, stabilizer descent, host verification; X and Z side, time budget split | `/u/zdliu/lp/dopt/bdirect_v2/estimator_gpu_v2.py` |
| `gpu_core_v2.py` | batched GPU kernels of the estimator (packed GF(2) arithmetic, BP, OSD, descent) | `/u/zdliu/lp/dopt/bdirect_v2/gpu_core_v2.py` |
| `gpu_graphs.py` | CUDA-graph capture of the batched kernels (shape bucketing `CSSDIST_SHAPE_BUCKET` is in the estimator) | `/u/zdliu/lp/dopt/bdirect_v2/gpu_graphs.py` |
| `rref_panel.py` | blocked (panel) GF(2) row reduction used by the OSD step | `/u/zdliu/lp/dopt/bdirect_v2/rref_panel.py` |
| `fast_gf2.py` | bit-packed GF(2) linear algebra on the CPU: rank, kernel, CSS logical operators (used by the evaluators and the circuit code as well) | `/u/zdliu/lp/dopt/bdirect_v2/fast_gf2.py` |
| `mc_rate.py` | batched min-sum BP with arbitrary syndromes (`bp_minsum_syn`); imported at load time by `../circuit_schedule/circ_1side.py` | `/u/zdliu/lp/dopt/bdirect_v2/mc_rate.py` |
| `dem_distance_v2.py` | the evaluators' interface: `d2s_batch(items, budget)` runs `estimate_d_gpu` on a batch of codes in a fresh child process and returns d with all verified minimum-weight operators; its circuit-level (DEM) functions are not used by the paper runs | `/u/zdliu/lp/circmean/dem_distance_v2.py` |
| `remeasure/select_cc_champions.py` | picks the champion program of every rate cell of a FOM run (highest fitness in the ledger) and writes its measured codes to a codes file | `/u/zdliu/lp/circmean/select_cc_champions.py` |
| `remeasure/sel61/select.sbatch` | the selection job of the gpt-6.1-sol FOM run | `/u/zdliu/lp/jobs/sel61/select.sbatch` |
| `remeasure/deepverify_cc.py` | 600 s re-measurement: `--budget 600 --reps 2`, minimum over the repetitions, results appended per code | `/u/zdliu/lp/plevolve_20260912/deepverify_cc.py` |
| `remeasure/deepverify_cc.sbatch` | GPU job around `deepverify_cc.py`; several jobs share one codes file with `PART=i/N` | `/u/zdliu/lp/plevolve_20260912/deepverify_cc.sbatch` |

## How it was invoked

- In the runs: `../evolve/eval_circmean_v3.py` and `../evolve/eval_circmean_pl.py` import
  `dem_distance_v2` (and through it `estimator_gpu_v2`) from `/u/zdliu/lp/circmean` and
  `/u/zdliu/lp/dopt/bdirect_v2`; the screen is `d2s_batch(..., budget=CM_D2S_BUDGET)` with
  `CM_D2S_BUDGET=2`, the FOM measurement `d2s_batch(..., budget=CM_D2S_DEEP)` with `CM_D2S_DEEP=30` and
  `CM_DIST_MODE=cc` (worker exports `DIST=cc`, `D2SDEEP` default 30; see `../evolve/circmean_worker.sbatch`).
- Re-measurement of the gpt-6.1-sol FOM elite codes:

```
# on the cluster, from /u/zdliu/lp/circmean
python -u select_cc_champions.py --run /u/zdliu/lp/jobs/sel61 --arm gpt61sol \
    --out /u/zdliu/lp/jobs/codes_gpt61sol_20261004.json          # remeasure/sel61/select.sbatch
sbatch --export=ALL,CODES=<codes.json>,OUT=<results.jsonl>,PART=0/2 deepverify_cc.sbatch
sbatch --export=ALL,CODES=<codes.json>,OUT=<results.jsonl>,PART=1/2 deepverify_cc.sbatch
# BUDGET=600 and REPS=2 are the defaults
```

## Dependencies

Python 3.12, numpy, torch 2.5.1 (CUDA 12.1 build) on an NVIDIA GPU (RTX 6000). The 2 s
and 30 s budgets are wall-clock budgets, so a slower GPU gives slightly higher (more conservative)
readings. Slurm modules `gcc/11`, `cuda/12.2`, `python-waterboa/2024.06`.
