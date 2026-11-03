# evolve

The search loop: the modified OpenEvolve engine, the run state and prompt building, the evaluators of the
two objectives (FOM = mean k d^2/n of the five measured codes of a child, with the code-capacity distance;
CLER = circuit-level logical error rate score of the five measured codes), the launchers and job scripts of
the paper runs, their configurations and seed programs, and the random-mutation control.

The code ran on the cluster from the original server paths listed below (hard-coded in the scripts; this
folder does not change them, except that the name of the deployment directory of the random-mutation
and hill-climbing controls is shortened to `/u/zdliu/deploy/`). To run it elsewhere, place each file at its
server path or edit the paths.

## How one run works

1. A controller job (`oe_circmean.sbatch` for FOM, `oe_circmean_pl.sbatch` for CLER) starts the RUN STATE
   sidecar (`runstate.py` / `runstate_pl.py`) and `run_circmean.py`, which evaluates all seed programs of the
   seed directory and then runs the engine (`openevolve/`) for ITERS iterations with the given config.
2. Each iteration the engine draws a parent and a card level, builds the prompt (`openevolve/prompt/`), asks
   the model for a diff of the EVOLVE block and hands the child to `oe_evaluator_circmean_v3.py` (FOM) or
   `oe_evaluator_circmean_pl.py` (CLER). The adapter runs the child's `get_candidates()` (60 slots) and writes one request to the spool
   directory; the size window and the check-weight cap are enforced by the program scaffold of every
   seed and again by the worker evaluator (`eval_circmean_v3.py`).
3. GPU worker jobs (`circmean_worker.sbatch` / `worker_pl.sbatch`) run `circmean_worker.py`, which takes
   requests from the spool and runs `eval_circmean_v3.py` (FOM) or `eval_circmean_pl.py` (CLER) on each:
   build H_X, H_Z of every slot (`random_search_all_levels._lift` / `_lifted_product`), check CSS, drop
   duplicates by matrix hash and by isomorphism key (`iso_dedup.py`), drop codes already in the run's code
   archive, 2 s distance screen of every distinct candidate, then the top 5 by k d^2/n are measured: 30 s
   distance search (FOM, `DIST=cc`) or the circuit-level score (CLER, `../decoder/pl_score.py`). The
   fitness is returned through the spool and appended to the run's ledger.

## Files

| file | what it does | server path |
|---|---|---|
| `openevolve/` | OpenEvolve 0.2.11 with our modifications (rate-cell MAP-Elites, assignment cards, per-cell parent pools, prompt sections, RUN STATE hook); the modified files are listed at the end of this file | `/u/zdliu/lpenv/lib/python3.12/site-packages/openevolve/` |
| `run_circmean.py` | controller entry point: evaluates every seed, adds them to the database, runs the engine; `--checkpoint` resumes | `/u/zdliu/lp/circmean/run_circmean.py` |
| `oe_circmean.sbatch` | FOM controller job: exports the run settings, starts `runstate.py` and `run_circmean.py` | `/u/zdliu/lp/circmean/oe_circmean.sbatch` |
| `oe_circmean_pl.sbatch` | CLER controller job (same as above with the CLER config, evaluator and `runstate_pl.py`) | `/u/zdliu/lp/plevolve_20260912/oe_circmean_pl.sbatch` |
| `circmean_worker.sbatch` | FOM GPU worker job (1 GPU, 8 cores): budgets `CM_D2S_BUDGET=2`, `CM_TOPK=5`, `CM_DIST_MODE=$DIST`, `CM_D2S_DEEP=${D2SDEEP:-30}` | `/u/zdliu/lp/circmean/circmean_worker.sbatch` |
| `worker_pl.sbatch` | CLER GPU worker job (1 GPU, 16 cores): `CM_D2S_BUDGET=2`, `CM_TOPK=5`, `PL_SCORE_P`, `PL_SCORE_ROUNDS`, `PL_SCORE_SHOTS`, `PL_SCORE_MAX_ERRORS`, schedule search time | `/u/zdliu/lp/plevolve_20260912/worker_pl.sbatch` |
| `circmean_worker.py` | worker loop: spool requests, leases, per-request logs, evaluation timeout | `/u/zdliu/lp/circmean/circmean_worker.py` |
| `oe_evaluator_circmean_v3.py` | FOM adapter on the controller: runs the child program, admissibility checks, spool request, metrics and artifacts back to the engine | `/u/zdliu/lp/circmean/oe_evaluator_circmean_v3.py` |
| `oe_evaluator_circmean_pl.py` | CLER adapter (generated from the FOM adapter; differs in the reported score) | `/u/zdliu/lp/circmean/oe_evaluator_circmean_pl.py` (linked from `/u/zdliu/lp/plevolve_20260912/`) |
| `eval_circmean_v3.py` | FOM worker evaluator: build, CSS check, dedup, code archive, 2 s screen, 30 s measurement of the top 5, fitness | `/u/zdliu/lp/circmean/eval_circmean_v3.py` |
| `eval_circmean_pl.py` | CLER worker evaluator: the same pipeline, the top 5 scored by `pl_score.py` | `/u/zdliu/lp/circmean/eval_circmean_pl.py` (linked from `/u/zdliu/lp/plevolve_20260912/`) |
| `eval_circmean.py` | first evaluator version; the runs import only its rate-cell grid (`rate_bin_of`, `RATE_*`) | `/u/zdliu/lp/circmean/eval_circmean.py` |
| `random_search_all_levels.py` | lifted-product construction used by the evaluators (`_lift`, `_lifted_product`); the rest of the file is not used | `/u/zdliu/lp/lifted_product_search/random_search_all_levels.py` |
| `iso_dedup.py` | isomorphism-aware dedup key of a CSS code (Tanner-graph fingerprints) | `/u/zdliu/lp/circmean/iso_dedup.py` |
| `group_fp.py` | group fingerprint from a multiplication table (order histogram, centre, classes) | `/u/zdliu/lp/circmean/group_fp.py` |
| `runstate.py` | FOM sidecar: renders the RUN STATE text (beds, moves, frontier, shapes) every 300 s | `/u/zdliu/lp/circmean/runstate.py` |
| `runstate_pl.py` | CLER sidecar (generated from `runstate.py`; reads the CLER score instead of d) | `/u/zdliu/lp/plevolve_20260912/runstate_pl.py` |
| `launch_w_line.sh` | launcher of the ten-cell lines with a check-weight cap (`WMAX`), model tag, config, seed directory, effort and resume overrides | `/u/zdliu/lp/circmean/launch_w_line.sh` |
| `launch_10cells.sh` | launcher of the ten-cell setting that `launch_w_line.sh` extends (same exports) | `/u/zdliu/lp/circmean/launch_10cells.sh` |
| `launch_pair.sh` | launcher of a FOM/CLER pair of lines with identical settings (`launch_pair.sh <ITERS> <TAG> cc|pl|both`); its `launch_pl` block is the submit line of the CLER language-model runs (controller `oe_circmean_pl.sbatch`, workers `worker_pl.sbatch` with `SHOTS=200000,MAXERR=100,PLP=0.002`) | `/u/zdliu/lp/plevolve_20260912/launch_pair.sh` |
| `run_configs/*.yaml` | engine configurations of the paper runs (model, effort, prompt texts, population, islands, timeouts) | `/u/zdliu/lp/circmean/<name>.yaml` |
| `seed_dirs/seeds_rates_w14_plus_v5_nocyc_v3/` | the eight seed programs of every ten-cell run | `/u/zdliu/lp/circmean/seeds_rates_w14_plus_v5_nocyc_v3/` |
| `seed_dirs/seeds_rates_w14_plus_v5_nocyc_v3_w9/` | the same eight seeds with the cap fallback 9, for the check-weight-9 run | `/u/zdliu/lp/circmean/seeds_rates_w14_plus_v5_nocyc_v3_w9/` |
| `random_mutation/` | random-mutation control, see below | |

### Run configurations (`run_configs/`)

| config | runs |
|---|---|
| `oe_circmean_cc_v7_gpt61sol_high_w12_p16_nocyc_v3_10cells.yaml` | FOM, gpt-6.1-sol, high effort (`cc_v7_gpt61sol500_w12_10c_out`) |
| `oe_circmean_cc_v7_gpt61sol_high_w9_p16_nocyc_v3_10cells.yaml` | FOM, gpt-6.1-sol, check weight at most 9 (`cc_v7_gpt61sol500_w9_10c_out`) |
| `oe_circmean_cc_v7_gpt6sol_high_w12_p16_nocyc_v3_10cells.yaml` | FOM, gpt-6-sol, high effort (`cc_v7_gpt6sol50_w12_10c_cmp_out`) |
| `oe_circmean_cc_v7_gpt6luna_xhigh_w12_p16_nocyc_v3_10cells_rep2.yaml` | FOM, gpt-6-luna extra-high effort (`cc_v7_gpt6luna500_w12_10c_rep2_out`) |
| `oe_circmean_cc_v7_gpt6luna_{high,low}_w12_p16_nocyc_v3_10cells.yaml` | FOM, gpt-6-luna high and low effort (`cc_v7_gpt6lunahigh500_w12_10c_prod_out`, `cc_v7_gpt6lunalow500_w12_10c_prod_out`) |
| `oe_circmean_cc_v7_rnd_w12_10cells.yaml` | FOM, random mutation (`__PORT__` is filled in by the job) |
| `oe_circmean_pl_v3_gpt61sol_high_p16_nocyc_v3_10cells.yaml` | CLER, gpt-6.1-sol (`pl10_gpt61sol_out`) |
| `oe_circmean_pl_v3_gpt6sol_high_p16_nocyc_v3_10cells.yaml`, `oe_circmean_pl_v3_gpt6luna_{xhigh,high,low}_p16_nocyc_v3_10cells.yaml` | CLER, gpt-6-sol and gpt-6-luna |
| `oe_circmean_pl_v3_rnd_10cells.yaml` | CLER, random mutation |

### `openevolve/` (installed package, `/u/zdliu/lpenv/lib/python3.12/site-packages/openevolve/`)

| file | what it does |
|---|---|
| `__init__.py`, `_version.py`, `cli.py` | package entry, version 0.2.11, command line (not used by the runs; `run_circmean.py` uses the library API) |
| `config.py` | configuration classes read from the YAML |
| `controller.py` | the evolution controller |
| `database.py` | program database: islands, rate-cell MAP-Elites grid (`LAYOUT_BIN_ISLANDS`, `LAYOUT_RATE_BIN_N`, `LAYOUT_RATE_EDGE_CELLS`), per-cell pools (`LAYOUT_CELL_POOL`), parent sampling, checkpoints |
| `process_parallel.py` | parallel iterations: target cell and card level of each round (`V2_CARD_SELF`), inspirations (`CM_INSP_BINS`), model call, evaluation |
| `iteration.py` | one iteration in a worker process |
| `evaluator.py`, `evaluation_result.py` | calls the evaluator file with the config's `evaluator.timeout`, collects metrics and artifacts |
| `llm/base.py`, `llm/ensemble.py`, `llm/openai.py` | model interface and the OpenAI client (reasoning effort, reasoning summary log) |
| `prompt/assignment.py` | the assignment card (level 1, 2 or 3) drawn before the prompt is built |
| `prompt/sampler.py`, `prompt/templates.py` | prompt assembly: parent, lineage, top and inspiration programs with their slot tables, RUN STATE (`EM_ARCH`), card and target cell (`LAYOUT_CELL_FOCUS_LEVELS`) |
| `prompts/defaults/*.txt` | prompt templates, including the level-2 and level-3 toolboxes |
| `utils/*.py` | diff application, formatting, metric helpers, async helpers |

## Paper runs: how they were started

All jobs were submitted with `sbatch --export=ALL,...` (the cluster sets `SBATCH_EXPORT=NONE`).

FOM controller (gpt-6.1-sol run; the other LLM runs differ in `CONFIG`, `EFFORT` and the output names):

```
sbatch --export=ALL,ITERS=520,EFFORT=high,NMIN=100,NMAX=600,WMAX=12,NSLOTS=60,TABLETOP=0,TCELLS=1,\
RATELO=0.10,RATESTEP=0.05,RATEBINS=10,BINISL=1,INSP=4,TOPG=1,FULLTBL=1,CARDSELF=0,QUOTA=0,CELLPOOL=12,\
SHAPEK=0,EDGECELLS=1,OUT=<run dir>,SPOOL=<spool>,ARCHIVE=<code archive>,\
CONFIG=oe_circmean_cc_v7_gpt61sol_high_w12_p16_nocyc_v3_10cells.yaml,\
EVALUATOR=oe_evaluator_circmean_v3.py,SEEDS=seeds_rates_w14_plus_v5_nocyc_v3 oe_circmean.sbatch
```

FOM workers (six for the gpt-6.1-sol run, eight for most others):

```
sbatch --export=ALL,NMIN=100,NMAX=600,WMAX=12,RATELO=0.10,RATESTEP=0.05,RATEBINS=10,SPOOL=<spool>,\
EVAL=/u/zdliu/lp/circmean/eval_circmean_v3.py,DIST=cc circmean_worker.sbatch
```

- Budgets: 2 s screen of every distinct candidate (`CM_D2S_BUDGET=2`), 30 s measurement of the top 5
  (`CM_D2S_DEEP=30`, `CM_TOPK=5`), controller wait 7200 s (`OE_EVAL_TIMEOUT` in the adapter, `evaluator.timeout`
  in the config).
- A run that stopped was resumed with the same submit line plus `CKPT=<checkpoint dir>` and
  `ARCHIVE=<the run's archive>` (ITERS is then the number of further iterations).
- The gpt-6-luna and gpt-6-sol runs were started with `launch_w_line.sh` (same exports as above), e.g.
  `MODEL=gpt6luna CFG=<config> SEEDDIR=seeds_rates_w14_plus_v5_nocyc_v3 EFFORT=xhigh launch_w_line.sh 12 500 prod 8`.
  The check-weight-9 run (`WMAX=9`) was continued with its resume mode (`CKPT=... RESUME_OUT=...`),
  `CFG=oe_circmean_cc_v7_gpt61sol_high_w9_p16_nocyc_v3_10cells.yaml`, seeds `seeds_rates_w14_plus_v5_nocyc_v3_w9`,
  `EFFORT=high`.

CLER controller and workers (gpt-6.1-sol run; the gpt-6-sol and gpt-6-luna runs differ in `CONFIG` and `EFFORT`):

```
sbatch --export=ALL,ITERS=...,<the same settings as the FOM controller>,OUT=<run dir>,SPOOL=<spool>,\
CONFIG=oe_circmean_pl_v3_gpt61sol_high_p16_nocyc_v3_10cells.yaml,SEEDS=seeds_rates_w14_plus_v5_nocyc_v3,\
OETIMEOUT=70000 oe_circmean_pl.sbatch
sbatch --export=ALL,NMIN=100,NMAX=600,WMAX=12,RATELO=0.10,RATESTEP=0.05,RATEBINS=10,SPOOL=<spool>,\
SHOTS=200000,MAXERR=100,PLP=0.002,WKTIMEOUT=62000,PLTIMEOUT=60000,WKLEASE=70000 worker_pl.sbatch
```

- CLER protocol: p = 0.002, R = 6 rounds (`PLR` default), at most 2e5 shots per basis, stop at 100 failed
  shots, CP-SAT schedule search 120 s per depth (`SCHEDTIME` default), 2 s distance screen before the
  circuit-level stage. The decoder settings are the defaults of `../decoder/pl_score.py`
  (`PLC_MODE=gpu2`, `PLG_STAGE1=gpu`, `PLC_STAGE3=lsd`).

## Random-mutation control (`random_mutation/`)

The model is replaced by a local OpenAI-compatible endpoint (`rnd_llm_server.py`) that answers every request
with a random child of the prompt's current program at the dealt card level (`rnd_mutator.py`). Engine,
seeds, evaluator, cells, archive and budgets are those of the LLM runs.

| file | what it does | server path |
|---|---|---|
| `rnd_mutator.py` | random proposal at levels 1/2/3 (`RND_V5=1`: protograph shapes drawn as in the LLM runs, level 2 keeps the group family, level 3 redraws the shapes with the measured gpt-6.1-sol proportions); `RND_LIBRARY=smallgrp` draws the level-3 group from the GAP small-groups library, `RND_ORDER_RULE=llm_n` matches the block length to the gpt-6.1-sol children; `RND_WCAPS` lists the values from which every child's check-weight cap (its `W_MAX` line) is drawn uniformly, `12` in the FOM runs and `9,10,11,12` in the CLER runs (`launch_rnd_v5.sh`) | `/u/zdliu/lp/circmean/rnd_mutator.py` |
| `rnd_llm_server.py` | the local endpoint; logs every mutation to `rndmut.jsonl` | `/u/zdliu/lp/circmean/rnd_llm_server.py` |
| `launch_rnd_v5.sh` | `launch_rnd_v5.sh fom|cler <REP> [NWORK] [ITERS]`: engine seeds 42..46, wait 400000 s, protocol manifest per run | `/u/zdliu/lp/circmean/launch_rnd_v5.sh` |
| `deploy/oe_circmean_rnd_v5.sbatch`, `deploy/circmean_worker_v5.sbatch` | FOM controller and worker jobs (start the endpoint, fill `__PORT__`, run the protocol check) | `/u/zdliu/lp/circmean/` |
| `deploy/oe_circmean_pl_rnd_v5.sbatch`, `deploy/worker_pl_v5.sbatch` | CLER controller and worker jobs | `/u/zdliu/lp/plevolve_20260912/` |
| `deploy/protocol_check_v5.py` | checks the effective environment and configuration of every job against the run's manifest | `/u/zdliu/lp/circmean/protocol_check_v5.py` |
| `deploy/test_protocol_check.py` | tests of the protocol check (reads the random config `../run_configs/oe_circmean_cc_v7_rnd_w12_10cells.yaml`; adjust its path) | test directory on the cluster |
| `rnd_v5_test.py` | offline tests of the v5 rules | `/u/zdliu/deploy/rnd_v5_20261015/test/` |
| `smallgrp_lib.jsonl` | GAP small groups of order 5..299 except 256 plus 300 groups of order 256, as permutation generators (8,660 lines) | `/u/zdliu/lp/circmean/smallgrp_lib.jsonl` |
| `smallgrp_sd.tsv` | GAP StructureDescription of every library group (level 2 keeps the group family) | `/u/zdliu/lp/circmean/smallgrp_sd.tsv` |
| `template_ids.tsv` | GAP IdGroup of every metacyclic and dicyclic template presentation | `/u/zdliu/lp/circmean/template_ids.tsv` |
| `llm_n_distribution.json` | block lengths n of the gpt-6.1-sol children (keys `fom`, `cler`) | `/u/zdliu/lp/circmean/llm_n_distribution.json` |
| `gap_library/export.g`, `gap_library/export.sbatch` | GAP 4.15.1 script and job that wrote `smallgrp_lib.jsonl` | `/u/zdliu/lp/jobs/smallgrp_lib_20261011/` |
| `gap_sd/`, `gap_tmpl/` | GAP scripts that wrote `smallgrp_sd.tsv` and `template_ids.tsv` | `/u/zdliu/deploy/rnd_v5_20261015/` |

The five FOM and five CLER repeats of the paper (`cc_v7_rnd500v5_w12_10c_rep1..5_out`;
`pl10_rnd500v5_rep1,2,3,6,5_out`, where the run `rep6` is the fourth CLER repeat of the paper) were started
with `launch_rnd_v5.sh fom <REP>` and `launch_rnd_v5.sh cler <REP>` (engine seed 41 + REP).

### Hill-climbing control (`random_mutation/local_v5/`)

The second non-LLM source of mutations of the paper (hill-climbing in the paper, local mutation in the file names). Same endpoint and rules as the random-mutation control
(`RND_V5=1`), with one change behind `RND_LOCAL=1`: a level-1 child carries the parent's best measured code
(`LOCAL_BASE`) and each of its 60 slots is that code after one to three edits of a protograph entry
(replace, add or remove one group element; edits that only rename qubits and checks are redrawn). Levels 2
and 3 are the random-mutation proposals, and the check-weight cap of every child is drawn as in the
random-mutation control (`RND_WCAPS`: `12` for FOM, `9,10,11,12` for CLER).

| file | what it does | server path |
|---|---|---|
| `local_v5/rnd_mutator.py` | `rnd_mutator.py` plus the local level-1 move (`LOCAL_WRAPPER`, `make_response_local`); the run's file has sha1 `08f87436ac3a6b94` (recorded by every local run); the deposited copy differs from it in comments only | `/u/zdliu/deploy/rnd_local_v5_20261018/rnd_mutator.py` |
| `local_v5/rnd_llm_server.py` | the endpoint with `parent_best` (reads the parent's evaluation table from the prompt) | `/u/zdliu/deploy/rnd_local_v5_20261018/rnd_llm_server.py` |
| `local_v5/oe_circmean_rnd_v5local.sbatch`, `local_v5/oe_circmean_pl_rnd_v5local.sbatch` | FOM and CLER controller jobs (start the endpoint with `RND_LOCAL=1`, run the protocol check); workers are `deploy/circmean_worker_v5.sbatch` and `deploy/worker_pl_v5.sbatch` | `/u/zdliu/deploy/rnd_local_v5_20261018/` |
| `local_v5/launch_rnd_v5local.sh` | `launch_rnd_v5local.sh fom|cler <REP> [NWORK] [ITERS]`, engine seeds 42..46, 520 iterations | `/u/zdliu/deploy/rnd_local_v5_20261018/launch_rnd_v5local.sh` |

The five FOM repeats (`cc_v7_rnd520locv5_w12_10c_rep1..5_out`) were started with
`launch_rnd_v5local.sh fom <REP>` on 2026-10-18; the five CLER repeats (`pl10_rnd520locv5_rep1..5_out`) with
`launch_rnd_v5local.sh cler <REP>`.

## Refinement runs (`run_configs/oe_circmean_cc_v7_gpt61sol_w9c2fixed_seq20.yaml`)

The three refinement runs of the weight-9 program (`programs/plain_w9c2_r1,2,3_20261021`) start from the
program of the 258th child of the weight-9 run as their only seed and use this configuration: 20 steps,
one evaluation at a time (`parallel_evaluations: 1`), program-size limit 120000 characters; the rest of the
file is the weight-9 configuration. The controller is `oe_circmean.sbatch` with the exports of the weight-9
run (`WMAX=9`, `NSLOTS=60`, ten rate cells) plus `LAYOUT_LOCK_CELL=2` (the target cell is fixed to the
rate-0.2 cell) and `LAYOUT_CELL_FOCUS_LEVELS=1` (every step receives the level-1 instruction); the code
archive of each run starts empty. The three runs differ only in the model's sampling.

## Changes to the OpenEvolve package

`openevolve/` is OpenEvolve 0.2.11 with these files modified: `config.py`, `controller.py`, `database.py`,
`iteration.py`, `llm/ensemble.py`, `llm/openai.py`, `process_parallel.py`, `prompt/sampler.py`,
`prompt/templates.py`, `prompts/defaults/diff_user.txt`, `prompts/defaults/full_rewrite_user.txt`,
`prompts/defaults/inspiration_program.txt`, `prompts/defaults/previous_attempt.txt`,
`prompts/defaults/top_program.txt`, `utils/format_utils.py`; these files added:
`prompt/assignment.py`, `prompts/defaults/toolbox_l2.txt`, `prompts/defaults/toolbox_l3.txt`; and
`prompts/defaults/fragments.json` of the package not included.

## Dependencies

Python 3.12 (`/u/zdliu/lpenv`), numpy, scipy, networkx, PyYAML, openai (engine), torch 2.5.1 + triton 3.1.0 on
NVIDIA GPUs (RTX 6000), stim, ldpc, relay-bp, ortools (CLER evaluation, see `../decoder`,
`../circuit_schedule`), GAP 4.15.1 with the smallgrp package (random-mutation library). The workers import
`../distance_estimates`, `../circuit_schedule` and `../decoder` from their server paths. Slurm with the modules
`python-waterboa/2024.06`, `gcc/11`, `cuda/12.2`.
