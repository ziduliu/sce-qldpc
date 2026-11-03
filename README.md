# sce-qldpc

Code, programs and records of "Large-Language-Model Discovery of Quantum LDPC Codes through
Structured Concept Evolution" (Zidu Liu and Florian Marquardt; v2 of arXiv:2606.24808).

## Layout

Code (each folder has a README that maps every file to the server path it ran from and gives the
commands of the runs):

- `evolve/` the modified OpenEvolve engine (`openevolve/`), run state and prompt building
  (`runstate.py`, `runstate_pl.py`), the FOM and CLER evaluators (`eval_circmean_v3.py` with
  `oe_evaluator_circmean_v3.py`; `eval_circmean_pl.py` with `oe_evaluator_circmean_pl.py`; the
  lifted-product construction in `random_search_all_levels.py`, isomorphism dedup in `iso_dedup.py`),
  the job scripts and launchers, the engine configuration of every run (`run_configs/`), the eight seed
  programs of the weight-12 runs and of the weight-9 run (`seed_dirs/`), the random-mutation
  control (`random_mutation/`: mutator, mutation server, launcher, protocol check, GAP small-group
  library with the scripts that exported it) and the hill-climbing control (`random_mutation/local_v5/`)
- `distance_estimates/` the GPU distance estimator (2 s screen, 30 s measurement, 600 s
  re-measurement) and the re-measurement drivers (`remeasure/`)
- `exact_distance/` the exact-distance certifier (`src/`, regression tests, cluster scripts)
- `circuit_schedule/` CP-SAT syndrome-extraction schedule search, perfect-matching block decomposition,
  Stim circuit builder; `bb144_published_schedule.json`, the published syndrome cycle of the
  [[144,12,12]] reference code of Fig. 2
- `decoder/` circuit-level sampling and decoding (relay-BP stages on the GPU, BP+LSD; runtime modules in
  `cpu_accel_20260911/` and `plstage2_20260912/`), the CLER score, and the p-sweep drivers (`psweep/`)
- `prompts/` one complete request of the gpt-6.1-sol run (`example_prompt.json`), its user message and
  an abridged version

Data:

- `programs/<run>/` the stored child programs of the 34 runs of the paper and of one further run
  (`cc_v7_gpt61sol_n1k_w12_10c_out`, below 1500 qubits): the text as written by the model, by the random
  mutator or by the hill-climbing move, plus the
  fixed scaffold. Three namings, by the way the programs were exported: `it<iteration>_<id>.py` (programs
  taken from the engine checkpoints; the checkpoints keep a second copy of some programs under iteration 0,
  so these appear twice, as `it0000_<id>.py` and `it<iteration>_<id>.py`); `it<iteration>_<id>.py` with
  `manifest.json` (id and iteration of every file; a program copied between islands is listed once per
  copy) for the gpt-6-luna and gpt-6-sol runs; `<hash>.py` (the first 16 hex digits of the SHA-1 of the
  program text) for the weight-9 run, the hill-climbing FOM runs, the refinement runs and the run below
  1500 qubits. Every folder has `index.jsonl`, one line per stored program: `file`, `child` (position of
  its evaluation in the run, 0-based: the seed programs first, then the children in the order they were
  evaluated; null when the text matches no evaluation record), `score` (its program score), `rate_bin`
  (rate cell 0-9), `n`, `family`, and, where the export kept them, `id`, `parent_id`, `iteration_found`,
  `generation`, `island` and `changes` (the engine's summary of the edit). The run-cell tables of the
  Supplementary Information give, per rate cell, the best score among the eight seed programs and the
  children of the first 500 steps of a run: `iteration_found` at most 500 for the language-model runs
  (a step whose response is rejected produces no child), and `child` at most 507 for the random-mutation
  and hill-climbing runs, where every step produces one child
- `codes/` the parity-check matrices (MatrixMarket text) of every code of Table 1, every gpt-6.1-sol code
  of the Supplementary elite tables, the eleven codes of the weight-9 table and the best code of the best
  program of its cell 2, with `index.json`
  (run, cell, `[[n,k,d]]`, check weight, group, protograph shapes, archive hash), and `codes/reference/`,
  the [[144,12,12]] reference code of Fig. 2; see `codes/README.md`
- `records/exact_bb_20261006_61/`, `records/w9cert_20261013/`, `records/recert_v5_20261014/`,
  `records/exact_v5_20261017_w9cell0/`, `records/exact_v5_20261022_refinement/` the exact-distance
  certificates behind Table 1, the weight-9 table, the five codes of the best program of its cell 2 and
  the code marked R (per-weight slices, aggregates and controller logs; `exact_bb_20261006_61/c2_W24_operators.json`
  holds the verified weight-24 logical operators of the [[585,121,24]] code). The In-loop, 600 s, 2 h and
  dist-m4ri columns of the verification table are not backed by files in this repository
- `records/pl_sweep_20261006_61/`, `records/pl_sweep_20261007_topup/`, `records/pl_sweep_20261009_c0dgx/`,
  `records/pl_sweep_20261009_bbibm/` the circuit-level sweeps (one JSON per code and physical error rate, with
  the schedule and the decoder log), and `records/pl_sweep_20261030_onecircuit/` the measurements of the Fig. 2
  codes on one fixed schedule per code (`sched/<code>.json`, the same circuit in both bases): `legs.tsv` lists
  each measurement (code, p, basis, shot cap, failure cap, shards) with its result in `out/`, `codes.json` the codes

The CLER random-mutation repeat 4 of the paper is the folder `pl10_rnd500v5_rep6_out`. The refinement
folders hold the start program (child 0) and the children of each run; their configuration is
`evolve/run_configs/oe_circmean_cc_v7_gpt61sol_w9c2fixed_seq20.yaml` (see `evolve/README.md`).

Figures and tables:

- Fig. 1: panel (a) is a drawing; panel (b) plots `kd^2/n` against `k/n` for the FOM codes of the ten rate
  cells (the gpt-6.1-sol rows of the Supplementary elite table) and the codes of the weight-9 table
  (`codes/index.json`)
- Fig. 2: `pl_sweep_onecircuit_build.py` builds `figures/data/pl_sweep_20261030_onecircuit_fig.json` from the
  records above. Every code is measured on its fixed schedule in both bases at every p: the counts of each basis
  come from a measurement in `records/pl_sweep_20261030_onecircuit/` (row field `one_circuit`) or from the sweep
  record of that point, whose run used the same schedule (row field `kept_basis`); the build checks this for every
  row and adds the [[144,12,12]] reference rows (published depth-7 schedule). `pl_sweep_61_ab_v.py` draws the
  figure (rows, markers and intervals through `pl_sweep_61_clean.py`)
- Fig. 3: `line_score_arms.py` from `figures/data/line_score_arms_iter.json` (the run score of every
  run after each step)
- Table 1: `n`, `k`, `d` and `w` from `codes/index.json`; CNOTs is the number of nonzero entries of `H_X` and
  `H_Z`; Depth is the schedule depth of the `p = 0.002` rows of `figures/data/pl_sweep_20261030_onecircuit_fig.json`

## License

Apache License 2.0 (`LICENSE`); `evolve/openevolve/` is a modified copy of OpenEvolve 0.2.11 (`NOTICE`).
