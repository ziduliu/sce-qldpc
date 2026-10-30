# decoder

Circuit-level sampling and decoding of the CLER objective, the CLER score, and the drivers of the
circuit-level re-measurements (the p sweep).

- Shots of the six-round memory experiment (circuit from `../circuit_schedule`) are sampled with Stim in
  chunks of 250, both bases (Z, X), until a stop at `PL_MAX_ERRORS` failed shots or `PL_SHOTS` shots.
- Decoding is a three-stage cascade on the detector error model (parameters of arXiv:2608.02773):
  1. relay-BP stage 1 (gamma0 = 0.1, 200 pre-iterations, up to 20 relay sets of 100 iterations, gamma drawn
     from (-0.24, 0.6), stop at the first convergent solution). The pre-iterations of all shots of a chunk run
     batched on the GPU (Triton kernels, `cpu_accel_20260911/`); the relay part of the unconverged shots runs
     on the GPU as well (`PLG_STAGE1=gpu`).
  2. relay-BP stage 2 on the shots left unsatisfied (500 pre-iterations, up to 200 sets of 200 iterations),
     on the GPU (`plstage2_20260912/gpu_relay2.py`, which reproduces relay-bp's `RelayDecoderF32`, including its
     ChaCha12 gamma stream, `chacha12.py`).
  3. BP+LSD from `ldpc` on the remainder (min-sum, 300 iterations, adaptive scaling, `lsd_cs` order 1;
     `PLC_STAGE3=lsd`).
- Score: the shot failure rate of the worse basis is converted into a logical error rate per round and per
  logical qubit, and s = -log10 of the upper end of its 95% Clopper-Pearson interval.

The code ran on the cluster from the original server paths listed below (hard-coded in the scripts; this
folder does not change them). To run it elsewhere, place each file at its server path or edit the paths.

## Files

| file | what it does | server path |
|---|---|---|
| `pl_score.py` | `score_codes(codes, ...)`: schedule search if needed (`../circuit_schedule/asc_schedule.py`), starts `pl_sample.py` per code and basis (as a subprocess, or as `srun` ranks with `PL_SCORE_SRUN=1`), computes s; `PL_SCORE_BASES` (default `Z,X`) selects the bases; default sampler environment `PLC_MODE=gpu2`, `PLG_STAGE1=gpu`, `PLC_STAGE3=lsd`, `PLF_WORKERS=0`, `PL_CHUNK=250` | `/u/zdliu/lp/circmean/pl_score.py` (linked from `/u/zdliu/lp/plevolve_20260912/`) |
| `pl_sample.py` | sampling in chunks, batched GPU first leg, cascade on the unconverged shots, stopping rule, failure counts per basis, sharding over ranks | `/u/zdliu/lp/plevolve_20260912/pl_sample.py` |
| `pl_attrib.py` | block-level failure map from the scored shots (reported back to the model in the CLER runs) | `/u/zdliu/lp/plevolve_20260912/pl_attrib.py` |
| `pl_at_p.py` | relay-BP stage parameters `STAGE1`, `STAGE2`, stage-3 settings, Clopper-Pearson interval | `/u/zdliu/lp/circmean/pl_at_p.py` |
| `rankgpu.sh` | gives each `srun` rank one GPU of the allocation (used with `PL_SCORE_SRUN=1`) | `/u/zdliu/lp/circmean/rankgpu.sh` |
| `plstage2_20260912/exact_cascade.py` | the cascade in `gpu2` mode: stages 1 and 2 on the GPU, stage 3 from the module below | `/u/zdliu/lp/plstage2_20260912/exact_cascade.py` |
| `plstage2_20260912/gpu_relay2.py` | GPU relay-BP, same results as relay-bp's `RelayDecoderF32` shot by shot | `/u/zdliu/lp/plstage2_20260912/gpu_relay2.py` |
| `plstage2_20260912/chacha12.py` | the relay gamma random stream (ChaCha12 as in the Rust crate) | `/u/zdliu/lp/plstage2_20260912/chacha12.py` |
| `plcost_20260912/exact_cascade.py` | stage hand-off and the cached stage-3 decoder (BP+LSD with `PLC_STAGE3=lsd`, BP-OSD otherwise); loaded by absolute path from the module above | `/u/zdliu/lp/plcost_20260912/exact_cascade.py` |
| `cpu_accel_20260911/gpu_backend.py` | `make_first_leg`: chooses the GPU first-leg implementation (`PL_GPU_IMPL`, default `padded`) | `/u/zdliu/lp/circmean/cpu_accel_20260911/gpu_backend.py` |
| `cpu_accel_20260911/gpu_padded.py` | first leg with aligned shot strides (the implementation used) | `/u/zdliu/lp/circmean/cpu_accel_20260911/gpu_padded.py` |
| `cpu_accel_20260911/gpu_prepass.py` | Triton min-sum kernels of the first leg (check, syndrome, logical updates) | `/u/zdliu/lp/circmean/cpu_accel_20260911/gpu_prepass.py` |
| `cpu_accel_20260911/variable_kernels.py`, `variable_streamed.py` | Triton variable-node updates in the summation order of the Rust decoder | `/u/zdliu/lp/circmean/cpu_accel_20260911/` |
| `cpu_accel_20260911/screen_contract.py` | fingerprints that tie a GPU screen to its exact CPU inputs and parameters | `/u/zdliu/lp/circmean/cpu_accel_20260911/screen_contract.py` |
| `psweep/deepverify_pl.py`, `psweep/deepverify_pl.sbatch` | circuit-level re-measurement of named codes from a run archive (`--hash`) or a matrix file (`--npz`) at the legs `LEGS="p:shots ..."`; `PL_SCORE_SRUN=1`, schedule search 300 s, stop at `MAXERR` (1000) failures | `/u/zdliu/lp/plevolve_20260912/` |
| `psweep/select_pl61_champions.py` | the ten cell-champion codes of the gpt-6.1-sol CLER run | `/u/zdliu/lp/jobs/select_pl61_champions.py` |
| `psweep/sweep61_launch.sh` | p sweep of the gpt-6.1-sol elite codes of both objectives: p = 0.0015 / 0.002 / 0.003 / 0.004 / 0.005 with 2e6 / 1e6 / 3e5 / 1e5 / 5e4 shots, one job per code and p | `/u/zdliu/lp/jobs/sweep61_launch.sh` |
| `psweep/topup61_launch.sh` | legs at p = 0.001, 0.002, 0.0025, 0.003, 0.0035, 0.004, 0.0045, 0.005 with the schedule pinned to the p = 0.002 leg, and more shots (up to 3e8, no stop at 1e7 and above) where the worse basis had fewer than ten failures; includes the [[144,12,12]] reference | `/u/zdliu/lp/jobs/topup61_launch.sh` |
| `psweep/pl_sweep_gen2.py` | sweep generation of the earlier gpt-6-sol sweep (2026-09-28), whose [[144,12,12]] matrix is the reference used by `topup61_launch.sh` | `/u/zdliu/lp/circmean/pl_sweep_gen2.py` |

## How it was invoked

- During the CLER runs: `../evolve/eval_circmean_pl.py` calls `pl_score.score_codes` for the five measured
  codes of each child, with the worker settings of `../evolve/worker_pl.sbatch`: `PL_SCORE_P=0.002`,
  `PL_SCORE_ROUNDS=6`, `PL_SCORE_SHOTS=200000`, `PL_SCORE_MAX_ERRORS=100`, `PL_SCORE_SCHED_TIME=120`,
  `PL_SCORE_TIMEOUT=60000`.
- Re-measurements of the gpt-6.1-sol elite codes (cluster, from `/u/zdliu/lp/plevolve_20260912`):

```
python select_pl61_champions.py                      # -> /u/zdliu/lp/jobs/codes_pl61_champions_20261006.json
DRY=1 /u/zdliu/lp/jobs/sweep61_launch.sh             # list the legs, then run without DRY
DRY=1 /u/zdliu/lp/jobs/topup61_launch.sh             # top-up legs, then run without DRY
# each leg is one job:
sbatch -p gpu --nodes=2 --ntasks=2 --ntasks-per-node=1 --gres=gpu:1 ... \
  --export=ALL,ARCHIVE=<run archive>,OUT=<dir>,TAG=<tag>,LEGS=<p>:<shots>,MAXERR=1000,HASH=c<cell>:<q>:<sha> deepverify_pl.sbatch
```

- The one-circuit measurements of Fig. 2 (`../records/pl_sweep_20261030_onecircuit/`): one run per line of its
  `legs.tsv` (code, p, basis, shot cap, failure cap, shards), each with the code's fixed schedule
  (`sched/<code>.json`) copied into the work directory as `asc_schedules.json`, so that no schedule is solved
  (`sched_note` "cached" in every result):

```
# inside a Slurm allocation of <shards> tasks with one GPU each; pl_score starts the shards with srun
PL_SCORE_ROUNDS=6 PL_SCORE_SRUN=1 PL_SCORE_P=<p> PL_SCORE_BASES=<basis> PL_SCORE_SHOTS=<shot_cap> \
PL_SCORE_MAX_ERRORS=<max_errors> PL_SCORE_SHARDS=<shards> PLF_WORKERS=5 \
  python deepverify_pl.py --npz c<cell>:<q>:<code npz> --workdir <work dir> --out <result json>
```

## Dependencies

stim 1.16.0, ldpc 2.4.1 (BP+LSD, BP-OSD), relay-bp 0.2.2 (Rust crate with Python bindings from
github.com/trmue/relay, built with maturin and installed from `/u/zdliu/lp/relayrepo`; reference decoder of
`gpu_relay2.py` and stage 1 on the CPU when `PLG_STAGE1` is not `gpu`), torch 2.5.1 and triton 3.1.0 on NVIDIA
GPUs (RTX 6000), numpy, scipy, Slurm (`srun` for sharded legs), modules `gcc/11`, `cuda/12.2`,
`python-waterboa/2024.06`.
