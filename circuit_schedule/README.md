# circuit_schedule

Syndrome-extraction circuits of the circuit-level (CLER) objective: the CP-SAT search for CNOT schedules,
the perfect-matching decomposition of the lifted blocks into layers, and the Stim memory
circuit with its detector error model.

- Each q x q block between a check block and a data block of a regular lift has constant row and column
  weight a and is split into a perfect matchings (`perfect_matchings`), a minimum edge colouring of the
  block. The matchings are taken from the matrix and are not in general the permutation matrices of the
  group elements of the protograph cell. All q CNOTs of such a layer unit share one time step.
- `solve_schedule` assigns a time step to every unit with OR-Tools CP-SAT: two units sharing a check block
  or a data block get different steps, and every X-check / Z-check pair satisfies the X-Z commutation
  (even inversion parity) condition of Zhang et al. (arXiv:2603.21499), merged into one condition per way
  of linking the ancillas of two blocks when possible and imposed per pair of ancillas otherwise. The depth is searched upward from the degree bound with a time limit per depth;
  the first feasible depth is returned.
- `build_circuit` writes the memory experiment (R rounds, one basis) with the noise model of the Methods:
  depolarizing noise after every CNOT, on every ancilla after reset and before measurement, and on every data
  qubit before the final readout; no idling noise. A noiseless copy must give no detector event and the
  CNOT count must equal nnz(H) times the number of rounds.
- `circ_1side.dem_matrices` turns the Stim detector error model into the check matrix, logical matrix and
  priors that the decoders use.

The code ran on the cluster from the original server paths listed below (hard-coded in the scripts; this
folder does not change them). To run it elsewhere, place each file at its server path or edit the paths.

## Files

| file | what it does | server path |
|---|---|---|
| `asc_schedule.py` | layer units (`build_units`, `perfect_matchings`), commutation conditions, CP-SAT schedule search (`solve_schedule`), circuit builder (`build_circuit`), detector check | `/u/zdliu/lp/circmean/asc_schedule.py` |
| `pl_at_p_asc.py` | builds the Stim memory circuit of one code and basis from its stored schedule (`build`); `PL_NOISE_MODEL` unset is the noise model of the paper | `/u/zdliu/lp/circmean/pl_at_p_asc.py` |
| `circ_1side.py` | one-sided memory circuits on group layers, `perfect_matchings` / `group_layers` (the decomposition `asc_schedule.py` reuses), `dem_matrices` (DEM to H, L, priors) | `/u/zdliu/lp/lps_graftgate/circ_1side.py` |

## How it was invoked

These modules are libraries; the runs reach them through `../decoder/pl_score.py`:

- `pl_score.py` calls `asc_schedule.solve_schedule(..., time_limit=PL_SCORE_SCHED_TIME)` for every code
  that has no stored schedule: 120 s per depth in the CLER runs (`SCHEDTIME` default of
  `../evolve/worker_pl.sbatch`), 300 s in the re-measurements (`../decoder/psweep/deepverify_pl.sbatch`).
  The schedule is stored per code and basis (`asc_schedules.json`); the circuit-level elite codes carry it
  in the run archive, and the top-up of the p sweep (`../decoder/psweep/topup61_launch.sh`) pins every leg
  of a code to the schedule of its p = 0.002 leg.
- `../decoder/pl_sample.py` calls `pl_at_p_asc.build(npz, q, basis, name, schedule)` and
  `circ_1side.dem_matrices` for every (code, basis) unit.
- p = 0.002 and R = 6 during evolution; p from 0.001 to 0.005, R = 6 in the re-measurements.

## Dependencies

stim 1.16.0, OR-Tools CP-SAT 9.15 (installed in `/ptmp/zdliu/ortools_local`, put on the path by
`pl_score.py` through `PL_SCORE_ORTOOLS`), numpy, scipy; `circ_1side.py` also imports torch, ldpc and
`fast_gf2`, `gpu_core_v2`, `mc_rate` from `../distance_estimates` (server path `/u/zdliu/lp/dopt/bdirect_v2`)
at load time.

`bb144_published_schedule.json` is the published depth-7 syndrome cycle of the [[144,12,12]] reference code of
Fig. 2 (Table 5 of Bravyi et al. 2024) in the schedule format of `asc_schedule.py`: one unit per monomial
term of the check matrices `codes/reference/bb144_{HX,HZ}.mtx`, with the time step of a term one less than its
round in that table (terms of H_X take the rounds of the Z checks of that paper and terms of H_Z those of its
X checks, because the two roles are exchanged in the deposited matrices).
