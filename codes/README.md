# codes

Parity-check matrices of the reported codes: every code of Table 1 and every gpt-6.1-sol code of the
Supplementary elite tables (21 codes), and twelve codes of the run with check weight at most 9 (the eleven
codes of its Supplementary table and the best code of the best program of cell 2), 33 codes in all. They are the fixed matrices the runs
stored in their code archives; the distances of the paper were certified, re-measured and sampled on exactly
these matrices.

- `fom/`, `cler/`: the codes of the gpt-6.1-sol FOM run (`cc_v7_gpt61sol500_w12_10c_out`) and CLER run
  (`pl10_gpt61sol_out`). One file pair per code, `cell<c>[_best]_n<n>_k<k>_<sha16>_HX.mtx` and `..._HZ.mtx`.
- `fom_w9/`: the codes of the Supplementary table of the gpt-6.1-sol FOM run with check weight at most 9
  (`cc_v7_gpt61sol500_w9_10c_out`): the best code of the best program of every rate cell, the best
  code of cell 2 (`cell2_best_n585_k119_<sha16>`), and the code marked R, from the level-1 refinement of the
  program of child 258 (`cell2_refinement_n585_k117_<sha16>`).
- Format: MatrixMarket coordinate, integer, 1-based; rows are checks, columns are qubits; every listed
  entry is 1 and the matrices are over GF(2). `scipy.io.mmread` reads them.
- `<sha16>`: the archive name of the code, the first 16 hex digits of the SHA-1 of the unpacked 0/1 arrays
  `H_X` followed by `H_Z` (row-major `uint8` bytes). `index.json` lists for every code the objective, run,
  rate cell, selection (elite = best code of the elite program of the cell, best = best code of the cell of the weight-9 run,
  two-block = the second code of cell 0, refinement (R) = the code marked R of the weight-9 table), `n`, `k`,
  `d` with its status (exact or upper bound), the check weight `w`, the group, the protograph shapes and
  `q = |G|`, and the two files.

Block structure (the lifting convention of the Supplementary Information): with protographs `A` of shape
`m_A x n_A` and `B` of shape `m_B x n_B` over the group algebra of `G`, `q = |G|`, the matrices are

    H_X = [ A (x) I_{n_B} | I_{m_A} (x) B^T ],    H_Z = [ I_{n_A} (x) B | A^T (x) I_{m_B} ],

where every group-algebra entry is replaced by its `q x q` lift (left regular representation for the
entries of `A`, right regular representation for the entries of `B`). Qubit columns are ordered as the
`n_A n_B q` columns indexed by `(a, b)` (row-major, `a` over the columns of `A`, `b` over the columns of
`B`), followed by the `m_A m_B q` columns indexed by `(i, j)` (`i` over the rows of `A`, `j` over the rows
of `B`); the rows of `H_X` are indexed by `(i, b)` and the rows of `H_Z` by `(a, j)`, each block of `q`
consecutive rows or columns carrying one group-element index.

The files were written from the code archives of the runs; the hash, `H_X H_Z^T = 0`, `n`, `k`, `w` and the
protograph shapes were checked against the tables, and every file was read back with scipy.

`reference/`: the check matrices of the bivariate-bicycle code [[144,12,12]] of Bravyi et al. (Nature 627, 778
(2024)) in the form used by the circuit-level sweeps of Fig. 2 (`records/pl_sweep_20261009_bbibm/`), with the
X and Z roles and the two data blocks exchanged relative to that paper; its published depth-7 syndrome cycle is
`circuit_schedule/bb144_published_schedule.json`.
