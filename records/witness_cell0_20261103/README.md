# witness_cell0_20261103

Lightest logical operators found on 2026-11-03 for the two codes of cell 0 of Table 1 whose distances are upper
bounds, on the matrices in `codes/fom/`:

| code | matrices | file | lightest operators found |
|---|---|---|---|
| [[510,40,42]] | `cell0_n510_k40_fd925ff20fe235cc_{HX,HZ}.mtx` | `cell0-510-40-42.json` | weight 42, X and Z |
| [[578,32,46]] | `cell0_n578_k32_6612ba96961ef2c1_{HX,HZ}.mtx` | `cell0-578-32-46.json` | weight 46, X and Z |

Each file holds ten X and ten Z operators of the lightest weight found (`wx`, `wz`: supports as 0-based qubit
indices, the column order of the MatrixMarket files), `d`, `dx`, `dz`, the run time in seconds and the estimator's
summary (`info`: `multiplicity`, the number of distinct operators of the minimum weight it collected, and its
search statistics `hits` and `spectrum`). The operators come from one 600 s run per code of a later version of
the GPU estimator (`gpu-directed-bp-osd-v5`, with a larger OSD step than `distance_estimates/estimator_gpu_v2.py`;
not part of this repository). Their weights equal the bounds of Table 1; no lighter operator was found.

`python codes/verify.py <HX.mtx> <HZ.mtx> <file.json>` checks every operator against the matrices: an X operator
must satisfy `H_Z x = 0` and lie outside the row space of `H_X`, so that its weight is an upper bound on the X
distance; Z operators likewise.
