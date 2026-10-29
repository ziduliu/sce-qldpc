#!/bin/bash -l
#SBATCH -J recert5
#SBATCH -p standard
#SBATCH -N 1 -n 1 -c 32
#SBATCH --mem=32G
#SBATCH -t 12:00:00
#SBATCH --export=ALL
#SBATCH -o /u/zdliu/lp/oe_run/recert_v5_20261014/recert_%A_%a.out
#SBATCH --array=0-2
# Re-certifies the three FOM Table I codes whose v4 records lack a stored time limit:
# v5 driver, both types, W = d-1 (must be NONE) and W = d (FOUND gives the exact value).
set -uo pipefail
module load gcc/11 python-waterboa/2024.06 || true
export PATH=/u/zdliu/lpenv/bin:$PATH
CERT=/u/zdliu/lp/cert_v5
A=/u/zdliu/lp/jobs/sel61/archive/codes
OUT=/u/zdliu/lp/oe_run/recert_v5_20261014
CODES=(3c7722431bc2662e:592:18 d233cc7273de92c2:560:14 b51b14afe214768f:588:12)
IFS=: read SHA N D <<< "${CODES[$SLURM_ARRAY_TASK_ID]}"
sha256sum $CERT/src/dkp_bb $CERT/src/exact_bb.py $A/$SHA.npz
python $CERT/src/exact_bb.py --npz $A/$SHA.npz --n $N --type both --wmin $((D-1)) --wmax $D \
  --threads 32 --time 20000 --bin $CERT/src/dkp_bb --out $OUT/${SHA}_v5.json
echo "exit $?"
