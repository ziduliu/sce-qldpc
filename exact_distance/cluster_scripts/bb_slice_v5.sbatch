#!/bin/bash -l
#SBATCH -J bbs5
#SBATCH -p standard
#SBATCH -N 1 -n 1 -c 32
#SBATCH --mem=32G
#SBATCH -t 120:00:00
#SBATCH -o /u/zdliu/lp/jobs/bbs5_%j.out
# One frontier slice of one level. The slice reads its settings from the run's manifest and
# refuses to run if the matrix, orbit file or binary on disk differ from it; its record carries
# the run id and fingerprints and no bound on the code. The exit status is non-zero unless the
# search completed (NONE or verified FOUND).
set -euo pipefail
module load gcc/11 python-waterboa/2024.06 || true
export PATH=/u/zdliu/lpenv/bin:$PATH
export OMP_NUM_THREADS=32
echo "host=$(hostname) run=$(basename $RUNDIR) slice $SLICE/$NSL tlim=$TLIM"
python "$CERT/src/exact_bb.py" --expect "$RUNDIR/manifest.json" --slice "$SLICE/$NSL" --threads 32 --time "$TLIM" \
    --bin "$CERT/src/dkp_bb" --out "$RUNDIR/s${SLICE}.json"
