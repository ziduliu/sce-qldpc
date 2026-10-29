#!/bin/bash -l
#SBATCH -J deepvpl
#SBATCH -o /u/zdliu/lp/jobs/deepvpl_%j.out
#SBATCH -e /u/zdliu/lp/jobs/deepvpl_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=5
#SBATCH --ntasks=10
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --export=ALL
#
# Circuit-level re-measurement of selected codes (deepverify_pl.py) at one or
# more (p, max shots) legs.
#
# Ten tasks: five codes x two bases = ten units, one rank and one GPU each
# (PL_SCORE_SRUN=1). Packing ranks onto a GPU also needs PLF_GROUP (rankgpu.sh).
#
# Submit with --export=ALL on the command line (SBATCH_EXPORT=NONE overrides
# the in-script --export=ALL):
#   sbatch --export=ALL,PICK='fam:q:6,7,8,3,2',TAG=band2 deepverify_pl.sbatch
set -uo pipefail
module load gcc/11 2>/dev/null || true
module load cuda/12.2 2>/dev/null || true
module load python-waterboa/2024.06
export PATH=/u/zdliu/lpenv/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ARCHIVE=${ARCHIVE:-/u/zdliu/lp/oe_run/pl_v2_out/archive}
OUT=${OUT:-/u/zdliu/lp/oe_run/deepverify_20260913}
TAG=${TAG:-run}
mkdir -p "$OUT"

# fixed across legs: rounds, decoder, schedule search
export PL_SCORE_ROUNDS=${PLR:-6}
export PL_SCORE_SCHED_TIME=${SCHEDTIME:-300}
export PL_SCORE_TIMEOUT=${PLTIMEOUT:-40000}
export PL_SCORE_SRUN=1
export PL_SCORE_KEEP=1

# legs p:max_shots; max_errors defaults to 1000 so the stopping rule does not
# end a leg while the interval is still wide
LEGS=${LEGS:-"0.002:200000 0.003:100000 0.004:50000"}

# PICK and HASH select codes from the archive, NPZ reads codes from matrix
# files; at least one is required, entries are separated by ";". Use "+" inside
# a value, never a comma: --export=ALL,VAR=v treats a comma as a variable
# separator.
PICK=${PICK:-}
NPZ=${NPZ:-}
HASH=${HASH:-}
ARGS=""
for spec in $(echo "$PICK" | tr ';' ' '); do
  [ -n "$spec" ] && ARGS="$ARGS --pick $spec"
done
for spec in $(echo "$NPZ" | tr ';' ' '); do
  [ -n "$spec" ] && ARGS="$ARGS --npz $spec"
done
for spec in $(echo "$HASH" | tr ';' ' '); do
  [ -n "$spec" ] && ARGS="$ARGS --hash $spec"
done
if [ -z "$ARGS" ]; then echo "FATAL: none of PICK/NPZ/HASH given"; exit 2; fi

echo "archive=$ARCHIVE"
echo "args=$ARGS"
echo "legs=$LEGS"
nvidia-smi --query-gpu=index,name --format=csv,noheader 2>/dev/null | head -3

for leg in $LEGS; do
  P=${leg%%:*}; S=${leg##*:}
  echo
  echo "================ p=$P shots<=$S ================"
  export PL_SCORE_P=$P
  export PL_SCORE_SHOTS=$S
  export PL_SCORE_MAX_ERRORS=${MAXERR:-1000}
  /u/zdliu/lpenv/bin/python -u /u/zdliu/lp/plevolve_20260912/deepverify_pl.py \
      --archive "$ARCHIVE" $ARGS \
      --workdir "$OUT/work_${TAG}_p${P}" \
      --out "$OUT/${TAG}_p${P}_s${S}.json"
  rc=$?
  echo "leg p=$P exit=$rc"
done
echo "DONE $TAG"
