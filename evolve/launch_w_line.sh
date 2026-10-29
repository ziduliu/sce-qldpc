#!/bin/bash
# Launch one ten-cell run with row-weight cap WMAX: one controller job and
# NWORK GPU worker jobs. Shape-stratified parent selection is off (SHAPEK=0).
#
#   launch_w_line.sh <WMAX> <ITERS> <TAG> [NWORK]
#     launch_w_line.sh 9  6   smoke 8      throwaway smoke
#     launch_w_line.sh 14 500 prod  16     production
#
# RESUME: CKPT=<checkpoint dir> and RESUME_OUT=<the run's existing OUT dir>
# continue that run in place - same OUT, same spool, same archive
# (ARCHIVE=OUT/archive keeps the duplicate-code check), same ledger appended.
# ITERS is then the ADDITIONAL iterations.
#     CKPT=.../checkpoints/checkpoint_110 RESUME_OUT=.../cc_v7_luna500_w9_10c_prod_out \
#       launch_w_line.sh 9 390 prod 16
#
# The cap is written in three places the model reads (config prompt, seed
# scaffold fallback) plus the evaluator's environment. This script refuses to
# start unless all of them say the same number - make_w_lines.py writes the
# per-w config and seeds.
#
# Never pipe this script into head: SIGPIPE kills the loop after one line.
set -euo pipefail
WMAX=${1:?wmax}; ITERS=${2:?iters}; TAG=${3:?tag}; NWORK=${4:-8}
SHAPEK=${SHAPEK:-0}
EFFORT=${EFFORT:-xhigh}
# CFG override: a config that differs from the default one only in model name
# and effort; the cap guards below still apply to it.
CFG=${CFG:-oe_circmean_cc_v7_luna_xhigh_w${WMAX}_p16_nocyc_v3_10cells.yaml}
SEEDDIR=${SEEDDIR:-seeds_rates_w14_plus_v5_nocyc_v3_w${WMAX}}
# LOCKCELL=<b>: every round targets that cell (engine LAYOUT_LOCK_CELL)
LOCKCELL=${LOCKCELL:-}
CTIME=${CTIME:-24:00:00}; WTIME=${WTIME:-24:00:00}

cd /u/zdliu/lp/circmean
[ -f "$CFG" ] || { echo "no config $CFG - run make_w_lines.py --apply"; exit 2; }
[ -d "$SEEDDIR" ] || { echo "no seed dir $SEEDDIR - run make_w_lines.py --apply"; exit 2; }
# --- the three places must agree on the cap ---
grep -q "max row weight <= $WMAX\b" "$CFG" \
  || { echo "GUARD: $CFG does not state 'max row weight <= $WMAX'"; exit 2; }
if grep -q "<= 12\b" "$CFG" && [ "$WMAX" != 12 ]; then
  echo "GUARD: $CFG still carries a '<= 12' literal"; exit 2; fi
NSEED=$(ls "$SEEDDIR"/seed_*.py | wc -l)
NOK=$(grep -l "\"CM_WMAX\", \"$WMAX\"" "$SEEDDIR"/seed_*.py | wc -l)
# every seed must carry this cap as its fallback; the standard set has 8, a
# single-cell run has 1
[ "$NSEED" -ge 1 ] && [ "$NOK" = "$NSEED" ] \
  || { echo "GUARD: seeds: $NSEED files, $NOK with fallback $WMAX (need all)"; exit 2; }
grep -q "10 cells centred" "$CFG" || { echo "GUARD: config is not the ten-cell prompt"; exit 2; }
grep -q "LAYOUT_RATE_EDGE_CELLS" oe_circmean.sbatch || { echo "GUARD: launcher not patched"; exit 2; }
grep -q "LAYOUT_RATE_EDGE_CELLS" /u/zdliu/lpenv/lib/python3.12/site-packages/openevolve/database.py \
  || { echo "GUARD: engine not patched"; exit 2; }
echo "guards passed: cap $WMAX in config, $NOK/$NSEED seeds ($SEEDDIR), ten-cell prompt, lock ${LOCKCELL:-none}, SHAPEK=$SHAPEK, model $(grep -m1 "name: gpt" "$CFG" | tr -d " ") effort $EFFORT"
# --- file-quota gate: refuse to launch when the /u fileset has fewer than
# MIN_FREE_FILES free files (a 500-step run writes ~80k archive files).
_FREE=$(/usr/lpp/mmfs/bin/mmlsquota -u zdliu 2>/dev/null | awk '$2=="root"{print $12-$10}')
if [ -n "$_FREE" ] && [ "$_FREE" -lt "${MIN_FREE_FILES:-250000}" ]; then
  echo "GUARD: only $_FREE files free on /u (need ${MIN_FREE_FILES:-250000}); move finished archives first (archive_move_*.sbatch)"; exit 2
fi
echo "quota gate: $_FREE files free"

CKPT=${CKPT:-}; RESUME_OUT=${RESUME_OUT:-}
if [ -n "$CKPT" ]; then
  [ -d "$CKPT/programs" ] && [ -f "$CKPT/metadata.json" ] || { echo "GUARD: $CKPT is not a complete checkpoint"; exit 2; }
  [ -d "$RESUME_OUT" ] || { echo "GUARD: RESUME_OUT must be the existing run dir"; exit 2; }
  case "$CKPT" in "$RESUME_OUT"/checkpoints/*) ;; *) echo "GUARD: checkpoint is not inside RESUME_OUT"; exit 2 ;; esac
  OUT=$RESUME_OUT
  SPOOL=/u/zdliu/lp/jobs/$(basename "$OUT" _out | sed "s/cc_v7_/cc7_/")_spool
  [ -d "$SPOOL" ] || { echo "GUARD: spool $SPOOL missing"; exit 2; }
  EXTRA=",CKPT=$CKPT,ARCHIVE=$OUT/archive"
  echo "RESUME from $(basename "$CKPT") (last_iteration $(grep -o '"last_iteration": *[0-9]*' "$CKPT/metadata.json" | grep -o '[0-9]*$')), +$ITERS iterations, into $OUT"
else
  # MODEL tag in the run name (default luna); the actual model comes from CFG
  MODEL=${MODEL:-luna}
  OUT=/u/zdliu/lp/oe_run/cc_v7_${MODEL}${ITERS}_w${WMAX}_10c_${TAG}_out
  SPOOL=/u/zdliu/lp/jobs/cc7_${MODEL}${ITERS}_w${WMAX}_10c_${TAG}_spool
  mkdir -p "$OUT" "$SPOOL"
  EXTRA=""
fi

COMMON="ITERS=$ITERS,EFFORT=$EFFORT"
COMMON="$COMMON,NMIN=100,NMAX=600,WMAX=$WMAX,NSLOTS=60,TABLETOP=0,TCELLS=1"
COMMON="$COMMON,RATELO=0.10,RATESTEP=0.05,RATEBINS=10,BINISL=1"
COMMON="$COMMON,INSP=4,TOPG=1,FULLTBL=1,CARDSELF=0,QUOTA=0"
COMMON="$COMMON,CELLPOOL=12,SHAPEK=$SHAPEK,EDGECELLS=1"
[ -n "$LOCKCELL" ] && COMMON="$COMMON,LAYOUT_LOCK_CELL=$LOCKCELL"
WCOMMON="NMIN=100,NMAX=600,WMAX=$WMAX,RATELO=0.10,RATESTEP=0.05,RATEBINS=10,WK_IDLE_EXIT=${WK_IDLE_EXIT:-1800}"

C=$(sbatch --parsable --time=$CTIME \
    --export=ALL,$COMMON,OUT=$OUT,SPOOL=$SPOOL,CONFIG=$CFG,EVALUATOR=oe_evaluator_circmean_v3.py,SEEDS=$SEEDDIR$EXTRA \
    oe_circmean.sbatch)
echo "controller $C -> $OUT"
# exclude node zpg002 only while Slurm still knows it; an unknown node name
# makes sbatch fail
EXCL=""; sinfo -h -n zpg002 -o %N 2>/dev/null | grep -q zpg002 && EXCL="--exclude=zpg002"
for i in $(seq 1 "$NWORK"); do
  W=$(sbatch --parsable $EXCL --time=$WTIME \
      --export=ALL,$WCOMMON,SPOOL=$SPOOL,EVAL=/u/zdliu/lp/circmean/eval_circmean_v3.py,DIST=cc \
      circmean_worker.sbatch)
  echo "worker $W"
done
