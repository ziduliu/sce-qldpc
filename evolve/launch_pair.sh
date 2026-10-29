#!/bin/bash
# Launch the paired runs: distance objective and circuit-level pL objective,
# identical in everything else (engine, seeds, grid, window, weight cap, slots,
# cards, effort, step count).
#
#   launch_pair.sh <ITERS> <TAG> [cc|pl|both]     e.g. launch_pair.sh 6 smoke both
#
# Every value that is NOT the objective is set ONCE below and passed to both.
set -euo pipefail
ITERS=${1:?iters}; TAG=${2:?tag}; WHICH=${3:-both}
NWORK=${NWORK:-8}
PL_SHOTS=${PL_SHOTS:-200000}          # pL line: shot cap per code-basis
CC_CONFIG=${CC_CONFIG:-oe_circmean_cc_v7.yaml}
PL_CONFIG=${PL_CONFIG:-oe_circmean_pl_v3.yaml}

COMMON="ITERS=$ITERS,SEEDS=seeds_rates_w14_plus_v5,EFFORT=xhigh"
COMMON="$COMMON,NMIN=100,NMAX=600,WMAX=12,NSLOTS=60,TABLETOP=0,TCELLS=1"
COMMON="$COMMON,RATELO=0.10,RATESTEP=0.05,RATEBINS=10,BINISL=1"
COMMON="$COMMON,INSP=4,TOPG=1,FULLTBL=1,CARDSELF=0,QUOTA=0"
WCOMMON="NMIN=100,NMAX=600,WMAX=12,RATELO=0.10,RATESTEP=0.05,RATEBINS=10"
# workers must outlive the run: long idle-exit and a wall limit above the run
# length; timeouts are sized for the 100k-shot pL stage (one program can take
# several hours).
WCOMMON="$WCOMMON,WK_IDLE_EXIT=130000"
WTIME=${WTIME:-120:00:00}                 # partition max is 7 days; the pL line can need >47 h
CTIME=${CTIME:-120:00:00}

launch_cc () {
  OUT=/u/zdliu/lp/oe_run/cc_v7_${TAG}_out; SPOOL=/u/zdliu/lp/jobs/cc7_${TAG}_spool
  mkdir -p "$OUT" "$SPOOL"
  cd /u/zdliu/lp/circmean
  C=$(sbatch --parsable --time=$CTIME --export=ALL,$COMMON,OUT=$OUT,SPOOL=$SPOOL,CONFIG=$CC_CONFIG,EVALUATOR=oe_evaluator_circmean_v3.py oe_circmean.sbatch)
  echo "cc  controller $C  -> $OUT"
  for i in $(seq 1 $NWORK); do
    W=$(sbatch --parsable --exclude=zpg002 --time=$WTIME --export=ALL,$WCOMMON,SPOOL=$SPOOL,EVAL=/u/zdliu/lp/circmean/eval_circmean_v3.py,DIST=cc circmean_worker.sbatch)
    echo "cc  worker $W"
  done
}

launch_pl () {
  OUT=/u/zdliu/lp/oe_run/pl_v3_${TAG}_out; SPOOL=/u/zdliu/lp/jobs/pl3_${TAG}_spool
  mkdir -p "$OUT" "$SPOOL"
  cd /u/zdliu/lp/plevolve_20260912
  C=$(sbatch --parsable --time=$CTIME --export=ALL,$COMMON,OUT=$OUT,SPOOL=$SPOOL,CONFIG=$PL_CONFIG,OETIMEOUT=70000 oe_circmean_pl.sbatch)
  echo "pl  controller $C  -> $OUT"
  for i in $(seq 1 $NWORK); do
    W=$(sbatch --parsable --exclude=zpg002 --time=$WTIME --export=ALL,$WCOMMON,SPOOL=$SPOOL,SHOTS=$PL_SHOTS,MAXERR=100,PLP=0.002,WKTIMEOUT=62000,PLTIMEOUT=60000,WKLEASE=70000 worker_pl.sbatch)
    echo "pl  worker $W"
  done
}

case "$WHICH" in
  cc) launch_cc ;;
  pl) launch_pl ;;
  both) launch_cc; launch_pl ;;
  *) echo "which: cc|pl|both"; exit 2 ;;
esac
