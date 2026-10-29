#!/bin/bash
# Launch one run on the ten-cell rate grid (k/n 0.10 ... 0.55, both boundary
# cells developed): one controller job and NWORK GPU worker jobs.
#
#   launch_10cells.sh <ITERS> <TAG> [NWORK]
#     launch_10cells.sh 50 smoke 8      short check that prompt and engine agree
#     launch_10cells.sh 500 prod 16     production run
#
# Other settings: 60 slots, full tables, per-cell pools of 12, parent drawn
# uniformly among the top 3 shapes.
#
# Never pipe this script into head: SIGPIPE kills the loop after one line.
set -euo pipefail
ITERS=${1:?iters}; TAG=${2:?tag}; NWORK=${3:-8}
CFG=${CFG:-oe_circmean_cc_v7_luna_xhigh_w12_p16_nocyc_v3_10cells.yaml}
SEEDDIR=${SEEDDIR:-seeds_rates_w14_plus_v5_nocyc_v3}
CTIME=${CTIME:-24:00:00}; WTIME=${WTIME:-24:00:00}

cd /u/zdliu/lp/circmean
[ -f "$CFG" ] || { echo "no config $CFG - run patch_ten_cells.py --apply first"; exit 2; }
grep -q "LAYOUT_RATE_EDGE_CELLS" oe_circmean.sbatch || { echo "launcher not patched"; exit 2; }
grep -q "LAYOUT_RATE_EDGE_CELLS" /u/zdliu/lpenv/lib/python3.12/site-packages/openevolve/database.py \
  || { echo "engine not patched"; exit 2; }

# --- file-quota gate: refuse to launch when the /u fileset has fewer than
# MIN_FREE_FILES free files (a 500-step run writes ~80k archive files).
_FREE=$(/usr/lpp/mmfs/bin/mmlsquota -u zdliu 2>/dev/null | awk '$2=="root"{print $12-$10}')
if [ -n "$_FREE" ] && [ "$_FREE" -lt "${MIN_FREE_FILES:-250000}" ]; then
  echo "GUARD: only $_FREE files free on /u (need ${MIN_FREE_FILES:-250000}); move finished archives first (archive_move_*.sbatch)"; exit 2
fi
echo "quota gate: $_FREE files free"
OUT=/u/zdliu/lp/oe_run/cc_v7_luna${ITERS}_10cells_${TAG}_out
SPOOL=/u/zdliu/lp/jobs/cc7_luna${ITERS}_10cells_${TAG}_spool
mkdir -p "$OUT" "$SPOOL"

# EDGECELLS=1 develops the two boundary rate cells.
COMMON="ITERS=$ITERS,EFFORT=xhigh"
COMMON="$COMMON,NMIN=100,NMAX=600,WMAX=12,NSLOTS=60,TABLETOP=0,TCELLS=1"
COMMON="$COMMON,RATELO=0.10,RATESTEP=0.05,RATEBINS=10,BINISL=1"
COMMON="$COMMON,INSP=4,TOPG=1,FULLTBL=1,CARDSELF=0,QUOTA=0"
COMMON="$COMMON,CELLPOOL=12,SHAPEK=3,EDGECELLS=1"
WCOMMON="NMIN=100,NMAX=600,WMAX=12,RATELO=0.10,RATESTEP=0.05,RATEBINS=10,WK_IDLE_EXIT=${WK_IDLE_EXIT:-1800}"

C=$(sbatch --parsable --time=$CTIME \
    --export=ALL,$COMMON,OUT=$OUT,SPOOL=$SPOOL,CONFIG=$CFG,EVALUATOR=oe_evaluator_circmean_v3.py,SEEDS=$SEEDDIR \
    oe_circmean.sbatch)
echo "controller $C -> $OUT"
for i in $(seq 1 "$NWORK"); do
  W=$(sbatch --parsable --exclude=zpg002 --time=$WTIME \
      --export=ALL,$WCOMMON,SPOOL=$SPOOL,EVAL=/u/zdliu/lp/circmean/eval_circmean_v3.py,DIST=cc \
      circmean_worker.sbatch)
  echo "worker $W"
done
