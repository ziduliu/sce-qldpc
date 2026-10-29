#!/bin/bash -l
#SBATCH -J bbagg5
#SBATCH -p standard
#SBATCH -N 1 -n 1 -c 2
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /u/zdliu/lp/jobs/bbagg5_%j.out
# Aggregates one level: only the records in this run's directory, each checked against the
# manifest. CERTIFIED -> next level up to WMAX; FOUND -> upper bound, climb ends; anything else ->
# INCOMPLETE, the climb stops and nothing is claimed.
set -uo pipefail
module load gcc/11 python-waterboa/2024.06 || true
export PATH=/u/zdliu/lpenv/bin:$PATH
LOG=$OUTDIR/${TAG}_${TYPE}_climb.log
rm -f "$RUNDIR/verdict.json"      # a verdict of an earlier attempt must not be read as this attempt's result
python "$CERT/src/exact_bb.py" --aggregate $RUNDIR/s*.json --expect "$RUNDIR/manifest.json" --out "$RUNDIR/verdict.json" > "$RUNDIR/verdict.txt" 2>&1
AGG_RC=$?
RES=$(python -c "import json;print(json.load(open('$RUNDIR/verdict.json'))['result'])" 2>/dev/null || echo INCOMPLETE)
[ $AGG_RC -ne 0 ] && [ "$RES" = CERTIFIED ] && RES=INCOMPLETE
echo "$(date +%F_%T) W=$W run $(basename $RUNDIR): $RES - $(tail -1 $RUNDIR/verdict.txt | cut -c1-200)" >> "$LOG"
if [ "$RES" = CERTIFIED ] && [ "$W" -lt "$WMAX" ]; then
  "$CERT/cluster_scripts/bb_climb_v5.sh" "$TAG" "$NPZ" "$N" "$TYPE" $((W + 1)) "$ORB" "$WMAX" "$SPEC"
elif [ "$RES" = CERTIFIED ]; then
  echo "$(date +%F_%T) reached WMAX=$WMAX with d_${TYPE} > $WMAX certified; climb ends" >> "$LOG"
elif [ "$RES" = FOUND ]; then
  echo "$(date +%F_%T) operator found at W=$W (upper bound); climb ends" >> "$LOG"
else
  echo "$(date +%F_%T) W=$W INCOMPLETE; climb stops, nothing claimed for this level" >> "$LOG"
fi
