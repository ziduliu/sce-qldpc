#!/bin/bash
# One level of a certified-distance climb. Every submission gets its own run
# directory and a manifest written BEFORE the slices are submitted (matrix, orbit file and binary
# fingerprints, problem fingerprint, slice count). Slices refuse to run if the inputs on disk no
# longer match the manifest; the aggregator reads only this run's directory and checks every record
# against the manifest, so an older result can never be reused.
#   bb_climb_v5.sh TAG NPZ N TYPE W ORBITS|- WMAX SLICESPEC
#   SLICESPEC "23:4,25:16,99:60" = 4 slices up to W=23, 16 up to 25, then 60
set -euo pipefail
TAG=$1; NPZ=$2; N=$3; TYPE=$4; W=$5; ORB=$6; WMAX=$7; SPEC=${8//;/,}
TLIM=${TLIM:-400000}
CERT=${CERT:-/u/zdliu/lp/cert_v5}            # src/ with exact_bb.py, code_orbits.py and the dkp_bb binary
OUTDIR=${OUTDIR:-/u/zdliu/lp/oe_run/exact_v5}
BIN=$CERT/src/dkp_bb
NSL=60
for kv in $(echo "$SPEC" | tr "," " "); do lim=${kv%%:*}; ns=${kv##*:}; if [ "$W" -le "$lim" ]; then NSL=$ns; break; fi; done
RUN_ID=${TAG}_${TYPE}_W${W}_$(date +%Y%m%d%H%M%S)_$$_$RANDOM
RUNDIR=$OUTDIR/runs/$RUN_ID
mkdir -p "$OUTDIR/runs"; mkdir "$RUNDIR"                      # fails if the directory exists
ORBARG=(); [ "$ORB" != "-" ] && ORBARG=(--orbits "$ORB")
module load gcc/11 python-waterboa/2024.06 >/dev/null 2>&1 || true
export PATH=/u/zdliu/lpenv/bin:$PATH
python "$CERT/src/exact_bb.py" --make-manifest --run-id "$RUN_ID" --npz "$NPZ" --n "$N" --type "$TYPE" --W "$W" \
    --slices "$NSL" --bin "$BIN" "${ORBARG[@]}" --out "$RUNDIR/manifest.json"
LOG=$OUTDIR/${TAG}_${TYPE}_climb.log
echo "$(date +%F_%T) level W=$W: $NSL slices, run $RUN_ID" >> "$LOG"
IDS=""
for s in $(seq 0 $((NSL - 1))); do
  J=$(sbatch --parsable --export=ALL,CERT=$CERT,RUNDIR=$RUNDIR,SLICE=$s,NSL=$NSL,TLIM=$TLIM "$CERT/cluster_scripts/bb_slice_v5.sbatch")
  IDS="$IDS${IDS:+:}$J"
done
A=$(sbatch --parsable --dependency=afterany:$IDS --export=ALL,CERT=$CERT,RUNDIR=$RUNDIR,TAG=$TAG,NPZ=$NPZ,N=$N,TYPE=$TYPE,W=$W,ORB=$ORB,WMAX=$WMAX,SPEC=${SPEC//,/;},TLIM=$TLIM,OUTDIR=$OUTDIR "$CERT/cluster_scripts/bb_agg_v5.sbatch")
echo "$IDS" | tr ":" "\n" > "$RUNDIR/slice_jobs.txt"
echo "$(date +%F_%T)   slices $IDS aggregator $A" >> "$LOG"
echo "W=$W: $NSL slice jobs, aggregator $A, run $RUN_ID"
