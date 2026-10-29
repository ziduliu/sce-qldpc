#!/bin/bash -l
#SBATCH -J w9certctl
#SBATCH -p standard
#SBATCH -N 1 -n 1 -c 4
#SBATCH --mem=16G
#SBATCH -t 120:00:00
#SBATCH -o /u/zdliu/lp/oe_run/w9cert_20261013/ctl_%j.out
# Controller for the exact certification of the codes of the weight-9 table, 2026-10-13.
# 1) select codes + checks, 2) Tanner-graph automorphism orbits, 3) submit one climb per code and
# type with bb_climb61.sh (dkp_bb_v4, frontier slices), at most CAP certification jobs at a time
# so that the account's 250-job limit never blocks the GPU workers of the random lines.
# Climb per code: start at W0 = d30s-2 (d30s-1 for d30s >= 20) and stop at WMAX = d30s-1;
# NONE at every level up to d30s-1 together with the estimator's weight-d30s operator is exact.
set -uo pipefail
module load gcc/11 python-waterboa/2024.06 || true
export PATH=/u/zdliu/lpenv/bin:$PATH
D=/u/zdliu/lp/oe_run/w9cert_20261013; cd $D
CAP=${CAP:-90}
python select_w9.py > select.log 2>&1; tail -2 select.log
python - <<'PY' || { echo "CHECKS FAILED, stopping"; exit 1; }
import json
m=json.load(open("manifest.json")); bad=[x["tag"] for x in m if not (x["check"]["css"] and x["check"]["k"]==x["k"] and x["check"]["n"]==x["n"] and x["check"]["w"]<=9)]
print("checks failed:",bad); assert not bad and len(m)==50
PY
mkdir -p orbits
python - <<'PY' > order.txt
import json
m=json.load(open("manifest.json"))
m.sort(key=lambda x:(x["d30s"]>=20, x["d30s"], x["cell"], x["slot"]))   # cheap first, the d=24 cell last
for x in m: print(x["tag"], x["n"], x["d30s"])
PY
cd /u/zdliu/lp/circmean
export TLIM=400000
while read TAG N D30; do
  python code_orbits.py --npz $D/npz/$TAG.npz --n $N --out $D/orbits/$TAG.json > $D/orbits/$TAG.log 2>&1 || { echo "orbits failed $TAG"; continue; }
  if   [ $D30 -le 9 ];  then SPEC="99:1"; NS=1
  elif [ $D30 -le 14 ]; then SPEC="99:4"; NS=4
  elif [ $D30 -le 19 ]; then SPEC="99:8"; NS=8
  else SPEC="99:60"; NS=60; fi
  W0=$((D30 - 2)); [ $D30 -ge 20 ] && W0=$((D30 - 1)); [ $W0 -lt 1 ] && W0=1
  for T in X Z; do
    grep -q "submitted $TAG $T " $D/submitted.log 2>/dev/null && continue
    while :; do
      CUR=$(squeue -u zdliu -h -n bbs61 | wc -l); ALL=$(squeue -u zdliu -h -t RUNNING | wc -l)
      [ $((CUR + NS)) -le $CAP ] && [ $ALL -le 200 ] && break
      sleep 600
    done
    ./bb_climb61.sh $TAG $D/npz/$TAG.npz $N $T $W0 $D/orbits/$TAG.json $((D30 - 1)) "$SPEC" | tail -1
    echo "$(date +%F_%T) submitted $TAG $T W0=$W0 WMAX=$((D30-1)) slices=$NS" >> $D/submitted.log
  done
done < $D/order.txt
echo "$(date +%F_%T) all climbs submitted" >> $D/submitted.log
