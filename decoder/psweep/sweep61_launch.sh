#!/bin/bash
# Five-p circuit-level sweep of the gpt-6.1-sol codes. Distance side: the ten
# cell-champion codes of the distance run (codes_gpt61sol_20261004.json; cell 2
# uses dc33333253c79e16) plus three further codes (the C17xC17 two-block code
# and the best single codes of cells 1 and 3). Circuit side: the ten cell
# champions of the circuit-level run (codes_pl61_champions_20261006.json).
# Shots 2M/1M/300k/100k/50k for p = 0.0015/0.002/0.003/0.004/0.005, 1000
# errors, R=6. The p=0.002 legs of the codes in RUNNING_0002 are measured
# separately and not submitted here. Submission order = priority: p=0.002
# first, then 0.003, 0.0015, 0.004, 0.005.
# One job = one code, one p, two ranks (the two bases) on two GPUs.
#   DRY=1 sweep61_launch.sh   # list only
set -u
cd /u/zdliu/lp/plevolve_20260912
OUT=/u/zdliu/lp/oe_run/pl_sweep_20261006_61
DIST_ARCH=/u/zdliu/lp/jobs/sel61/archive
PLC_ARCH=/u/zdliu/lp/oe_run/pl10_gpt61sol_out/archive
PY=/u/zdliu/lpenv/bin/python
mkdir -p $OUT
DIST=$($PY - <<'EOF'
import json
led = [json.loads(l) for l in open("/u/zdliu/lp/oe_run/cc_v7_gpt61sol500_w12_10c_out/eval_ledger.jsonl") if l.strip()]
fam2q = {r["family"]: r["q"] for r in led}
rows = []
for x in json.load(open("/u/zdliu/lp/jobs/codes_gpt61sol_20261004.json")):
    rows.append((x["cell"], "dc33333253c79e16" if x["cell"] == 2 else x["sha"]))
rows += [(0, "6612ba96961ef2c1"), (1, "30e219762cabf54d"), (3, "b028585ed55c4437")]  # C17xC17, Table I best codes of cells 1 and 3
for c, sha in rows:
    m = json.load(open("/u/zdliu/lp/jobs/sel61/archive/codes/%s.json" % sha))
    q = fam2q.get(m["family"])
    if q is None:
        raise SystemExit("no q for family %s (sha %s)" % (m["family"], sha))
    print("%d %s %s %s %s" % (c, sha, q, m["n"], m["k"]))
EOF
) || exit 1
echo "$DIST" | sed 's/^/dist /'
PLC=$($PY - <<'EOF'
import json
for x in json.load(open("/u/zdliu/lp/jobs/codes_pl61_champions_20261006.json")):
    print("%d %s %s %s %s" % (x["cell"], x["sha"], x["q"], x["n"], x["k"]))
EOF
) || exit 1
echo "$PLC" | sed 's/^/plc  /'
[ "${DRY:-0}" = "1" ] && exit 0
submit() {  # submit ARM ARCH CELL SHA Q P SHOTS
  local arm=$1 arch=$2 c=$3 sha=$4 q=$5 p=$6 shots=$7
  local tag=${arm}61_c${c}_${sha:0:8}_p${p}
  if ls $OUT/${tag}_* >/dev/null 2>&1; then echo "skip $tag (result exists)"; return; fi
  local J
  J=$(sbatch --parsable -p gpu --nodes=1 --ntasks=2 --ntasks-per-node=2 --gres=gpu:2 --cpus-per-task=6 --mem=100G --time=12:00:00 \
      --export=ALL,ARCHIVE=$arch,OUT=$OUT,TAG=$tag,LEGS=$p:$shots,MAXERR=1000,HASH=c$c:$q:$sha deepverify_pl.sbatch)
  echo "$(date +%F_%T) sweep61 $arm cell $c $sha q=$q p=$p shots=$shots job $J" | tee -a $OUT/LAUNCH.txt
}
RUNNING_0002="6612ba96961ef2c1 fd925ff20fe235cc dc33333253c79e16 d233cc7273de92c2"
for p_shots in 0.002:1000000 0.003:300000 0.0015:2000000 0.004:100000 0.005:50000; do
  p=${p_shots%%:*}; shots=${p_shots##*:}
  while read c sha q n k; do
    [ "$p" = "0.002" ] && echo "$RUNNING_0002" | grep -q "$sha" && { echo "skip dist c$c $sha p=0.002 (Table I job running or done)"; continue; }
    submit dist $DIST_ARCH $c $sha $q $p $shots
  done <<< "$DIST"
  while read c sha q n k; do
    submit plc $PLC_ARCH $c $sha $q $p $shots
  done <<< "$PLC"
done
echo "deepvpl jobs in queue: $(squeue -u zdliu -h -n deepvpl | wc -l)"
