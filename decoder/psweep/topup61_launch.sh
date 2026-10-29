#!/bin/bash
# Top-up of the gpt-6.1-sol five-p sweep so that every leg has at least ten
# failures in its worse basis, with one fixed schedule per code.
#
# Per code (ten distance champions, ten circuit champions, the [[144,12,12]]
# reference):
#  1. every leg uses the schedule of the code's p=0.002 leg: that file is
#     copied into the new work directory before the job starts, so
#     ensure_schedule finds it cached and nothing is re-solved;
#  2. legs whose worse basis has fewer than ten failures get the shot budget
#     in TOPUP; legs with ten failures but a different schedule are re-run with
#     their original budget; the rest are left alone.
# R=6, stop at 1000 failures, PLF_WORKERS=5 (chunk-level worker pool).
#   DRY=1 topup61_launch.sh   # list only
set -u
cd /u/zdliu/lp/plevolve_20260912
SW=/u/zdliu/lp/oe_run/pl_sweep_20261006_61
S28=/u/zdliu/lp/oe_run/pl_sweep_20260928
OUT=/u/zdliu/lp/oe_run/pl_sweep_20261007_topup
DIST_ARCH=/u/zdliu/lp/jobs/sel61/archive
PLC_ARCH=/u/zdliu/lp/oe_run/pl10_gpt61sol_out/archive
BBNPZ=$S28/work_bb144_p0002_p0.002/bb144.npz
mkdir -p $OUT
LIST=$(python3 - <<'PY'
import json, glob, os
SW="/u/zdliu/lp/oe_run/pl_sweep_20261006_61"; S28="/u/zdliu/lp/oe_run/pl_sweep_20260928"
CODES=[("dist",0,"fd925ff20fe235cc",255),("dist",0,"6612ba96961ef2c1",289),("dist",1,"3c7722431bc2662e",37),("dist",2,"dc33333253c79e16",117),("dist",3,"d233cc7273de92c2",28),
       ("dist",4,"b51b14afe214768f",84),("dist",5,"15d39679803f52d4",16),("dist",6,"db18259eb1856d92",56),("dist",7,"b7bb3363de23fb63",21),
       ("dist",8,"612dfb68fc96d396",40),("dist",9,"cf5015a213992503",24),
       ("plc",0,"ba8e654b964da969",46),("plc",1,"30b645d7a9a46846",36),("plc",2,"ba427de15c84301e",29),("plc",3,"239bca5f913aa8b1",29),
       ("plc",4,"1048a36b0f576939",42),("plc",5,"8031fd1f7b90d19b",13),("plc",6,"d8d92f24197070fc",55),("plc",7,"ccf067d3f16c1b1e",29),
       ("plc",8,"f4eda3e53eb8fdb2",19),("plc",9,"217b1dab11f6cb77",32)]
# shot budget per p (the stopping rule at 1000 failures ends easy legs early);
# the hard cases are raised in TOPUP
ORIG={"0.001":5000000,"0.002":1000000,"0.0025":4000000,"0.003":300000,"0.0035":1000000,"0.004":100000,"0.0045":300000,"0.005":50000}
# legs with no failure at all in 5e6 shots get 1e8
TOPUP={("bb",None,"0.001"):100000000,("bb",None,"0.002"):6000000,
       ("dist",0,"0.002"):10000000,("dist",0,"0.001"):100000000,
       ("dist","0:6612ba96","0.002"):20000000,("dist","0:6612ba96","0.001"):100000000,
       ("dist",1,"0.002"):10000000,("dist",1,"0.001"):100000000,
       ("dist",2,"0.001"):16000000,("dist",3,"0.001"):50000000,
       ("plc",0,"0.003"):6000000,("plc",0,"0.002"):20000000,("plc",0,"0.001"):100000000,
       ("plc",1,"0.002"):5000000,("plc",1,"0.001"):100000000,
       ("plc",2,"0.002"):3000000,("plc",2,"0.001"):100000000,
       ("plc",3,"0.001"):100000000,
       ("plc",0,"0.0025"):50000000,("dist","0:6612ba96","0.0025"):20000000}
# legs that cannot reach ten failures by sampling: never submitted
SKIP={("dist",0,"0.001"),("dist","0:6612ba96","0.001"),("dist",1,"0.001"),("plc",0,"0.001"),("plc",1,"0.001"),("plc",2,"0.001"),("plc",3,"0.001")}
def ticks(path):
    try: S=json.load(open(path))
    except Exception: return None
    return {k:v.get("ticks") for k,v in S.items()}
out=[]
for (arm,c,sha,q) in CODES:
    tag="%s61_c%d_%s"%(arm,c,sha[:8])
    ref="%s/work_%s_p0.002_p0.002/asc_schedules.json"%(SW,tag)
    RT=ticks(ref)
    for p in ORIG:
        if (arm,c,p) in SKIP or (arm,"%d:%s"%(c,sha[:8]),p) in SKIP:
            continue
        # the second code of a cell is keyed by "cell:sha8" in TOPUP
        shots=TOPUP.get((arm,"%d:%s"%(c,sha[:8]),p), TOPUP.get((arm,c,p)) if sha[:8]!="6612ba96" else None)
        legdir="%s/work_%s_p%s_p%s"%(SW,tag,p,p)
        T=ticks(legdir+"/asc_schedules.json")
        same=(T is not None and RT is not None and all(T.get(k)==RT[k] for k in RT))
        if shots is None and same: continue
        reason="topup" if shots else ("new" if T is None else "schedule")
        out.append("%s %d %s %d %s %d %s %s"%(arm,c,sha,q,p,shots or ORIG[p],reason,ref))
ref=S28+"/work_bb144_p0002_p0.002/asc_schedules.json"; RT=ticks(ref)
for p in ORIG:
    shots=TOPUP.get(("bb",None,p)); d="%s/work_bb144_p%s_p%s"%(S28,p.replace(".",""),p)
    T=ticks(d+"/asc_schedules.json"); same=(T is not None and all(T.get(k)==RT[k] for k in RT))
    if shots is None and same: continue
    out.append("bb 0 bb144 72 %s %d %s %s"%(p,shots or ORIG[p],"topup" if shots else ("new" if T is None else "schedule"),ref))
print("\n".join(out))
PY
)
n=0; gpuh=0
while read arm c sha q p shots reason ref; do
  [ -z "$arm" ] && continue
  if [ "$arm" = "bb" ]; then tag=bb144_p${p}; else tag=${arm}61_c${c}_${sha:0:8}_p${p}; fi
  if ls $OUT/${tag}_* >/dev/null 2>&1; then echo "skip $tag (result exists)"; continue; fi
  if ls -d $OUT/work_${tag}_* >/dev/null 2>&1; then echo "skip $tag (running)"; continue; fi
  TLIM=12:00:00; nice=""; k=1; ME=1000
  [ "$shots" -ge 10000000 ] && { k=2; ME=0; }
  [ "$shots" -ge 40000000 ] && { k=4; TLIM=24:00:00; nice="--nice=500"; }
  printf "%-5s c%d %s p=%s shots=%-9d %-8s tlim=%s shards=%d\n" $arm $c ${sha:0:8} $p $shots $reason $TLIM $k
  n=$((n+1))
  [ "${DRY:-0}" = "1" ] && continue
  W=$OUT/work_${tag}_p${p}; mkdir -p $W; cp $ref $W/asc_schedules.json
  if [ "$arm" = "bb" ]; then
    J=$(sbatch --parsable $nice -p gpu --nodes=$((2*k)) --ntasks=$((2*k)) --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=6 --mem=50G --time=$TLIM \
        --export=ALL,ARCHIVE=$DIST_ARCH,OUT=$OUT,TAG=$tag,LEGS=$p:$shots,MAXERR=$ME,NPZ=bb144:72:$BBNPZ,PLF_WORKERS=5,PL_SCORE_SHARDS=$k deepverify_pl.sbatch)
  else
    arch=$DIST_ARCH; [ "$arm" = "plc" ] && arch=$PLC_ARCH
    J=$(sbatch --parsable $nice -p gpu --nodes=$((2*k)) --ntasks=$((2*k)) --ntasks-per-node=1 --gres=gpu:1 --cpus-per-task=6 --mem=50G --time=$TLIM \
        --export=ALL,ARCHIVE=$arch,OUT=$OUT,TAG=$tag,LEGS=$p:$shots,MAXERR=$ME,HASH=c$c:$q:$sha,PLF_WORKERS=5,PL_SCORE_SHARDS=$k deepverify_pl.sbatch)
  fi
  echo "$(date +%F_%T) topup61 $arm cell $c $sha q=$q p=$p shots=$shots $reason shards=$k sched=$ref job $J" | tee -a $OUT/LAUNCH.txt
done <<< "$LIST"
echo "legs: $n"
