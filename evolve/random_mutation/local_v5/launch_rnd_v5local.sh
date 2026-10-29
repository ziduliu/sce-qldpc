#!/bin/bash
# Local-mutation control on the v5 random mutator: launch_rnd_v5.sh with
#   RND_LOCAL=1, the mutator and server of /u/zdliu/deploy/rnd_local_v5_20261018, its own run names and a smoke mode.
#   launch_rnd_v5local.sh fom|cler <REP> [NWORK] [ITERS]   (REP=smoke: ITERS 6, engine seed 42, run names *_repsmoke_out)
# Seeds, evaluator, cells, archive, 60 slots and parallel_evaluations 16 are as in the LLM runs;
# full GAP library, order matched to the gpt-6.1-sol block lengths, weight cap 12 (FOM) / 9-12 (CLER).
#   RND_V5=1          rnd_mutator.py v5 rules (see its header)
#   RND_L3_SHAPE_P    0.63 (FOM) / 0.53 (CLER): share of level-3 children of the gpt-6.1-sol run of the
#                     same objective that changed their shapes (91/145, 68/129)
#   engine seed       random_seed 41+REP
#   wait limit        400000 s for both objectives (engine OE_EVAL_TIMEOUT and config evaluator
#                     timeout), so a queued evaluation is not dropped
#   workers           default 13 (FOM) / 16 (CLER), submitted before the controller, which starts
#                     only after the first worker has started (--dependency=after); FOM workers
#                     wait up to 6 h idle so a late controller still finds them
#   ITERS             default 520; the analysis cuts every run at 500 children
#   CTRLP             partition of the controller job (default highfreq; overrides the sbatch scripts'
#                     standard partition)
#   fresh state       CKPT and ARCHIVE are unset, the archive directory must not exist, and ARCHIVE
#                     is passed explicitly
#   runtime check     the *_v5.sbatch scripts run protocol_check_v5.py right before Python starts:
#                     every value of the run's manifest ($O/protocol_controller.env,
#                     $O/protocol_worker.env) must hold, or the job exits; the controller also checks
#                     the configuration as the engine loads it (random seed, evaluator timeout,
#                     parallel evaluations, every model's api_base = this run's mutation server)
#   clean environment every variable the sbatch scripts and evaluators read is unset in this shell
#                     first; the values used are then passed explicitly and written to launch.log
set -euo pipefail
unset BASH_ENV
for v in ARCHIVE BINISL CARDSELF CELLPOOL CKPT CONFIG EDGECELLS EFFORT EVALUATOR FOCUSLV FULLTBL INSP ITERS \
         NMAX NMIN NSLOTS OETIMEOUT OUT QUOTA RATEBINS RATELO RATESTEP RCAP RCODE RND_SEED RNOTE RUNSTATE \
         RUNSTATE_INTERVAL SEEDS SHAPEK SIDECAR SPOOL TABLETOP TCELLS TOPG WMAX D2SDEEP DIST EVAL MAXERR PLP \
         PLR PLTIMEOUT SCHEDTIME SHOTS WKLEASE WKTIMEOUT WK_IDLE_EXIT; do unset "$v"; done
for v in $(env | cut -d= -f1 | grep -E '^(CM|EM|OE|LAYOUT|V2|PL|RND|OPENAI|PROTO|WK)_[A-Za-z0-9_]*$' || true); do unset "$v"; done
OBJ=${1:?fom|cler}; REP=${2:?rep}
case "$OBJ" in fom) NW=${3:-13} ;; cler) NW=${3:-16} ;; *) echo "objective must be fom or cler"; exit 2 ;; esac
ITERS=${4:-520}
if [ "$REP" = smoke ]; then ITERS=${4:-6}; fi
C0=/u/zdliu/lp/circmean
D=/u/zdliu/deploy/rnd_local_v5_20261018
LIB=$C0/smallgrp_lib.jsonl; SD=$C0/smallgrp_sd.tsv; ND=$C0/llm_n_distribution.json; TM=$C0/template_ids.tsv
# ---- guards (all before anything is created) ----
[ "$(wc -l < "$LIB")" -ge 8000 ] || { echo "GUARD: library"; exit 2; }
[ "$(wc -l < "$SD")" -eq "$(wc -l < "$LIB")" ] || { echo "GUARD: $SD does not cover the library"; exit 2; }
[ -f "$ND" ] || { echo "GUARD: $ND missing"; exit 2; }
[ "$(wc -l < "$TM")" -ge 3270 ] || { echo "GUARD: $TM missing or short"; exit 2; }
grep -q "def draw_child_v5" $D/rnd_mutator.py || { echo "GUARD: $D/rnd_mutator.py has no v5 rules"; exit 2; }
grep -q "def make_response_local" $D/rnd_mutator.py || { echo "GUARD: $D/rnd_mutator.py has no local moves"; exit 2; }
grep -q "RM.OPEN_LO" $D/rnd_llm_server.py || { echo "GUARD: $D/rnd_llm_server.py has no v5 band"; exit 2; }
grep -q "def parent_best" $D/rnd_llm_server.py || { echo "GUARD: $D/rnd_llm_server.py has no parent_best"; exit 2; }
for f in $C0/protocol_check_v5.py $D/oe_circmean_rnd_v5local.sbatch $C0/circmean_worker_v5.sbatch \
         $D/oe_circmean_pl_rnd_v5local.sbatch /u/zdliu/lp/plevolve_20260912/worker_pl_v5.sbatch; do
  [ -f "$f" ] || { echo "GUARD: $f missing"; exit 2; }
done
_FREE=$(/usr/lpp/mmfs/bin/mmlsquota -u zdliu 2>/dev/null | awk '$2=="root"{print $12-$10}' || true)
QMIN=${RND5_QUOTA_MIN:-250000}   # minimum free files on /u before launch
if [ -n "$_FREE" ] && [ "$_FREE" -lt "$QMIN" ]; then echo "GUARD: only $_FREE files free on /u (gate $QMIN)"; exit 2; fi
if [ "$OBJ" = fom ]; then
  export RND_WCAPS="12"; L3P=0.63; NKEY=fom
  O=/u/zdliu/lp/oe_run/cc_v7_rnd520locv5_w12_10c_rep${REP}_out
  S=/u/zdliu/lp/jobs/cc7_rnd520locv5_w12_10c_rep${REP}_spool
  A=/ptmp/zdliu/archives_live/cc_v7_rnd520locv5_w12_10c_rep${REP}_out/archive
  BASE=$C0/oe_circmean_cc_v7_rnd_w12_10cells.yaml; OLDT=40000
else
  export RND_WCAPS="9,10,11,12"; L3P=0.53; NKEY=cler
  O=/u/zdliu/lp/oe_run/pl10_rnd520locv5_rep${REP}_out
  S=/u/zdliu/lp/jobs/pl10_rnd520locv5_rep${REP}_spool
  A=/ptmp/zdliu/archives_live/pl10_rnd520locv5_rep${REP}_out/archive
  BASE=$C0/oe_circmean_pl_v3_rnd_10cells.yaml; OLDT=80000
fi
[ -e "$O" ] && { echo "$OBJ rep$REP: $O exists, refusing"; exit 2; }
[ -e "$S" ] && { echo "$OBJ rep$REP: spool $S exists, refusing"; exit 2; }
[ -e "$A" ] && { echo "$OBJ rep$REP: archive $A exists, refusing"; exit 2; }
grep -q "^random_seed:" "$BASE" && { echo "GUARD: $BASE already sets random_seed"; exit 2; }
[ "$(grep -c "^  timeout: $OLDT$" "$BASE")" -eq 1 ] || { echo "GUARD: $BASE evaluator timeout is not $OLDT"; exit 2; }
if [ "$REP" = smoke ]; then ESEED=42; else ESEED=$((41 + REP)); fi
CFGTXT=$({ echo "random_seed: $ESEED"; sed "s/^  timeout: $OLDT$/  timeout: 400000/" "$BASE"; })
echo "$CFGTXT" | grep -q "^  timeout: 400000$" || { echo "GUARD: timeout edit failed"; exit 2; }
# ---- create and submit ----
# claim the run atomically: plain mkdir fails if another launcher created the directory in between
mkdir "$O" || { echo "$OBJ rep$REP: could not claim $O"; exit 2; }
mkdir "$S" || { echo "$OBJ rep$REP: could not claim $S"; exit 2; }
mkdir -p "$(dirname "$A")"; mkdir "$A" || { echo "$OBJ rep$REP: could not claim $A"; exit 2; }
ln -sfn "$A" "$O/archive"
CFG=$O/config_base_seed${ESEED}.yaml
echo "$CFGTXT" > "$CFG"
# ---- manifests checked inside the jobs (protocol_check_v5.py) ----
SEEDS=seeds_rates_w14_plus_v5_nocyc_v3
COMMONM="CM_NMIN=100
CM_NMAX=600
CM_WMAX=12
CM_RATE_LO=0.10
CM_RATE_STEP=0.05
CM_RATE_BINS=10
EM_SPOOL=$S"
{ echo "$COMMONM"
  echo "CM_NSLOTS=60
CM_TABLE_TOP=0
CM_TABLE_CELLS=1
CM_INSP_BINS=4
CM_TOP_GLOBAL=1
CM_SHOW_FULL_TABLE=1
CM_ARCHIVE=$A
LAYOUT_BIN_ISLANDS=1
LAYOUT_RATE_BIN_N=10
LAYOUT_CELL_POOL=12
LAYOUT_SHAPE_PARENT_K=0
LAYOUT_RATE_EDGE_CELLS=1
LAYOUT_CELL_FOCUS_LEVELS=2
V2_CARD_SELF=0
V2_CARD_QUOTA=0
OE_EVAL_TIMEOUT=400000
RND_V5=1
RND_LIBRARY=smallgrp
RND_LIB_PATH=$LIB
RND_ORDER_RULE=llm_n
RND_N_DIST=$ND
RND_N_KEY=$NKEY
RND_SD_PATH=$SD
RND_TMPL_PATH=$TM
RND_L3_SHAPE_P=$L3P
RND_WCAPS=$RND_WCAPS
RND_LOCAL=1
PROTO_SEEDS=$SEEDS
PROTO_ITERS=$ITERS
PROTO_CONFIG=$O/config_rnd.yaml
CM_LEDGER=$O/eval_ledger.jsonl
EM_ARCH=$O/runstate.txt
CONFIGFIELD:random_seed=$ESEED
CONFIGFIELD:evaluator.timeout=400000
CONFIGFIELD:evaluator.parallel_evaluations=16"
  if [ "$OBJ" = fom ]; then echo "PROTO_EVALUATOR=oe_evaluator_circmean_v3.py"; else echo "PROTO_EVALUATOR=oe_evaluator_circmean_pl.py
CM_DIST_MODE=pl"; fi
} > "$O/protocol_controller.env"
{ echo "$COMMONM"
  echo "CM_D2S_BUDGET=2
CM_TOPK=5"
  if [ "$OBJ" = fom ]; then echo "EM_SCRIPT=$C0/eval_circmean_v3.py
CM_DCIRC_BUDGET=60
CM_DIST_MODE=cc
CM_D2S_DEEP=30
WK_EVAL_TIMEOUT=5400
WK_LEASE=7200"; else echo "EM_SCRIPT=$C0/eval_circmean_pl.py
CM_DIST_MODE=pl
PL_SCORE_P=0.002
PL_SCORE_ROUNDS=6
PL_SCORE_SHOTS=200000
PL_SCORE_MAX_ERRORS=100
PL_SCORE_SCHED_TIME=120
PL_SCORE_TIMEOUT=60000
WK_EVAL_TIMEOUT=62000
WK_LEASE=70000"; fi
} > "$O/protocol_worker.env"
PROTOC="PROTO_MANIFEST=$O/protocol_controller.env,PROTO_RECORD_DIR=$O"
PROTOW="PROTO_MANIFEST=$O/protocol_worker.env,PROTO_RECORD_DIR=$O"
export RND_LOCAL=1 RND_V5=1 RND_SD_PATH=$SD RND_TMPL_PATH=$TM RND_L3_SHAPE_P=$L3P RND_LIBRARY=smallgrp RND_LIB_PATH=$LIB RND_ORDER_RULE=llm_n RND_N_DIST=$ND RND_N_KEY=$NKEY
COMMON="EFFORT=high,NMIN=100,NMAX=600,WMAX=12,NSLOTS=60,TABLETOP=0,TCELLS=1,RATELO=0.10,RATESTEP=0.05,RATEBINS=10,BINISL=1,INSP=4,TOPG=1,FULLTBL=1,CARDSELF=0,QUOTA=0,CELLPOOL=12,SHAPEK=0,EDGECELLS=1,SEEDS=$SEEDS"
RNDV="RND_LOCAL=1,RND_V5=1,RND_SD_PATH=$SD,RND_TMPL_PATH=$TM,RND_L3_SHAPE_P=$L3P,RND_LIBRARY=smallgrp,RND_LIB_PATH=$LIB,RND_ORDER_RULE=llm_n,RND_N_DIST=$ND,RND_N_KEY=$NKEY"
W=""
if [ "$OBJ" = fom ]; then
  cd $C0
  for i in $(seq 1 "$NW"); do
    W="$W $(sbatch --parsable --time=48:00:00 --export=ALL,NMIN=100,NMAX=600,WMAX=12,RATELO=0.10,RATESTEP=0.05,RATEBINS=10,WK_IDLE_EXIT=21600,SPOOL=$S,EVAL=$C0/eval_circmean_v3.py,DIST=cc,D2SDEEP=30,$PROTOW circmean_worker_v5.sbatch)"
  done
  FIRST=$(echo $W | awk '{print $1}')
  C=$(sbatch --parsable --partition=${CTRLP:-highfreq} --dependency=after:$FIRST --time=48:00:00 --export=ALL,ITERS=$ITERS,$COMMON,$RNDV,OUT=$O,SPOOL=$S,ARCHIVE=$A,CONFIG=$CFG,OETIMEOUT=400000,EVALUATOR=oe_evaluator_circmean_v3.py,$PROTOC $D/oe_circmean_rnd_v5local.sbatch)
else
  cd /u/zdliu/lp/plevolve_20260912
  for i in $(seq 1 "$NW"); do
    W="$W $(sbatch --parsable --time=120:00:00 --export=ALL,NMIN=100,NMAX=600,WMAX=12,RATELO=0.10,RATESTEP=0.05,RATEBINS=10,WK_IDLE_EXIT=36000,SPOOL=$S,SHOTS=200000,MAXERR=100,PLP=0.002,PLR=6,SCHEDTIME=120,WKTIMEOUT=62000,PLTIMEOUT=60000,WKLEASE=70000,$PROTOW worker_pl_v5.sbatch)"
  done
  FIRST=$(echo $W | awk '{print $1}')
  C=$(sbatch --parsable --partition=${CTRLP:-highfreq} --dependency=after:$FIRST --time=120:00:00 --export=ALL,ITERS=$ITERS,$COMMON,$RNDV,OUT=$O,SPOOL=$S,ARCHIVE=$A,CONFIG=$CFG,OETIMEOUT=400000,EVALUATOR=oe_evaluator_circmean_pl.py,$PROTOC $D/oe_circmean_pl_rnd_v5local.sbatch)
fi
{ echo "exported RND_/CM_/EM_/OE_/LAYOUT_/V2_/PL_/OPENAI_ variables at submission:"; env | grep -E '^(CM|EM|OE|LAYOUT|V2|PL|RND|OPENAI)_' | sort; echo "controller partition: ${CTRLP:-highfreq}"; echo "sha1 of the files in $D:"; sha1sum $D/rnd_mutator.py $D/rnd_llm_server.py $D/oe_circmean_rnd_v5local.sbatch $D/oe_circmean_pl_rnd_v5local.sbatch; } >> "$O/launch.log"
echo "$(date +%F_%T) v5 local-mutation $OBJ rep$REP (engine seed $ESEED, $ITERS iterations, L3 shape p $L3P): controller $C workers$W -> $O" | tee -a "$O/launch.log"
