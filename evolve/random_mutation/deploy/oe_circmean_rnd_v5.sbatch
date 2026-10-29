#!/bin/bash -l
# Controller job of the random-mutation arm: evolve/oe_circmean.sbatch with rnd_llm_server.py
# started in-job as the model endpoint (the config's __PORT__ filled in), no API key, and
# reasoning summary and reasoning-in-prompt off. Submitted by launch_rnd_v5.sh, which passes
# the environment on the command line (the cluster sets SBATCH_EXPORT=NONE, which overrides the
# in-script --export=ALL).
#SBATCH -J oe_cmrnd
#SBATCH -o /u/zdliu/lp/oe_run/oe_cmrnd_%j.out
#SBATCH -e /u/zdliu/lp/oe_run/oe_cmrnd_%j.err
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --export=ALL
set -uo pipefail
module load python-waterboa/2024.06 || true
export PATH=/u/zdliu/lpenv/bin:$PATH
export OPENAI_API_KEY=${OPENAI_API_KEY:-random-mutator}   # rnd arm: the local endpoint ignores it
export OPENAI_REASONING_EFFORT=${EFFORT:-high}
export OPENAI_REASONING_SUMMARY=0   # rnd arm: no responses API
export REASONING_IN_PROMPT=0        # rnd arm
OUT=${OUT:-/u/zdliu/lp/oe_run/cmean_out}
export OPENAI_REASONING_LOG=$OUT/reasoning.jsonl
# card level: V2_CARD_SELF=0 turns the engine's NEXT_MOVE level self-selection off
# (CARDSELF=1 turns it on)
export V2_CARD_SELF=${CARDSELF:-0}
export V2_CARD_QUOTA=${QUOTA:-0}
# reasoning-text limits (engine env-gated)
export OE_REASONING_CAP=${RCAP:-10000}
export OE_REASONING_NOTE_CHARS=${RNOTE:-10000}
export OE_REASONING_CODE_CHARS=${RCODE:-8000}
# genotype visibility in the slot table
export CM_TABLE_CELLS=${TCELLS:-0}
export EM_SPOOL=${SPOOL:-/u/zdliu/lp/jobs/circmean_spool}
# size window (scaffold, adapter, sidecar); the workers serving SPOOL must use the same
export CM_NMIN=${NMIN:-100} CM_NMAX=${NMAX:-600}
export CM_WMAX=${WMAX:-12}
# MAP-Elites rate grid. Read by the evaluator (which bins the child) and by
# database.rate_grid() (which names the band in the round's prompt), so the
# controller needs the same values as the worker. Defaults: 15 cells of width 0.025.
export CM_RATE_LO=${RATELO:-0.15} CM_RATE_STEP=${RATESTEP:-0.025} CM_RATE_BINS=${RATEBINS:-15}
# Per-cell rounds: each round targets one cell, drawn so that every cell
# gets the same share, and the prompt names it. CM_RATE_BINS must equal the
# config's feature_bins.rate_bin; rate_grid() logs an error if it does not.
export LAYOUT_BIN_ISLANDS=${BINISL:-0} LAYOUT_RATE_BIN_N=${RATEBINS:-15}
export LAYOUT_CELL_POOL=${CELLPOOL:-0} LAYOUT_SHAPE_PARENT_K=${SHAPEK:-0}
# EDGECELLS=1: develop the two boundary rate cells too, so ten cells are live
# instead of eight.
export LAYOUT_RATE_EDGE_CELLS=${EDGECELLS:-0}
# The engine and the prompt must agree about how many cells are live; otherwise
# rounds ask for cells the prompt never names. Checked here because this is the
# one place that sees both the switch and the config.
_CFGF=${CONFIG:-oe_circmean.yaml}
case "$_CFGF" in /*) ;; *) _CFGF=/u/zdliu/lp/circmean/$_CFGF ;; esac
if [ "$LAYOUT_RATE_EDGE_CELLS" = "1" ]; then
  grep -q "10 cells centred" "$_CFGF" || {
    echo "FATAL: EDGECELLS=1 but $_CFGF does not state the ten-cell grid"; exit 1; }
elif grep -q "10 cells centred" "$_CFGF"; then
  echo "FATAL: $_CFGF states the ten-cell grid but EDGECELLS is not 1"; exit 1
fi
export CM_NSLOTS=${NSLOTS:-20} CM_TABLE_TOP=${TABLETOP:-30}
# which card levels name the round's target cell in the prompt (engine default 2 = the
# shape card only). FOCUSLV=1+2+3 names it on every card; "+" because a comma inside
# an --export value splits the export list.
export LAYOUT_CELL_FOCUS_LEVELS=$(echo "${FOCUSLV:-2}" | tr "+" ",")
export OE_EVAL_TIMEOUT=${OETIMEOUT:-40000}   # long enough that queued requests do not time out as fitness-0 children
# run-wide code archive: slots reproducing any earlier program's code
# score nothing
export CM_ARCHIVE=${ARCHIVE:-$OUT/archive}
# prompt composition:
#   CM_INSP_BINS   inspirations = elites of that many OTHER rate bins,
#                  drawn uniformly from the whole population (all islands)
#   CM_TOP_GLOBAL  top programs = the parent's own rate band, all islands
#   CM_SHOW_FULL_TABLE  every shown program carries its full 20-slot table
export CM_INSP_BINS=${INSP:-4} CM_TOP_GLOBAL=${TOPG:-1} CM_SHOW_FULL_TABLE=${FULLTBL:-1}
mkdir -p "$OUT"
# live RUN STATE in every prompt (bed map / move history / frontier):
# the adapter appends one ledger line per evaluation, the sidecar
# renders $OUT/runstate.txt every 5 min, the prompt sampler injects it
# under the current program's artifacts ("archive" key). RUNSTATE=0 turns
# all of it off; SIDECAR=0 when the sidecar already runs as its own job.
if [ "${RUNSTATE:-1}" = "1" ]; then
  export CM_LEDGER=$OUT/eval_ledger.jsonl
  export EM_ARCH=$OUT/runstate.txt
  if [ "${SIDECAR:-1}" = "1" ]; then
    /u/zdliu/lpenv/bin/python -u /u/zdliu/lp/circmean/runstate.py --out "$OUT" \
      --interval ${RUNSTATE_INTERVAL:-300} > "$OUT/runstate.log" 2>&1 &
    RUNSTATE_PID=$!
    trap 'kill $RUNSTATE_PID 2>/dev/null' EXIT
  fi
fi
cd /u/zdliu/lp/circmean
# ---- random-mutation arm: a local endpoint stands in for the LLM ----------
# rnd_llm_server.py answers every completion with a random child of the
# prompt's current program (rnd_mutator.py); the engine is unchanged.
# The server binds a free port chosen by the OS and writes it to a file; the run
# accepts it only when the answer on that port carries this run's token and the server's pid
RND_TOKEN="${SLURM_JOB_ID}-$(date +%s%N)-$$"
export RND_SEED=${RND_SEED:-$SLURM_JOB_ID}   # distinct mutator stream per run
rm -f "$OUT/rnd_port"
/u/zdliu/lpenv/bin/python -u /u/zdliu/lp/circmean/rnd_llm_server.py --port 0 --port-file "$OUT/rnd_port" --token "$RND_TOKEN" \
  --log "$OUT/rndmut.jsonl" ${RND_SEED:+--seed $RND_SEED} > "$OUT/rnd_server.log" 2>&1 &
RND_PID=$!
trap 'kill $RND_PID ${RUNSTATE_PID:-} 2>/dev/null' EXIT
for _i in $(seq 1 1500); do
  [ -s "$OUT/rnd_port" ] && break
  kill -0 $RND_PID 2>/dev/null || { echo "FATAL: rnd_llm_server exited before binding"; cat "$OUT/rnd_server.log"; exit 1; }
  sleep 0.2
done
[ -s "$OUT/rnd_port" ] || { echo "FATAL: rnd_llm_server wrote no port"; exit 1; }
RND_PORT=$(cat "$OUT/rnd_port")
/u/zdliu/lpenv/bin/python -c "import json, sys, urllib.request; d = json.load(urllib.request.urlopen('http://127.0.0.1:$RND_PORT/', timeout=10)); sys.exit(0 if d.get('token') == '$RND_TOKEN' and d.get('pid') == $RND_PID else 1)" \
  || { echo "FATAL: the server answering on $RND_PORT is not this run's server"; exit 1; }
kill -0 $RND_PID 2>/dev/null || { echo "FATAL: rnd_llm_server is not running"; exit 1; }
export PROTO_RND_PORT=$RND_PORT
sed "s#__PORT__#$RND_PORT#" "${CONFIG:?CONFIG}" > "$OUT/config_rnd.yaml"
grep -q "127.0.0.1:$RND_PORT/v1" "$OUT/config_rnd.yaml" || { echo "FATAL: $CONFIG has no __PORT__ placeholder"; exit 1; }
CONFIG=$OUT/config_rnd.yaml
echo "rnd arm: endpoint 127.0.0.1:$RND_PORT, config $CONFIG"
# protocol check after all startup files and module loads (protocol_check_v5.py); a v5 run
# never resumes from a checkpoint
[ -z "${CKPT:-}" ] || { echo "FATAL: CKPT is set ($CKPT) in a fresh v5 run"; exit 1; }
export PROTO_CONFIG="$CONFIG" PROTO_SEEDS="${SEEDS:-}" PROTO_EVALUATOR="${EVALUATOR:-oe_evaluator_circmean.py}" PROTO_ITERS="${ITERS:-60}"
/u/zdliu/lpenv/bin/python /u/zdliu/lp/circmean/protocol_check_v5.py "${PROTO_MANIFEST:?}" "${PROTO_RECORD_DIR:?}" controller || { echo "FATAL: protocol check failed"; exit 1; }
/u/zdliu/lpenv/bin/python -u run_circmean.py --config ${CONFIG:-oe_circmean.yaml} \
  --seeds ${SEEDS:-seeds_rates} --evaluator ${EVALUATOR:-oe_evaluator_circmean.py} \
  --iterations ${ITERS:-60} --output "$OUT" ${CKPT:+--checkpoint "$CKPT"}
