#!/bin/bash -l
# Controller job: runs the evolution loop (run_circmean.py) for one run.
# Submit with the environment on the command line, e.g.
#   sbatch --export=ALL,OUT=/u/zdliu/lp/oe_run/cmean_out,ITERS=60 oe_circmean.sbatch
# (the cluster sets SBATCH_EXPORT=NONE, which overrides the in-script
# --export directive, so inline "VAR=x sbatch ..." settings are dropped).
#SBATCH -J oe_cmean
#SBATCH -o /u/zdliu/lp/oe_run/oe_cmean_%j.out
#SBATCH -e /u/zdliu/lp/oe_run/oe_cmean_%j.err
#SBATCH --partition=standard
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=24:00:00
#SBATCH --export=ALL
set -uo pipefail
module load python-waterboa/2024.06 || true
export PATH=/u/zdliu/lpenv/bin:$PATH
if [ -z "${OPENAI_API_KEY:-}" ]; then echo "FATAL: no OPENAI_API_KEY"; exit 1; fi
export OPENAI_REASONING_EFFORT=${EFFORT:-high}
export OPENAI_REASONING_SUMMARY=1
export REASONING_IN_PROMPT=1
OUT=${OUT:-/u/zdliu/lp/oe_run/cmean_out}
export OPENAI_REASONING_LOG=$OUT/reasoning.jsonl
# level choice: menu card, the model declares LEVEL: 1/2/3 at
# generation time from parent+inspirations (V2_CARD_SELF=0 turns the
# engine's NEXT_MOVE mechanism off; pass CARDSELF=1 to restore it)
export V2_CARD_SELF=${CARDSELF:-0}
export V2_CARD_QUOTA=${QUOTA:-0}
# limits on the reasoning text carried into prompts (engine env-gated)
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
# controller and the workers must use the same values.
export CM_RATE_LO=${RATELO:-0.15} CM_RATE_STEP=${RATESTEP:-0.025} CM_RATE_BINS=${RATEBINS:-15}
# Per-cell rounds: each round targets one cell, drawn so that every cell
# gets the same share, and the prompt names it. CM_RATE_BINS must equal the
# config's feature_bins.rate_bin; rate_grid() logs an error if it does not.
export LAYOUT_BIN_ISLANDS=${BINISL:-0} LAYOUT_RATE_BIN_N=${RATEBINS:-15}
export LAYOUT_CELL_POOL=${CELLPOOL:-0} LAYOUT_SHAPE_PARENT_K=${SHAPEK:-0}
# EDGECELLS=1: develop the two boundary rate cells too, so ten cells are live
# instead of eight.
export LAYOUT_RATE_EDGE_CELLS=${EDGECELLS:-0}
# The engine and the prompt must agree about how many cells are live;
# otherwise rounds ask for cells the prompt never names.
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
export OE_EVAL_TIMEOUT=7200
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
# live RUN STATE in every prompt: the evaluator appends one ledger line per
# evaluation, the runstate.py sidecar renders $OUT/runstate.txt every 5 min,
# and the prompt sampler injects it under the program's artifacts ("archive"
# key). RUNSTATE=0 turns all of it off; SIDECAR=0 when the sidecar runs as
# its own job.
if [ "${RUNSTATE:-1}" = "1" ]; then
  export CM_LEDGER=$OUT/eval_ledger.jsonl
  export EM_ARCH=$OUT/runstate.txt
  if [ "${SIDECAR:-1}" = "1" ]; then
    /u/zdliu/lpenv/bin/python -u "${RUNSTATE_PY:-/u/zdliu/lp/circmean/runstate.py}" --out "$OUT" \
      --interval ${RUNSTATE_INTERVAL:-300} > "$OUT/runstate.log" 2>&1 &
    RUNSTATE_PID=$!
    trap 'kill $RUNSTATE_PID 2>/dev/null' EXIT
  fi
fi
cd /u/zdliu/lp/circmean
# CKPT=<checkpoint dir> resumes (ITERS then = ADDITIONAL iterations);
# pair it with ARCHIVE=<the original run's archive> so the code archive
# carries over
/u/zdliu/lpenv/bin/python -u run_circmean.py --config ${CONFIG:-oe_circmean.yaml} \
  --seeds ${SEEDS:-seeds_rates} --evaluator ${EVALUATOR:-oe_evaluator_circmean.py} \
  --iterations ${ITERS:-60} --output "$OUT" ${CKPT:+--checkpoint "$CKPT"}
