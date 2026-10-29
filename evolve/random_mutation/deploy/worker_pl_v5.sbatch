#!/bin/bash -l
#SBATCH -J cmworkpl
#SBATCH -o /u/zdliu/lp/jobs/cmworkpl_%j.out
#SBATCH -e /u/zdliu/lp/jobs/cmworkpl_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --export=ALL
# GPU worker for the circuit-level pL line: the worker loop of circmean_worker_v5.sbatch
# running eval_circmean_pl.py with the pL protocol below.
# Pass the environment on the command line (sbatch --export=ALL,VAR=x,...): the cluster
# sets SBATCH_EXPORT=NONE, which overrides the in-script --export=ALL.
set -euo pipefail
module load gcc/11 2>/dev/null || true
module load cuda/12.2 2>/dev/null || true
module load python-waterboa/2024.06
export TRITON_CACHE_DIR=/ptmp/zdliu/triton_cache   # compiled-kernel cache outside the /u file quota
export PATH=/u/zdliu/lpenv/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EM_SPOOL=${SPOOL:-/u/zdliu/lp/jobs/pl_spool}
export EM_SCRIPT=/u/zdliu/lp/circmean/eval_circmean_pl.py
# score-contract budgets: identical for every program in a run
export CM_D2S_BUDGET=2 CM_TOPK=5 CM_DIST_MODE=pl
export CM_NMIN=${NMIN:-100} CM_NMAX=${NMAX:-600} CM_WMAX=${WMAX:-14}
# MAP-Elites rate grid. Read by eval_circmean_pl.py (which bins the child)
# and by database.rate_grid() (which names the band in the round's prompt),
# so the controller needs the same values as the worker. Defaults: 15 cells of width 0.025.
export CM_RATE_LO=${RATELO:-0.15} CM_RATE_STEP=${RATESTEP:-0.025} CM_RATE_BINS=${RATEBINS:-15}
# pL protocol. SHOTS/MAXERR are the stopping rule; they fix every score in the run,
# so they are set here and nowhere else.
export PL_SCORE_P=${PLP:-0.002} PL_SCORE_ROUNDS=${PLR:-6}
# MAXERR counts failed shots (shots with any logical qubit wrong).
export PL_SCORE_SHOTS=${SHOTS:-20000} PL_SCORE_MAX_ERRORS=${MAXERR:-100}
export PL_SCORE_SCHED_TIME=${SCHEDTIME:-120}
# must stay below WK_EVAL_TIMEOUT, or a hung sampler is killed from outside
# with no message instead of being reported by pl_score
export PL_SCORE_TIMEOUT=${PLTIMEOUT:-6000}
export WK_EVAL_TIMEOUT=${WKTIMEOUT:-9000} WK_LEASE=${WKLEASE:-12000}
# protocol check after all startup files and module loads (protocol_check_v5.py)
/u/zdliu/lpenv/bin/python /u/zdliu/lp/circmean/protocol_check_v5.py "${PROTO_MANIFEST:?}" "${PROTO_RECORD_DIR:?}" worker || { echo "FATAL: protocol check failed"; exit 1; }
/u/zdliu/lpenv/bin/python -u /u/zdliu/lp/circmean/circmean_worker.py
