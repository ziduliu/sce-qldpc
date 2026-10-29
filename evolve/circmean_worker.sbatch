#!/bin/bash -l
#SBATCH -J cmwork
#SBATCH -o /u/zdliu/lp/jobs/cmwork_%j.out
#SBATCH -e /u/zdliu/lp/jobs/cmwork_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --export=ALL
set -euo pipefail
module load gcc/11 2>/dev/null || true
module load cuda/12.2 2>/dev/null || true
module load python-waterboa/2024.06
export TRITON_CACHE_DIR=/ptmp/zdliu/triton_cache   # compiled-kernel cache outside the /u file quota
export PATH=/u/zdliu/lpenv/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export EM_SPOOL=${SPOOL:-/u/zdliu/lp/jobs/circmean_spool}
export EM_SCRIPT=${EVAL:-/u/zdliu/lp/circmean/eval_circmean.py}
# score-contract budgets: identical for every program in a run
export CM_D2S_BUDGET=2 CM_DCIRC_BUDGET=60 CM_TOPK=5
export CM_DIST_MODE=${DIST:-circ} CM_D2S_DEEP=${D2SDEEP:-30}
export CM_NMIN=${NMIN:-100} CM_NMAX=${NMAX:-600} CM_WMAX=${WMAX:-12}
# MAP-Elites rate grid. Read by the evaluator (which bins the child) and by
# database.rate_grid() (which names the band in the round's prompt), so the
# controller and the workers must use the same values.
export CM_RATE_LO=${RATELO:-0.15} CM_RATE_STEP=${RATESTEP:-0.025} CM_RATE_BINS=${RATEBINS:-15}
export WK_EVAL_TIMEOUT=5400 WK_LEASE=7200
/u/zdliu/lpenv/bin/python -u /u/zdliu/lp/circmean/circmean_worker.py
