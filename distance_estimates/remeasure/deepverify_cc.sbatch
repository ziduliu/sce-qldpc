#!/bin/bash -l
#SBATCH -J deepvcc
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH -o /u/zdliu/lp/jobs/%x_%j.out
#SBATCH --export=ALL
# Deep distance verification of selected archived codes: 600 s x 2 per code.
#   sbatch --export=ALL,CODES=codes.json,OUT=results.jsonl,PART=0/4 deepverify_cc.sbatch
module load gcc/11 2>/dev/null || true
module load cuda/12.2 2>/dev/null || true
module load python-waterboa/2024.06
export PATH=/u/zdliu/lpenv/bin:$PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export LAYOUT_SQFAST_PATH=/u/zdliu/lp/dopt/agentA
cd /u/zdliu/lp/plevolve_20260912
python -u deepverify_cc.py --codes "${CODES:?}" --out "${OUT:?}" --budget ${BUDGET:-600} --reps ${REPS:-2} --part ${PART:-0/1}
