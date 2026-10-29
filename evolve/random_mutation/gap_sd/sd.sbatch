#!/bin/bash
#SBATCH --job-name=gapsd
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --array=0-15
#SBATCH --output=/u/zdliu/deploy/rnd_v5_20261015/gap_sd/sd_%a.out
cd /u/zdliu/deploy/rnd_v5_20261015/gap_sd
echo "TI := ${SLURM_ARRAY_TASK_ID};; NT := 16;;" > ti_${SLURM_ARRAY_TASK_ID}.g
/u/zdliu/deploy/gapenv/bin/gap -q -b ti_${SLURM_ARRAY_TASK_ID}.g sd.g
echo "exit $?"
