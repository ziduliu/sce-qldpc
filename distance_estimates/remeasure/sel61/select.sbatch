#!/bin/bash
#SBATCH -J sel61
#SBATCH -p highfreq
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH -o /u/zdliu/lp/jobs/sel61_%j.out
#SBATCH --export=ALL
cd /u/zdliu/lp/circmean
/u/zdliu/lpenv/bin/python -u select_cc_champions.py --run /u/zdliu/lp/jobs/sel61 --arm gpt61sol --out /u/zdliu/lp/jobs/codes_gpt61sol_20261004.json
