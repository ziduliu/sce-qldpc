#!/bin/bash
#SBATCH --job-name=gaptmpl
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/u/zdliu/deploy/rnd_v5_20261015/gap_tmpl/ids_%j.out
cd /u/zdliu/deploy/rnd_v5_20261015/gap_tmpl
/u/zdliu/deploy/gapenv/bin/gap -q -b ids.g
echo "exit $?"
