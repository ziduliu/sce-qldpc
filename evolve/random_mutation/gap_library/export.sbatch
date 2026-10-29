#!/bin/bash
#SBATCH -J sgexport
#SBATCH -p standard
#SBATCH -N 1 -n 1 -c 2
#SBATCH --mem=8G
#SBATCH -t 06:00:00
#SBATCH -o /u/zdliu/lp/jobs/smallgrp_lib_20261011/export_%j.out
/u/zdliu/deploy/gapenv/bin/gap -q -b /u/zdliu/lp/jobs/smallgrp_lib_20261011/export.g
