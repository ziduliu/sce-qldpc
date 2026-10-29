#!/bin/bash -l
#SBATCH -J certv5test
#SBATCH -p standard
#SBATCH -n 1 -c 8
#SBATCH --mem=8G
#SBATCH -t 01:00:00
#SBATCH -o /u/zdliu/lp/cert_v5/build_test_%j.out
# Builds the certifier (portable flags) and runs the regression tests on a compute node.
set -uo pipefail
module load gcc/11 python-waterboa/2024.06 || true
export PATH=/u/zdliu/lpenv/bin:$PATH
cd /u/zdliu/lp/cert_v5/src
g++ -O3 -march=x86-64-v3 -fopenmp -Wall -o dkp_bb dkp_bb.cpp && sha256sum dkp_bb dkp_bb.cpp exact_bb.py code_orbits.py
cd ../tests && DKP_BIN=../src/dkp_bb python test_regression.py
echo "tests exit $?"
