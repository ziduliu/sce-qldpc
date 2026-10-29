#!/bin/bash
# Give each rank one of the node's allocated GPUs, round-robin by local rank,
# then exec the command. The list indexed is Slurm's CUDA_VISIBLE_DEVICES (the
# allocation), not the node's full set of physical devices.
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  IFS=',' read -ra _D <<< "$CUDA_VISIBLE_DEVICES"
else
  _D=($(seq 0 $(( $(nvidia-smi -L 2>/dev/null | wc -l) - 1 ))))
fi
_N=${#_D[@]}
[ "$_N" -gt 0 ] || _N=1

# Ranks that share a GPU need an explicit PLF_GROUP: pl_sample's auto_group
# sizes from the free memory at each rank's start, so ranks starting together
# on one card would all size for the whole card, and a fixed group keeps the
# GPU batching identical across rank layouts.
_PERGPU=$(( ${SLURM_NTASKS_PER_NODE:-1} / _N ))
if [ "${SLURM_NTASKS_PER_NODE:-1}" -gt "$_N" ] && [ "${PLF_GROUP:-0}" = "0" ]; then
  echo "rankgpu.sh: ${SLURM_NTASKS_PER_NODE} ranks share $_N GPU(s) on $(hostname)" \
       "(~$_PERGPU per GPU) with PLF_GROUP unset." >&2
  echo "rankgpu.sh: set PLF_GROUP explicitly (a whole-GPU run uses 16; divide by" \
       "the ranks per GPU) or give each rank its own GPU. Refusing to run." >&2
  exit 78
fi

export CUDA_VISIBLE_DEVICES=${_D[$(( ${SLURM_LOCALID:-0} % _N ))]}
echo "rank ${SLURM_PROCID:-?} local ${SLURM_LOCALID:-?} on $(hostname)" \
     "gpu $CUDA_VISIBLE_DEVICES group ${PLF_GROUP:-auto}" >&2
exec "$@"
