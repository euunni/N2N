#!/bin/bash
#SBATCH -A m4138
#SBATCH -C gpu
#SBATCH --qos=regular
#SBATCH --time=14:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --job-name=Classical_TCN
#SBATCH --output=%x-%j.out

# Notes:
# - This script launches a single Python process that uses nn.DataParallel to
#   utilize all GPUs on the node. For multi-node scaling with DDP, use torchrun instead.
# - Adjust account (-A), qos, time, and resources as needed.

set -euo pipefail
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

srun bash -lc '
  set -euo pipefail
  source ~/.bashrc && conda activate N2N

  export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
  export TORCH_DISTRIBUTED_DEBUG=${TORCH_DISTRIBUTED_DEBUG:-OFF}

  OUT_DIR=${OUT_DIR:-/global/homes/h/haeun/QML/denoising/N2N/TCN/n2n/output/batch}
  mkdir -p "$OUT_DIR"

  MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n1)
  MASTER_PORT=${MASTER_PORT:-29500}

  torchrun \
      --nnodes ${SLURM_JOB_NUM_NODES:-${SLURM_NNODES}} \
      --nproc_per_node ${SLURM_GPUS_ON_NODE:-4} \
      --rdzv_id ${SLURM_JOB_ID} \
      --rdzv_backend c10d \
      --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
      /global/homes/h/haeun/QML/denoising/N2N/TCN/n2n/scripts/train_waveforms.py \
      --runlist /global/homes/h/haeun/QML/denoising/N2N/TCN/n2n/scripts/runlist.txt \
      --output_dir "$OUT_DIR" \
      --batch_size 1000 \
      --events_per_file 500
'
