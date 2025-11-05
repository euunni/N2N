#!/bin/bash
#SBATCH -A m4138
#SBATCH -C gpu               
#SBATCH --qos=regular
#SBATCH --time=20:00:00    
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4     
#SBATCH --cpus-per-task=32
#SBATCH --job-name=N2N_QML
#SBATCH --output=/global/homes/h/haeun/QML/denoising/N2N/TCN_QML/N2N/batch/log/%x-%j.out
#SBATCH --mail-user=haeun@cern.ch

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export PYTHONUNBUFFERED=1

# Allow external overrides for output/checkpoint/total epochs
OUTPUT_DIR=${OUTPUT_DIR:-/global/homes/h/haeun/QML/denoising/N2N/TCN_QML/N2N/output/exp1}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-${OUTPUT_DIR}/checkpoints}
TOTAL_EPOCHS=${TOTAL_EPOCHS:-100}

srun bash -lc '
  source ~/.bashrc && conda activate N2N_QML
  
  mkdir -p "$OUTPUT_DIR" "$CHECKPOINT_DIR"

  MASTER_ADDR=$(scontrol show hostnames $SLURM_NODELIST | head -n1)
  MASTER_PORT=${MASTER_PORT:-29500}

  torchrun \
      --nnodes ${SLURM_NNODES} \
      --nproc_per_node ${SLURM_GPUS_ON_NODE:-4} \
      --rdzv_id ${SLURM_JOB_ID} \
      --rdzv_backend c10d \
      --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
      /global/homes/h/haeun/QML/denoising/N2N/TCN_QML/N2N/scripts/train_waveforms.py \
      --runlist /global/homes/h/haeun/QML/denoising/N2N/TCN_QML/N2N/scripts/runlist.txt \
      --output_dir "$OUTPUT_DIR" \
      --checkpoint_dir "$CHECKPOINT_DIR" \
      --resume \
      --total-epochs "$TOTAL_EPOCHS" \
      --epochs 1 \
      --batch_size 200 \
      --events_per_file 200
