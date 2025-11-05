#!/bin/bash
set -euo pipefail

# Usage: ./submit_chain.sh [exp_name] [total_epochs]
# Submits a chain of Slurm jobs, each running 1 epoch and resuming from the latest checkpoint.

SCRIPT_PATH="/global/homes/h/haeun/QML/denoising/N2N/TCN_QML/N2N/batch/n2n_script.sl"

EXP_NAME="${1:-exp1}"
TOTAL_EPOCHS="${2:-100}"

BASE_OUTPUT_DIR="/global/homes/h/haeun/QML/denoising/N2N/TCN_QML/N2N/output/${EXP_NAME}"
CHECKPOINT_DIR="${BASE_OUTPUT_DIR}/checkpoints"

mkdir -p "${BASE_OUTPUT_DIR}" \
         "/global/homes/h/haeun/QML/denoising/N2N/TCN_QML/N2N/batch/log" \
         "${CHECKPOINT_DIR}"

echo "Submitting ${TOTAL_EPOCHS} jobs chained (1 epoch/job) to OUTPUT_DIR=${BASE_OUTPUT_DIR}" >&2

jid=""
for i in $(seq 1 "${TOTAL_EPOCHS}"); do
  if [[ -z "${jid}" ]]; then
    jid=$(sbatch --parsable \
      --export=ALL,OUTPUT_DIR="${BASE_OUTPUT_DIR}",CHECKPOINT_DIR="${CHECKPOINT_DIR}",TOTAL_EPOCHS="${TOTAL_EPOCHS}" \
      "${SCRIPT_PATH}")
  else
    jid=$(sbatch --parsable --dependency=afterok:${jid} \
      --export=ALL,OUTPUT_DIR="${BASE_OUTPUT_DIR}",CHECKPOINT_DIR="${CHECKPOINT_DIR}",TOTAL_EPOCHS="${TOTAL_EPOCHS}" \
      "${SCRIPT_PATH}")
  fi
  echo "Submitted job for epoch ${i}: ${jid}"
done


