#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

NTRY="${1:-5}"

declare -a DATASETS=(
  "bnci2014001"
  "bnci2015001"
  "lee2019"
  "stieger2021"
)

declare -a TASKS=(
  "session"
  "subject"
)

for ds in "${DATASETS[@]}"; do
  for task in "${TASKS[@]}"; do
    echo "Submitting dataset=${ds} task=${task} ntry=${NTRY}"
    sbatch experiments/scripts/run_da500_single_gpu.slurm.sh \
      --dataset "${ds}" \
      --task "${task}" \
      --ntry "${NTRY}"
  done
done

