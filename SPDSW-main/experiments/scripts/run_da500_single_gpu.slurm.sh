#!/bin/bash
#SBATCH --job-name=spdsw_da500_1gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --partition=boost_usr_prod
#SBATCH --account=EUHPC_D33_186
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

DATASET="bnci2014001"
TASK="session"
NTRY=5
DEVICE="cuda:0"
SUBJECTS="auto"
CHECKPOINT_EVERY=1
RESUME_FLAG="--resume"
EXTRA_ARGS=()

print_usage() {
    echo "Usage: sbatch $0 [options]"
    echo ""
    echo "Options:"
    echo "  --dataset NAME              bnci2014001 | bnci2015001 | lee2019 | stieger2021"
    echo "  --task NAME                 session | subject"
    echo "  --ntry N"
    echo "  --device DEV                cuda:0 | cpu | auto"
    echo "  --subjects LIST             auto | all | 1,2,3"
    echo "  --checkpoint-every N"
    echo "  --resume | --no-resume"
    echo "  --extra \"ARGS\"             extra args passed to da_transfs_500.py"
    echo "  -h | --help"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --ntry) NTRY="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --subjects) SUBJECTS="$2"; shift 2 ;;
        --checkpoint-every) CHECKPOINT_EVERY="$2"; shift 2 ;;
        --resume) RESUME_FLAG="--resume"; shift ;;
        --no-resume) RESUME_FLAG="--no-resume"; shift ;;
        --extra)
            read -r -a EXTRA_ARGS <<< "$2"
            shift 2
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1"
            print_usage
            exit 1
            ;;
    esac
done

module purge
module load profile/deeplrn
module load python/3.11.6--gcc--8.5.0
module load cuda/12.1

VENV_CANDIDATES=(
  "$WORK/venvs/corsw/bin/activate"
  "/leonardo_work/EUHPC_D33_186/venvs/corsw/bin/activate"
  "$WORK/icmlrick/bin/activate"
  "/leonardo_work/EUHPC_D33_186/icmlrick/bin/activate"
)
FOUND_VENV=""
for v in "${VENV_CANDIDATES[@]}"; do
  if [[ -f "$v" ]]; then
    FOUND_VENV="$v"
    break
  fi
done
if [[ -z "$FOUND_VENV" ]]; then
  echo "[FATAL] venv activate not found." >&2
  exit 1
fi
source "$FOUND_VENV"

PROJECT_CANDIDATES=(
  "${PROJECT_DIR:-}"
  "${SLURM_SUBMIT_DIR:-}"
  "$WORK/code/corsw/SPDSW-main"
  "/leonardo_work/EUHPC_D33_186/code/corsw/SPDSW-main"
)
FOUND_PROJECT=""
for p in "${PROJECT_CANDIDATES[@]}"; do
  if [[ -n "$p" && -f "$p/experiments/scripts/da_transfs_500.py" ]]; then
    FOUND_PROJECT="$p"
    break
  fi
done
if [[ -z "$FOUND_PROJECT" ]]; then
  echo "[FATAL] project dir not found. Tried:" >&2
  for p in "${PROJECT_CANDIDATES[@]}"; do
    [[ -n "$p" ]] && echo "  - $p" >&2
  done
  exit 1
fi
PROJECT_DIR="$FOUND_PROJECT"
cd "${PROJECT_DIR}"

FAST="/leonardo_scratch/fast/EUHPC_D33_186"
export MNE_DATA="${FAST}/eeg_datasets/moabb_mne_data"
TSMNET_CANDIDATES=(
  "${TSMNET_ROOT:-}"
  "$WORK/code/corsw/TSMNet"
  "/leonardo_work/EUHPC_D33_186/code/corsw/TSMNet"
  "$WORK/code/corsw/TSMNet-main"
  "/leonardo_work/EUHPC_D33_186/code/corsw/TSMNet-main"
)
for t in "${TSMNET_CANDIDATES[@]}"; do
  if [[ -n "$t" && -f "$t/datasetio/eeg/moabb/__init__.py" ]]; then
    export TSMNET_ROOT="$t"
    break
  fi
done
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

echo "Project: ${PROJECT_DIR}"
echo "Python: $(python -c 'import sys; print(sys.executable)')"
echo "Dataset=${DATASET} Task=${TASK} NTRY=${NTRY} Device=${DEVICE} Subjects=${SUBJECTS}"
echo "MNE_DATA=${MNE_DATA}"
echo "TSMNET_ROOT=${TSMNET_ROOT:-<not-found>}"

python experiments/scripts/da_transfs_500.py \
    --dataset "${DATASET}" \
    --task "${TASK}" \
    --ntry "${NTRY}" \
    --device "${DEVICE}" \
    --subjects "${SUBJECTS}" \
    --checkpoint_every "${CHECKPOINT_EVERY}" \
    ${RESUME_FLAG} \
    "${EXTRA_ARGS[@]}"

echo "Done. Summary appended to ${PROJECT_DIR}/acc.txt"
