#!/bin/bash
#SBATCH --job-name=corswmat_var_1gpu
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
DISTANCE="all"
NTRY=5
EPHO=500
LR_COV=2.5
LR_COR=2.5
LR_MIX=0.5
POWER=0.25
MAX_ITER=100
USE_COV_NET=1
USE_COR_NET=1
CHECKPOINT_EVERY=1
CHECKPOINT_DIR=""
RESUME_FLAG="--resume"
SEED_LIST="929,1884,2473,7066,7490"
PARALLEL_SEEDS=1
EXTRA_ARGS=()

print_usage() {
    echo "Usage: sbatch $0 [options]"
    echo ""
    echo "Options:"
    echo "  --dataset NAME              bnci2014001 | bnci2015001 | lee2019 | stieger2021"
    echo "  --task NAME                 session | subject"
    echo "  --distance NAME             all | ecm | lecm | olm | lsm"
    echo "  --ntry N"
    echo "  --epho N"
    echo "  --lr-cov FLOAT"
    echo "  --lr-cor FLOAT"
    echo "  --lr-mix FLOAT"
    echo "  --power FLOAT"
    echo "  --max-iter N"
    echo "  --use-cov-net 0|1"
    echo "  --use-cor-net 0|1"
    echo "  --checkpoint-dir PATH"
    echo "  --checkpoint-every N"
    echo "  --seed-list LIST            comma-separated seeds, default 929,1884,2473,7066,7490"
    echo "  --parallel-seeds            run one process per seed in parallel on this GPU"
    echo "  --serial-seeds              disable parallel-seeds mode"
    echo "  --resume | --no-resume"
    echo "  --extra \"ARGS\""
    echo "  -h | --help"
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset) DATASET="$2"; shift 2 ;;
        --task) TASK="$2"; shift 2 ;;
        --distance) DISTANCE="$2"; shift 2 ;;
        --ntry) NTRY="$2"; shift 2 ;;
        --epho) EPHO="$2"; shift 2 ;;
        --lr-cov) LR_COV="$2"; shift 2 ;;
        --lr-cor) LR_COR="$2"; shift 2 ;;
        --lr-mix) LR_MIX="$2"; shift 2 ;;
        --power) POWER="$2"; shift 2 ;;
        --max-iter|--max_iter) MAX_ITER="$2"; shift 2 ;;
        --use-cov-net) USE_COV_NET="$2"; shift 2 ;;
        --use-cor-net) USE_COR_NET="$2"; shift 2 ;;
        --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
        --checkpoint-every) CHECKPOINT_EVERY="$2"; shift 2 ;;
        --seed-list) SEED_LIST="$2"; shift 2 ;;
        --parallel-seeds) PARALLEL_SEEDS=1; shift ;;
        --serial-seeds) PARALLEL_SEEDS=0; shift ;;
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
  "$WORK/code/corsw/corsw_mat"
  "/leonardo_work/EUHPC_D33_186/code/corsw/corsw_mat"
)
FOUND_PROJECT=""
for p in "${PROJECT_CANDIDATES[@]}"; do
  if [[ -n "$p" && -f "$p/experiments/scripts/da_transfs_changeall_Rd_matrix_var.py" ]]; then
    FOUND_PROJECT="$p"
    break
  fi
done
if [[ -z "$FOUND_PROJECT" ]]; then
  echo "[FATAL] project dir not found." >&2
  exit 1
fi
PROJECT_DIR="$FOUND_PROJECT"
cd "${PROJECT_DIR}"

FAST="/leonardo_scratch/fast/EUHPC_D33_186"
export MNE_DATA="${FAST}/eeg_datasets/moabb_mne_data"
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8

echo "Project: ${PROJECT_DIR}"
echo "Python: $(python -c 'import sys; print(sys.executable)')"
echo "Dataset=${DATASET} Task=${TASK} Distance=${DISTANCE} NTRY=${NTRY}"
echo "MNE_DATA=${MNE_DATA}"
echo "CHECKPOINT_EVERY=${CHECKPOINT_EVERY}"
echo "CHECKPOINT_DIR=${CHECKPOINT_DIR:-<default>}"
echo "SEED_LIST=${SEED_LIST:-<default>}"
echo "PARALLEL_SEEDS=${PARALLEL_SEEDS}"

CHECKPOINT_ARGS=()
if [[ -n "${CHECKPOINT_DIR}" ]]; then
  CHECKPOINT_ARGS=(--checkpoint_dir "${CHECKPOINT_DIR}")
fi

if [[ "${PARALLEL_SEEDS}" -eq 1 ]]; then
  IFS=',' read -r -a SEEDS <<< "${SEED_LIST}"
  PIDS=()
  for raw_seed in "${SEEDS[@]}"; do
    seed="$(echo "${raw_seed}" | tr -d '[:space:]')"
    if [[ -z "${seed}" ]]; then
      continue
    fi
    echo "[PARALLEL] start seed=${seed}"
    python experiments/scripts/da_transfs_changeall_Rd_matrix_var.py \
        --dataset "${DATASET}" \
        --task "${TASK}" \
        --distance "${DISTANCE}" \
        --ntry 1 \
        --seed_list "${seed}" \
        --results_suffix "seed${seed}" \
        --epho "${EPHO}" \
        --lr_cov "${LR_COV}" \
        --lr_cor "${LR_COR}" \
        --lr_mix "${LR_MIX}" \
        --power "${POWER}" \
        --max_iter "${MAX_ITER}" \
        --use_cov_net "${USE_COV_NET}" \
        --use_cor_net "${USE_COR_NET}" \
        --checkpoint_every "${CHECKPOINT_EVERY}" \
        ${RESUME_FLAG} \
        "${CHECKPOINT_ARGS[@]}" \
        "${EXTRA_ARGS[@]}" &
    PIDS+=($!)
  done

  FAIL=0
  for pid in "${PIDS[@]}"; do
    wait "${pid}" || FAIL=1
  done
  if [[ "${FAIL}" -ne 0 ]]; then
    echo "[FATAL] At least one parallel seed process failed." >&2
    exit 4
  fi
else
  python experiments/scripts/da_transfs_changeall_Rd_matrix_var.py \
      --dataset "${DATASET}" \
      --task "${TASK}" \
      --distance "${DISTANCE}" \
      --ntry "${NTRY}" \
      --seed_list "${SEED_LIST}" \
      --epho "${EPHO}" \
      --lr_cov "${LR_COV}" \
      --lr_cor "${LR_COR}" \
      --lr_mix "${LR_MIX}" \
      --power "${POWER}" \
      --max_iter "${MAX_ITER}" \
      --use_cov_net "${USE_COV_NET}" \
      --use_cor_net "${USE_COR_NET}" \
      --checkpoint_every "${CHECKPOINT_EVERY}" \
      ${RESUME_FLAG} \
      "${CHECKPOINT_ARGS[@]}" \
      "${EXTRA_ARGS[@]}"
fi
