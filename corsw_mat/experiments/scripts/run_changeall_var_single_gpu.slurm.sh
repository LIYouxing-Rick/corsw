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

ORIGINAL_ARGS=("$@")

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
MAX_RUNTIME_HOURS=23
AUTO_RESUBMIT=1
MERGE_PARALLEL_RESULTS=1
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
    echo "  --max-runtime-hours N       auto-stop after N hours (default: 23)"
    echo "  --auto-resubmit             auto sbatch re-submit after timeout (default: on)"
    echo "  --no-auto-resubmit          disable auto re-submit after timeout"
    echo "  --merge-parallel-results    merge per-seed csv into one csv (default: on)"
    echo "  --no-merge-parallel-results keep per-seed csv files only"
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
        --max-runtime-hours) MAX_RUNTIME_HOURS="$2"; shift 2 ;;
        --auto-resubmit) AUTO_RESUBMIT=1; shift ;;
        --no-auto-resubmit) AUTO_RESUBMIT=0; shift ;;
        --merge-parallel-results) MERGE_PARALLEL_RESULTS=1; shift ;;
        --no-merge-parallel-results) MERGE_PARALLEL_RESULTS=0; shift ;;
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
echo "MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS}"
echo "AUTO_RESUBMIT=${AUTO_RESUBMIT}"
echo "MERGE_PARALLEL_RESULTS=${MERGE_PARALLEL_RESULTS}"

if ! [[ "${MAX_RUNTIME_HOURS}" =~ ^[0-9]+$ ]] || [[ "${MAX_RUNTIME_HOURS}" -le 0 ]]; then
  echo "[FATAL] --max-runtime-hours must be a positive integer, got '${MAX_RUNTIME_HOURS}'" >&2
  exit 9
fi

if [[ "${DATASET}" == "bnci2014001" || "${DATASET}" == "bnci2015001" ]]; then
  BNCI_ROOT="${MNE_DATA}/MNE-bnci-data/database/data-sets"
  if [[ "${DATASET}" == "bnci2014001" ]]; then
    BNCI_SUBDIR="${BNCI_ROOT}/001-2014"
    echo "BNCI_DIR=${BNCI_SUBDIR}"
    if [[ ! -d "${BNCI_SUBDIR}" ]]; then
      echo "[FATAL] BNCI2014001 cache dir not found: ${BNCI_SUBDIR}" >&2
      exit 5
    fi
    REQUIRED_SUBJECTS=(1 3 7 8 9)
    REQUIRED_SPLITS=("T")
    if [[ "${TASK}" == "session" ]]; then
      REQUIRED_SPLITS+=("E")
    fi
    for sid in "${REQUIRED_SUBJECTS[@]}"; do
      for split in "${REQUIRED_SPLITS[@]}"; do
        f="${BNCI_SUBDIR}/A0${sid}${split}.mat"
        if [[ ! -f "${f}" ]]; then
          echo "[FATAL] Missing BNCI2014001 file: ${f}" >&2
          exit 6
        fi
      done
    done
    BNCI_FILE_COUNT=$(find "${BNCI_SUBDIR}" -maxdepth 1 -type f -name 'A*.mat' | wc -l | tr -d ' ')
    echo "BNCI2014001 MAT count=${BNCI_FILE_COUNT}"
  else
    BNCI_SUBDIR="${BNCI_ROOT}/001-2015"
    echo "BNCI_DIR=${BNCI_SUBDIR}"
    if [[ ! -d "${BNCI_SUBDIR}" ]]; then
      echo "[FATAL] BNCI2015001 cache dir not found: ${BNCI_SUBDIR}" >&2
      exit 7
    fi
    BNCI_FILE_COUNT=$(find "${BNCI_SUBDIR}" -maxdepth 1 -type f -name 'S*.mat' | wc -l | tr -d ' ')
    echo "BNCI2015001 MAT count=${BNCI_FILE_COUNT}"
    if [[ "${BNCI_FILE_COUNT}" -eq 0 ]]; then
      echo "[FATAL] No BNCI2015001 MAT files found under ${BNCI_SUBDIR}" >&2
      exit 8
    fi
  fi
fi

CHECKPOINT_ARGS=()
if [[ -n "${CHECKPOINT_DIR}" ]]; then
  CHECKPOINT_ARGS=(--checkpoint_dir "${CHECKPOINT_DIR}")
fi

TIMEOUT_SECONDS=$((MAX_RUNTIME_HOURS * 3600))
TIMEOUT_FLAG_FILE=$(mktemp /tmp/corsw_timeout_flag.XXXXXX)
WATCHDOG_PID=""
RESUBMIT_NEEDED=0

cleanup_timeout_resources() {
  if [[ -n "${WATCHDOG_PID}" ]]; then
    kill "${WATCHDOG_PID}" 2>/dev/null || true
    wait "${WATCHDOG_PID}" 2>/dev/null || true
    WATCHDOG_PID=""
  fi
}

start_watchdog() {
  local target_pids=("$@")
  (
    sleep "${TIMEOUT_SECONDS}"
    echo "timeout" > "${TIMEOUT_FLAG_FILE}"
    echo "[TIMEOUT] Reached ${MAX_RUNTIME_HOURS}h; stopping current run to allow checkpoint resume."
    for pid in "${target_pids[@]}"; do
      kill -TERM "${pid}" 2>/dev/null || true
    done
    sleep 20
    for pid in "${target_pids[@]}"; do
      kill -KILL "${pid}" 2>/dev/null || true
    done
  ) &
  WATCHDOG_PID=$!
}

run_and_monitor() {
  local run_pids=("$@")
  local fail=0
  start_watchdog "${run_pids[@]}"
  for pid in "${run_pids[@]}"; do
    wait "${pid}" || fail=1
  done
  cleanup_timeout_resources
  if [[ -s "${TIMEOUT_FLAG_FILE}" ]]; then
    RESUBMIT_NEEDED=1
    return 0
  fi
  return "${fail}"
}

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

  if [[ "${#PIDS[@]}" -eq 0 ]]; then
    echo "[FATAL] No valid seeds parsed from --seed-list." >&2
    exit 10
  fi
  if ! run_and_monitor "${PIDS[@]}"; then
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
      "${EXTRA_ARGS[@]}" &
  SERIAL_PID=$!
  if ! run_and_monitor "${SERIAL_PID}"; then
    echo "[FATAL] Serial run failed before timeout." >&2
    exit 4
  fi
fi

if [[ "${RESUBMIT_NEEDED}" -eq 1 ]]; then
  if [[ "${AUTO_RESUBMIT}" -eq 1 ]]; then
    RESUBMIT_SCRIPT="${SLURM_SUBMIT_DIR:-${PROJECT_DIR}}/experiments/scripts/run_changeall_var_single_gpu.slurm.sh"
    if [[ ! -f "${RESUBMIT_SCRIPT}" ]]; then
      RESUBMIT_SCRIPT="${PROJECT_DIR}/experiments/scripts/run_changeall_var_single_gpu.slurm.sh"
    fi
    echo "[AUTO-RESUBMIT] Re-submitting: sbatch ${RESUBMIT_SCRIPT} ${ORIGINAL_ARGS[*]}"
    sbatch "${RESUBMIT_SCRIPT}" "${ORIGINAL_ARGS[@]}"
    exit 0
  fi
  echo "[TIMEOUT] Auto-resubmit disabled; exiting with checkpoint saved."
  rm -f "${TIMEOUT_FLAG_FILE}" 2>/dev/null || true
  exit 0
fi

if [[ "${PARALLEL_SEEDS}" -eq 1 && "${MERGE_PARALLEL_RESULTS}" -eq 1 ]]; then
  export _MERGE_PROJECT_DIR="${PROJECT_DIR}"
  export _MERGE_DATASET="${DATASET}"
  export _MERGE_TASK="${TASK}"
  export _MERGE_DISTANCE="${DISTANCE}"
  export _MERGE_EPHO="${EPHO}"
  export _MERGE_LR_COV="${LR_COV}"
  export _MERGE_LR_COR="${LR_COR}"
  export _MERGE_LR_MIX="${LR_MIX}"
  export _MERGE_POWER="${POWER}"
  export _MERGE_MAX_ITER="${MAX_ITER}"
  export _MERGE_USE_COV_NET="${USE_COV_NET}"
  export _MERGE_USE_COR_NET="${USE_COR_NET}"
  export _MERGE_SEED_LIST="${SEED_LIST}"
  python - <<'PY'
import os
import pandas as pd

def fmt_lr_tag(lr: float) -> str:
    s = f"{lr:.0e}"
    return s.replace("e-0", "e-").replace("e+0", "e+")

def fmt_power_tag(p: float) -> str:
    return f"{p:g}"

project_dir = os.environ["_MERGE_PROJECT_DIR"]
dataset = os.environ["_MERGE_DATASET"]
task = os.environ["_MERGE_TASK"]
distance = os.environ["_MERGE_DISTANCE"]
epho = int(os.environ["_MERGE_EPHO"])
lr_cov = float(os.environ["_MERGE_LR_COV"])
lr_cor = float(os.environ["_MERGE_LR_COR"])
lr_mix = float(os.environ["_MERGE_LR_MIX"])
power = float(os.environ["_MERGE_POWER"])
max_iter = int(os.environ["_MERGE_MAX_ITER"])
use_cov_net = int(os.environ["_MERGE_USE_COV_NET"])
use_cor_net = int(os.environ["_MERGE_USE_COR_NET"])
seed_list = [s.strip() for s in os.environ["_MERGE_SEED_LIST"].split(",") if s.strip()]
run_ntry = len(seed_list)
task_tag = "cross_session" if task == "session" else "cross_subject"
run_name_base = (
    f"cormat__dataset-{dataset}__task-{task_tag}__distance-{distance}"
    f"__epho-{epho}__ntry-{run_ntry}"
    f"__lr_cov-{fmt_lr_tag(lr_cov)}"
    f"__lr_cor-{fmt_lr_tag(lr_cor)}"
    f"__lr_mix-{fmt_lr_tag(lr_mix)}"
    f"__power-{fmt_power_tag(power)}"
    f"__max_iter-{max_iter}"
    f"__covnet-{use_cov_net}__cornet-{use_cor_net}"
)
results_dir = os.path.join(project_dir, "experiments", "results")
merged_path = os.path.join(results_dir, f"{run_name_base}.csv")
seed_paths = [os.path.join(results_dir, f"{run_name_base}_seed{seed}.csv") for seed in seed_list]

frames = []
missing = []
for path in seed_paths:
    if os.path.exists(path):
        try:
            frames.append(pd.read_csv(path))
        except Exception:
            missing.append(path)
    else:
        missing.append(path)

if len(frames) == 0:
    print("[MERGE] No seed result csv found, skip merged output.")
else:
    merged = pd.concat(frames, ignore_index=True)
    if "__param_key" in merged.columns:
        merged = merged.drop_duplicates(subset=["__param_key"], keep="last")
    merged.to_csv(merged_path, index=False)
    print(f"[MERGE] Wrote merged csv: {merged_path} rows={len(merged)}")

if len(missing):
    print("[MERGE] Missing/invalid seed csv files:")
    for p in missing:
        print(f"  - {p}")
PY
fi

rm -f "${TIMEOUT_FLAG_FILE}" 2>/dev/null || true
