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

ORIGINAL_ARGS=("$@")

DATASET="bnci2014001"
TASK="session"
NTRY=5
DEVICE="cuda:0"
SUBJECTS="auto"
CHECKPOINT_EVERY=1
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
    echo "  --ntry N"
    echo "  --device DEV                cuda:0 | cpu | auto"
    echo "  --subjects LIST             auto | all | 1,2,3"
    echo "  --checkpoint-every N"
    echo "  --seed-list LIST            comma-separated seeds, e.g. 0,1,2,3,4"
    echo "  --parallel-seeds            run one process per seed in parallel on this GPU"
    echo "  --serial-seeds              disable parallel-seeds mode"
    echo "  --max-runtime-hours N       auto-stop after N hours (default: 23)"
    echo "  --auto-resubmit             auto sbatch re-submit after timeout (default: on)"
    echo "  --no-auto-resubmit          disable auto re-submit after timeout"
    echo "  --merge-parallel-results    merge per-seed csv into one csv (default: on)"
    echo "  --no-merge-parallel-results keep per-seed csv files only"
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
echo "SEED_LIST=${SEED_LIST:-<default>}"
echo "PARALLEL_SEEDS=${PARALLEL_SEEDS}"
echo "MAX_RUNTIME_HOURS=${MAX_RUNTIME_HOURS}"
echo "AUTO_RESUBMIT=${AUTO_RESUBMIT}"
echo "MERGE_PARALLEL_RESULTS=${MERGE_PARALLEL_RESULTS}"

if ! [[ "${MAX_RUNTIME_HOURS}" =~ ^[0-9]+$ ]] || [[ "${MAX_RUNTIME_HOURS}" -le 0 ]]; then
  echo "[FATAL] --max-runtime-hours must be a positive integer, got '${MAX_RUNTIME_HOURS}'" >&2
  exit 9
fi

if [[ "${DATASET}" == "stieger2021" ]]; then
  STIEGER_DIR="${MNE_DATA}/MNE-Stieger2021-data"
  STIEGER_MAT_COUNT=0
  if [[ -d "${STIEGER_DIR}" ]]; then
    STIEGER_MAT_COUNT=$(find "${STIEGER_DIR}" -maxdepth 1 -type f -name '*.mat' | wc -l | tr -d ' ')
  fi
  echo "Stieger local MAT count=${STIEGER_MAT_COUNT} at ${STIEGER_DIR}"
  if [[ "${STIEGER_MAT_COUNT}" -eq 0 ]]; then
    echo "[FATAL] No local Stieger2021 MAT files found under ${STIEGER_DIR}." >&2
    echo "[FATAL] Current cluster nodes cannot access api.figshare.com, so online download will fail." >&2
    echo "[FATAL] Please pre-populate MNE_DATA cache (MNE-Stieger2021-data/*.mat) or copy files from another machine." >&2
    exit 2
  fi
fi

TIMEOUT_SECONDS=$((MAX_RUNTIME_HOURS * 3600))
TIMEOUT_FLAG_FILE=$(mktemp /tmp/spdsw_timeout_flag.XXXXXX)
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
  if [[ -z "${SEED_LIST}" ]]; then
    echo "[FATAL] --parallel-seeds requires --seed-list." >&2
    exit 3
  fi
  IFS=',' read -r -a SEEDS <<< "${SEED_LIST}"
  PIDS=()
  for raw_seed in "${SEEDS[@]}"; do
    seed="$(echo "${raw_seed}" | tr -d '[:space:]')"
    if [[ -z "${seed}" ]]; then
      continue
    fi
    echo "[PARALLEL] start seed=${seed}"
    python experiments/scripts/da_transfs_500.py \
        --dataset "${DATASET}" \
        --task "${TASK}" \
        --ntry 1 \
        --seed_list "${seed}" \
        --results_suffix "seed${seed}" \
        --device "${DEVICE}" \
        --subjects "${SUBJECTS}" \
        --checkpoint_every "${CHECKPOINT_EVERY}" \
        ${RESUME_FLAG} \
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
  SEED_ARGS=()
  if [[ -n "${SEED_LIST}" ]]; then
    SEED_ARGS=(--seed_list "${SEED_LIST}")
  fi
  python experiments/scripts/da_transfs_500.py \
      --dataset "${DATASET}" \
      --task "${TASK}" \
      --ntry "${NTRY}" \
      --device "${DEVICE}" \
      --subjects "${SUBJECTS}" \
      --checkpoint_every "${CHECKPOINT_EVERY}" \
      "${SEED_ARGS[@]}" \
      ${RESUME_FLAG} \
      "${EXTRA_ARGS[@]}" &
  SERIAL_PID=$!
  if ! run_and_monitor "${SERIAL_PID}"; then
    echo "[FATAL] Serial run failed before timeout." >&2
    exit 4
  fi
fi

if [[ "${RESUBMIT_NEEDED}" -eq 1 ]]; then
  if [[ "${AUTO_RESUBMIT}" -eq 1 ]]; then
    RESUBMIT_SCRIPT="${SLURM_SUBMIT_DIR:-${PROJECT_DIR}}/experiments/scripts/run_da500_single_gpu.slurm.sh"
    if [[ ! -f "${RESUBMIT_SCRIPT}" ]]; then
      RESUBMIT_SCRIPT="${PROJECT_DIR}/experiments/scripts/run_da500_single_gpu.slurm.sh"
    fi
    echo "[AUTO-RESUBMIT] Re-submitting: sbatch ${RESUBMIT_SCRIPT} ${ORIGINAL_ARGS[*]}"
    sbatch "${RESUBMIT_SCRIPT}" "${ORIGINAL_ARGS[@]}"
    rm -f "${TIMEOUT_FLAG_FILE}" 2>/dev/null || true
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
  export _MERGE_SEED_LIST="${SEED_LIST}"
  python - <<'PY'
import os
import pandas as pd

project_dir = os.environ["_MERGE_PROJECT_DIR"]
dataset = os.environ["_MERGE_DATASET"]
task = os.environ["_MERGE_TASK"]
seed_list = [s.strip() for s in os.environ["_MERGE_SEED_LIST"].split(",") if s.strip()]
results_dir = os.path.join(project_dir, "experiments", "results")
task_tag = "cross_subject" if task == "subject" else "cross_session"
merged_path = os.path.join(results_dir, f"da_{dataset}_{task_tag}_epho500.csv")
seed_paths = [os.path.join(results_dir, f"da_{dataset}_{task_tag}_epho500_seed{seed}.csv") for seed in seed_list]

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
    merged = pd.concat(frames, ignore_index=True).drop_duplicates(keep="last")
    merged.to_csv(merged_path, index=False)
    print(f"[MERGE] Wrote merged csv: {merged_path} rows={len(merged)}")

if len(missing):
    print("[MERGE] Missing/invalid seed csv files:")
    for p in missing:
        print(f"  - {p}")
PY
fi

rm -f "${TIMEOUT_FLAG_FILE}" 2>/dev/null || true

echo "Done. Summary appended to ${PROJECT_DIR}/acc.txt"
