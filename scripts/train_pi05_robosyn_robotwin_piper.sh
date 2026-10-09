#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PI05_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
CONFIG_NAME="${CONFIG_NAME:-pi05_robosyn_robotwin_piper}"
NUM_GPUS="${NUM_GPUS:-4}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PRECISION="${PRECISION:-float32}"
MAX_STEPS="${MAX_STEPS:-30000}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SAVE_INTERVAL="${SAVE_INTERVAL:-1000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
SMOKE="${SMOKE:-0}"

if [[ "${SMOKE}" == "1" ]]; then
  MAX_STEPS="${SMOKE_STEPS:-10}"
  SAVE_INTERVAL="${SMOKE_SAVE_INTERVAL:-10}"
  KEEP_PERIOD="${SMOKE_KEEP_PERIOD:-10}"
fi

test -x "${PYTHON_BIN}"
test -f "${REPO_ROOT}/pi05_base_pytorch/model.safetensors"
test -f "${REPO_ROOT}/assets/${CONFIG_NAME}/robosyn_robotwin_piper/norm_stats.json"

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-${CONFIG_NAME}_${NUM_GPUS}gpu_bs${BATCH_SIZE}_${MAX_STEPS}steps_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

"${PYTHON_BIN}" -m torch.distributed.run \
  --standalone --nnodes=1 --nproc_per_node="${NUM_GPUS}" \
  scripts/train_pytorch.py "${CONFIG_NAME}" \
  --exp-name "${EXP_NAME}" \
  --pytorch-training-precision "${PRECISION}" \
  --num-train-steps "${MAX_STEPS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --save-interval "${SAVE_INTERVAL}" \
  --keep-period "${KEEP_PERIOD}" \
  --no-wandb-enabled 2>&1 | tee "${LOG_DIR}/${EXP_NAME}.log"
