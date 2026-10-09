#!/usr/bin/env bash
set -euo pipefail

# JAX PI05 full fine-tuning on Robosyn Sim + Robosyn Real + Robotwin Piper.
# The config keeps the training batch at 128 and uses Pi0Config's JAX bfloat16 default.

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PI05_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
CONFIG_NAME="${CONFIG_NAME:-pi05_robosyn_robotwin_piper_jax_full}"
FSDP_DEVICES="${FSDP_DEVICES:-2}"
BATCH_SIZE="${BATCH_SIZE:-128}"
MAX_STEPS="${MAX_STEPS:-30000}"
NUM_WORKERS="${NUM_WORKERS:-8}"
LOG_INTERVAL="${LOG_INTERVAL:-100}"
SAVE_INTERVAL="${SAVE_INTERVAL:-5000}"
KEEP_PERIOD="${KEEP_PERIOD:-5000}"
SMOKE="${SMOKE:-0}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-32}"
SMOKE_STEPS="${SMOKE_STEPS:-1}"
SMOKE_SAVE_INTERVAL="${SMOKE_SAVE_INTERVAL:-1}"
SMOKE_KEEP_PERIOD="${SMOKE_KEEP_PERIOD:-1}"
OVERWRITE="${OVERWRITE:-0}"
RESUME="${RESUME:-0}"
DRY_RUN="${DRY_RUN:-0}"
MODEL_DTYPE="${MODEL_DTYPE:-}"

if [[ "${SMOKE}" == "1" ]]; then
    BATCH_SIZE="${SMOKE_BATCH_SIZE}"
    MAX_STEPS="${SMOKE_STEPS}"
    SAVE_INTERVAL="${SMOKE_SAVE_INTERVAL}"
    KEEP_PERIOD="${SMOKE_KEEP_PERIOD}"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-${CONFIG_NAME}_${FSDP_DEVICES}gpu_bs${BATCH_SIZE}_${MAX_STEPS}steps_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/${EXP_NAME}.log}"

if [[ ! -d "${REPO_ROOT}" ]]; then
    echo "REPO_ROOT does not exist: ${REPO_ROOT}" >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -f "${REPO_ROOT}/assets/pi05_robosyn_robotwin_piper/robosyn_robotwin_piper/norm_stats.json" ]]; then
    echo "Missing norm stats under ${REPO_ROOT}/assets/pi05_robosyn_robotwin_piper" >&2
    exit 1
fi

GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l || true)"
if [[ "${GPU_COUNT}" -lt "${FSDP_DEVICES}" ]]; then
    echo "Need ${FSDP_DEVICES} GPUs for FSDP, found ${GPU_COUNT}" >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE="${WANDB_MODE:-disabled}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-${REPO_ROOT}/.jax_cache}"

mkdir -p "${LOG_DIR}" "${JAX_COMPILATION_CACHE_DIR}"
cd "${REPO_ROOT}"

ARGS=(
    scripts/train.py "${CONFIG_NAME}"
    --exp-name "${EXP_NAME}"
    --batch-size "${BATCH_SIZE}"
    --num-train-steps "${MAX_STEPS}"
    --num-workers "${NUM_WORKERS}"
    --log-interval "${LOG_INTERVAL}"
    --save-interval "${SAVE_INTERVAL}"
    --keep-period "${KEEP_PERIOD}"
    --fsdp-devices "${FSDP_DEVICES}"
    --no-wandb-enabled
)
if [[ -n "${MODEL_DTYPE}" ]]; then
    ARGS+=(--model.dtype "${MODEL_DTYPE}")
fi
if [[ "${OVERWRITE}" == "1" ]]; then
    ARGS+=(--overwrite)
fi
if [[ "${RESUME}" == "1" ]]; then
    ARGS+=(--resume)
fi
if [[ -n "${CHECKPOINT_BASE_DIR:-}" ]]; then
    ARGS+=(--checkpoint-base-dir "${CHECKPOINT_BASE_DIR}")
fi

{
    echo "repo=${REPO_ROOT}"
    echo "config=${CONFIG_NAME} fsdp_devices=${FSDP_DEVICES} batch_size=${BATCH_SIZE} steps=${MAX_STEPS} workers=${NUM_WORKERS}"
    echo "exp_name=${EXP_NAME} log=${LOG_FILE}"
    echo "xla_preallocate=${XLA_PYTHON_CLIENT_PREALLOCATE} jax_cache=${JAX_COMPILATION_CACHE_DIR}"
    echo "command=${PYTHON_BIN} ${ARGS[*]}"
} | tee "${LOG_FILE}"

if [[ "${DRY_RUN}" == "1" ]]; then
    exit 0
fi

set -o pipefail
"${PYTHON_BIN}" "${ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"



