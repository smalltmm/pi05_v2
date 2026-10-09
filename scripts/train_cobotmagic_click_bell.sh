#!/usr/bin/env bash
set -euo pipefail

# Run from the openpi checkout, or override REPO_ROOT when the checkout is elsewhere.
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PI05_PYTHON:-${REPO_ROOT}/.venv/bin/python}"
CONFIG_NAME="${CONFIG_NAME:-pi05_cobotmagic_sim}"
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

if [[ -z "${DATASET_ID:-}" ]]; then
    case "${CONFIG_NAME}" in
        pi05_cobotmagic_sim)
            DATASET_ID="cobotmagic_Sim_click_bell"
            ;;
        pi05_cobotmagic_sim_*)
            DATASET_ID="cobotmagic_Sim_${CONFIG_NAME#pi05_cobotmagic_sim_}"
            ;;
        *)
            echo "Unsupported CobotMagic config: ${CONFIG_NAME}" >&2
            exit 1
            ;;
    esac
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${EXP_NAME:-${CONFIG_NAME}_${NUM_GPUS}gpu_bs${BATCH_SIZE}_${MAX_STEPS}steps_${TIMESTAMP}}"
LOG_DIR="${LOG_DIR:-${REPO_ROOT}/logs}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"

test -d "${REPO_ROOT}"
test -x "${PYTHON_BIN}"
test -f "${REPO_ROOT}/pi05_base_pytorch/model.safetensors"
test -d "${REPO_ROOT}/assets/${CONFIG_NAME}/${DATASET_ID}"

GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l || true)"
if [[ "${GPU_COUNT}" -lt "${NUM_GPUS}" ]]; then
    echo "Need ${NUM_GPUS} GPUs, found ${GPU_COUNT}" >&2
    exit 1
fi

export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}/packages/openpi-client/src"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

echo "repo=${REPO_ROOT}"
echo "config=${CONFIG_NAME} dataset=${DATASET_ID} precision=${PRECISION} gpus=${NUM_GPUS} batch_size=${BATCH_SIZE} steps=${MAX_STEPS}"
echo "exp_name=${EXP_NAME} log=${LOG_FILE}"

set -o pipefail
"${PYTHON_BIN}" -m torch.distributed.run \
    --standalone \
    --nnodes=1 \
    --nproc_per_node="${NUM_GPUS}" \
    scripts/train_pytorch.py "${CONFIG_NAME}" \
    --exp-name "${EXP_NAME}" \
    --pytorch-training-precision "${PRECISION}" \
    --num-train-steps "${MAX_STEPS}" \
    --batch-size "${BATCH_SIZE}" \
    --num-workers "${NUM_WORKERS}" \
    --save-interval "${SAVE_INTERVAL}" \
    --keep-period "${KEEP_PERIOD}" \
    --no-wandb-enabled \
    2>&1 | tee "${LOG_FILE}"
