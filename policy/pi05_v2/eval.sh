#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# bash policy/pi05_v2/eval.sh <task> <setting> <checkpoint> [gpu_id] [extra_opts...]
#
# Example:
# bash policy/pi05_v2/eval.sh click_bell random \
#   /path/to/checkpoint 7 \
#   --max_episodes 20 --headless true --eval_video_log true
# -----------------------------------------------------------------------------

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${ROBOSYN_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
REPO_ROOT="$(cd -- "$REPO_ROOT" && pwd)"
WORKSPACE_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
EMBODICHAIN_ROOT="${EMBODICHAIN_ROOT:-$WORKSPACE_ROOT/EmbodiChain}"
OPENPI_ROOT="${OPENPI_ROOT:-$SCRIPT_DIR}"

POLICY_NAME=pi05_v2
USAGE="Usage: bash policy/pi05_v2/eval.sh <task_name> <setting> <checkpoint_path> [gpu_id] [extra_opts...]"

TASK_NAME="${1:?$USAGE}"
SETTING="${2:?$USAGE}"
CHECKPOINT_PATH="${3:?$USAGE}"
shift 3
GPU_ID=0
if [[ $# -gt 0 && "$1" != --* ]]; then
    GPU_ID="$1"
    shift
fi
if [[ ! "$GPU_ID" =~ ^[0-9]+$ ]]; then
    echo "Error: gpu_id must be one physical GPU index." >&2
    exit 1
fi
EXTRA_ARGS=("$@")

if [[ -n "${PI05_TRAIN_CONFIG:-}" ]]; then
    TRAIN_CONFIG_NAME="$PI05_TRAIN_CONFIG"
else
    # The shared JAX checkpoint was trained across all RoboSyn tasks.
    TRAIN_CONFIG_NAME=pi05_robosyn_robotwin_piper_jax_full
fi

if [[ ! -d "$OPENPI_ROOT/src/openpi" ]]; then
    echo "Error: invalid OpenPI source directory: $OPENPI_ROOT" >&2
    exit 1
fi
OPENPI_ROOT="$(cd -- "$OPENPI_ROOT" && pwd)"

# Use an explicit policy interpreter; never borrow another user's environment.
PI05_PYTHON="${PI05_PYTHON:-$OPENPI_ROOT/.venv/bin/python}"

for i in "${!EXTRA_ARGS[@]}"; do
    case "${EXTRA_ARGS[$i]}" in
        true) EXTRA_ARGS[$i]=True ;;
        false) EXTRA_ARGS[$i]=False ;;
        none|null) EXTRA_ARGS[$i]=None ;;
    esac
done

resolve_python() {
    local executable
    executable="$(command -v -- "$1")" || return 1
    [[ -x "$executable" ]] || return 1
    printf '%s/%s\n' "$(cd -- "$(dirname -- "$executable")" && pwd)" "$(basename -- "$executable")"
}
# Explicit interpreter selections take precedence and fail without falling back.
# Automatic discovery checks package availability without importing CUDA libraries.
sim_python_available() {
    "$1" -c 'import importlib.util, sys; sys.exit(not all(importlib.util.find_spec(m) is not None for m in ("embodichain", "dexsim", "robosynchallenge", "torch")))' >/dev/null 2>&1
}
select_sim_python() {
    local candidate resolved
    if [[ -n "${PYTHON_BIN:-}" || -n "${ROBOSYN_VENV_DIR:-}" ]]; then
        candidate="${PYTHON_BIN:-${ROBOSYN_VENV_DIR}/bin/python}"
        resolved="$(resolve_python "$candidate")" || {
            echo "Error: selected simulator Python is not executable: $candidate" >&2
            return 1
        }
        sim_python_available "$resolved" || {
            echo "Error: $resolved cannot locate embodichain, dexsim, robosynchallenge and torch. Install the official simulator dependencies in this environment." >&2
            return 1
        }
        printf '%s\n' "$resolved"
        return 0
    fi
    local candidates=()
    [[ -z "${VIRTUAL_ENV:-}" ]] || candidates+=("$VIRTUAL_ENV/bin/python")
    [[ -z "${CONDA_PREFIX:-}" ]] || candidates+=("$CONDA_PREFIX/bin/python")
    candidates+=("$EMBODICHAIN_ROOT/.venv/bin/python" "$REPO_ROOT/.venv/bin/python"
                 "$WORKSPACE_ROOT/.venv/bin/python" python python3)
    for candidate in "${candidates[@]}"; do
        resolved="$(resolve_python "$candidate")" || continue
        if sim_python_available "$resolved"; then
            printf '%s\n' "$resolved"
            return 0
        fi
    done
    echo "Error: no simulator Python found. Activate the official simulator environment, or set PYTHON_BIN / ROBOSYN_VENV_DIR." >&2
    return 1
}
if [[ ! -f "$REPO_ROOT/scripts/eval_policy.py" ]]; then
    echo "Error: RoboSynChallenge not found at $REPO_ROOT; place pi05_v2 under RoboSynChallenge/policy or set ROBOSYN_ROOT." >&2
    exit 1
fi
PYTHON_BIN="$(select_sim_python)"
if ! PI05_PYTHON="$(resolve_python "$PI05_PYTHON")"; then
    echo "Error: policy interpreter unavailable; create pi05_v2/.venv or set PI05_PYTHON." >&2
    exit 1
fi
if [[ ! -d "$CHECKPOINT_PATH/params" ]]; then
    echo "Error: checkpoint must contain params: $CHECKPOINT_PATH" >&2
    exit 1
fi
if ! compgen -G "$CHECKPOINT_PATH/assets/*/norm_stats.json" >/dev/null; then
    echo "Error: checkpoint is missing assets/<repo_id>/norm_stats.json: $CHECKPOINT_PATH" >&2
    exit 1
fi

# Resolve relative checkpoint paths before changing to the repository root.
CHECKPOINT_PATH="$(cd -- "$CHECKPOINT_PATH" && pwd)"

# DexSim receives a physical GPU ordinal through --gpu_id. Keep the simulator
# process unmasked, then mask only the OpenPI worker subprocess.
unset CUDA_VISIBLE_DEVICES
export CUDA_DEVICE_ORDER=PCI_BUS_ID
# Match the official pi05 launcher: allow metrics saving after env.close().
export EMBODICHAIN_SIM_EXIT_PROCESS=0
export MPLCONFIGDIR="${MPLCONFIGDIR:-$REPO_ROOT/.cache/matplotlib}"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/policy:$EMBODICHAIN_ROOT${PYTHONPATH:+:$PYTHONPATH}"

echo "========================================="
echo "  PI0.5 JAX Evaluation"
echo "  Task:         $TASK_NAME ($SETTING)"
echo "  Config:       $TRAIN_CONFIG_NAME"
echo "  Checkpoint:   $CHECKPOINT_PATH"
echo "  GPU:          $GPU_ID"
echo "  RoboSyn Py:   $PYTHON_BIN"
echo "  OpenPI root:  $OPENPI_ROOT"
echo "  OpenPI Py:    $PI05_PYTHON"
echo "========================================="

cd "$REPO_ROOT"
PYTHONWARNINGS=ignore::UserWarning \
"$PYTHON_BIN" scripts/eval_policy.py \
    --config "policy/$POLICY_NAME/deploy_policy.yml" \
    --overrides \
    --task_name "$TASK_NAME" \
    --setting "$SETTING" \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --train_config_name "$TRAIN_CONFIG_NAME" \
    --model_name "$(basename "$(dirname "$CHECKPOINT_PATH")")_$(basename "$CHECKPOINT_PATH")" \
    --gpu_id "$GPU_ID" \
    --pi05_cuda_visible_devices "$GPU_ID" \
    --openpi_root "$OPENPI_ROOT" \
    --pi05_python "$PI05_PYTHON" \
    "${EXTRA_ARGS[@]}"
