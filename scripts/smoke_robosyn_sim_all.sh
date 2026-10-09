#!/usr/bin/env bash
set -uo pipefail
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PYTHON_BIN="${PI05_PYTHON:-$REPO_ROOT/.venv/bin/python}"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT/packages/openpi-client/src"
cd "$REPO_ROOT"
RUN_DIR="$1"
mkdir -p "$RUN_DIR"
export CUDA_VISIBLE_DEVICES=0,1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_COMPILATION_CACHE_DIR="${JAX_COMPILATION_CACHE_DIR:-$REPO_ROOT/.jax_cache}"
export PYTHONUNBUFFERED=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS=8
for task in click_bell handle_basket table_rearrangement drawer_open_place item_assembly items_handover mixer_operating manipulate_pipette sample_loading water_pouring; do
    echo "START $task $(date -Is)"
    "$PYTHON_BIN" scripts/smoke_robosyn_sim_task.py --task "$task" --output-dir "$RUN_DIR" >"$RUN_DIR/$task.log" 2>&1
    code=$?
    echo "END $task exit=$code $(date -Is)"
    if [[ "$code" != 0 ]]; then tail -n 25 "$RUN_DIR/$task.log"; fi
done
echo "ALL_FINISHED $(date -Is)"
