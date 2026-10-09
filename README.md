# pi05_v2 policy

This repository contains the pi05_v2 policy implementation used for RoboSynChallenge evaluation. It includes the policy adapter, OpenPI source, the locked dependency files, and the training entry points.

The repository intentionally does not contain a Python virtual environment, CUDA wheels, model checkpoints, training datasets, or machine-local caches. The evaluator should install the environment on the evaluation machine and provide the task checkpoints separately.

## Evaluation layout

The official evaluation checkout is expected to contain this repository at:

~~~
RoboSynChallenge/policy/pi05_v2/
~~~

Place the contents of this repository at the following location (including this README, eval.sh and src/). To avoid a nested Git repository, clone to a separate staging directory and export the tracked files:

~~~bash
git clone https://github.com/smalltmm/pi05_v2.git /path/to/pi05_v2-source
export ROBOSYN_ROOT=/path/to/RoboSynChallenge_ws/RoboSynChallenge
mkdir -p "$ROBOSYN_ROOT/policy/pi05_v2"
git -C /path/to/pi05_v2-source archive HEAD | tar -x -C "$ROBOSYN_ROOT/policy/pi05_v2"
export PI05_ROOT="$ROBOSYN_ROOT/policy/pi05_v2"
~~~

The launcher infers the RoboSynChallenge root from this layout. ROBOSYN_ROOT can override that root, but the official evaluator still requires the policy under its policy/pi05_v2 directory.

The official simulator environment and this policy environment are separate. The simulator uses the Python environment prepared by RoboSynChallenge; the PI0.5 worker uses pi05_v2/.venv.

## 1. Install the official simulator environment

Install RoboSynChallenge and EmbodiChain according to the official installation guide:

https://edem-ai.github.io/RoboSynChallenge/html/getting_started/installation.html

Use the simulator environment already prepared by the official installation. No deployment/local-env.sh is required; that file was a machine-specific helper and is not supplied by the official repository.

Either activate the environment as usual (virtualenv or Conda), or provide its interpreter explicitly:

~~~bash
# Any location is supported. This is the SIMULATOR Python, not the PI0.5 worker.
export PYTHON_BIN=/path/to/simulator-environment/bin/python
"$PYTHON_BIN" -c "import embodichain, dexsim, robosynchallenge, torch; print('Simulator imports OK')"
~~~

Alternatively, set ROBOSYN_VENV_DIR to the environment directory. With neither variable set, eval.sh checks these candidates in order and selects the first executable that can locate embodichain, dexsim, robosynchallenge and torch:

1. The activated virtualenv (VIRTUAL_ENV), then the activated Conda environment (CONDA_PREFIX).
2. EmbodiChain/.venv in the sibling EmbodiChain checkout (or EMBODICHAIN_ROOT/.venv).
3. RoboSynChallenge/.venv, then the workspace .venv, for other installation layouts.
4. python and python3 on PATH.

The official local installation example creates EmbodiChain/.venv; it is a supported candidate, not a required location. An explicit PYTHON_BIN takes priority over ROBOSYN_VENV_DIR. Invalid explicit selections produce an error instead of silently switching environments. The launcher prints both selected Python paths before evaluation.

The evaluation host must have:

- Python 3.11
- A working NVIDIA driver and CUDA runtime compatible with the installed JAX and PyTorch wheels
- An available physical GPU for the simulator and PI0.5 worker
- uv available on PATH
- The official RoboSynChallenge checkout and downloaded simulator assets

## 2. Install the locked PI0.5 environment

Set the policy checkout path and run the provided installer:

~~~
export PI05_ROOT="$ROBOSYN_ROOT/policy/pi05_v2"
cd "$PI05_ROOT"
bash setup_runtime.sh
~~~

setup_runtime.sh performs the following steps:

1. Creates PI05_ROOT/.venv with Python 3.11 when needed.
2. Installs the project and workspace packages with uv sync --frozen --no-dev.
3. Uses pyproject.toml and uv.lock without changing the lock file.
4. Runs uv pip check.
5. Verifies the bundled PaliGemma tokenizer checksum and downloads it only when it is absent.

For a normal online installation, the command used by the installer is:

~~~
uv sync --frozen --no-dev --python 3.11
uv pip check --python "$PI05_ROOT/.venv/bin/python"
~~~

If uv is not installed, install Astral uv using the official installer or the system package manager, then verify:

~~~
uv --version
python3.11 --version
~~~

The canonical dependency files are:

| File | Purpose |
| --- | --- |
| pyproject.toml | Project metadata, direct dependencies, Python requirement, and uv workspace configuration |
| uv.lock | Complete reproducible dependency lock with versions and hashes |
| runtime-requirements.txt | Hash-pinned requirements exported from uv.lock |
| requirements.txt | Dependency-list entry point that includes runtime-requirements.txt; does not install the local workspace packages |
| setup_runtime.sh | Environment creation, frozen installation, dependency check, and tokenizer verification |

The installer selects Python 3.11. The locked runtime includes JAX 0.5.3, Flax 0.10.2, Orbax Checkpoint 0.11.13, NumPy 1.x, PyTorch 2.7.1, Transformers 4.53.2, and the pinned LeRobot revision. The repository does not require Docker.

The optional offline installer requires BOTH runtime_wheels/ and a matching runtime-wheel-manifest.json with filenames and SHA256 hashes. Neither is distributed in this Git repository; use setup_runtime.sh for the normal online frozen installation. A requirements-only installation is not a replacement for installing the OpenPI workspace.

## 3. Checkpoint layout and model names

The evaluator supplies one checkpoint for each submitted task. The checkpoint directory must contain:

~~~
<checkpoint>/
├── params/
└── assets/
    └── <repo_id>/
        └── norm_stats.json
~~~

The submitted model directory should be named exactly after the task. For example:

~~~
/path/to/checkpoints/
├── click_bell/
├── handle_basket/
├── water_pouring/
├── table_rearrangement/
├── items_handover/
├── drawer_open_place/
├── mixer_operating/
├── item_assembly/
├── manipulate_pipette/
└── sample_loading/
~~~

No checkpoint is included in this repository.

Normalization statistics are always loaded from the selected checkpoint, without renaming or copying assets:

1. If PI05_NORM_ASSET_ID is set, load assets/<PI05_NORM_ASSET_ID>/norm_stats.json from this checkpoint. A missing explicit selection is an error.
2. Otherwise, if the training configuration's asset name exists under this checkpoint, use that file.
3. If it does not exist and exactly one assets/*/norm_stats.json exists in this checkpoint, load that file.
4. If several alternatives exist, stop with an error listing their asset IDs; specify the correct one explicitly.

No normalization statistics are read from the policy source tree, the simulator environment, or any other checkpoint.

For drawer_open_place, a checkpoint containing only assets/robosyn_robotwin_piper_sf/norm_stats.json is therefore detected automatically. To select it explicitly (for example when the checkpoint includes multiple statistics files):

~~~bash
PI05_NORM_ASSET_ID=robosyn_robotwin_piper_sf \\
  bash "$PI05_ROOT/eval.sh" drawer_open_place random \\
  "$MODEL_ROOT/drawer_open_place" 0 --max_episodes 20 --headless true
~~~

The selected statistics path is logged before model restoration. This changes only checkpoint asset lookup; it does not change training/model configuration, action transforms or checkpoint contents. The default configuration is pi05_robosyn_robotwin_piper_jax_full; set PI05_TRAIN_CONFIG only if the submitted checkpoint uses a different compatible training configuration.

## 4. Evaluation command format

The policy launcher accepts:

~~~
bash "$PI05_ROOT/eval.sh" \
  <task_name> random <checkpoint_path> <gpu_id> \
  --max_episodes 20 --headless true
~~~

From the deployment layout above, set the policy path and activate or select the official simulator environment:

~~~bash
export ROBOSYN_ROOT=/path/to/RoboSynChallenge_ws/RoboSynChallenge
export PI05_ROOT="$ROBOSYN_ROOT/policy/pi05_v2"
# Optional if a suitable simulator environment is already activated/detected:
export PYTHON_BIN=/path/to/simulator-environment/bin/python
~~~

gpu_id is a physical GPU index. The launcher leaves the simulator unmasked and passes the selected physical index to the PI0.5 worker. The worker always uses PI05_ROOT/.venv unless PI05_PYTHON is explicitly supplied.

The launcher sets EMBODICHAIN_SIM_EXIT_PROCESS=0 so that the official evaluator can save evaluation_metrics.json during normal environment cleanup.

## 5. Ten official task commands

The following commands assume that MODEL_ROOT contains one checkpoint directory per task and that the submitted directory name matches the task name. Replace 0 with an available physical GPU index.

~~~
export MODEL_ROOT=/path/to/checkpoints

bash "$PI05_ROOT/eval.sh" click_bell random \
  "$MODEL_ROOT/click_bell" 0 --max_episodes 20 --headless true

bash "$PI05_ROOT/eval.sh" handle_basket random \
  "$MODEL_ROOT/handle_basket" 0 --max_episodes 20 --headless true

bash "$PI05_ROOT/eval.sh" water_pouring random \
  "$MODEL_ROOT/water_pouring" 0 --max_episodes 20 --headless true

bash "$PI05_ROOT/eval.sh" table_rearrangement random \
  "$MODEL_ROOT/table_rearrangement" 0 --max_episodes 20 --headless true

bash "$PI05_ROOT/eval.sh" items_handover random \
  "$MODEL_ROOT/items_handover" 0 --max_episodes 20 --headless true

bash "$PI05_ROOT/eval.sh" drawer_open_place random \
  "$MODEL_ROOT/drawer_open_place" 0 --max_episodes 20 --headless true

bash "$PI05_ROOT/eval.sh" mixer_operating random \
  "$MODEL_ROOT/mixer_operating" 0 --max_episodes 20 --headless true

bash "$PI05_ROOT/eval.sh" item_assembly random \
  "$MODEL_ROOT/item_assembly" 0 --max_episodes 20 --headless true

bash "$PI05_ROOT/eval.sh" manipulate_pipette random \
  "$MODEL_ROOT/manipulate_pipette" 0 --max_episodes 20 --headless true

bash "$PI05_ROOT/eval.sh" sample_loading random \
  "$MODEL_ROOT/sample_loading" 0 --max_episodes 20 --headless true
~~~

Use clear instead of random when the official evaluation protocol for the checkpoint requires the clear setting.

## 6. Evaluation outputs

The official evaluator writes results under the RoboSynChallenge checkout:

~~~
$ROBOSYN_ROOT/eval_result/<task>/<policy>/<setting>/
~~~

Each run directory contains the recorded videos and evaluation_metrics.json. In the adapter contract, success stops the current action chunk without being reported as truncation. truncated is reserved for the environment time limit.

## 7. Training sources

The repository retains the OpenPI JAX/PyTorch training entry points and the RoboSyn/Robotwin configuration. Training data, initial weights, and training output directories are not included. The path overrides and training-only setup are documented in LOCAL_SETUP.txt.
