# pi05_v2 policy

This repository contains the pi05_v2 policy implementation used for RoboSynChallenge evaluation. It includes the policy adapter, OpenPI source, the locked dependency files, and the training entry points.

The repository intentionally does not contain a Python virtual environment, CUDA wheels, model checkpoints, training datasets, or machine-local caches. The evaluator should install the environment on the evaluation machine and provide the task checkpoints separately.

## Evaluation layout

The official RoboSynChallenge source tree and this policy repository are kept separate:

~~~
/path/to/
├── RoboSynChallenge/          # official RoboSynChallenge checkout
└── pi05_v2/                   # this repository
~~~

The policy launcher accepts the official checkout through ROBOSYN_ROOT. It can also be copied to:

~~~
RoboSynChallenge/policy/pi05_v2/
~~~

The official simulator environment and this policy environment are separate. The simulator uses the Python environment prepared by RoboSynChallenge; the PI0.5 worker uses pi05_v2/.venv.

## 1. Install the official simulator environment

Install RoboSynChallenge and EmbodiChain according to the official installation guide:

https://edem-ai.github.io/RoboSynChallenge/html/getting_started/installation.html

After installation, set the official checkout path and load its launcher environment:

~~~
export ROBOSYN_ROOT=/path/to/RoboSynChallenge
source "$ROBOSYN_ROOT/deployment/local-env.sh"
~~~

local-env.sh must be sourced in the shell that starts evaluation. It configures the official simulator, EmbodiChain, DexSim, FFmpeg, assets, and simulator-side Python environment. It is not part of this policy repository.

The evaluation host must have:

- Python 3.11
- A working NVIDIA driver and CUDA runtime compatible with the installed JAX and PyTorch wheels
- An available physical GPU for the simulator and PI0.5 worker
- uv available on PATH
- The official RoboSynChallenge checkout and downloaded simulator assets

## 2. Install the locked PI0.5 environment

Set the policy checkout path and run the provided installer:

~~~
export PI05_ROOT=/path/to/pi05_v2
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
| requirements.txt | pip-compatible entry point that includes runtime-requirements.txt |
| setup_runtime.sh | Environment creation, frozen installation, dependency check, and tokenizer verification |

The locked runtime includes Python 3.11, JAX 0.5.3, Flax 0.10.2, Orbax Checkpoint 0.11.13, NumPy 1.x, PyTorch 2.7.1, Transformers 4.53.2, and the pinned LeRobot revision. The repository does not require Docker.

For an offline installation, provide a directory named runtime_wheels containing the Linux x86_64 / CPython 3.11 wheels listed by the lock manifest before running setup_runtime.sh. Wheel bundles are intentionally excluded from this Git repository.

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

## 4. Evaluation command format

The policy launcher accepts:

~~~
bash "$PI05_ROOT/eval.sh" \
  <task_name> random <checkpoint_path> <gpu_id> \
  --max_episodes 20 --headless true
~~~

Set ROBOSYN_ROOT before calling eval.sh when the policy repository is outside the official checkout:

~~~
export ROBOSYN_ROOT=/path/to/RoboSynChallenge
export PI05_ROOT=/path/to/pi05_v2
source "$ROBOSYN_ROOT/deployment/local-env.sh"
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
