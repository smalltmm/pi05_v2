# PI0.5 v2 官方评测

本仓库只包含 PI0.5 policy、评测脚本和锁定的推理依赖，不包含仿真环境、模型权重或 checkpoint。请把代码放到 RoboSynChallenge 的 `policy/pi05_v2` 下；不要在该目录保留嵌套的 `.git`。

## 1. 获取 policy

```bash
export ROBOSYN_WS=/path/to/RoboSynChallenge_ws
export ROBOSYN_ROOT="$ROBOSYN_WS/RoboSynChallenge"
export POLICY_ROOT="$ROBOSYN_ROOT/policy/pi05_v2"

mkdir -p "$ROBOSYN_ROOT/policy"
git clone https://github.com/smalltmm/pi05_v2.git /tmp/pi05_v2
mkdir -p "$POLICY_ROOT"
git -C /tmp/pi05_v2 archive HEAD | tar -x -C "$POLICY_ROOT"
```

## 2. 安装并启动官方仿真器环境

按 RoboSynChallenge 官方安装说明安装 EmbodiChain v0.2.4.post1 和 RoboSynChallenge：

```bash
cd "$ROBOSYN_WS"
git clone https://github.com/DexForce/EmbodiChain.git
cd EmbodiChain
git checkout tags/v0.2.4.post1
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e . \
  --extra-index-url http://pyp.open3dv.site:2345/simple/ \
  --trusted-host pyp.open3dv.site

cd "$ROBOSYN_ROOT"
uv pip install -e .
```

评测前激活官方环境，并明确告诉脚本使用哪个 Python：

```bash
source "$ROBOSYN_WS/EmbodiChain/.venv/bin/activate"
export PYTHON_BIN="$ROBOSYN_WS/EmbodiChain/.venv/bin/python"
python -c "import embodichain, dexsim, robosynchallenge; print('simulator ok')"
```

如果官方安装把仿真环境建在 `$ROBOSYN_WS/.venv`，将上面的两处路径改为 `$ROBOSYN_WS/.venv`。评测脚本会在该 Python 进程中启动官方仿真器。

## 3. 安装 PI0.5 推理环境

```bash
cd "$POLICY_ROOT"
uv venv --python 3.11 .venv
uv sync --frozen --no-dev --python 3.11
uv pip check --python "$POLICY_ROOT/.venv/bin/python"
export PI05_PYTHON="$POLICY_ROOT/.venv/bin/python"
```

`pyproject.toml` 和 `uv.lock` 是推理环境的标准来源。仓库中的 `requirements.txt` 和 `runtime-requirements.txt` 仅用于兼容没有 uv 的安装流程；需要与官方版本一致时请优先使用上面的 `uv sync --frozen`。

## 4. 放置 checkpoint

每个 checkpoint 放在独立目录，至少包含 `params/` 和 checkpoint 自带的归一化统计：

```text
$MODEL_ROOT/<任务名>/
├── params/
└── assets/<checkpoint 使用的 repo_id>/norm_stats.json
```

例如：

```bash
export MODEL_ROOT="$ROBOSYN_WS/checkpoints"
mkdir -p "$MODEL_ROOT/drawer_open_place"
# 将主办方提供的 drawer_open_place checkpoint 解压到该目录
test -d "$MODEL_ROOT/drawer_open_place/params"
find "$MODEL_ROOT/drawer_open_place/assets" -name norm_stats.json
```

评测始终从当前 checkpoint 的 `assets/**/norm_stats.json` 读取归一化统计，不使用仓库外的固定 norm 文件。

## 5. 完整评测命令

下面的命令同时启动官方仿真器和 PI0.5 推理进程。第四个参数是 GPU 编号：

```bash
cd "$ROBOSYN_ROOT"
source "$ROBOSYN_WS/EmbodiChain/.venv/bin/activate"
export PYTHON_BIN="$ROBOSYN_WS/EmbodiChain/.venv/bin/python"
export PI05_PYTHON="$POLICY_ROOT/.venv/bin/python"
export GPU_ID=0

bash "$POLICY_ROOT/eval.sh" drawer_open_place random \
  "$MODEL_ROOT/drawer_open_place" "$GPU_ID" \
  --max_episodes 20 --headless true --seed 0
```

十个任务的命令如下（每个任务使用同名 checkpoint 目录）：

```bash
bash "$POLICY_ROOT/eval.sh" click_bell          random "$MODEL_ROOT/click_bell"          "$GPU_ID" --max_episodes 20 --headless true --seed 0
bash "$POLICY_ROOT/eval.sh" handle_basket       random "$MODEL_ROOT/handle_basket"       "$GPU_ID" --max_episodes 20 --headless true --seed 0
bash "$POLICY_ROOT/eval.sh" water_pouring       random "$MODEL_ROOT/water_pouring"       "$GPU_ID" --max_episodes 20 --headless true --seed 0
bash "$POLICY_ROOT/eval.sh" table_rearrangement random "$MODEL_ROOT/table_rearrangement" "$GPU_ID" --max_episodes 20 --headless true --seed 0
bash "$POLICY_ROOT/eval.sh" items_handover      random "$MODEL_ROOT/items_handover"      "$GPU_ID" --max_episodes 20 --headless true --seed 0
bash "$POLICY_ROOT/eval.sh" drawer_open_place   random "$MODEL_ROOT/drawer_open_place"   "$GPU_ID" --max_episodes 20 --headless true --seed 0
bash "$POLICY_ROOT/eval.sh" mixer_operating     random "$MODEL_ROOT/mixer_operating"     "$GPU_ID" --max_episodes 20 --headless true --seed 0
bash "$POLICY_ROOT/eval.sh" item_assembly       random "$MODEL_ROOT/item_assembly"       "$GPU_ID" --max_episodes 20 --headless true --seed 0
bash "$POLICY_ROOT/eval.sh" manipulate_pipette  random "$MODEL_ROOT/manipulate_pipette"  "$GPU_ID" --max_episodes 20 --headless true --seed 0
bash "$POLICY_ROOT/eval.sh" sample_loading      random "$MODEL_ROOT/sample_loading"      "$GPU_ID" --max_episodes 20 --headless true --seed 0
```

`eval.sh` 会设置官方支持的 `EMBODICHAIN_SIM_EXIT_PROCESS=0`，并将仿真器和 policy 使用的路径传给评测程序。

## 6. 评测输出

每次运行的结果位于：

```text
$ROBOSYN_ROOT/eval_result/<任务名>/pi05_v2/random/<train_config>/<model_name>/<时间戳>/
├── evaluation_metrics.json
└── videos/
```

`evaluation_metrics.json` 保存成功率等指标；终端日志会打印 `Evaluation Results Summary` 和结果文件路径。
