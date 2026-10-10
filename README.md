# PI0.5 v2 评测

本仓库包含 PI05_v2 policy 和锁定的依赖，不包含 Python 环境或 checkpoint。代码部署到 `RoboSynChallenge/policy/pi05_v2`。

## 1. 获取 policy

先准备官方 RoboSynChallenge 代码；已有工作区可跳过克隆官方仓库。将下方路径替换为实际位置：

```bash
export ROBOSYN_WS=/path/to/RoboSynChallenge_ws
export ROBOSYN_ROOT="$ROBOSYN_WS/RoboSynChallenge"
export POLICY_ROOT="$ROBOSYN_ROOT/policy/pi05_v2"
mkdir -p "$ROBOSYN_WS"
# 仅首次获取官方代码时执行：
git clone https://github.com/EDEM-AI/RoboSynChallenge.git "$ROBOSYN_ROOT"
```

为避免嵌套 Git 仓库，先克隆 policy 到临时目录，再导出到 `policy/pi05_v2`：

```bash
POLICY_DOWNLOAD_DIR="$(mktemp -d)"
git clone https://github.com/smalltmm/pi05_v2.git "$POLICY_DOWNLOAD_DIR"
mkdir -p "$POLICY_ROOT"
git -C "$POLICY_DOWNLOAD_DIR" archive HEAD | tar -x -C "$POLICY_ROOT"
```

## 2. 安装仿真器环境

已按[官方安装说明](https://edem-ai.github.io/RoboSynChallenge/html/getting_started/installation.html)安装的环境可跳过本节。官方版本为 `v0.2.4.post1`。

Docker 安装需要宿主机已安装 Docker、NVIDIA Container Toolkit 和 NVIDIA 驱动（官方要求驱动版本至少为 535）。按官方脚本创建容器：

```bash
export ROBOSYN_WS=/path/to/RoboSynChallenge_ws
mkdir -p "$ROBOSYN_WS"
docker pull dexforce/embodichain:ubuntu22.04-cuda12.8
git clone https://github.com/DexForce/EmbodiChain.git "$ROBOSYN_WS/EmbodiChain"
cd "$ROBOSYN_WS/EmbodiChain"
git checkout tags/v0.2.4.post1
./docker/docker_run.sh robosyn "$ROBOSYN_WS"
```

该脚本会将宿主机工作区挂载到容器内的 `/root/workspace`。若已创建过容器，后续只需执行第 3 节的 `docker start` 和 `docker exec`。无 Docker 权限时使用下面的本地安装方式。

以下为本地安装（Linux，已配置 NVIDIA 驱动及官方要求的渲染依赖）：

```bash
# 已有 uv 时跳过这两行
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

cd "$ROBOSYN_WS"
git clone https://github.com/DexForce/EmbodiChain.git
cd EmbodiChain
git checkout tags/v0.2.4.post1
uv venv --python 3.11 .venv
source .venv/bin/activate
uv pip install -e . \
  --extra-index-url http://pyp.open3dv.site:2345/simple/ \
  --trusted-host pyp.open3dv.site
uv pip install -e "$ROBOSYN_ROOT"
```

## 3. 激活官方仿真器环境

以下两种方式选一种。这里只选择仿真器使用的 Python，评测时 `eval.sh` 会启动仿真器。

### 3.1 Docker 安装

在宿主机启动并进入按官方教程创建的容器（名称替换为实际值）：

```bash
export CONTAINER_NAME=robosyn
docker start "$CONTAINER_NAME"
docker exec -it "$CONTAINER_NAME" bash
```

进入容器后执行；第 4–7 节也在同一容器内执行：

```bash
conda activate robosyn
export ROBOSYN_WS=/root/workspace
export ROBOSYN_ROOT="$ROBOSYN_WS/RoboSynChallenge"
export POLICY_ROOT="$ROBOSYN_ROOT/policy/pi05_v2"
export PYTHON_BIN="$(command -v python)"
"$PYTHON_BIN" -c "import embodichain, dexsim, robosynchallenge; print('simulator ok')"
```

### 3.2 本地安装

```bash
export ROBOSYN_WS=/path/to/RoboSynChallenge_ws
export ROBOSYN_ROOT="$ROBOSYN_WS/RoboSynChallenge"
export POLICY_ROOT="$ROBOSYN_ROOT/policy/pi05_v2"
export SIM_ENV="$ROBOSYN_WS/EmbodiChain/.venv"  # 改为实际仿真环境目录
source "$SIM_ENV/bin/activate"
export PYTHON_BIN="$(command -v python)"
"$PYTHON_BIN" -c "import embodichain, dexsim, robosynchallenge; print('simulator ok')"
```

若已激活其他位置的官方环境，只需设置 `PYTHON_BIN="$(command -v python)"` 并运行上述导入检查，无需固定为 `EmbodiChain/.venv`。

## 4. 安装 PI0.5 推理环境

确认 `uv` 可用后执行。Docker 中若未安装 uv，先执行第 2 节的 uv 安装命令。

```bash
cd "$POLICY_ROOT"
(
  unset VIRTUAL_ENV CONDA_PREFIX
  export UV_PROJECT_ENVIRONMENT="$POLICY_ROOT/.venv"
  export GIT_LFS_SKIP_SMUDGE=1
  uv sync --frozen --no-dev --python 3.11
)
export PI05_PYTHON="$POLICY_ROOT/.venv/bin/python"
uv pip check --python "$PI05_PYTHON"
"$PI05_PYTHON" -c "import jax, openpi; print('jax', jax.__version__)"
```

`pyproject.toml` 与 `uv.lock` 锁定本 policy 的依赖，使用 Python 3.11、JAX 0.5.3（CUDA 12）、PyTorch 2.7.1。`requirements.txt` 引用 `runtime-requirements.txt`（从 lock 导出的第三方依赖，不含本地 OpenPI/workspace 包），完整安装请使用上面的 `uv sync`。本流程无需安装训练数据集或 `rlds` 依赖组。

**无需再次 `source pi05_v2/.venv/bin/activate`。** 保留仿真环境，`eval.sh` 会通过 `PI05_PYTHON` 自动启动独立的推理进程。Docker 内的 PI05 环境也应在容器内安装。

## 5. 下载 checkpoint

模型仓库链接及只读 token 见提交页面。一个 Hugging Face 模型仓库包含 10 个同名任务目录。按提示输入仓库 ID（网址中 `huggingface.co/` 后的 `用户名/仓库名`）和 token；token 不回显、不写入 README。

```bash
export MODEL_ROOT="$ROBOSYN_WS/checkpoints/pi05_v2"
mkdir -p "$MODEL_ROOT"
read -r -p "Hugging Face 仓库 ID（用户名/仓库名）: " HF_REPO_ID
export HF_REPO_ID
"$PI05_PYTHON" - <<'PY'
import getpass
import os
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=os.environ["HF_REPO_ID"],
    local_dir=os.environ["MODEL_ROOT"],
    token=getpass.getpass("Hugging Face 只读 token（公开仓库可留空）: ") or False,
)
PY
```

下载后目录结构应为（不要在任务目录下再套一层 checkpoint 目录）：

```text
checkpoints/pi05_v2/
├── click_bell/
│   ├── params/
│   └── assets/<asset_id>/norm_stats.json
├── drawer_open_place/
│   ├── params/
│   └── assets/<asset_id>/norm_stats.json
└── ...其余 8 个任务同样结构
```

归一化统计始终从**当前 checkpoint** 的 `assets/*/norm_stats.json` 加载。`<asset_id>` 保留模型原目录名，无需将 `robosyn_robotwin_piper_sf` 改名。

## 6. 完整评测命令

新终端先执行第 3 节，选择 Docker 或本地仿真环境，再执行以下命令。两个环境的 Python 都在此明确指定；PI05 推理进程由评测脚本自动启动。

```bash
cd "$ROBOSYN_ROOT"
export PYTHON_BIN="$(command -v python)"  # 当前已激活的官方仿真环境
export PI05_PYTHON="$POLICY_ROOT/.venv/bin/python"
export MODEL_ROOT="$ROBOSYN_WS/checkpoints/pi05_v2"
export GPU_ID=0  # 改为空闲 GPU 编号；Docker 使用容器内可见编号

bash "$POLICY_ROOT/eval.sh" click_bell          random "$MODEL_ROOT/click_bell"          "$GPU_ID" --max_episodes 20 --headless true --seed 0
bash "$POLICY_ROOT/eval.sh" handle_basket       random "$MODEL_ROOT/handle_basket"       "$GPU_ID" --max_episodes 20 --headless true --seed 0
bash "$POLICY_ROOT/eval.sh" water_pouring       random "$MODEL_ROOT/water_pouring"       "$GPU_ID" --max_episodes 20 --headless true --seed 0
bash "$POLICY_ROOT/eval.sh" table_rearrangement random "$MODEL_ROOT/table_rearrangement" "$GPU_ID" --max_episodes 20 --headless true --seed 0
bash "$POLICY_ROOT/eval.sh" items_handover      random "$MODEL_ROOT/items_handover"      "$GPU_ID" --max_episodes 20 --headless true --seed 0
bash "$POLICY_ROOT/eval.sh" drawer_open_place   random "$MODEL_ROOT/drawer_open_place"   "$GPU_ID" --max_episodes 20 --headless true --seed 0
bash "$POLICY_ROOT/eval.sh" mixer_operating     random "$MODEL_ROOT/mixer_operating"     "$GPU_ID" --max_episodes 20 --headless true --seed 0
bash "$POLICY_ROOT/eval.sh" item_assembly       random "$MODEL_ROOT/item_assembly"       "$GPU_ID" --max_episodes 20 --headless true --seed 0
bash "$POLICY_ROOT/eval.sh" manipulate_pipette random "$MODEL_ROOT/manipulate_pipette"   "$GPU_ID" --max_episodes 20 --headless true --seed 0
bash "$POLICY_ROOT/eval.sh" sample_loading     random "$MODEL_ROOT/sample_loading"      "$GPU_ID" --max_episodes 20 --headless true --seed 0
```

可单独执行某一行；全部执行时依次评测 10 个任务。每个任务使用对应目录中的 checkpoint。`eval.sh` 自动设置 `EMBODICHAIN_SIM_EXIT_PROCESS=0` 以完成指标保存。首次运行会加载权重并进行 JAX 编译，请等待日志出现 worker ready 和 episode 进度。

## 7. 评测输出

终端打印每轮结果、最终成功率和指标路径。结果保存在：

```text
$ROBOSYN_ROOT/eval_result/<任务名>/pi05_v2/random/<train_config>/<model_name>/<时间戳>/
├── evaluation_metrics.json
└── videos/
```

`evaluation_metrics.json` 包含成功率、各轮 seed、动作步数和推理耗时；`videos/` 保存各轮视频。默认配置开启专家可行性筛选，筛除的 seed 也会记录在指标中。
