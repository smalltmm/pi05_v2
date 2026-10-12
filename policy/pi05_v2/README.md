# PI0.5 v2 — RoboSynChallenge 评测部署

本仓库提供 **PI0.5 v2（JAX）Policy** 的推理代码、部署配置、评测脚本和依赖锁定文件。本次提交覆盖 **RoboSynChallenge 全部 10 个任务**，每个任务使用与任务同名的 checkpoint。

- **Policy 代码：** https://github.com/smalltmm/pi05_v2
- **模型权重：** https://huggingface.co/wxhtmo/robosyn_model_v1
- **官方 RoboSynChallenge：** https://github.com/EDEM-AI/RoboSynChallenge
- **本地评测配置：** `random`、每任务 20 episodes、`seed=0`

**部署前提：** 评测服务器已安装并配置好官方 RoboSynChallenge 仿真环境（本提交使用 EmbodiChain `v0.2.4.post1`）。以下仅说明本 Policy 的模型下载、代码放置、推理环境与评测方法，不包含官方仿真环境的安装步骤。

## 1. 下载模型 Checkpoints

模型仓库：https://huggingface.co/wxhtmo/robosyn_model_v1

仓库采用 **Public + Gated Access**：仓库页面公开，但下载 checkpoint 需要访问授权。评测人员应使用**RoboSynChallenge** Hugging Face 账号认证；

### 1.1 登录并下载

请先确保系统已安装 Hugging Face CLI（`hf`），随后执行：

```bash
hf auth login

export HF_REPO_ID="wxhtmo/robosyn_model_v1"
export MODEL_ROOT="/path/to/model_root"

hf download "$HF_REPO_ID" \
    --repo-type model \
    --local-dir "$MODEL_ROOT"
```

`MODEL_ROOT` 是模型保存目录，请替换为评测服务器上的实际路径。`hf download` 将按仓库原有结构下载全部 10 个任务的 checkpoint。

### 1.2 Checkpoint 目录结构

下载后应保留如下目录结构：

```text
/path/to/model_root/
├── click_bell/
│   ├── params/
│   └── assets/
│       └── <asset_id>/norm_stats.json
├── handle_basket/
├── water_pouring/
├── table_rearrangement/
├── items_handover/
├── drawer_open_place/
├── mixer_operating/
├── item_assembly/
├── manipulate_pipette/
└── sample_loading/
```

其余各任务目录同样包含 `params/` 和 `assets/<asset_id>/norm_stats.json`。请勿修改目录层级，也不要在任务目录下再次嵌套 checkpoint 文件夹。推理时归一化统计从**当前任务的 checkpoint** 中加载。

## 2. 获取 Policy 代码

首先确保官方 RoboSynChallenge 仓库已存在于评测工作区。以下目录均为示例，按实际位置替换：

```bash
export ROBOSYN_ROOT="/path/to/RoboSynChallenge_ws/RoboSynChallenge"
export POLICY_ROOT="$ROBOSYN_ROOT/policy/pi05_v2"
```

先克隆到临时目录，再把仓库中的 `policy/pi05_v2/` 复制到官方项目的 adapter 目录，避免生成嵌套路径：

```bash
POLICY_DOWNLOAD_DIR="$(mktemp -d)"
git clone https://github.com/smalltmm/pi05_v2.git "$POLICY_DOWNLOAD_DIR"
mkdir -p "$POLICY_ROOT"
cp -a "$POLICY_DOWNLOAD_DIR/policy/pi05_v2/." "$POLICY_ROOT/"
```

完成后 Policy 路径为：

```text
/path/to/RoboSynChallenge_ws/RoboSynChallenge/policy/pi05_v2/
├── deploy_policy.py
├── deploy_policy.yml
├── eval.sh
├── pyproject.toml
├── uv.lock
├── src/openpi/
└── ...
```


## 3. 安装 PI0.5 推理环境（uv）

**请先激活官方 RoboSynChallenge 仿真环境，这里以本地安装好的RoboSynChallenge 仿真环境为例。**

本 Policy 使用两个 Python 环境，**只需在独立推理环境中安装 OpenPI；官方仿真环境不需要额外安装 OpenPI**。

| 环境 | Python 路径 | 用途 |
|---|---|---|
| 官方仿真环境 | `PYTHON_BIN` | 运行 EmbodiChain / RoboSynChallenge 仿真与 Policy Adapter |
| PI0.5 独立环境 | `PI05_PYTHON` | 安装 OpenPI 及 JAX 依赖，加载 checkpoint 并运行推理 Worker |

### 3.1 创建独立的 PI0.5 环境

在**已激活官方仿真环境**的终端中执行：

```bash
export ROBOSYN_ROOT="/path/to/RoboSynChallenge_ws/RoboSynChallenge"
export POLICY_ROOT="$ROBOSYN_ROOT/policy/pi05_v2"

# 保存已激活的官方仿真 Python 路径
export PYTHON_BIN="$(command -v python)"

cd "$POLICY_ROOT"

(
    unset VIRTUAL_ENV CONDA_PREFIX
    export UV_PROJECT_ENVIRONMENT="$POLICY_ROOT/.venv"
    export GIT_LFS_SKIP_SMUDGE=1
    uv sync --frozen --python 3.11
)

export PI05_PYTHON="$POLICY_ROOT/.venv/bin/python"
```

这一步使用项目的 `pyproject.toml` 和 `uv.lock` 安装独立 PI0.5 推理环境，路径为 `policy/pi05_v2/.venv`。Policy 使用 Python 3.11；依赖的具体版本以锁定文件为准。

### 3.2 验证推理环境与进程分工

检查OPENPI推理环境：

```bash
"$PI05_PYTHON" -c \
"import jax, openpi; print('policy JAX:', jax.__version__)"
```

检查官方仿真环境：

```bash
"$PYTHON_BIN" -c \
"import embodichain, dexsim, robosynchallenge; print('simulator OK')"
```

`eval.sh` 使用 `PYTHON_BIN` 启动仿真评测，使用 `PI05_PYTHON` 启动独立推理进程。

注意：若在 Docker 内评测，PI0.5 的独立 `.venv` 也应在容器内创建，并确保两个解释器路径均可访问。

## 4. 运行全部 10 个任务

评测前**激活官方仿真环境**，并设置以下环境变量（按实际部署位置替换路径）：

```bash
export ROBOSYN_ROOT="/path/to/RoboSynChallenge_ws/RoboSynChallenge"

export POLICY_ROOT="$ROBOSYN_ROOT/policy/pi05_v2"

export PYTHON_BIN="$(command -v python)"

export PI05_PYTHON="$POLICY_ROOT/.venv/bin/python"

export MODEL_ROOT="/path/to/model_root"

export GPU_ID=0

cd "$ROBOSYN_ROOT"
```

其中：

- `PYTHON_BIN`：已激活的官方仿真环境 Python。
- `PI05_PYTHON`：独立 PI0.5 Policy 环境 Python。
- `MODEL_ROOT`：包含全部 10 个任务 checkpoint 目录的根目录。
- `GPU_ID`：推理使用的 GPU 编号，根据实际可用设备修改。Docker 环境使用容器内可见的 GPU 编号。

### 4.1 评测命令

本地评测命令格式：

```bash
bash "$POLICY_ROOT/eval.sh" <task_name> <setting> <checkpoint_path> <gpu_id> [extra_options...]
```

以下分别使用各任务对应的 checkpoint，采用 `random` 设置、每任务 20 episodes、无界面运行、随机种子 0。

```bash
# 1. click_bell
bash "$POLICY_ROOT/eval.sh" click_bell random "$MODEL_ROOT/click_bell" "$GPU_ID" --max_episodes 20 --headless true --seed 0

# 2. handle_basket
bash "$POLICY_ROOT/eval.sh" handle_basket random "$MODEL_ROOT/handle_basket" "$GPU_ID" --max_episodes 20 --headless true --seed 0

# 3. water_pouring
bash "$POLICY_ROOT/eval.sh" water_pouring random "$MODEL_ROOT/water_pouring" "$GPU_ID" --max_episodes 20 --headless true --seed 0

# 4. table_rearrangement
bash "$POLICY_ROOT/eval.sh" table_rearrangement random "$MODEL_ROOT/table_rearrangement" "$GPU_ID" --max_episodes 20 --headless true --seed 0

# 5. items_handover
bash "$POLICY_ROOT/eval.sh" items_handover random "$MODEL_ROOT/items_handover" "$GPU_ID" --max_episodes 20 --headless true --seed 0

# 6. drawer_open_place
bash "$POLICY_ROOT/eval.sh" drawer_open_place random "$MODEL_ROOT/drawer_open_place" "$GPU_ID" --max_episodes 20 --headless true --seed 0

# 7. mixer_operating
bash "$POLICY_ROOT/eval.sh" mixer_operating random "$MODEL_ROOT/mixer_operating" "$GPU_ID" --max_episodes 20 --headless true --seed 0

# 8. item_assembly
bash "$POLICY_ROOT/eval.sh" item_assembly random "$MODEL_ROOT/item_assembly" "$GPU_ID" --max_episodes 20 --headless true --seed 0

# 9. manipulate_pipette
bash "$POLICY_ROOT/eval.sh" manipulate_pipette random "$MODEL_ROOT/manipulate_pipette" "$GPU_ID" --max_episodes 20 --headless true --seed 0

# 10. sample_loading
bash "$POLICY_ROOT/eval.sh" sample_loading random "$MODEL_ROOT/sample_loading" "$GPU_ID" --max_episodes 20 --headless true --seed 0
```

可以单独执行其中任意一行，也可以依次执行全部命令。

每个任务使用 `MODEL_ROOT` 中与任务名称对应的 checkpoint。

首次启动时需要恢复 checkpoint 并进行 JAX 编译，请等待推理 Worker 就绪后再开始评测。

**说明：** 以上是本地复现所使用的评测参数，不代表官方 held-out 测试一定采用相同配置。主办方可以根据赛事评测协议调整场景、episode 数量和随机种子。

### 4.2 模型与任务对应关系

| Task | Checkpoint Path |
|---|---|
| click_bell | `$MODEL_ROOT/click_bell` |
| handle_basket | `$MODEL_ROOT/handle_basket` |
| water_pouring | `$MODEL_ROOT/water_pouring` |
| table_rearrangement | `$MODEL_ROOT/table_rearrangement` |
| items_handover | `$MODEL_ROOT/items_handover` |
| drawer_open_place | `$MODEL_ROOT/drawer_open_place` |
| mixer_operating | `$MODEL_ROOT/mixer_operating` |
| item_assembly | `$MODEL_ROOT/item_assembly` |
| manipulate_pipette | `$MODEL_ROOT/manipulate_pipette` |
| sample_loading | `$MODEL_ROOT/sample_loading` |

## 5. 评测结果

评测程序会在终端报告各 episode 结果、最终成功率以及指标保存路径。

典型结果目录如下，具体以实际日志为准：

```text
$ROBOSYN_ROOT/eval_result/
└── <task_name>/
    └── pi05_v2/
        └── random/
            └── <train_config>/
                └── <model_name>/
                    └── <timestamp>/
                        ├── evaluation_metrics.json
                        └── videos/
```

- `evaluation_metrics.json`：记录成功率、episode 信息、动作步数、推理耗时等评测指标。
- `videos/`：保存评测过程视频，具体取决于评测配置。

本仓库提供的 `eval.sh` 会设置 `EMBODICHAIN_SIM_EXIT_PROCESS=0`，以便在任务结束时保存评测指标。

评测结果的实际保存路径以程序日志输出为准。