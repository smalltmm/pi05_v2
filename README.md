# pi05_v2

这是一个独立的 PI0.5 policy 仓库，包含模型推理适配器、OpenPI 源码和训练入口。仓库不包含 Python 虚拟环境、CUDA wheel、checkpoint、训练数据或本地缓存。

官方部署流程要求 policy 位于 RoboSynChallenge 的 policy/pi05_v2 下。本仓库也支持独立 clone，然后通过 ROBOSYN_ROOT 指向官方 RoboSynChallenge 工作区。

## 1. 准备官方仿真环境

先按照 RoboSynChallenge 官方安装说明准备仿真环境，并记录官方工作区路径：

~~~
export ROBOSYN_ROOT=/path/to/RoboSynChallenge
source "$ROBOSYN_ROOT/deployment/local-env.sh"
~~~

local-env.sh 是 RoboSynChallenge 仿真环境的启动脚本，不属于本仓库。它负责 EmbodiChain、DexSim、FFmpeg 和仿真资源的路径设置。

## 2. 安装 pi05_v2 推理环境

本仓库不提交环境本身，只提交锁定的依赖声明和安装脚本。需要 Python 3.11、uv，以及能访问 PyPI 的网络：

~~~
export PI05_ROOT=/path/to/pi05_v2
cd "$PI05_ROOT"
bash setup_runtime.sh
~~~

安装结果位于：

~~~
$PI05_ROOT/.venv
~~~

该环境使用本仓库的 pyproject.toml 和 uv.lock。推理时不使用其他用户的 Python 环境，也不要求 Docker。

如果网络受限，可以先把官方 Linux x86_64 / CPython 3.11 wheel 放入 runtime_wheels/，再运行 setup_runtime.sh；wheel 文件本身没有提交到 GitHub。

## 3. 依赖文件和环境变量

仓库中的依赖文件有明确分工：

| 文件 | 用途 |
| --- | --- |
| pyproject.toml | 项目元数据、直接依赖、Python 版本和 uv workspace 配置 |
| uv.lock | uv 的完整锁文件，固定传递依赖、版本和下载哈希；正式安装使用它 |
| runtime-requirements.txt | 从 uv.lock 导出的带哈希 requirements 文件，适合审计或离线下载 |
| requirements.txt | pip 兼容入口，内部引用 runtime-requirements.txt |
| setup_runtime.sh | 创建本地 .venv、执行 frozen 安装、运行依赖检查并校验 tokenizer |
| runtime_assets/openpi/big_vision/paligemma_tokenizer.model | PI0.5 推理需要的 tokenizer 资源 |

仓库不包含 uv 可执行文件。目标机器需要先安装 Astral uv，并确认版本：

~~~
uv --version
python3.11 --version
~~~

没有 uv 时，可以按官方方式安装：

~~~
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
~~~

也可以不使用安装脚本，手动执行与 setup_runtime.sh 相同的步骤：

~~~
cd "$PI05_ROOT"
unset VIRTUAL_ENV
export UV_PROJECT_ENVIRONMENT="$PI05_ROOT/.venv"
export UV_CACHE_DIR="$PI05_ROOT/.cache/uv"
export UV_LINK_MODE=hardlink
uv venv --python 3.11 .venv
uv sync --frozen --no-dev --python 3.11
uv pip check --python "$PI05_ROOT/.venv/bin/python"
~~~

uv sync --frozen 不会修改 uv.lock。不要使用 uv lock 或不带 --frozen 的同步来改变提交中的锁文件，除非你明确要升级依赖。

当前推理环境的关键版本由锁文件固定，包括 Python 3.11、JAX 0.5.3、Flax 0.10.2、Orbax Checkpoint 0.11.13、NumPy 1.x、PyTorch 2.7.1、Transformers 4.53.2 和 LeRobot 的固定 Git revision。JAX 使用 CUDA 12 wheel；机器仍需要可用的 NVIDIA 驱动。

常用环境变量：

| 变量 | 作用 |
| --- | --- |
| ROBOSYN_ROOT | 官方 RoboSynChallenge 工作区，包含 deployment/local-env.sh 和 scripts/eval_policy.py |
| PI05_ROOT | 本仓库 clone 的路径 |
| PI05_PYTHON | 覆盖 PI0.5 worker 的 Python，默认是 $PI05_ROOT/.venv/bin/python |
| PYTHON_BIN / ROBOSYN_VENV_DIR | 覆盖 RoboSynChallenge 仿真环境 Python |
| OPENPI_ROOT | 覆盖 OpenPI 源码目录，默认是 $PI05_ROOT |
| UV_CACHE_DIR | uv 下载缓存目录，建议放在数据盘 |
| OPENPI_DATA_HOME | OpenPI tokenizer 和运行时数据目录 |
| XLA_PYTHON_CLIENT_PREALLOCATE | JAX 显存预分配开关；显存共享时建议设为 false |

deployment/local-env.sh 只负责官方仿真环境，不会替代 PI0.5 的 .venv。两套环境由 eval.sh 分别调用。

## 3. 准备模型

每个模型目录都必须包含 JAX checkpoint 和对应归一化统计：

~~~
<checkpoint>/
├── params/
└── assets/
    └── <repo_id>/
        └── norm_stats.json
~~~

建议把模型目录按任务名保存，例如：

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

checkpoint 不包含在本仓库中。命令中的 <model_root>/<task_name> 需要替换成实际模型目录。

## 4. 通用推理命令

eval.sh 的参数格式为：

~~~
bash "$PI05_ROOT/eval.sh" \
  <task_name> random <checkpoint_path> <gpu_id> \
  --max_episodes 20 --headless true
~~~

如果 policy 仓库没有放到 RoboSynChallenge 的 policy/pi05_v2 目录，必须设置 ROBOSYN_ROOT：

~~~
export ROBOSYN_ROOT=/path/to/RoboSynChallenge
export PI05_ROOT=/path/to/pi05_v2
source "$ROBOSYN_ROOT/deployment/local-env.sh"
bash "$PI05_ROOT/eval.sh" click_bell random \
  /path/to/checkpoints/click_bell 0 \
  --max_episodes 20 --headless true
~~~

<gpu_id> 是服务器的物理 GPU 编号。脚本会让仿真器使用物理编号，并只将同一编号传给 JAX worker。模型推理使用 pi05_v2/.venv。

## 5. 十个任务的推理命令

假设模型目录为 /path/to/checkpoints/<task_name>，下面每条命令运行 20 个随机 episode。把 0 改成空闲的物理 GPU 编号。

~~~
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

若模型对应的官方评测 setting 是 clear，将命令中的 random 替换为 clear。模型目录名可以直接使用对应任务名，命令中的任务名和 checkpoint 目录必须保持一致。

## 6. 运行结果

官方评测器会在 RoboSynChallenge 工作区写入：

~~~
$ROBOSYN_ROOT/eval_result/<task>/<policy>/<setting>/
~~~

每个 run 目录包含视频和 evaluation_metrics.json。success 表示任务成功；truncated 只表示环境达到时间上限，不表示任务成功。

## 7. 训练代码

本仓库保留 OpenPI 的 JAX/PyTorch 训练入口和 RoboSyn/Robotwin 配置。训练数据、初始权重和训练输出需要在目标机器单独准备，默认路径和环境变量说明见 LOCAL_SETUP.txt。
