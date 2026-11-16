# 迁移记录（2026-11-11）

基线：`smalltmm/pi05_v2`，提交 `cb28fed0bc3db793ffa2eaf84b27d98cab1fc9a1`。
来源：原 `/data/openpi/pi05_v2` 的本地工作区，包含尚未提交的辅助代码；源目录只读。

| 原模块/内容 | 新位置 |
|---|---|
| deploy_policy.py / pi05_worker.py 的 horizon 修改 | sub_policy/policy/pi05_v2/ |
| grasp_dip、grip_hold、bell_press、basket_dip、handover、relative_handover、sample_placement、item_assembly、pipette_press、water_pouring、model_z_floor、timing | policy/pi05_v2/pi05_assists/ |
| move_left_to_xyz 中的 FK/IK/路径数值计算 | policy/pi05_v2/pi05_assists/kinematics.py |
| SDK 正运动学计算（不含 SDK 设备接口） | pi05_assists/piper_fk.py；保留原 MIT 许可 |
| 实机 monitor 中的辅助调度 | pi05_assists/runtime.py，propose / commit 逐帧协议 |
| 本地 README 的任务参数 | assist_profiles.json / checkpoint-map.example.json |
| piper、camera、monitor、runner、录制、回零、移动命令 | /data/openpi/real_robot/ |
| 实机调用旧辅助模块 | 兼容别名，引用同一份 pi05_assists 实现 |
| 临时相机探测和实机启动脚本 | /data/openpi/real_robot/scripts/ |
| 原日志、视频、缓存、权重 | 留在原位置，未复制到提交目录 |

行为调整：

- 决赛控制固定 10 Hz / 1000 帧，替代部分本地测试命令的 100 Hz / 10000 帧。
- 模型频率与 horizon 逐任务选择，辅助仍可检查完整预测尾部；仅执行选定前缀。
- 硬件反馈读取替换为官方逐帧观测。没有 CAN 时间戳时，以观测调用序号表示不同控制帧；外部适配器也可传入真实的观测序号。
- 执行后明确确认；等待、按压、下探等额外帧计入步数。回合重置会丢弃所有辅助状态。
- 涉及耗时 IK 的按压/插接规划后先输出保持帧，再用新观测检查关节是否漂移，避免使用旧反馈直接推进轨迹。
- 需要计时的纯辅助使用执行帧时钟；实机旧入口仍保留原实时计时。
- 默认 `items_handover` 遵循本地 README 的最后一条无辅助命令，早期辅助版单独保留。
- 校准 JSON 只保留数值参考和必要溯源，删除采样序列、CAN 端口和原绝对路径。完整原始校准仍保存在实机目录。

验证：提交侧 176 项离线测试通过；实机侧 462 项回归用例覆盖（迁移后单独更新了测试的 IK mock 目标和校准溯源路径）。100 组随机关节位姿的纯 FK 结果与原 SDK 数值实现逐元素相等。OpenPI 推理依赖导入成功；Python / Shell 语法检查通过。原目录 2234 个非 Git 文件的 SHA256 与迁移前快照一致。

未进行：权重推理、真实机器人执行、主办方未公开决赛接口联调、Git 提交/推送或比赛网站提交。

`/tmp/pi05-final-migration/clone-before-migration` 保存新克隆仓库中被修改或移除文件的安装前副本。原始工程未修改。原始文件哈希清单位于实机目录 `migration/source-sha256.json`。


## 辅助方法同步：2026-11-13

同步了原工程在首次迁移后的五个模块：`handover` 接触证据保持和接触丢失恢复、`handover_observation` 交接后右臂观察位及模型右臂输入替换、`assembly_pickup` 抓取后保持并垂直抬升、`item_assembly` 相对对准/偏移及分段插接、`pipette_press` 完整模型前缀稳定性触发与速度控制。新增模块已接入提交版逐帧 propose/commit 流程。最低高度约束也支持移液器任务。

默认配置按用户确认采用本地 README 最后一条完整命令；发生变化的旧配置保存为 `<task_name>_previous_20261111`，其算法实现随本次同步更新。`items_handover` 默认开启交接和观察位辅助，`manipulate_pipette`、`mixer_operating`、`sample_loading` 默认不启用辅助。只同步完整命令，不自动启用 README 中游离的调试参数。权重映射按实际命令中的 checkpoint 路径记录（不是段落中描述的实验名称）。

比赛端继续固定 10 Hz / 1000 步，辅助只读取传入观测、模型预测和固定参考参数。右臂观察位使用的模型等待状态参考来自原代码，物理反馈与模型输入保持分离；记录中保留真实观测。新校准文件剔除了设备端口、原绝对路径及采样状态等硬件元信息。

独立 `/data/openpi/real_robot` 的共享模块调用同步适配新接口；没有连接或控制机器人。原 `/data/openpi/pi05_v2` 的 84 个相关源文件与本次操作前的 SHA256 快照一致。

验证：提交侧 **260 项通过**；独立实机侧 **492 项离线回归通过**；新增代码未定义名称检查、原工程只读校验和实机入口帮助检查通过。未加载权重执行端到端评测，未推送 GitHub。操作前副本及本次测试日志位于 `/tmp/pi05-assist-sync-20261113/`。


## 比赛夹爪动作单位修正

根据比赛接口确认，模型和辅助内部仍使用 0～1 连续开度；`encode_action()` 在 `env.step()` 前仅将索引 6、13 乘以 0.05，转换为单侧夹爪关节位移（米）。0 为闭合，0.05 为最大张开，两指最大间距约 0.10 m。单步 `get_action()` 使用相同输出换算。转换操作复制动作，不修改模型输出缓存、辅助 pending target 或执行确认状态；其他关节保持弧度不变。本地实机 SDK 的总开度毫米换算不受影响。

已检查本地官方环境的 `normalize_robot_joint_data` 观测配置：夹爪观测已归一化为 0～1，因此输入不额外缩放。测试覆盖两个评测分支、单步接口、端点和中间值、裁剪，以及多次调用不会重复缩放。


## 辅助代码与任务命令同步：2026-11-15

以用户本次完整命令为准，源目录 `/data/openpi/pi05_v2/policy/pi05_v2/real_robot` 只读。

- 同步纯数值辅助：grasp_dip/cartesian/runtime、可选 grasp_dip_vertical、grip_hold、model_z_floor、pipette_press、water_pouring；新增 basket_release 和 table_release。保留纯 FK/IK 适配，不引入设备 SDK、相机、CAN、回零、录制或实机执行器。
- 提篮：10 Hz / horizon 16 / 前缀 8；左夹爪阈值 0.5/0.6；移除默认旧抓取下探，加入松爪前沿工具 +Z 前移 50 mm、100 mm/s。
- 桌面整理：松爪前端点 roll 校正并下降至 SDK Z=140 mm、50 mm/s、保持 0.6 s；源代码可选垂直抓取参数同步但默认关闭，因为本次命令未启用。
- 倒水：角度 90°、右夹爪命令上限 0.3，保留原模型松爪意图和辅助最终张开。
- 交接：观察位速度 60°/s；模型改为 SLA-JAX-handover-FP32-28000ft-10k。
- 样品装载：恢复 25 Hz / horizon 50 / 前缀 25、相对交接与固定点放置；使用 water_pouring 权重。保留已有校准数值，本次源校准差异仅为被清理的实机元数据。
- 移液器：完整原始预测块参与稳定触发；61 mm、三角形三点按压、给定 XYZ 偏移；接近速度 180 mm/s / 60°/s，按压速度 32 mm/s / 40°/s。修正与下探准备期间夹持保持器的关闭时机。
- 搅拌：前缀 8，最新完整命令不启用额外辅助；权重更新为 SLA-JAX-Real540-NoSample-FP32-40000ft-14k-20261114。插接按最后命令保留，用户已停止调试。
- 按铃、抽屉的动作配置保持一致；10 个任务的最新模型名、用户实机结果记录分别更新到 checkpoint-map.example.json 和 SUBMISSION.md；不把实机结果当作比赛测试结果。
- 更新前变更任务的 profile 保存为 *_previous_20261113；原 *_previous_20261111 等历史配置继续保留。

比赛执行仍是 10 Hz / 1000 帧。释放路径只在已执行动作确认后推进，可跨推理动作块继续。模型/辅助内部夹爪仍为 0～1，比赛出口只乘一次 0.05。保留用户要求的观测直传，不恢复已删除的观测异常检查、替换、日志或提前结束回合处理。

验证：336 项提交侧离线测试通过；未定义名称静态检查及 eval.sh 语法检查通过。原辅助源目录 132 个文件哈希保持一致；10 个任务使用的实际本地 checkpoint 均含 params/ 和 assets/*/norm_stats.json。未进行实机/仿真回合或权重推理，未复制权重、Git 提交/推送或网站提交。本次只修改 sub_policy，不同步其他运行副本。

本次修改前副本及源文件哈希位于 `/tmp/pi05-sync-20261115-_7dnctkb`。源文件溯源详见 `migration-provenance.json`。
