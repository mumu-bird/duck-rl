# Wonder Duck Simulation Lab

[拾取归位训练与逐物体结果](docs/tasks/retrieval-training-review.md) · [同次投掷视频索引](runs/fetch/retrieval-review/index.html) · [三段对照短片](runs/fetch/retrieval-review/highlights.mp4)

[拾取任务 PPO 实验](docs/tasks/retrieval-ppo.md)

[小黄鸭自抛接球训练和真机只读检查](docs/tasks/self-catch-ppo.md) · [明日现场验收单](docs/tasks/self-catch-field-validation.md)

[LINLEE 小鸭第一阶段实验](docs/tasks/linlee-duck-retrieval.md)

到货前的 Wonder Duck 仿真训练准备工程，附带从幻尔官方在线仿真器取得的机器人模型与基线策略。可视化仿真资产不等同于可修改的 WonderRL 强化学习训练工程。

[音频服务与任务 SDK](docs/audio/README.md) · [全部音频验收清单](docs/audio/acceptance.md) · [源码审阅与实施记录](docs/speaker-development-plan.md)

## 当前状态

- 自定义音频功能已实现：音频库、播放调度、暂停/跳转/循环、中文 TTS、浏览器及机身录音接口、任务 SDK、原厂提示音适配、嘴部包络回调和验收控制台。可安装包在 `dist/`；本机模拟服务及验收证据在 `runs/audio-acceptance/`。实机出声、声卡与嘴部动作仍待连接设备验收。

- [投掷物拾取、入箱、归位](docs/tasks/toss-retrieval.md) 已用头部相机训练并执行 180 个配对回合。开合鸭嘴无约束辅助时，未见过的马克笔通过 6/20，小球 7/20，鸭子玩偶 3/20。附三段物理仿真实录、训练权重和可复现命令。

- 拾取归位新增了独立的 on-policy PPO 训练器：PPO 学习夹取和投放参数，官方低层行走/拾取 ONNX 冻结；MuJoCo 负责物理回合，视觉推理与 PPO 更新使用 CUDA。训练轮次、基线配对评估、检查点和抽样成功/失败 MP4 独立写入 `runs/fetch/ppo-retrieval-*`，结果见[实验说明](docs/tasks/retrieval-ppo.md)。

- LINLEE 小鸭已有独立的 `tools/train_duck_ppo.py` 课程式 PPO 入口：并行 CPU MuJoCo rollout、批量 CUDA actor/critic、真实 PPO-Clip/GAE 更新，以及每回合 MP4、阶段 PNG、轨迹 JSON 和 GPU 指标，结果写入 `runs/fetch/linlee-duck-ppo-*`。旧的 `train_duck_retrieval.py` 仅保留作人工参数网格基线。

- [五米遥控直行测试](docs/tasks/straight-walking.md) 已在 216 个仿真回合比较原策略和两种纠偏。标准平地 0.12 m/s 时，5 米终点的横向绝对偏差中位数由 108.32 cm 降至 0.94 cm。

- 本仓库提供官方资源预检、任务规格和仿真评估工具。
- [48 项任务单](docs/task-catalog.md) 已有 19 项在 MuJoCo 实跑。分段视觉推送在 100 个新摆位上通过了 98 次。[同摆位视频](runs/push-feedback-20261121/review/paired-44100.mp4) · [实验记录](docs/simulation-results.md)
- [34 秒仿真实录](runs/showcase/wonder-duck.mp4) · [官方在线仿真器资源压缩包](runs/downloads/Wonder-Duck-MuJoCo.zip) · [11 月 21 日上午结束的批次](runs/nightly/2026-11-21/status.json)
- `tools/run_suite.py` 会运行官方 ONNX 基线、做接触/跌倒/关节限位验收，并逐轮落盘 CSV；可录制主视角和头部 RGB 画面的 MP4。桌面夹持对照显式区分“模拟夹持器辅助”和“纯物理接触”，不将辅助结果记成真机能力。
- 已下载官方在线仿真器使用的 Wonder Duck MJCF、19 个网格、运动学数据、可视化网格，以及步行、站立、起坐、拾取和左右脚踢球共 6 个 ONNX 基线。文件位于被 Git 忽略的 `vendor/wonderduck/`，约 131 MB。下载来源、版本注意事项和 SHA-256 校验记录见 [official-assets.md](docs/official-assets.md) 与 [wonderduck-assets.sha256](docs/wonderduck-assets.sha256)。
- 已下载幻尔官网列出的 WonderRL 0.1.32 Windows x64 安装包：`vendor/wonderrl/0.1.32/WonderRL-0.1.32-windows-x64.exe`。SHA-256：`7f5f0d7f73953e972b95c4373cecb8d60b3c6ec7fce5362f5a492dc3a216ed29`。该文件受 `.gitignore` 排除，不会进入 Git 提交。
- WonderRL 安装包未运行；InnoExtract 1.9 与 7-Zip 26.06 均无法读取该安装包，因此尚未确认安装包内部是否带有可训练项目。官方在线仿真器公开的模型和 ONNX 策略已下载，但可修改的 PPO 训练工程、训练配置和检查点仍待取得。
- **真机型号待核验：**官方论坛介绍称 15 自由度；当前在线仿真器引用的 `robot.xml` 则含 14 个受控铰链关节、1 个浮动根关节和 14 个执行器。开始训练或部署前，需向幻尔确认该模型对应的 Wonder Duck SKU/硬件版本。

官方入口：

- [Wonder Duck / WonderRL 官方介绍](https://forum.hiwonder.com.cn/t/wonderduck/1560)
- [Wonder Duck 在线模拟器](https://wonderrl.hiwonder.com.cn/WonderDuck/Simulator/)

## 目录

```text
configs/acceptance.toml     评估门槛
docs/official-assets.md     开始训练前的官方资源清单
docs/tasks/penalty-kick.md  固定场景点球任务规格
docs/tasks/push-recovery.md 抗推行走/恢复任务规格
tools/preflight.py          本机环境与官方资源预检
tools/evaluate.py           归一化 episode CSV 评估器
tests/                      标准库单元测试
ducklab/arena.py             任务场景、测量与裁判逻辑
ducklab/bench.py             头部 RGB 与桌面交互原型
ducklab/video.py             EGL 仿真录像与状态叠层
tools/run_suite.py           可恢复的有限/定时批量回归入口
```

GPU 已由主机确认可见（RTX 5060 Laptop GPU）。MuJoCo 接触动力学仍在 CPU；拾取 PPO 的 PyTorch 更新和视觉分割可用 CUDA，EGL 用于视频渲染。其它基线回归与规则控制器不等同于 PPO。

## 1. 查看已下载模型

MJCF 路径：`vendor/wonderduck/simulator/robot/wonderduck/robot.xml`。其相对网格目录 `meshes/` 已下载齐全；运动学数据和可视化资源也保存在同一目录。

如果当前 Python 环境已安装 MuJoCo，可先编译模型确认本机运行时接受该模型：

```bash
python3 -c 'import mujoco; m=mujoco.MjModel.from_xml_path("vendor/wonderduck/simulator/robot/wonderduck/robot.xml"); print(mujoco.__version__, "joint records:", m.njnt, "actuators:", m.nu)'
```

在线仿真器入口：[Wonder Duck Simulator](https://wonderrl.hiwonder.com.cn/WonderDuck/Simulator/)

## 2. 检查训练资源

下载 WonderRL 后，用实际路径运行预检：

```bash
python3 tools/preflight.py \
  --wonderrl-root /path/to/WonderRL \
  --robot-mjcf /path/to/wonderduck.xml \
  --training-project /path/to/WonderRL/training-project \
  --baseline-policy /path/to/official-kick-policy.onnx
```

也可通过 `WONDERRL_ROOT`、`WONDERDUCK_MJCF`、`WONDERDUCK_TRAINING_PROJECT`、`WONDERDUCK_BASELINE_POLICY` 环境变量提供路径。预检只检查路径、MJCF XML 格式和运行它的进程能看到的 GPU/Python 环境；它不会尝试启动未知版本的官方训练命令，也不会把“资源齐全”误报成“已确认兼容”。

预检输出 `ASSETS_READY_PLATFORM_UNVERIFIED` 时，表示资源路径存在；仍需按该版 WonderRL 的官方说明确认操作系统、驱动、CUDA 和训练入口。缺文件时输出 `BLOCKED_MISSING_ASSETS` 及缺项列表。

## 3. 运行测试

工具只依赖 Python 标准库：

```bash
python3 -m unittest discover -s tests -v
```

## 4. 运行基线、训练和评估

先用官方说明运行行走、站立/起身与踢球基线，至少记录 10 个 episode。训练入口、策略初始化方式、观察量/动作量和 ONNX 导出命令必须从下载到的 WonderRL 版本确认；本仓库不猜测这些私有接口。

仿真回归（官方策略 + 少量明确标记的桌面规则原型）：

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 .venv/bin/python tools/run_suite.py \
  --output runs/my-replay --episodes 160 --seed 71000 \
  --video-every 10000 --exploratory
```

这条命令让 16 类任务各跑 10 个种子，输出到 `runs/my-replay/`。桌面定位与推送读取头部 RGB，`table_fixture` 在嘴端接触后开启三维点约束，`table_contact` 提供关闭夹持辅助的对照。18 项完整队列还包含固定球位的助跑点球和原地对照。

夜间入口为 `tools/start-night`，北京时间 22:00 至次日 07:00 内可启动或续跑。运行中逐轮写入 `runs/nightly/`，同一晚重复启动会复用已有实例。停止与复现说明见[夜间队列](docs/simulation-results.md#夜间队列)。

评估器使用归一化 CSV。若 WonderRL 日志格式不同，应先按下表转换，不能把字段名称或单位不一致的原始日志直接当成结果：

```text
task,episode,seed,success,fell,joint_limit_violations,nonfinite
penalty_kick,0,1000,1,0,0,0
```

- `task`：`penalty_kick` 或 `push_recovery`。
- `episode`、`seed`：整数；每个 CSV 内 `episode` 唯一。
- `success`、`fell`、`nonfinite`：`0`/`1` 或 `false`/`true`。
- `fell` 表示 episode 以失控倒地结束，不把任务设计中的受控扰动本身算作失败。
- `joint_limit_violations`：该 episode 的关节限制违规次数。

运行示例：

```bash
python3 tools/evaluate.py penalty_kick results/penalty-kick/candidate.csv \
  --baseline results/penalty-kick/baseline.csv
python3 tools/evaluate.py push_recovery results/push-recovery/candidate.csv
```

评估器会输出 JSON 指标、验收项和候选策略相对基线的成功率差值；未达到门槛时以非零状态退出。它评估仿真日志，不会部署策略到机器人。

## 安全边界

ONNX 文件只有在官方部署流程确认 observation/action 形状、关节映射、归一化和控制频率一致后才可上机。首次真机测试使用软垫、低动态动作、有人可立即停止；未经仿真评估的高动态策略不上机。
