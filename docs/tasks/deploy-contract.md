# 可上真机的训练契约（deploy contract）

2026 年 11 月 23 日。本篇记录 `ducklab/deploy/` 训练栈对准的真机接口契约。
所有数字转录自官方交付包 `vendor/official-handoff-20261113/`（只读快照），
不许"顺手改进"：观测错一个下标不会报错，只会造出一个看起来正常一上机就倒的机器人。

## 契约本体

| 项 | 值 | 来源 |
|---|---|---|
| 观测 | `obs[1,61]`：陀螺仪 3 + 重力投影 3 + 关节位置（相对 home）14 + 关节速度 14 + 上一步动作 14 + 指令 13 | `reference/robotd/duck-control/src/obs.rs` |
| 动作 | `actions[1,14]`，**目标 = home 姿态 + 1.0 × 动作**，跳过嘴部 | `reference/robotd/robotd/src/control.rs` |
| 关节顺序 | 左腿 20–24、颈/头 30–33、右腿 10–14；嘴部 34 不在策略内（MOUTH_INDEX=9） | `duck-control/src/model.rs` |
| 控制 | 50 Hz 策略 / 5 ms（200 Hz）物理，每 4 步推理一次 | `robotd.toml` + 训练元数据 |
| 舵机 | 纯 P，kp=15，±2 计数死区，目标斜率 ≤100 rad/s，电压 7.7 V | `fitted_control.py` + `identified.json` |
| 延迟 | 指令延迟 3–6 个物理步（每回合采样）；执行器内部 3.03 ms | `DELAY_STEPS['official']` |
| 指令块 | vx/vy/vyaw + 颈/头四轴绝对角 + body z/roll/pitch；body x/y/yaw 恒 0 | `obs.rs` 命令块注释 |

## 感知与嘴部的分工（和真机一致）

- **策略看不见物体**。61 维里没有相机。物体位置由外部跟踪器（原厂取色追踪页或自研视觉节点）
  转成指令块：扭转向量 + 头部目标角。仿真里用「带噪声和 2 tick 延迟的 oracle」模拟这个跟踪器。
- **嘴部是独立总线舵机 34**，不在动作空间里。咬合由规则控制器 `BeakController`
  （靠近张嘴、到位闭嘴、夹住保持）驱动，和真机上用 GOAL_POSITION 写舵机 34 一一对应。
  嘴部行程：编码器 2048（闭）→1600（开），39.375°，正角度打开。

## 物理与执行器模型

- 物理模型用官方 identified `robot.xml`（124 网格、200 Hz implicitfast、地面摩擦
  `1 0.005 0.0001`），嘴部关节和上下喙触点为本训练栈新增（官方 XML 里嘴部是刚性视觉体）。
- 执行器是官方拟合参数（`identified.json`，m6/servo_p：kt、R、armature、误差增益 0.2183、
  Stribeck/粘性/二次摩擦）的参数级移植：**不是 BAM 库的逐位实现**。静摩擦写入 MuJoCo
  `dof_frictionloss`（约束式，200 Hz 下不抖振），速度阻尼显式施加。
- 部署结论依赖的是契约（观测/动作/频率/增益/延迟），执行器近似是明确的 sim-to-real 风险，
  靠现场验收闭环。

## 训练入口

```bash
# 叼起林里小鸭：走近、低头、闭嘴、抬起并保持 1 秒
.venv/bin/python tools/train_deploy.py --task duck_lift --episodes 8000 --workers 12

# 抛球/抛方块接回：从 0.9–1.6 m 外抛向身前，鸭子往前走迎击，触地前咬住并保持
.venv/bin/python tools/train_deploy.py --task toss_catch --episodes 8000 --workers 12
# --object cube 可切换抛方块
```

流程：官方行走策略行为克隆预热（给出站立/行走先验）→ PPO 微调（同一 61→14 契约）。
输出 `policy-best.pt` / `policy-best.onnx` / `training.json` / `evaluations.json`。

导出与上机：

```bash
.venv/bin/python tools/export_deploy_policy.py <run>/policy-best.pt --task toss_catch
```

产出 `policy.onnx`（opset 18，obs[1,61]→actions[1,14]，与 robotd 策略槽同构）和
`deploy-manifest.json`（关节表、home、增益、延迟、嘴部与指令块映射、校验值）。
经 Studio「模型管理」页替换对应槽位即可，**替换前记录原模型的名称与 SHA-256**。

## 内存与并行

官方 identified 模型单份约 4.8 GB（264 万网格顶点）。训练时**一个 mjModel 跨 worker
共享**、每个环境只持自己的 `MjData`（`DeployEnv(..., shared=...)`）：12 worker 约 5 GB。
不要给每个 worker 单独编译模型（12 份 ≈ 58 GB，会被 OOM 杀）。

## 已知边界

- 嘴部在仿真里是新增关节 + 球形触点；官方物理没有可动喙。咬合几何（间隙约 4 mm、
  张开约 22 mm）与实物 HX-08LCM/34 号舵机的实测行程需现场核对。
- 抛接任务的成功判据是「触地前喙部接触并保持 1 秒」，不要求特定抓取姿态。
- 训练用 CPU torch；策略网络很小，GPU 不是瓶颈（MuJoCo 物理才是）。
- `discardvisual`/`maxhullvert` 不能用于本模型：大部分碰撞几何 contype=0，靠 contact
  exclude 过滤，丢弃视觉会连碰撞一起删掉。

## 阻塞记录（2026-11-23 深夜）：执行器移植未复现实机动态

第一轮 8000+8000 回合训练（`runs/deploy/duck-lift-*`、`runs/deploy/toss-catch-*`）
**全部失败**：评估 0/6，所有回合在 0.7 秒内跌倒，策略连站立都没学会。

**根因不在策略，在环境**：官方自带的站立/行走 ONNX（真机上验证好用）在本仿真里
**1 秒内必倒**——证明环境没有复现 BAM 执行器的真实动态。已尝试并排除的组合：

| 尝试 | 结果 |
|---|---|
| 拟合软增益（V=kp×0.2183×error）+ 拟合静摩擦 0.517 N·m | 关节被摩擦焊死，0.5 s 只动 0.02 rad |
| 增益 1.0 + 同上摩擦 | 仍跟不上，官方策略 0.7 s 倒 |
| 增益 1.0 + XML 原生摩擦 0.0132 | 官方策略 0.9 s 倒 |
| 完全绕过自研伺服，用 XML 自带 kp=50 位置执行器 | 官方策略 0.5–0.9 s 倒 |
| 负载比例摩擦（load_friction∝电机力矩） | NaN，不稳定 |

**结论**：BAM 行为执行器（`bam.actuator` + `bam.model`）不在 PyPI、mjlab 1.6.1
也不包含它；仅凭 `identified.json` 的参数无法无歧义地重建其动态（`load_friction_*`
的物理含义有多解）。在执行器闭环之前，训练出的策略不具备任何上机意义。

**下一步（按优先级）**：
1. 用 mjlab 1.6.1（PyPI 可得）的执行器框架重写 `ducklab/deploy/servo.py`：
   理想位置伺服（kp/kv/effort 限幅）+ DOF 级摩擦/阻尼 + 3–6 步指令延迟，
   以「官方站立策略在本仿真连续站立 ≥ 30 s」为验收标准。
2. 寻找 BAM 源码（mjlab GitHub 组织/论文附带仓库）；若可得则直接集成。
3. 执行器达标后重跑训练；不达标不上真机、不出结论。

**同时可并行推进（零风险）**：真机遥测 61 维观测对账、Studio 读嘴部舵机 34 行程、
IMU 校准核对、相机取帧——见 docs/cross-computer-validation-20261123.md。
