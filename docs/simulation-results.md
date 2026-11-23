# Wonder Duck 仿真实验记录 · 2026 年 11 月 21 日

[34 秒实录](../runs/showcase/wonder-duck.mp4) · [影片章节及对应种子](../runs/showcase/chapters.json) · [48 项任务单](task-catalog.md) · [MuJoCo 资料包](../runs/downloads/Wonder-Duck-MuJoCo.zip)

## 桌面叼放和推送

每组实验使用相同的桌面、两块 12 mm 立方体、绿垫与初始关节扰动。红蓝两种指令各 50 组，种子为 20500–20599，摆位逐组变化。桌面台架给躯干提供支撑。头部 RGB 相机分辨率为 192×144，用颜色掩膜定位目标和绿垫，再用相机运动学及已知桌面高度将像素投影到桌面。程序根据检测结果规划嘴尖轨迹，用颈部关节反馈跟踪，撤回后再拍照。真实位姿供评分器比较，算法仅验证了这个标定环境下的彩色积木。

| 实验 | 通过 | 测量 |
|---|---:|---|
| 根据指令定位红块或蓝块 | 100/100 | 平面定位误差中位数 1.91 mm，门槛 5 mm |
| 嘴端接触后启用模拟夹持，搬至绿垫再释放 | 96/100 | 抬升中位数 24.40 mm，落点中心误差中位数 12.13 mm |
| 同一套动作，只保留碰撞接触 | 0/100 | 抬升中位数 0 mm，最终中心误差中位数 49.57 mm |
| 从侧面用嘴推到绿垫 | 11/100 | 最终中心误差中位数 49.92 mm |

叼放判定要求目标抬升至少 15 mm，释放后在桌面静止至少半秒，中心距绿垫中心不超过 15 mm，另一块积木在过程中移动不足 4 mm。公开 MJCF 的嘴部是固定刚体。辅助组在嘴端接触时启用 MuJoCo 点约束，靠物理求解搬运，释放时解除。真机夹持机构和本批裸模型的对应关系待实物验证。接下来的视觉推送优先解决失败后找回目标和第二次修正推向。

[400 条配对数据](../runs/table-paired/episodes.csv) · [四组汇总](../runs/table-paired/summary.json) · [辅助叼放实录](../runs/table-paired/videos/table_fixture-20500.mp4) · [同摆位纯接触实录](../runs/table-paired/videos/table_contact-20500.mp4) · [辅助组的失败实录](../runs/table-paired/videos/table_fixture-20501.mp4)

## 新实验 · 每推一段就抬头看

11 月 21 日晚，第 19 项在 100 个新的桌面摆位上通过了 98 次。当鸭子撤回嘴尖后，它用头部 RGB 重新定位目标，再决定下一段的推向。每段最多推 25 mm，每回合最多四段。躯干由台架支撑，三组实验都只通过嘴部与积木的物理接触进行推送。[15.6 秒同摆位对照，左为原方法、右为新方法](../runs/push-feedback-20261121/review/paired-44100.mp4)

| 推送方法 | 成功回合 | 落点误差中位数 | 回合时长中位数 |
|---|---:|---:|---:|
| 原来的一次长推 | 11/100 | 53.1 mm | 5.84 秒 |
| 分段推，只在开始拍照，根据已发指令估计后续位置 | 0/100 | 32.6 mm | 10.46 秒 |
| 分段推，途中复查并修正方向 | **98/100** | **5.1 mm** | **14.84 秒** |

开发阶段在种子 310–319 上比较了六组嘴尖偏移、高度和接近距离。冻结参数后用 44100–44199 做独立测试，每个种子三种方法的目标块、干扰块、绿垫和初始关节扰动相同。原方法成功的 11 例也全部通过新方法，另有 87 例从失败变为成功。两种分段方法使用相同的嘴尖几何和退回动作。只看首帧的方法假设积木每次都按指令走足推程，全部 100 例都高估了进度。

沿用原评分，要求方块在绿垫中心 15 mm 内停稳，干扰块移动不足 4 mm，并通过关节及数值检查。反馈组的两个失败例都用完四次推送，最终误差分别为 20.26 mm 和 15.76 mm。三个实验组均无跌倒、关节违规或非有限值，夹持约束启用回合数均为零。[失败例一](../runs/push-feedback-20261121/review/table_push_feedback-44179.mp4) · [失败例二](../runs/push-feedback-20261121/review/table_push_feedback-44185.mp4) · [300 条回合记录](../runs/push-feedback-20261121/episodes.csv) · [配对与录像校验](../runs/push-feedback-20261121/paired-checks.json)

## 点球和抗推

固定练习场的球心在前方 18 cm、偏右 4 cm，门线位于前方 35 cm。裁判在每个 5 ms 物理步检查指定脚与球的接触、过线插值位置和球的高度。全球从两柱内通过，动作结束后鸭子站稳，且通过限位及数值检查，才计分。机器人策略接收原有的观测和命令。

在开发种子 401–404 上搜索了 54 种助跑时序。冻结的组合使用 0.14 m/s 的前进指令 0.56 秒，停 0.24 秒，再运行原有右脚踢球策略。基线直接从同一球位踢球。在 18400–18499 这 100 个保留种子上，助跑 **94/100**，原地 **0/100**。助跑组终止跌倒、限位违规和非有限数值均为 0。结果适用于这个固定练习场和 ±0.001 rad 的初始关节扰动。官方 ONNX 权重用于推理，本次优化对象为助跑控制参数。

[参数搜索](../runs/penalty-study/search.csv) · [冻结的控制器](../configs/penalty-approach.json) · [100 对点球的验收结果](../runs/penalty-study/penalty-metrics.json) · [进球回放](../runs/penalty-study/videos/penalty_approach-18400.mp4) · [未通过回放](../runs/penalty-study/videos/penalty_approach-18405.mp4)

抗推实验在站立状态下对躯干施加 0.12 秒的水平力，方向随机、大小在 0.75–1.8 N 之间。100/100 个种子在终段站稳且水平位移小于 8 cm，关节限位违规和终止跌倒均为 0。受控推倒再起身、边走边恢复各有独立的后续任务。[抗推统计](../runs/penalty-study/push-metrics.json)

## 六个官方策略和本地运行条件

16 类基础任务各跑了 10 回合。站立、前进、后退、左右横移、左右转身均为 10/10。左右脚近球练习分别为 10/10 和 9/10。起坐和空载俯身点地各完成了 10/10 次目标动作，由于关节角超过模型范围，两项的完整验收均为 0/10，最大越限分别为 1.787° 和 0.426°。限位按 50 Hz 采样，容差 0.0001 rad，次数按每个超限关节在每个采样时刻分别计数。[逐回合数据](../runs/validation-20261121/episodes.csv) · [汇总和 16 段录像](../runs/validation-20261121/summary.json)

实验使用 MuJoCo 3.3.7 和 ONNX Runtime 1.23.2。机器人动力学用 200 Hz，关节控制用 50 Hz。物理和 ONNX 推理运行在 CPU。主机提供 NVIDIA GeForce RTX 5060 Laptop GPU，批量桌面相机和回归录像用 NVIDIA EGL 渲染。早期诊断及部分重新录制的影片在 Mesa EGL 下渲染。可视模型来自官方 GLB 的简化网格，场景添加的台架、相机内参和绿垫为本地实验配置。

本地资料包附带原始 MJCF、19 个依赖网格、两份 GLB、运动学文件、6 个 ONNX 策略及配置。30 份原始文件和压缩包内的副本均逐项校验了 SHA-256。整包 SHA-256 为 `43360994c7d39c9220f00ed27489b94f8316765c2d4db0c1372c88a98476428e`。另存有 WonderRL 0.1.32 Windows 安装程序。对应 SKU 的完整可编辑 PPO 工程、训练检查点和真机部署映射仍待取得。幻尔说明购机用户可通过官方客服或技术售后群领取学习资料。[资源清单及出处](official-assets.md) · [厂家说明](https://forum.hiwonder.com.cn/t/wonderduck/1560)

## 夜间队列

按北京时间 22:00 到次日 07:00 运行，目录日期取结束当天。夜间任务按轮切换 18 项仿真，同轮任务使用相同种子，下一轮更新种子。每个回合完成后同步写 CSV 并更新状态。每晚给每项任务留一段录像，再间隔抽样，最多 64 段。剩余磁盘空间低于 2 GiB 时结束。机器睡眠会暂停计算，后续每天的调度需要本机及桌面应用可用。

11 月 21 日 07:00，上一批完成 11,046 个回合并保存了 39 段视频。11 月 21 日 22 时启动下一批时，自动审批拦下了连续队列和低频后台采样，要求用户明确授权后台资源占用及项目写入。本节的桌面推送对照由已完成的有限前台批次产出。[上一批结束状态](../runs/nightly/2026-11-21/status.json) · [上一批逐回合数据](../runs/nightly/2026-11-21/episodes.csv) · [汇总与视频清单](../runs/nightly/2026-11-21/summary.json)

在项目根目录执行 `tools/start-night` 可以启动或续跑，已有实例会继续工作。`status.json` 显示进程、最近写入时间和已完成数。若只想停止本夜，在对应夜间目录创建一个名为 `STOP` 的空文件，运行器会在当前回合落盘后结束。后续夜间计划可在应用中暂停“Wonder Duck 夜间仿真与任务单”。

## 复现

以下命令会写入新的输出目录，保留本次记录。

```bash
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1
.venv/bin/python tools/table_study.py --output runs/replay-table --seeds 100
.venv/bin/python tools/run_suite.py --output runs/replay-feedback --tasks table_push table_push_initial_image table_push_feedback --episodes 300 --seed 44100 --video-every 10007
.venv/bin/python tools/calibrate_kick.py --output runs/replay-penalty
python3 tools/evaluate.py penalty_kick runs/replay-penalty/candidate.csv --baseline runs/replay-penalty/baseline.csv
python3 tools/evaluate.py push_recovery runs/replay-penalty/push.csv
.venv/bin/python tools/run_suite.py --output runs/replay-batch --episodes 160 --exploratory --video-every 10007
```
