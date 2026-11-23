# 拾取归位 PPO

这是拾取任务的强化学习实验入口，与先前“官方 ONNX + 手动选定抓取参数”的 V6 实验分开记分。当前使用 MuJoCo 物理接触、官方低层 ONNX 步态/低头策略冻结，并用 PyTorch PPO-Clip 学习上层夹取及投放参数。它不是 14 个关节端到端 PPO，也不是可直接刷入真机的策略。

## MDP

- 每回合随机投掷落点与朝向并改变背景颜色/光照。第二轮采用课程：先用普通训练物体、窄箱位/摩擦扰动，再加入薄件并逐步扩大随机范围。
- 策略在两个有序决策点各输出 6 维有界连续动作。第一个决定鸭嘴接近距离、闭合时刻、起身时机、扫掠俯仰/颈部和夹取朝向；抓持成立后，第二个决定箱前距离、横向站位、头部俯仰、释放时刻和开嘴速度。
- 夹取与投放使用独立 actor/critic head。观测使用头部视觉定位出的目标位置/面积尺度/轮廓长宽比/轮廓填充率、局部里程计的目标/箱/起点相对位置、目标轴方向、IMU、关节编码器、嘴部状态和阶段计时。MuJoCo 物体真值只供独立裁判和训练奖励使用，不输入策略。
- 训练奖励依据真实接触、抬升、释放时是否仍持有、物体静止留箱、回到初始位置/姿态、误触干扰物和摔倒。PPO 对完整两阶段回报用 GAE 结算。
- 学到的检查点只覆盖上述上层动作。训练任务仍依赖当前视觉模型、定位管线与冻结 ONNX；不能据此推断真机泛化。

## 运行

使用仓库 CUDA 环境和独立的新输出目录。每 24 个回合录制一段训练回放，验证与测试集的 PPO、固定 V6 基线都逐回合录 MP4：

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  runs/fetch/train-env/bin/python tools/train_retrieval_ppo.py \
  --episodes 72 --batch-decisions 64 --eval-trials 2 \
  --record-every 24 --out runs/fetch/ppo-retrieval-round1
```

产物包括 `policy-latest.pt`、按验证集选择的 `policy-best-val.pt` / `policy-final.pt`、`policy-last.pt`、`updates.json`、逐回合训练记录、验证/测试配对 JSON、摘要和按轮次区分的 `videos/`。每 48 回合在验证集做配对比较，盲测只在训练结束并选出验证 checkpoint 后运行。从训练断点续跑时将 `--resume` 指向检查点，输出目录必须与这次 run 一致；脚本拒绝覆盖已有训练记录。

## 评估约定

验证物体用于选轮，最终 split 由命令行指定。成功沿用独立任务裁判：真实抬升、释放时持有、整件物体静止留在箱底、机器人回到起点姿势，且没有误抓干扰物。除完整成功率外，单独报告抬升、留箱、回位和跌倒率；策略不能靠只优化其中一项报喜。`challenge` 默认不参与 PPO 训练，只有显式指定 `--include-challenge` 才加入扩展课程；盲测 `blind*` 仍不参与训练或选 checkpoint。

首次训练是可运行性与学习信号检查，不把少量 episode 结果当作可靠泛化结论。第二轮采用小方差、近零初始化的有界 residual PPO，并分开阶段策略、课程化随机化和周期性验证选模。若配对验证没有稳定超过固定 V6 基线，先按失败分解与视频调整奖励/动作范围/课程，再用新的 checkpoint 目录启动下一轮；不在测试集上反复调参。即使仿真通过，仍须先确认设备 SKU、执行器及传感器接口和厂家部署合同，不能直接上真机。

第三轮从第二轮按验证集选择的权重继续 PPO。课程采用 replay：前 20% 仅易例；20%–50% 按 65% 易例、35% 中等例采样；之后按 45% 易例、35% 中等例、20% 随机化难例采样，避免后半程遗忘已学分布。续训显式重设 optimizer 学习率。检查点选取仍只看验证物体；选 checkpoint 使用一组固定配对种子，最终验证改用另一组种子，测试集只在最后报告。

当前任务已可用多进程同时运行独立的 CPU MuJoCo 场景，并在主进程将目标检测和 PPO 决策合批送至 CUDA。各批场景使用同一份冻结策略，整批结束后才执行一次 PPO 更新；渲染、低层 ONNX 和物理步进仍在 CPU/EGL worker 中。该模式提升了环境吞吐，但还不是 GPU 刚体物理，也不保证显卡持续高利用率。

启动并行训练时设定 `--workers`。RTX 5060 Laptop 已测到 16 worker 稳定完成，正式轮先用 16，不继续增到 24，以留出桌面和显存余量。视频按 `--record-every` 保存 MP4，录像器同时输出初始、发现目标、抓取、释放和终态 PNG。每批记录 episode/s、检测和策略 batch 大小、GPU 利用率/显存快照。

```bash
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  runs/fetch/train-env/bin/python tools/train_retrieval_ppo.py \
  --episodes 384 --workers 16 --batch-wait-ms 50 --batch-decisions 64 \
  --record-every 24 --eval-every 96 --curriculum replay --include-challenge \
  --selection-split blind --final-splits blind2 blind3 \
  --resume runs/fetch/ppo-retrieval-round4/policy-best-val.pt \
  --out runs/fetch/ppo-retrieval-round5-parallel
```

并行 CPU rollout 只能利用 CUDA 上的视觉、动作批推理和 PPO 更新；要进一步提高 GPU 占用，下一阶段需要把场景与状态机迁移至批量 GPU 物理后端，并批量化官方低层控制。迁移前必须对动态鸭嘴、free-joint 物品、接触/摩擦、焊接持有约束、传感器、`mj_jacSite`、奖励裁判和官方 ONNX 每 4 个 5 ms 步推理一次的行为逐项做 CPU/GPU 单回合对齐。不可仅以策略能够运行视为仿真等价。

## 第三轮结果与第四轮

第三轮从第二轮验证 checkpoint 续训 384 回合。最终独立种子验证为 PPO `9/15`、V6 `8/15`；此前指定测试集两者都是 `9/15`。测试逐物体结果为大号记号笔 PPO `3/5` 对 V6 `2/5`、高尔夫球 `5/5` 对 `3/5`、Sony duck `1/5` 对 `4/5`。测试配对回合 PPO 单独成功 4 次、V6 单独成功 4 次，整体仍打平。录像共 166 段。Sony duck 失败多发生在已抬起并持有、但释放后未入箱；不同测试物体下投放动作变化也很小。第三轮不足以证明泛化提升。

第四轮保持 PPO 算法与 6 维动作不变，视觉观测增加分割轮廓长宽比和填充率；从第三轮 checkpoint 扩展输入层时新列置零，并重置 optimizer。训练显式加入 YCB challenge 网格中的香蕉、勺子、杯子，专门增加几何种类；新鲜 `blind` 物体只用于选 checkpoint，最终评估使用不同种子的 `blind` 以及未见过的 `blind2`、`blind3`，不再用已看过的第三轮 test 物体做调参。每个回合额外记录开嘴瞬间物体相对箱体的位置，以便分辨释放偏差和落地滑出。

第四轮完成 384 回合。固定 `blind` 选模的最终评估中，PPO 在 `blind/blind2/blind3` 为 `10/15、4/15、6/15`，V6 基线分别为 `6/15、4/15、5/15`。总体有改善但在不同留出物体上的表现不一致，不能称为稳定泛化。

并行实现验证：2 worker、2 回合成功完成并保存 MP4/PNG；8/12/16 worker 的试跑吞吐分别为 `0.864/1.316/1.739 episode/s`，最大视觉 batch 分别达到 8/12/16，PPO 更新均成功。16 worker 的 PyTorch 分配峰值约 `155 MiB`，`nvidia-smi` 显存采样峰值约 `4.0 GiB / 8.0 GiB`。计算时 GPU 利用率抽样约 `2–10%`，批次末采样更低；因此并行 CPU rollout 已明显提高 episode 吞吐，但 GPU 物理利用率仍待 GPU 物理后端迁移解决。正式并行轮使用第四轮选出的 checkpoint、新输出目录与保留的 `blind2/blind3` 留出集。

## 第五轮结果

16 worker 并行 PPO 完成 384 回合，训练成功 `156/384`（40.6%），抬起率 69.0%、释放时持有率 60.2%、留箱率 41.7%、回位率 78.1%。验证选模选回了初始 checkpoint，而不是任何训练中期权重。blind2 的 PPO 与基线均为 `3/15`；blind3 为 PPO `5/15`、基线 `3/15`。这组留出结果不支持稳定泛化提升，整轮 CUDA 利用率均值为 2.4%。因此不原样追加更多 PPO 回合；优先解决批量 GPU 物理的一致性，再设计下一轮算法对照。[逐回合训练结果](../../runs/fetch/ppo-retrieval-round5-parallel/summary.json) · [blind2](../../runs/fetch/ppo-retrieval-round5-parallel/evaluation-blind2.json) · [blind3](../../runs/fetch/ppo-retrieval-round5-parallel/evaluation-blind3.json)

## MJWarp GPU 物理试验（2026-11-23）

为评估 GPU 物理吞吐，在 `runs/fetch/mjwarp-retrieval-env` 建立独立环境，安装 MuJoCo `3.10.0`、MJWarp `3.8.1`、Warp `1.12.0`；原 `runs/fetch/train-env` 的 MuJoCo `3.3.7` 和训练权重未改动。试验入口为 `tools/benchmark_retrieval_mjwarp.py`，使用训练 split 的 `bath_bird` 场景，每档 400 个物理步；所有场景初态相同于配对 CPU/GPU，控制输入为零，因此这是**物理吞吐/数值差异基准，不是 PPO 训练或任务成功率**。CPU 列是单进程顺序步进，不能当作现有 16 worker rollout 的加速比。

|并行场景|CPU 顺序 env-steps/s|GPU MJWarp env-steps/s|GPU/CPU|终态最大 `qpos` 差|
|---:|---:|---:|---:|---:|
|1|10,873|947|0.09x|0.212|
|8|9,941|7,141|0.72x|0.217|
|16|8,139|11,855|1.46x|0.223|
|32|6,284|23,910|3.81x|0.236|
|64|5,773|39,603|6.86x|0.831|

64 场景的 GPU 吞吐峰值明显高于单线程 CPU；`nvidia-smi` 抽样曾达到 100% GPU 利用率，显存读数约 `380 MiB`。这是短时物理步进测试，采样有 0%–100% 波动，不代表持续满载，也还未与现有多进程 CPU 基线做同口径的闭环任务比较。

低层策略也不能直接合批：当前步态 ONNX 的输入/输出形状固定为 `[1,61] -> [1,14]`，`ducklab/policy.py` 明确选 `CPUExecutionProvider`；本机 ONNX Runtime provider 只有 Azure 与 CPU，没有 CUDA。要在保持低层动作一致的前提下利用 GPU，需另行导出/转换为动态 batch 模型并做逐输出校验，不能改写或覆盖发布的 ONNX。

**当前不能接回 PPO。** 接触后 CPU/GPU 轨迹迅速分化：同一初态首步最大差约 `3e-6`，40 步后最大 `qpos` 差约 `0.020`，400 步下 64 场景的最大差升至 `0.831`；最终平均接触数量约为 CPU `14.03`、GPU `21.89`。生产场景启用 `noslip_iterations=3`、求解容差 `1e-10`；MJWarp `3.8.1` 不支持正数 `noslip_iterations`，且将 GPU 求解容差下限设为 `1e-6`，所以本试验关闭 no-slip 并使用 `1e-6`。该改动会影响夹持摩擦、腿脚支撑和托盘碰撞，不能用当前结果训练或评估任务策略。[MJWarp 3.8.1 源码限制](https://github.com/google-deepmind/mujoco_warp/blob/v3.8.1/mujoco_warp/_src/io.py) · [MJWarp 批量 API](https://github.com/google-deepmind/mujoco_warp/tree/v3.8.1)

MJWarp `3.8.1` 的这轮结果不能接回 PPO。下一步使用较新 MJWarp 测试保留 no-slip 的 GPU 配置，并先验证任务事件、不是只比较自由物体的终态坐标；CPU 16 worker 继续作为训练吞吐和任务成功率基线。

试验摘要：[`summary.json`](../../runs/fetch/retrieval-mjwarp-probe-final-20261123/summary.json) · 视频：[`mjwarp-bath_bird-64world-batch.mp4`](../../runs/fetch/retrieval-mjwarp-probe-final-20261123/videos/mjwarp-bath_bird-64world-batch.mp4) · [接触关键帧](../../runs/fetch/retrieval-mjwarp-probe-final-20261123/videos/mjwarp-bath_bird-64world-batch-frames/01-parcel-contact.png)

## MJWarp 4.0.1 no-slip 吞吐试验（2026-11-23）

为避免影响生产训练，MuJoCo Core `4.0.1`、MJWarp `4.0.1`、Warp `1.19.0` 安装在独立 package overlay `runs/fetch/mjwarp-retrieval-site-4.0.1`；`runs/fetch/train-env` 未更改。新版允许本任务用 `implicit + noslip_iterations=3` 跑 GPU，但不支持生产积分器 `implicitfast`。为配对比较，本试验 CPU/GPU 候选都显式设为容差 `1e-6`，生产配置是 `1e-10`。[上游兼容说明](https://github.com/google-deepmind/mujoco_warp#mujoCo-api-compatibility) 这里的“生产参数 CPU”是在 MuJoCo `4.0.1` 下套用生产求解参数，不是 MuJoCo `3.3.7` 训练环境本身；因此该结果仍非生产构建的一比一复现。

同一 `bath_bird` 模型、相同随机初态、零电机控制，每档步进 400 步。有效档均使用 `nconmax=256`、`njmax=512`，逐世界检查 MJWarp overflow；最大 `nefc` 为 99/120，最大求解迭代 8/9，未触发 overflow。

|GPU worlds|GPU env-steps/s|相对候选 CPU 顺序步进|CPU/GPU 最大 `qpos` 差|平均接触数绝对差|GPU 显存|
|---:|---:|---:|---:|---:|---:|
|128|27,287|2.60x|1.875|1.75|982 MiB|
|256|39,251|3.80x|1.765|1.59|1,686 MiB|

256 worlds 的 timed kernel 期间 GPU 利用率抽样多次达到 100%；这约 2.6 秒的物理微基准不等于训练期间持续满载，也没有和现有 16-worker 完整 episode 吞吐做同口径比较。512 worlds 在相同高容量设置下产生非有限 `qpos`，该档被拒绝。128/256 worlds 的轨迹仍与候选 CPU 不一致；同一 MuJoCo 版本下，候选 `implicit + 1e-6` CPU 相对生产参数 CPU 的 256-world 最大 `qpos` 差约 `1.78`，平均接触数也由 `20.07` 变成 `19.16`。因此**目前仍不能把该后端用于 PPO 训练**。

这段试验只推进了 batched GPU physics 的容量和吞吐验证，不是“256 只鸭子完成拾取任务”的训练结果：256 个相同拓扑世界只做零控制物理步进，保存的视频是其中第 0 号世界的代表性回放。接入 PPO 前还需按顺序完成：

1. 在冻结策略和固定种子下，对生产 CPU、候选 CPU、候选 GPU 做闭环任务配对，记录抓持、抬起、释放、留箱、回位、摔倒与完整成功；覆盖笔状、球状、扁平、异形物体，并要求无 overflow/NaN。
2. 把官方低层 recurrent ONNX 控制器转换为可合批推理的实现，对输入 `[1,61]`、输出 `[1,14]` 与隐藏状态逐项做数值等价校验。按照机器鸭交接资料，当前 ONNX 是静态 batch 1，不能直接把 256 份观测拼起来调用。
3. 按碰撞模型拓扑把不同物品分组进 GPU worlds，再实现批量观测、Episode 状态机和奖励结算；测量口径改成完整任务 episodes/s，并与现有 16-worker CPU rollout 对比。
4. 只有任务事件差异和 held-out 成功率达到预设门槛，才启动 GPU PPO；每轮继续每 24 回合保存 MP4，并保留初始、发现目标、抓取、释放、回位 PNG。训练后的 checkpoint 仍在未参与选模的对象集上评估。

有效产物：128-world [`summary.json`](../../runs/fetch/retrieval-mjwarp-implicit-noslip-128world-capacity512-20261123/summary.json) · 256-world [`summary.json`](../../runs/fetch/retrieval-mjwarp-implicit-noslip-256world-capacity512-20261123/summary.json) · [256-world MP4](../../runs/fetch/retrieval-mjwarp-implicit-noslip-256world-capacity512-20261123/videos/mjwarp-bath_bird-256world-batch.mp4) · [接触关键帧 PNG](../../runs/fetch/retrieval-mjwarp-implicit-noslip-256world-capacity512-20261123/videos/mjwarp-bath_bird-256world-batch-frames/01-parcel-contact.png)
