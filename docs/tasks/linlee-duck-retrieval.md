# LINLEE 小鸭拾取第一阶段

2026 年 11 月 23 日。这个实验把目标限定为一只小鸭的完整闭环：视觉发现、接近、鸭嘴叼起、携带到托盘、放下、回到起点。多只鸭子的第一版采用逐只批处理，尚未把多目标视觉和同一物理场景中的鸭子互相碰撞纳入成功标准。

## 外观和物理假设

项目里没有 LINLEE 小鸭的实拍、尺寸、重量或鸭嘴间隙测量，所以当前配置是一个可替换的工作假设，而不是 LINLEE 的精确标定：

- 圆润的小型鸭形，约 4.8–5.8 cm 的长向尺寸；
- 光滑或略带柔性的 PVC/橡胶外皮；
- 约 10–34 g 的质量范围；
- 黄色、浅色或粉彩主体，橙色鸭嘴，黑色眼睛；
- 物体与鸭嘴之间的摩擦和接触顺应性做随机化，防止只对某一个硬度和某一个摆放姿态有效。

这组假设覆盖的是“形状相似、颜色和质地有变化”的小鸭家族。拿到实际 LINLEE 样品后，优先测量每只的三轴外接尺寸、质量、最大鸭嘴开口、表面是否有绒布/涂层以及被夹后是否明显变形，再更新 `configs/linlee-duck-variants.json`。

## GPU PPO 训练

最终训练入口是下面的真正 on-policy PPO，不是人工参数网格搜索。每个 episode 从四类程序化小鸭中抽样质量、尺寸、摩擦、软接触和颜色；多个 CPU MuJoCo rollout worker 将动作请求交给 CUDA 上的批量 actor/critic，PPO-Clip 和 GAE 更新也在 CUDA 上执行：

```bash
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
runs/fetch/train-env/bin/python tools/train_duck_ppo.py \
  --out runs/fetch/linlee-duck-ppo-v1 \
  --device cuda \
  --episodes 128 \
  --rollout-workers 8 \
  --batch-decisions 32 \
  --action-batch-size 32 \
  --record-every 1 \
  --eval-every 32
```

训练输出会保留：

- `training-episodes.json`：每回合事件、阶段动作、奖励、失败原因、鸭子物理参数和媒体路径；
- `videos/**/*.mp4` 及同名 `*-frames/*.png`：初始、目标发现、叼起、释放和最终状态；
- `updates.json`：PPO decisions、policy/value loss、KL、更新耗时、CUDA 显存；
- `policy-latest.pt`、`policy-best-val.pt`、`policy-final.pt`：可恢复的 PPO checkpoint；
- `summary.json`、`rollout-batches.json`：GPU 型号、CUDA/PyTorch 版本、rollout 吞吐、GPU update 吞吐和峰值显存。

当前 MuJoCo 3.3.7 的接触动力学仍在 CPU；这里的 GPU 使用边界是视觉推理、PPO actor/critic 前向和 PPO 反向更新。不能把这次实验描述成 GPU 物理仿真，也不能直接把 checkpoint 刷入真机。

## 参数网格基线

下面的脚本只用于人工候选参数的基线对照，不是强化学习：

```bash
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
runs/fetch/train-env/bin/python tools/train_duck_retrieval.py \
  --out runs/fetch/linlee-duck-training \
  --device cuda \
  --candidate-limit 9 \
  --train-trials 1 \
  --validation-trials 2
```

训练结果中的 `frozen.json` 是可传给批处理运行器的上层抓取/投放参数：

```bash
runs/fetch/train-env/bin/python tools/run_duck_batch.py \
  --out runs/fetch/linlee-duck-batch \
  --profiles runs/fetch/linlee-duck-training/frozen.json \
  --device cuda \
  --trials 2 \
  --include-reference \
  --video-all
```

批处理输出的每一行都会记录 `lifted`、`held_at_release`、`in_tray_at_end`、`pose_restored`、跌倒原因和完整 `success`。这些字段分别对应“叼起来”“走到托盘仍然叼住”“物体在托盘内稳定”“机器人回位”和完整任务，不把只抬起一次记成成功。

## 边界

- 这一步训练的是上层抓取和投放参数；官方低层行走、站立、低头 ONNX 策略仍被冻结。
- 当前批处理是“一只鸭一轮”，便于先验证动作。若把多只鸭同时放在相机里，现有 before/after 变化分割会把多个前景合并，不能可靠决定下一只要叼哪一只。
- 真机部署前还需要在实际机位拍摄空场/单鸭/多鸭数据，训练或校准目标选择，并确认嘴部开口、舵机电流/力矩和软物变形。仿真通过不等于可以直接把参数写入设备。
- 下一阶段应先把一只 LINLEE 鸭子的近距离实拍闭环跑通，再加入“逐只送到集合点”的共享场景；最后才加入绕障碍、排队、展示和舞步等动作。
