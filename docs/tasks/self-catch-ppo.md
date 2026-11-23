# 小黄鸭：自抛、前追、空中接回

这是独立的 MuJoCo 仿真技能。原有的「看见外部投掷物、等它落地、捡入浅箱、回到原位」仍是另一个任务。这里不调用它的回合控制器，不升级其默认策略，也不把地面拾球计入空中接球成绩。

真机可做的检查、迁移失败的录像与留出成绩见[现场验收单](self-catch-field-validation.md)。

## 录像和结果

- [六只鸭子的实际训练采样（11.2 秒）](../../runs/self-catch-ppo/delivery/training-six.mp4)：第一阶段第一批与第四阶段最后一批的六路轨迹，1/4 倍速，包括探索动作。
- [同一起点的训练前后对照（7.52 秒）](../../runs/self-catch-ppo/delivery/paired-catches.mp4)：最终保留组的首个种子 990000，原速后接 1/4 倍速，带鸭嘴特写。
- [全部 100 对回合及策略指纹](../../runs/self-catch-ppo/short-hop-04/assessment/paired.json) · [最终策略](../../runs/self-catch-ppo/short-hop-04/selected.pt) · [最终阶段训练日志](../../runs/self-catch-ppo/short-hop-04/training-report.json)

先在 24 个独立验证种子上，按严格成功数、再按平均奖励选择检查点；冻结后才使用种子 990000 至 990099。最终两组均用同样的 5 毫秒鸭嘴时序接口和初始随机条件。对照使用第一阶段未训练网络，两组均取策略均值，不挑动作。

| 100 个新起点 | 严格通过 | 球碰地 |
|---|---:|---:|
| 训练前 | 0 | 100 |
| 四阶段 PPO 训练后 | 100 | 0 |

这是近距离短抛，不是长距离奔跑。成功回合中位数：自由飞行 **95 毫秒**；离嘴后球前移 **29.1 毫米**、再升高 **15.1 毫米**，同期鸭身前移 **9.7 毫米**。练习球直径 20 毫米、质量 2.5 克，回合从球已在嘴中开始，初始含球位置与关节姿态有小幅扰动。没有测试不同球重、长距离、高速跑或真机。策略读取仿真测得的球位置和速度，而不是用 RGB 视觉估计球路。

## 动作、奖励与验收

每回合策略决策两次：先设置甩头动作、张嘴时机和起步时机，再根据当时的球相对位置、速度与关节状态设置截接姿态、闭嘴时机和前进速度。官方走路/站立 ONNX 冻结；上下嘴唇的开合与摩擦来自实验模型。球只在初始化时放入鸭嘴，后续无传送、外加初速度或抓持约束。

六个独立 CPU 进程各自运行 MuJoCo，上层使用 PyTorch PPO-Clip、价值网络与 GAE，从当批带探索的回合更新神经网络。后续阶段用上阶段**训练**日志中的动作初始化均值，再继续 PPO。地面参数搜索不在这 24,960 个 PPO 训练回合内。四阶段分别为 6,912、5,184、6,720、6,144 回合；最终阶段把鸭嘴时序细化到每个物理步，验收阈值没有降低。

独立裁判每 5 毫秒读取接触对及位姿，要求甩头前已连续双唇夹持、球无支撑上升再下降、自由飞行至少 80 毫秒、离嘴后升高至少 12 毫米、球前移至少 15 毫米、鸭身同期前移至少 5 毫米。球下落时须再次触嘴；到回合结束前连续 250 毫秒双唇夹持，全程未触地或撞到其他部位，鸭身保持站立。短暂碰嘴、弹地后再夹或只把球含在嘴里均不算成功。

训练时写入的原始状态用于录像，编码时逐帧比对球位和裁判；MP4 已全片解码，见[训练录像清单](../../runs/self-catch-ppo/delivery/training-video.json)和[配对录像清单](../../runs/self-catch-ppo/delivery/comparison-video.json)。第三阶段 70 次更新与采样轨迹均已保存；该阶段末的保护清单检测到旧 `ducklab/retrieval/episode.py` 在运行期间发生独立变化，因而没有签发该阶段的汇总报告。未回滚这处变化。最终阶段的 18 个受保护文件起止哈希一致。

## 复现

从项目根目录运行，`OUT` 必须是新实验路径；训练脚本拒绝写入已有训练目录。可复用本机现有的 PyTorch/MuJoCo 环境，无需第三方 PPO 库。下面的前四条训练命令各创建一个新目录；最后使用 `d` 的冻结策略和 `a` 的未训练策略做同种子对照，导出两个视频。

```bash
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MUJOCO_GL=egl
PY=runs/fetch/train-env/bin/python
OUT=runs/self-catch-ppo-repro
T=tools/train_self_catch.py
"$PY" "$T" --out "$OUT/a" --workers 6 --updates 72
"$PY" "$T" --out "$OUT/b" --workers 6 --updates 54 --seed 5538 --first-seed 150000 --validation-seed 840000 --clearance --bootstrap-log "$OUT/a/episodes.jsonl"
"$PY" "$T" --out "$OUT/c" --workers 6 --updates 70 --seed 6612 --first-seed 180000 --validation-seed 860000 --clearance --bootstrap-log "$OUT/b/episodes.jsonl"
"$PY" "$T" --out "$OUT/d" --workers 6 --updates 64 --seed 7241 --first-seed 210000 --validation-seed 880000 --clearance --fine-jaw --bootstrap-log "$OUT/c/episodes.jsonl"
"$PY" "$T" --out "$OUT/d" --workers 6 --assess 100 --test-seed 990000 --baseline "$OUT/a/initial.pt"
"$PY" tools/record_self_catch.py --runs "$OUT/a" "$OUT/b" "$OUT/c" "$OUT/d" --out "$OUT/video"
```
