# WonderDuck 看见新物体，捡入箱后归位

后续训练结果与同次投掷视频已汇总在[拾取训练与视频](retrieval-training-review.md)。以下为第一版基准。

2026-11-22 实验。小鸭自由站在平地，前方有两个原先就存在的物体。它记住起始姿态和头部相机画面。物体出现并连续约 0.6 秒保持在同一位置后，它走近、复查、低头夹取、走到指定浅箱，张嘴让物体落入箱内，然后依据 IMU、编码器与箱沿的视觉位置回到原地。细长目标会触发横向夹取。失手可以回到起点重新找一次。当前每轮一个新物体，箱的位置在起点坐标系中给出。

## 这一版在仿真中能做到什么

冻结视觉权重和抓取参数，三种此前未参与训练的物体各丢 20 次，每次在三种鸭嘴条件复用同一随机种子，共 180 轮。下面单列带开合关节、靠接触和摩擦抓持的成绩。视觉、拾取、松嘴和返回按同一回合计数。

|测试物体|仿真物体最大尺寸|抬离地面 ≥3.5 cm|箱口松嘴时仍持有|最终留在箱内|整轮通过|
|---|---:|---:|---:|---:|---:|
|马克笔|110 mm|11/20|8/20|6/20|6/20|
|小球（高尔夫球外形）|27 mm|13/20|8/20|7/20|7/20|
|鸭子玩偶|38 mm|8/20|4/20|3/20|3/20|

完整通过 16/60。通过的回合回家位置误差中位数为 1.37 cm，朝向 1.86°，关节均方根误差 0.010 rad。27/60 因跌倒结束。相同投掷下，固定鸭嘴只有物理接触时通过 0/60，在已出现嘴部接触后允许模拟约束辅助搬运时通过 42/60。这些数字描述仿真中的表现。第二个箱位（前方 40 cm、右侧 13 cm）做了 12 次独立物理抓取，完整通过 3 次。另测 12 个只有旧物的场景，均保持待命。

整轮验收要求记录到目标接触和提升，松嘴时目标仍在嘴边，释放后整个物体投影在箱内，接触箱底并静止，直到回合末仍连续满足 0.5 秒。此时机器人也须连续 0.5 秒保持在起点 3.5 cm、原朝向 7° 以内，关节均方根 ≤0.10 rad、任一关节误差 ≤0.22 rad，同时满足身高、倾角和静止条件。实际开合时连鸭嘴角度一起验收。碰到旧物并误夹的回合记为失败。回家误差由独立仿真世界状态算出。

[各阶段结果、失败原因和种子](retrieval-v1-results.json) · [完整逐轮记录](../../runs/fetch/retrieval-v1/episodes.json) · [第二个箱位](../../runs/fetch/retrieval-v1-alt/episodes.json)

[笔的成功回放](../../runs/fetch/retrieval-v1-replay/marker/040_large_marker-jaw-97204.mp4) · [球的成功回放](../../runs/fetch/retrieval-v1-replay/ball/058_golf_ball-jaw-97201.mp4) · [玩偶的成功回放](../../runs/fetch/retrieval-v1-replay/duck/sony_duck-jaw-97202.mp4)

## 数据、训练与适用范围

- 先按物体身份拆成 6 个训练、3 个验证、3 个测试物体，再用 MuJoCo 的头部相机生成 864 / 216 / 216 对 RGB 画面。地面有旧物、颜色及光照变化、轻微镜头晃动和无投掷负样本。监督掩膜来自渲染器。运行时模型接收两张 RGB 画面，不读取物体名字或世界坐标。它找的是新出现的东西。在原地移动旧物、多人或多件物体同时进入画面，以及完全遮挡都还需要专门评估。[身份划分](../../configs/retrieval-objects.json) · [资源 SHA-256](../../configs/retrieval-sha256.json)
- 实际使用 YCB 的草莓、网球、泡沫砖、壁球、马克笔、高尔夫球 6 个 16k 网格。数据由 Berk Calli 等发布，许可为 [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)。本项目缩放和重定向网格，并压缩纹理。鸭子玩偶使用 © 2006 Sony 模型，按 SCEA Shared Source License 1.0 使用；转换为 OBJ 并由顶点生成分段接触轮廓。源文件与原许可在本机 `vendor/retrieval/sony_duck/`。训练与验证还包含本地生成的笔和鸟形玩具。冻结训练后另引入 YCB 的香蕉、勺和带把小杯，使用简化凸包表示接触面。[YCB 官方资源表](https://ycb-benchmarks.s3.amazonaws.com/index.html) · [Khronos Duck 作品及授权](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/e124e8d8d7330a5372bc2b4d00d39a4a9d7997ea/Models/Duck/README.md)
- RTX 5060 Laptop GPU 上以 CUDA 12.8、PyTorch 2.9.1、batch 32 实训 20 轮。按验证集选第 19 轮。测试有目标的 183 对图中，179 对预测掩膜与标签的交并比达到 0.5，马克笔 59/61、球 59/61、玩偶 61/61。33 对空画面有 3 次单帧误报。多帧停稳判定用于抑制误启动。预加载训练数据和模型的峰值显存约 1.09 GiB，梯度训练 28.23 秒。物理步进和已有 ONNX 步态推理使用 CPU，画面用 EGL 渲染。[训练记录](../../runs/fetch/vision/metrics.json) · [模型权重](../../runs/fetch/vision/change.pt)
- 用训练物体的物理接触结果选了两套动作。通用抓取搜索 36 组参数 × 24 次，在 13/24 个训练摆位保持提升。细杆横向夹取搜索 48 组 × 8 次，选中的参数在固定方向附近的 8/8 次保持提升。两项为选参成绩。到了测试回合，机器人需要自己根据画面决定从哪一侧接近。用原有 ONNX 控制行走和低头，学习的参数包括对准距离、合嘴及开始上抬的时刻。[最终参数](../../configs/retrieval-grasps.json)
- 幻尔公开真机控制代码将嘴部的 0–39.375° 行程和 14 维运动策略分开。仿真在 CAD 的 mouth body 上增加关节，鸭嘴两片接触面用简化几何，力矩上限 0.09 N·m、摩擦、相机外参和接触感知均待实机标定。三个测试物体都按轻小刚体、4 g 质量建模，真笔、常规高尔夫球和软布玩偶的重量、接触形变可能差很多。现实部署还需要板端视觉运行时以及能反映夹紧/滑脱的反馈。本次没有向真机写入模型。[幻尔控制代码](https://github.com/Hiwonder/wonderduck/blob/f9617fe44e7188e67aa15d8f1725294de334b66c/duck-control/src/model.rs#L98-L110)

## 本机复现

在项目根目录执行。已有 `.venv` 可用于准备资产和 CPU 物理拟合，`runs/fetch/train-env` 是独立的 CUDA 环境。新机器可在独立 Python 3.12 环境安装 `ducklab/requirements.txt`，随后从 PyTorch cu128 官方源安装 `torch==2.9.1`。重新导出 ONNX 时安装 `onnx==1.19.1`，运行 `tools/export_retrieval.py`。先取得 README 中的机器人模型和 6 个基线策略。[机器人资产说明](../official-assets.md)

```bash
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1
.venv/bin/python tools/retrieval_assets.py
.venv/bin/python tools/render_retrieval.py --train 144 --eval 72
runs/fetch/train-env/bin/python tools/train_retrieval.py --epochs 20 --batch 32 --device cuda
.venv/bin/python tools/fit_retrieval_grasp.py --out runs/fetch/grasp-fit-final.json
.venv/bin/python tools/fit_retrieval_grasp.py --thin --out runs/fetch/grasp-thin.json
runs/fetch/train-env/bin/python tools/run_retrieval.py --split test --seed 97200 --trials 20 --out runs/fetch/retrieval-v1
python3 tools/report_retrieval.py runs/fetch/retrieval-v1/episodes.json --out runs/fetch/retrieval-v1/summary.json
```

修改箱位时使用起点坐标系，单位米，前方是正 X，左侧是正 Y。例如把箱放到前方 40 cm、右侧 13 cm 并录下一个回合：

```bash
runs/fetch/train-env/bin/python tools/run_retrieval.py --object 058_golf_ball --mode jaw --bin .40 -.13 --seed 90102 --trials 1 --video-seed 90102 --out runs/fetch/custom-bin
```

## 未见类别和负载测试

冻结策略后，把小香蕉、勺子、带把小杯各投掷 8 次。小鸭在 23/24 次启动了接近动作，10/24 次将指定物品抬起，完整任务有 1/24 次通过（带把小杯）。启动接近也可能由测量误差触发。常规测试的 4 g 负载改为 12 g，在相同形状和前 8 个配对种子上，完整成功从 8/24 降至 4/24。除了外形，重量也值得单独标定。[陌生类别逐轮结果](../../runs/fetch/retrieval-v1-novel/episodes.json) · [12 g 逐轮结果](../../runs/fetch/retrieval-v1-12g/episodes.json)

## 用自己的摄像头照片试识别

已有视觉权重同时提供 [ONNX 导出](../../runs/fetch/vision/change.onnx)。在全量 216 对独立测试照片上核对了 CPU ONNX Runtime 和 PyTorch，7,741,440 个像素的前景判定全部一致。[数值校验](../../runs/fetch/vision/change.parity.json)

用同一机位拍摄投掷前、物体落稳后的两张 RGB 照片，使用现有 CPU 仿真环境即可输出框的位置和标注图。

```bash
.venv/bin/python tools/detect_retrieval.py before.jpg after.jpg --annotated runs/fetch/my-detection.png
```

实际物体用秤测得重量后可直接修改仿真参数，也可使用独立的 `challenge` 类别组。

```bash
runs/fetch/train-env/bin/python tools/run_retrieval.py --split challenge --seed 98300 --trials 8 --mode jaw --out runs/fetch/retrieval-v1-novel
runs/fetch/train-env/bin/python tools/run_retrieval.py --split test --seed 97200 --trials 8 --mass-grams 12 --mode jaw --out runs/fetch/retrieval-v1-12g
```

最值得补充的真机数据是同一件物品的成功夹起、夹空和半路滑落。短视频配物体尺寸、重量与鸭嘴舵机角度，能直接校准这批实验中的负重失稳和夹持丢失。优先用现有三件物体做实机近距离验收，再逐步增加重量、距离和软物材质。
