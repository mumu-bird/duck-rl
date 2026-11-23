# WonderDuck 拾取归位训练与视频

2026 年 11 月 22 日。实测入选的是第六轮策略。相同的 60 次投掷，旧版完成 16 次，这一版完成 32 次。32 次成功里，机器人结束时离起点的距离中位数是 1.21 厘米，朝向误差中位数是 2.59°。23 个回合由失败变成功，7 个由成功变失败，另有 9 个新旧均成功。完整成功要求目标从地面抬起、在箱口松嘴、整件物体留在箱内，机器人回到原位并恢复起始姿势。

[三段同次投掷短片](../../runs/fetch/retrieval-review/highlights.mp4) · [556 段离线回放](../../runs/fetch/retrieval-review/index.html) · [逐段解码与校验值](../../runs/fetch/retrieval-review/video-qa.json)

短片用两倍速同步播放马克笔、鸭子玩偶和挂锁。完整回放用实时速度，带头部相机。索引覆盖 168 个不同的「物体 × 投掷」组合，允许筛选进步、退步、成功、失败。每个测试面板都可以点开旧版与入选版的逐回合记录。页面底部还保留了中间候选的成败样例、换箱位、加重和空场景。

## 各物体实测

每轮投入一件缩放后的 4 克刚体，等它落稳才接近。用带开合关节的鸭嘴靠摩擦抓持。物体身份和世界坐标供裁判评分，运行策略使用相机画面、里程计、姿态、关节和模拟嘴部接触。箱位相对起点可配置，常规测试为前方 34 厘米、左侧 19 厘米。每组中的版本共用同一组初始状态和投掷参数。测试箱体固定，不会随机移动。

|测试组与种子|物体|旧版整轮|新版抬起|新版在箱口仍持有|新版整轮|
|---|---|---:|---:|---:|---:|
|原对照 97200–97219|马克笔|6/20|12/20|10/20|8/20|
|同上|小球|7/20|16/20|14/20|12/20|
|同上|鸭子玩偶|3/20|20/20|16/20|12/20|
|第一批 112300–112311|小柠檬|2/12|12/12|11/12|9/12|
|同上|小挂锁|0/12|5/12|2/12|0/12|
|同上|骰子|1/12|8/12|8/12|6/12|
|第二批 115300–115311|果冻粉纸盒|0/12|0/12|0/12|0/12|
|同上|锅铲|0/12|2/12|1/12|0/12|
|同上|桃子|6/12|12/12|12/12|8/12|
|最终封存 118300–118311|小罐头|0/12|5/12|2/12|0/12|
|同上|夹子|0/12|4/12|0/12|0/12|
|同上|橙子|5/12|12/12|10/12|10/12|

第一批在第四轮冻结时首次开封。该批的挂锁失手随后推动了薄件实验，第六轮在此属于复测。第二批在第六轮选定后首次打开，随后研究投放。第三批用于最终评估，未参与开发和选参。最终封存组整轮 10/36，旧版 5/36，两版全部成功均来自橙子。各物体每组 12 次，统计量描述这些种子和几何。普通物品的重量、软玩偶变形、实物鸭嘴摩擦和舵机夹持力需实机验证。

[原对照明细](../../runs/fetch/retrieval-review/v6-test/episodes.json) · [第一批](../../runs/fetch/retrieval-review/v6-blind/episodes.json) · [第二批](../../runs/fetch/retrieval-review/v6-blind2/episodes.json) · [最终封存组](../../runs/fetch/retrieval-review/v6-blind3/episodes.json) · [模型尺寸、身份和划分](../../configs/retrieval-objects.json)

**其它条件**　起点坐标中的箱位移到前方 40 厘米、右侧 13 厘米，完成 3/12。常规箱位各取前 8 次投掷，4 克时完成 14/24，12 克时完成 15/24，后者按笔、球、玩偶分别为 1/8、8/8、6/8。空场景 12/12 保持待命。成功标准包括物体连续 0.5 秒停在箱底和机器人连续 0.5 秒保持起点附近，最大位置误差 3.5 厘米，朝向误差 7°，关节均方根误差 0.10 rad、单关节 0.22 rad，另检查高度、倾斜和静止。换重量改变了接触动力学，这 24 次配对不足以推出承重能力随重量的规律。

[换箱位](../../runs/fetch/retrieval-review/alternate-bin/episodes.json) · [12 克](../../runs/fetch/retrieval-review/payload-12g/episodes.json) · [待命](../../runs/fetch/retrieval-review/no-toss/episodes.json) · [验收实现](../../ducklab/retrieval/referee.py)

## 为什么逐轮改这些动作

|轮次|观察与训练方法|验证和取舍|
|---|---|---|
|2|旧版 27 次跌倒中 25 次发生在低头时。8 套夹取距离、合嘴时刻、起身时刻在 6 个训练物体各测 5 次。|独立验证原参数 3/30，提前起身 2/30，延后 3/30 但摔倒 14 次，原参数 7 次。保留原参数。[逐组记录](../../runs/fetch/round2-fit/selection.json)|
|3|旧轨迹在起身指令处跳过一段相位。用平滑曲线比较 10 种运动，同时扩充视觉训练形状、尺寸、光照和视角。|同一新版视觉下，45 个控制步渐进过渡 15/30，原轨迹 10/30。常规成套对照达到 29/60，跌倒 7/60。[动作选参](../../runs/fetch/round3-fit/selection.json)|
|4|重抓前将原背景投影到新的相机姿态。用失手前的估计位置缩小搜索范围，并从新的角度接近。|验证 14/36，原重抓 11/36。常规整轮仍为 29/60，跌倒 6/60。第一批陌生物体 14/36。[验证](../../runs/fetch/round4-fit/selection.json)|
|5|挂锁 12/12 碰到，均未抬起。新增三件训练薄件和两件验证薄件，检验第二次偏向边缘的 9 种动作。|验证薄件所有候选整轮均为 0/30。仅调整平面落点不足。[薄件实验](../../runs/fetch/round5-fit/selection.json)|
|6|在近距离摆位上比较 144 套鸭嘴角度、颈部动作、距离和时序，每套 3 个训练物体 × 3 个摆位，共 1,296 次接触仿真。前 3 名再跑全流程。|选中的第二次动作将仰俯关节目标减小 0.32 rad，调整到约 8 cm 的接近距离，较晚合嘴。薄件整轮由 0/24 升到 6/24，普通验证物体由 20/36 升到 22/36。常规测试整轮 32/60，跌倒 10/60，较第四轮的 6/60 多 4 次。[拟合](../../runs/fetch/round6-fit/fit.json) · [验证](../../runs/fetch/round6-fit/selection.json) · [入选参数](../../configs/retrieval-grasps-v6.json)|
|7|挂锁两次带到箱口后仍未留箱。由训练回合的箱口落点中位数产生候选站位，再比较头部姿态。|该验证组原方案 19/36，入围的新站位最好 7/36。[选参记录](../../runs/fetch/round7-fit/selection.json)|
|8|固定箱沿站位，比较持握到松嘴、松嘴速度、提前释放和头部俯仰。|该验证组原方案 23/36，新方案最好 21/36。[选参记录](../../runs/fetch/round8-fit/selection.json)|
|9|比较低头靠近箱底再释放的 9 种动作。训练和验证种子独立。|验证松嘴高度中位数由 21.3 降到 8.8 厘米，完整成绩由 16/36 降到 8/36。其中低位候选 29 次带至箱口，最终仅 9 次留箱。正式入口继续使用第六轮。[选参记录](../../runs/fetch/round9-fit/selection.json) · [低位候选回放](../../runs/fetch/round9-replay/episodes.json)|

视觉训练使用原来 864 对训练画面和 12 个新训练外形各 80 对画面，合计 1,824 对。验证图像 216 对。RTX 5060 Laptop GPU 上 CUDA 12.8、batch 64、24 轮，验证选中第 23 轮，梯度训练约 69 秒，CUDA 分配的峰值显存约 2.10 GiB。独立图像中新版定位达到 IoU 0.5 的正样本为 174/183，旧版 179/183。新版空背景单帧误报 1/33，旧版 3/33。下降主要在细笔，新版 52/61，旧版 59/61。全流程中靠动作和重试得到的进步与单帧分割成绩应分别解读。接触物理和官方步态 ONNX 在 CPU 上并行运行，CUDA 用于视觉训练和推理。ONNX 与 PyTorch CPU 在 216 对画面上的 7,741,440 个像素前景判断一致。

[训练日志](../../runs/fetch/round3-vision/metrics.json) · [CUDA 视觉权重](../../runs/fetch/round3-vision/change.pt) · [CPU ONNX](../../runs/fetch/round3-vision/change.onnx) · [导出校验](../../runs/fetch/round3-vision/change.parity.json)

## 调研依据、适用边界与后续实测

|来源|论文或上游实现验证了什么|本项目如何检验|
|---|---|---|
|[上游 WonderDuck GroundPick 源码，提交 10c3b6c](https://github.com/mumu-bird/wonderduck-rl/blob/10c3b6cc15fb09a364dcfbdaa96ea6fb87b1e531/src/mjlab_microduck/wonderduck/tasks/pick_motion.py)|上游分下降、短暂停留、起身阶段，并检查身体转动。|使用本地已有 ONNX，调整外部相位和鸭嘴动作。可编辑 mjlab 任务来源见[交接说明](https://github.com/mumu-bird/wonderduck-rl/blob/10c3b6cc15fb09a364dcfbdaa96ea6fb87b1e531/HANDOFF_ZH.md)。|
|[Mysore 等，CAPS，2021](https://ai.bu.edu/caps/)|时间平滑正则针对连续控制指令的高频变化。|试验三次平滑相位过渡，衡量鸭子的跌倒率和完整完成率。这里实现的是相位曲线。|
|[Tobin 等，域随机化，2017](https://arxiv.org/abs/1703.06907) 和 [YCB 官方资源](https://ycb-benchmarks.s3.amazonaws.com/index.html)|前者研究改变模拟渲染条件来训练定位，后者发布可下载的生活物体扫描和纹理，数据采用 CC BY 4.0。|按物体身份拆分，光照、颜色、投掷位置作变化。用公开 16k 网格搭建三批新身份面板，保留来源和 [SHA-256](../../configs/retrieval-sha256.json)。|
|[Morrison 等，闭环视觉抓取，2018](https://arxiv.org/abs/1804.05172) 和 [Calandra 等，接触后的重抓，2018](https://arxiv.org/abs/1805.11085)|前者用深度持续更新抓姿，后者用视触觉调整第二次抓取。|鸭子从成对 RGB 图像重新定位，并依模拟鸭嘴的接触信号决定是否重来。机器人没有使用这些论文的深度或 GelSight 模型。|
|[Odhner、Ma、Dollar，平面薄件抓取，2012](https://www.eng.yale.edu/grablab/pubs/odhner_icra2012.pdf)|机械手利用支撑面将薄件重定向后夹取。|在训练薄件上比较鸭嘴俯仰和合嘴时序，记录抬起、持续持有和完整任务三个指标。真实指尖结构与鸭嘴不同。|
|[Jiang 等，新物体放置，2012](https://arxiv.org/abs/1202.1694)|研究放置区域、物品姿态和最终稳定性的关系。|比较实际释放位置及高度、最终留箱和全身归位。第 7–9 轮揭示只更换这些参数的收益有限。|

挂锁和罐头各有 2 次随鸭子运至箱口，均未满足最终留箱判据。对 20 个箱口失手和 7 个成功对照做了 200 Hz 接触回放，27 个回合与原记录逐字段相符。挂锁 112308 在释放后 0.295 秒同时碰到场地平面与箱底，0.320 秒碰到近侧箱壁。罐头 118303 在 0.335 秒同时碰到平面和近侧箱壁，最终中心落在箱中心后方 10.84 厘米，越过 8.4 厘米的箱内边界。另一只挂锁 112310 碰到了箱底，仍未满足终局判据。浅箱与地面的接触几何、实物嘴边的滑动路径和碰沿后的反弹值得对照侧视实拍标定。普通纸盒首先需要解决贴地夹起。[逐 5 毫秒接触监测与轨迹](../../runs/fetch/retrieval-review/release-traces.json)

## 运行与录像

从本项目根目录运行。`reviewed` 入口默认使用第六轮权重、物理开合鸭嘴和全程录像，自动创建独立的时间戳目录。每段原始 MP4 包含头部画面和真实仿真时间。

```bash
export MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
runs/fetch/train-env/bin/python tools/run_retrieval.py --preset reviewed --split test --trials 3 --seed 97200
runs/fetch/train-env/bin/python tools/run_retrieval.py --preset reviewed --split blind3 --trials 3 --bin .40 -.13
```

重训时使用新目录。从头搭建本机环境的资产及依赖步骤见[初版报告](toss-retrieval.md)。

```bash
.venv/bin/python tools/augment_retrieval_frames.py --out runs/fetch/repro-frames --count 80
runs/fetch/train-env/bin/python tools/train_retrieval.py --frames runs/fetch/repro-frames --out runs/fetch/repro-vision --device cuda --batch 64 --epochs 24
.venv/bin/python tools/fit_beak_scoop.py --out runs/fetch/repro-scoop
runs/fetch/train-env/bin/python tools/validate_scoop.py --fit runs/fetch/repro-scoop --weights runs/fetch/repro-vision/change.pt
```

原对照的 60 段重放与封存的逐回合 JSON 全字段一致。每轮候选样例同样比对了原始记录。视频清单包含每段的 SHA-256、帧数和时长。浏览器验证了本地双路视频实际播放，以及四个测试面板的进步和失败筛选。[浏览器实测](../../runs/fetch/retrieval-review/browser-qa.json)
