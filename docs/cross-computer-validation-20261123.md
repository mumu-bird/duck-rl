# 小黄鸭：开发盘点与另一台电脑验证

盘点日期：2026 年 11 月 23 日，北京时间。以下命令中的相对路径均以本交接包解压后的根目录为准。`用户名`必须替换为这只机器鸭已确认的 SSH 账号；热点密码不是 SSH 密码。

## 1. 按用户确认的五项功能验收

本次范围固定为：**扬声器播放、语音唤醒、叼起林里的小鸭子、抛球接球、遇到大蒜避开**。林里小鸭子的代码命名使用 `LINLEE`。以下把用户目标、已找到的实现和实机证据分别记录。

| 功能 | 已定位的实现与证据 | 另一台电脑验证什么 / 当前缺口 |
|---|---|---|
| 1. 扬声器播放功能 | `duckaudio/` 及板端安装包；自定义音频、播放控制、音量与调度、中文 TTS、任务 SDK、网页控制台。上一轮本机 61 项音频及关联回归通过 | 安装预检后验证机身实际出声、首字、音量、暂停/停止、任务调用和原厂声音共存；仍无实机通过记录 |
| 2. 语音唤醒功能 | `voice-dev/wonderduck-main/scripts/voice_assistant.py`；录音后用离线 ASR 识别“小鸭小鸭”，再处理口令；上一轮 12 项逻辑测试通过 | 需板端 Whisper 与中文模型；验证无唤醒词不执行普通口令、唤醒后响应、误唤醒及延迟。目前是固定录音窗口的原型，并非已验证的低延迟唤醒引擎 |
| 3. 叼起林里的小鸭子 | `tools/train_duck_ppo.py`、`ducklab/retrieval/duck.py`、`configs/linlee-duck-*.json` 和独立仿真记录；新增拾取现场检查/影子推理工具源码 | 使用实际林里小鸭测尺寸、重量、材质、嘴部开口，再验证找到目标、接近、叼起和持续夹稳。搬到托盘/归位是后续扩展；通用物体拾取成绩不能直接当作林里小鸭成绩 |
| 4. 抛球接球 | `ducklab/self_catch.py`、训练/评估记录、只读检查包及独立相机取帧包。原仿真接口有 100/100 记录，但 20 ms 接口迁移严格成功仍为 0/100 | 先验相机、球状态估计、舵机映射与响应时序；当前不能直接验收实体自动抛接。接球须在触地前完成并夹稳，落地后捡起不算 |
| 5. 遇到大蒜避开 | 用户已明确这项工作目标；截至本次检索，在主工程与语音开发副本的代码/文档/配置中，未定位到“大蒜 / 蒜头 / 蒜瓣 / garlic / 避蒜”的实现或记录 | 状态为“待定位实现”，不据此判断从未开发。取得对应目录或聊天线索后再确认识别方式、避让动作和部署入口；验收目标为遇到大蒜主动避开、普通物体不误触发、移走后恢复正常行为 |

直行纠偏、基础仿真任务、视频评估、并行训练和 GPU 物理试验归入上述功能的配套工程，不再列为额外的用户功能。当前 GPU 物理轨迹一致性未达到接回正式 PPO 的条件。

连接证据仍是 2026-11-23 13:40:06（北京时间）的 `NO_LINK`，未向设备发请求。新增相机包的既有记录是 11 项模拟帧/Unix socket 软件测试通过，真实相机及实体接球仍为 `NOT_TESTED`。本次更新只重整交接范围并核对文件，没有连接机器人或新增大蒜避让实现。

实际开发分布在两个目录：

- 主工程：`/home/harrisonwang/文档/ChatGPT/wonder`。
- 离线语音副本：`/home/harrisonwang/WonderDuck机器鸭/voice-dev/wonderduck-main`。

主工程 Git 当前没有提交，文件均未跟踪；语音副本也没有 `.git`。所以换机不能靠 `git clone` 取得这些开发。本包包含独立源码快照。语音副本相对原厂 ZIP：新增 4 个有效文件、修改 1 份文档，另外 3 个新增文件只是 Python 缓存。主工程中另一个 `vendor` 源码审阅副本与这里的语音开发副本应分开理解。

## 2. 本包内容与另一台电脑准备

- `delivery/selfcatch-check.pyz`：约 56 KB，只读设备检查与离线数值自检。
- `delivery/selfcatch-camera.pyz`：独立相机只读取帧包；附原有校验清单，实机相机尚未验收。
- `audio/wonderduck-audio-1.0.0.tar.gz`：约 1.1 MB，板端音频发布包及其 `.sha256`。
- `voice-prototype/`：语音原型两个脚本、逻辑测试及原说明。
- `source/wonderduck-source-20261123.tar.gz`：主工程代码、配置、文档、测试、默认 WAV 和源码摘要；供继续开发与排错。
- `docs/audio/`、`examples/audio/`：音频使用/验收文档及 SDK 示例。
- `evidence/`：原有软件、浏览器、发布包与 NO_LINK 记录，以及上一版交接包的离线复核日志。本次五项范围更新不把旧回归记录扩展为新增源码的通过结论。历史浏览器验收使用合成麦克风；本次没有重新运行浏览器测试。
- `FIELD-RESULTS.md`：现场填写模板；`reports/`：新报告存放处。
- `CHECKSUMS.sha256`、`verify_files.py`：跨平台文件完整性检查，只读文件，不访问机器人。

普通电脑即可操作 Studio、SSH 和网页，不需要 NVIDIA GPU、MuJoCo 或 PyTorch。先准备浏览器和可用的 SSH/SCP；Linux 的自动只读检查需要 Python 3（建议 3.10+）和 `ip` 命令。Windows 可用已安装的 OpenSSH 和 Python；macOS 可用其 SSH 和已安装的 Python 3。

本包不包含本机 `.venv`、训练视频、仿真模型目录、训练权重、Whisper 可执行文件或模型、登录令牌和音频数据库。仅验证连接和音频不需要训练资产。完整训练复现还要按需同步 `vendor/`（约 332 MB）和所需 `runs/` 子目录（当前全部约 9.9 GB），并重建依赖环境；不能把本机虚拟环境当成另一系统的运行环境。

解压后先校验：

```bash
# Linux / macOS
python3 verify_files.py
```

```powershell
# Windows（已安装 Python Launcher）
py -3 verify_files.py
```

若没有 `py` 而已安装 `python`，改用 `python verify_files.py`。预期输出 `FILES_OK`。

## 3. 第一阶段：连鸭子，确认设备状态

1. 鸭子放稳，嘴里不放球，按原厂操作开机，等启动完成。
2. 另一台电脑连接这只鸭子 `WD-` 开头的 Wi-Fi。资料包记录的默认 Wi-Fi 密码为 `hiwonder`；若已改密码，以实际设备为准。
3. 浏览器打开 `http://192.168.149.1:8090`，核对设备身份/序列号、固件版本、设备状态、电量、IMU 和原厂相机画面。记录在 `FIELD-RESULTS.md`。这是原厂 Studio；新音频网页是另一个端口。
4. Windows/macOS 当前先走上述人工检查。现有 `.pyz` 的在线检查只实现 Linux 路由检测；不要把它在 Windows/macOS 的拒绝当成硬件故障。WSL 是否存在热点直连路由未验证，不承诺可用。
5. Linux 电脑直接运行：

```bash
python3 -I -B -S delivery/selfcatch-check.pyz --out-dir reports
```

脚本只读匿名 `GET /api/info`、`GET /api/state`，先检查 `192.168.149.0/24` 直连路由。经网关转发会在发请求前停止；默认不 SSH、不调用运动服务。报告写入 `reports/field-*.json`。

| 结果 | 解释 |
|---|---|
| `BASIC_READONLY_OK` | 设备身份、连续状态、IMU、电池基础遥测正常；公开舵机诊断可能不完整 |
| `PASS_READONLY` | 公开接口还提供了 15 个新鲜、无故障的实测舵机反馈；仍不等于动作/限位/相机通过 |
| `CHECK` | 缺字段、过期、告警或接口版本不匹配；按 JSON 原因核对原厂设备状态页 |
| `NO_LINK` | 直连路由不符或连不上；先查电脑的 Wi-Fi 和地址 |
| `NOT_TESTED` | 实体接球未测；正常且应保留 |

退出码 0 为完整只读通过；1 为缺项/告警；2 为连接或报告保存失败。仅基础遥测正常时也可能返回 1。

在未连设备时可验证包本身：

```bash
python3 -I -B -S delivery/selfcatch-check.pyz --offline
```

预期 `LOCAL_PARITY_OK`，仅说明电脑端数值一致。已有可信 SSH 账号时，Linux 在线命令可追加 `--ssh-user 用户名` 做板端固定输入回放；这仍不会测试视觉跟踪或接球动作。

## 4. 第二阶段：先验收最成熟的音频服务

此阶段需要已确认的 SSH 账号和板端 sudo 权限。无需 SSH 账号也可以完成前面的 Studio 检查；本次未确认这只设备的账号，不提供猜测密码。

在另一台电脑的终端，从交接包目录复制发布包：

```bash
scp audio/wonderduck-audio-1.0.0.tar.gz audio/wonderduck-audio-1.0.0.tar.gz.sha256 用户名@192.168.149.1:/tmp/
ssh 用户名@192.168.149.1
```

以下在机器人 SSH 终端执行：

```bash
cd /tmp
sha256sum -c wonderduck-audio-1.0.0.tar.gz.sha256
tar -xzf wonderduck-audio-1.0.0.tar.gz
cd wonderduck-audio-1.0.0
python3 --version
sudo python3 deploy/audio/install.py check
```

需要板端 Python >=3.10、`robot` 组、原厂功放脚本，以及 `alsa-utils`、`ffmpeg`、`espeak-ng`；Python 3.10 还需要 `tomli`。预检失败先保留输出，按缺项处理。若系统支持 apt 且可联网，依赖安装命令为：

```bash
sudo apt-get install alsa-utils ffmpeg espeak-ng python3-tomli
sudo python3 deploy/audio/install.py check
```

机器人热点连接不保证机器人本身能联网；需要离线安装时，准备匹配板端发行版和 CPU 架构的依赖。不要用本机 x86 的二进制替换板端程序。若板端 Python 版本不满足，先记录并停在预检，不更换系统 Python。

原厂源码默认路径为 `/home/wonderduck/wonderduck`，不同则在 `check` 和 `install` 都添加 `--robot-root /实际目录`。如果仅提示原厂提示音源码摘要不匹配，使用 `--without-prompt-integration` 重新 `check` 和 `install`；这会保留原厂提示音，提示音统一调度项目暂不算通过。

预检通过且机器人空闲时：

```bash
sudo python3 deploy/audio/install.py install
sudo systemctl status wonderduck-audio --no-pager
sudo duckaudio doctor
sudo duckaudio settings --volume 30 --no-mute
sudo duckaudio play task.started --wait
sudo duckaudio say '你好，我是小黄鸭，正在做音频验证。' --wait
```

安装器会创建音频服务、导入默认短句，默认改原厂配置的 `audio.player` 为共享功放包装器并保存备份；它不自动重启 `robotd`。若输出 `robotd_restart_required: true`，在安全空闲状态正常重启机器人后再验收新旧声音共存。完整路径、配置和例外见 `docs/audio/README.md`。

在另一台电脑另外打开终端并保持运行：

```bash
ssh -N -L 8092:127.0.0.1:8092 用户名@192.168.149.1
```

浏览器打开 `http://localhost:8092`。在机器人 SSH 终端运行 `sudo cat /var/lib/wonderduck-audio/access-token`，把显示的令牌仅输入这个页面。本包没有携带本机演示令牌；使用机器人新生成的令牌。

按顺序完成：

1. 上传一个短 WAV/MP3，实际听到机身扬声器出声，验证首字、暂停/继续、跳转和停止。
2. 测试音量 30/70、静音与恢复、循环后停止、排队与按任务取消。
3. 输入中文 TTS 并听清内容；再录一段浏览器声音并回放。浏览器录音来自这台电脑的麦克风。
4. 单独测试机身麦克风的录音和回放；它与浏览器录音是两项验收。
5. 初步通过后按 `docs/audio/acceptance.md` 验证任务 SDK、与原厂运动并行、系统提示音、异常恢复等；嘴部联动仍需实际控制器和标定。
6. 在网页记录每项结果并导出 JSON。HTTP 提交成功或播放状态 `finished` 都不能代替实际听到声音。

机身录音的 CLI 示例（听完后再开始录音）：

```bash
sudo duckaudio record start --seconds 5
# 录制过程中提前结束并回放
sudo duckaudio record stop --playback
```

若原厂宠物声音识别正在占用麦克风，先按原厂操作在空闲状态切换这项功能后再试，记录切换和恢复情况。

音频安装问题在机器人上收集：

```bash
sudo duckaudio doctor
journalctl -u wonderduck-audio -u wonderduck-prompt-sounds -b -n 120 --no-pager
aplay -l
arecord -l
```

回退/卸载入口（在发布包解压目录执行）：

```bash
sudo python3 deploy/audio/install.py rollback
# 仅有上一已安装版本时可回退；首次安装需要移除服务时使用：
sudo python3 deploy/audio/install.py uninstall
```

用户音频、配置备份和数据会保留。若原厂配置已被其他操作改动，安装器可能要求人工合并；不要覆盖后续改动。

## 5. 第三阶段：单独验证离线语音原型

这是另一套原型，尚未合并为 `duckaudio` 的统一语音助手。一次只运行一个机身录音入口，避免与音频控制台录音争用设备。

先准备适用于板端的 `whisper-cli` 和支持中文的多语言 ggml 模型；这些不在包中，不把单元测试通过算成 ASR 已可用。原型要求 `audio.enabled=true`，且 `audio.pet_detect=false`；后者会释放单客户端麦克风，需记录并在测试后恢复用户原设置。更改原厂配置的生效方式以当前固件为准。

从另一台电脑复制到专用验证目录：

```bash
ssh 用户名@192.168.149.1 'mkdir -p ~/wonderduck-verify-20261123'
scp -r voice-prototype 用户名@192.168.149.1:~/wonderduck-verify-20261123/
```

以下在机器人 SSH 终端执行。默认原厂路径不同则替换两条链接的目标：

```bash
cd ~/wonderduck-verify-20261123/voice-prototype
ln -s /home/wonderduck/wonderduck/scripts/audio-output.py scripts/audio-output.py
ln -s /home/wonderduck/wonderduck/scripts/audio_speaker.py scripts/audio_speaker.py
python3 scripts/audio-diagnostic.py --seconds 4 --play-tone
export WONDERDUCK_ASR_MODEL=/var/lib/robot/voice/models/ggml-tiny.bin
python3 scripts/voice_assistant.py --once
```

链接复用板端自己的功放实现，不替换原厂文件。若链接已存在先核对目标，不覆盖。模型路径改成实际准备的文件。`--once` 只录一次，不要求唤醒词；先说“你好”或“帮助”，记录识别文本、延迟和回复声音。完成后再用 `python3 scripts/voice_assistant.py` 验证“小鸭小鸭”唤醒。

“停止/停下/别动”会调用 `robot.stop`，不是只读命令；在机器人空闲且操作者知情时单独验收。首次不配置任何运动技能别名。模型缺失、IPC 拒绝、麦克风占用、音色与速度问题都记为待修复，不扩展为任意命令执行。

不接机器人也能复核逻辑（在 `voice-prototype` 目录）：

```bash
python3 -B -m unittest discover -s tests -p 'test_voice_assistant.py' -v
```

## 6. 叼起林里小鸭、抛球接球与大蒜避让

当前不把 `.pt` 上传到 Studio，不替换原厂走路/站立/拾球模型槽。直行、拾取、LINLEE 和自抛接球的权重都是实验产物，仍缺具体设备的映射、限位、视觉输入、时序和实物参数验证。

叼小鸭先以“抬离地面并持续夹稳”为基本目标，记录是否滑落、夹取位置和实物状态，再扩展到搬运与归位。

本次先核对原厂相机是否能正常显示。源码快照另含 `tools/selfcatch_field/camera_probe.py`，用于已知 SSH 账号下的相机缓冲取帧；已另打为 `delivery/selfcatch-camera.pyz`，不在 `selfcatch-check.pyz` 默认流程中，也没有实际相机通过证据。帧率采样不等于端到端视觉延迟或看球能力。后续运动验收应先完成设备/视觉/低层接口对齐，再在对应时序仿真通过后安排实体动作。

已核实 Linux 热点直连和 SSH 账号时，可单独采集相机帧：

```bash
python3 -I -B -S delivery/selfcatch-camera.pyz --ssh-user 用户名 --count 30 --out-dir reports/camera
```

该入口只采集，不驱动舵机。软件时间戳帧率不代表硬件曝光或端到端视觉延迟。补齐顺序见 `docs/tasks/self-catch-completion.md`。

**大蒜避让仍待定位代码或此前开发记录。** 现场记录表已单列此项，待入口确认后分别测试大蒜出现、近似普通物体和大蒜移走三类场景。当前交接包没有大蒜识别器、专用策略或避让启动命令，不把一般障碍物回避当成“大蒜专门识别”。

## 7. 验证后带回什么

- 填好的 `FIELD-RESULTS.md`：电脑系统、设备序列号/固件、连接方式、成功项和失败项。
- `reports/field-*.json`（Linux 自动检查产生）。
- 音频网页导出的验收 JSON、`duckaudio doctor` 和相关日志。
- 实际声音/机身录音/语音识别的短视频或日志；记录发生时间和具体命令。
- 不把访问令牌、SSH 私钥或密码写入报告。

建议先验证设备连接与身份，再验扬声器播放和语音唤醒；叼小鸭、抛球接球按各自实机接口推进，大蒜避让先定位实现。五项分别记录结果，不用一项的通过代替另一项。
