# WonderDuck 扬声器源码审阅与实施记录

## 2026-11-23 实施更新

用户将范围扩展为“全部功能开发并验收”后，已实现独立的 `duckaudio` 公共音频服务、Python 任务 SDK、音频库、播放控制与调度、离线中文 TTS、浏览器/机身录音、网页控制台、原厂提示音适配、嘴型包络回调、安装回退和验收记录。

最终采用独立 Python 服务，纯音频接口不经过 robotd 的运动控制权判断，原厂 `audio-output.py` 继续管理功放。原厂提示音通过版本校验的可回退适配器入队；原厂其他声音通过统一 player 包装器共享锁。资源目录为 `/var/lib/wonderduck-audio`。没有修改原厂源码审阅副本或 Rust 运动控制循环。

真实任务接入已落在 `tools/run_retrieval.py`、`ducklab/retrieval/episode.py`、`ducklab/suite.py` 和 `tools/run_suite.py`。具体接口与安装见 [使用文档](audio/README.md)，验收边界和步骤见 [验收清单](audio/acceptance.md)。本机测试并非实机出声结论；嘴部物理映射仍由已授权的真机任务控制器完成。

以下保留最初的源码审阅和设计依据，包含当时尚未实现的拟定方案；若与上述实施更新冲突，以 `docs/audio/` 和实际代码为准。

---

审阅日期：2026-11-23。需求补充：用户确认开机已有音频播放；优先目标是让任务模块调用自定义音频，在任务开始、识别目标、完成或失败时播报。

## 结论与范围

首个交付应是「任务音频调用层 + 自定义 WAV 资源部署 + 独立播放控制」，并接入真实任务事件。网页上传管理、文字转语音和嘴部联动作为后续扩展。

用户确认开机能出声，说明至少开机提示音输出链路有效。源码中开机提示音由独立 Python 服务调用 `audio-output.py`，任务调用则可复用 `robot.soundClip`；两条路径不完全相同，因此仍需验证任务播放路径的实际配置与版本。

本次完成源码审阅及规划更新，没有连接真机。开机出声是用户提供的实际观察；任务播放接口是否已经部署、能否在自主任务期间调用，仍需真机验证。下文的任务 SDK、队列、停止和状态接口是开发设计，尚未实现。

审阅材料：

- 用户提供的 `wonderduck-main.zip`，SHA-256 为 `42bc961c5ab5c05580925dfce53a50b00599449b0be2f7dac15aac7c1d640818`；ZIP 条目时间为 2026-11-18 16:10:58（压缩包未提供时区和提交号）。[本地源码副本](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main)。
- 主仓库的 [.gitmodules](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/.gitmodules:1) 指向 `Hiwonder/wonderduck_studio`，用户 ZIP 未包含该子模块。补充审阅官方 Studio 固定提交 `86f31bfee90ba96cd4990320b7bdcb9dd7864612`，提交时间为 2026-11-18 08:10:39 UTC。该提交是补充审阅基线，不能据此认定它与设备版本一致。[来源清单](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/studio-source-manifest.json)。
- [Studio 使用说明](</home/harrisonwang/WonderDuck机器鸭/WonderDuck机器鸭/1.教程资料/2.WonderDuck Studio 使用说明.pdf>) 第 42 页，确认当前产品界面有系统音量与最长 2 分钟的机身录音回放。已核对页面截图。

审阅副本位于工作区已忽略的 `vendor/` 下，Studio 仅下载了相关源码，不能直接当作完整构建目录。实际开发应在完整主仓库中初始化匹配版本的 Studio 子模块；它的 Rust 包使用 `../../../duck-ipc-proto` 等相对依赖，并有独立 workspace，见 [webpadd/Cargo.toml](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck_studio/webpadd/Cargo.toml:8)。

## 任务模块调用方式（首版目标）

当前工作区的任务代码主要是 Python 仿真实验；本次检索未发现 `robotd.sock`、`soundClip` 或可复用的音频调用封装。首版建议提供小型 Python SDK；在机器人本机通过 Unix socket 调用音频执行层。若后续任务运行在电脑上，再为同一 SDK 增加经过访问控制的网络传输，不直接暴露机器人原始控制 socket。

下面是**拟定接口示意，尚未实现**：

```python
# 在任务阶段切换时调用，play() 提交后立即返回，不等待声音播完。
job = audio.play("retrieval.target_found", task_id=task_id)

# 完成判定来自任务自己的结果，不因声音播放成功就认定任务成功。
audio.play("retrieval.success", task_id=task_id)

# 用户取消任务时，只停止该任务的当前音频和待播音频。
audio.stop_task(task_id)
```

需要等播完再进入下一阶段时，可在任务协程中使用拟定的 `await job.wait(timeout=...)`；不得在舵机控制或物理步进循环中阻塞等待。`job.status()` 区分排队、播放中、完成、取消与失败，失败原因记录到任务日志。默认音频故障不终止任务动作，需要音频成功才能继续的任务必须显式等待并处理失败。

建议以任务资源清单映射语义名称，例如 `retrieval.target_found` 对应设备中的 `retrieval_target_found.wav`。录音、音效或预先生成的语音都可以成为资源；首版用脚本转换和部署，不要求先做网页上传。

| 任务事件 | 示例音频 | 触发规则 |
| --- | --- | --- |
| 开始执行 | 「我出发啦」 | 每个任务实例一次 |
| 识别到目标 | 「找到目标了」 | 从未找到转为找到时触发；去重、冷却 |
| 完成并确认结果 | 「任务完成」 | 根据正式成功判定触发一次 |
| 失败或需要帮助 | 「我需要帮助」 | 状态转换时触发，避免每帧重播 |
| 用户取消 | 停止任务自己的提示音 | 不触发机器人全局急停，不取消其他任务或系统提示 |

本地拾取任务的正式成功判断在 [episode.py](/home/harrisonwang/文档/ChatGPT/wonder/ducklab/retrieval/episode.py:764)，上层执行入口在 [run_retrieval.py](/home/harrisonwang/文档/ChatGPT/wonder/tools/run_retrieval.py:24)。未来应在任务编排层接音频事件，不把播放塞进底层仿真或运动计算。仿真与批量训练默认使用记录事件的后端，只有显式选择真机后端才驱动小鸭子的扬声器。

## 已经实现了什么

| 能力 | 源码中的实际实现 | 对本次开发的意义 |
| --- | --- | --- |
| 声卡与功放 | TLV320AIC3104，ALSA 设备默认 `plughw:aic3104`；播放期间保持 `gpiochip1` offset 9，即 GPIO1_B1 / 物理引脚 37 | 复用现有输出适配，不先重写驱动；板卡差异需要真机核实 |
| 播放适配 | `audio-output.py` 包装 `aplay`，持有 `/run/lock/audio-output.lock`，包含 500 ms 预热，并保持功放使能 | 后续所有音频出口应遵守同一互斥和退出清理机制 |
| 自定义 WAV | `robot.soundClip` 从 `/var/lib/robot/audio-clips/<name>.wav` 播放；可单次播放或保持循环 | 可作为首个播放原型的执行基础 |
| 文件校验 | 名称限制为单个文件名；禁止路径分隔符；打开时拒绝符号链接，检查普通文件及 WAV 头 | 保留这些约束，在上传层增加格式、大小、时长与存储配额校验 |
| 系统音量 | Studio 提供 `GET/PUT /audio-volume`，通过 `amixer` 控制 PCM 音量与静音 | 第一版直接复用，不重复做第二套音量滑杆 |
| 机身录音回放 | `arecord` 录制最多 120 秒，保存为 WAV，随后通过同一播放适配器回放；有状态和删除接口 | 录音链路已有，可增加「保存到我的音频库」 |
| 浏览器语音消息 | `POST /voice-message` 接收音频，FFmpeg 解码后播放，HTTP 请求等待播放完成 | 可以借鉴解码过程，但当前实现适合短语音，不能直接充当完整播放器 |
| UI 接入程度 | 当前系统设置页有音量、录音和删除录音；`voice.js` 有按住说话逻辑，但当前页面没有 `#voice-talk`，也没有加载 `/voice.js` | 不应把语音消息后端误认为已可在当前页面操作的完整功能 |

主要证据：

- [声卡与播放器默认值](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/robotd-params/src/lib.rs:788)、[功放 GPIO](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/scripts/audio_speaker.py:7)、[输出锁和预热](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/scripts/audio-output.py:13)、[启动 aplay](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/scripts/audio-output.py:215)。
- [WAV 播放与循环](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/robotd/src/sound/clip.rs:49)、[文件打开校验](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/robotd/src/sound/clip.rs:178)、[现有 RPC 参数](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/duck-ipc-proto/src/lib.rs:2718)。
- [Studio 音频路由](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck_studio/webpadd/src/main.rs:545)、[音量实现](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck_studio/webpadd/src/audio.rs:21)、[机身录音](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck_studio/webpadd/src/audio_record.rs:15)、[语音解码与播放](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck_studio/webpadd/src/voice.rs:51)。
- [实际系统设置界面](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck_studio/webpadd/web/index.html:192)、[页面加载的脚本](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck_studio/webpadd/web/index.html:233)、[未接入页面的喊话脚本](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck_studio/webpadd/web/voice.js:1)。

当前代码还会枚举 WAV 文件供嘴部音效设置使用，但没有发现通用音频库的上传、重命名、文件删除、独立播放和播放进度页面。现有枚举并不等于素材管理，见 [list_mouth_sounds](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck_studio/webpadd/src/settings.rs:537)。

## 第一阶段必须处理的技术问题

1. **实际播放器配置。** Rust 默认 `audio.player = "aplay"`；示例配置没有显式指定功放适配脚本，提示音服务却直接调用该脚本。因此必须读取真机 `/etc/robot/robotd.toml` 和 systemd 生效配置，确认 robotd、Studio 和系统提示音是否使用一致的音频出口。这是待验证项，不能仅凭源码认定当前设备配置有错。[默认值](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/robotd-params/src/lib.rs:814)、[示例配置](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/deploy/robotd.toml:293)、[提示音播放器](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/scripts/prompt-sounds.py:25)。
2. **缺少独立停止与完成反馈。** 现有 `SoundClipParams` 只有 `name`、`hold`。`hold:false` 用来释放保持循环，不是通用的一次性音频停止按钮。RPC 在提交意图后就返回 accepted；文件丢失、播放器失败可能稍后才写日志。需要明确的播放任务 ID、停止命令、状态和失败原因。[请求接受](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/robotd/src/main.rs:10927)、[异步失败](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/robotd/src/sound/clip.rs:101)、[释放循环](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/robotd/src/sound/clip.rs:95)。
3. **循环语义与播放队列。** `hold:true` 必须在 300 ms 内持续刷新，断开后自动释放，这是按住音效的机制。单次请求通过一个原子槽传递，快速提交可能覆盖未消费的请求；持续音效占用时单次音频也可能被跳过。首版使用「单任务 + 明确替换或拒绝」策略；后续的有限次循环和队列要单独设计。[300 ms 常量](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/robotd/src/intents.rs:218)、[请求槽](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/robotd/src/intents.rs:978)、[保持判断](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/robotd/src/intents.rs:1026)、[播放竞争](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/robotd/src/sound/clip.rs:72)。
4. **音频仍受运动控制会话限制。** `RobotSoundClip` 被列入 `is_motion_intent`，会进入控制权和自主运动状态检查。新增独立音频播放应有明确的音频权限，不能靠抢占运动控制权播放，也不能用 `robot.stop` 充当停止声音。验证休息、行走、跟随和维护页面各自的预期行为。[分类](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/robotd/src/main.rs:9058)、[控制权判断](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/robotd/src/main.rs:9856)。
5. **多个播放来源的协调。** robotd 音效、Studio 语音/录音回放和独立提示音服务会分别启动播放器；文件锁能串行化出口，但不提供统一任务状态和优先级。当前提示音遇忙会重试，低电量逻辑还会临时调整音量。需要明确提示音优先级、用户音量恢复与长音频的处理，不能假设已经自动抢占。[提示音等待](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/scripts/prompt-sounds.py:522)、[提示音音量](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck-main/scripts/prompt-sounds.py:252)。
6. **长音频的资源使用。** 现有 `/voice-message` 关闭 HTTP 默认体积限制，并一次性接收、解码成完整 PCM，再写播放器。通用文件播放需要上传大小限制、解码超时、最大时长和流式输出，避免随着音频变长持续增加内存占用。各路径还应统一执行 `audio.enabled` 和静音约定。[路由](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck_studio/webpadd/src/main.rs:549)、[完整解码](/home/harrisonwang/文档/ChatGPT/wonder/vendor/wonderduck-source-review-20261123/wonderduck_studio/webpadd/src/voice.rs:51)。

## 推荐的实现结构

```mermaid
flowchart LR
    T[任务模块：开始 / 发现目标 / 成功 / 失败] --> C[任务音频 SDK]
    C --> P[音频请求：播放 / 停止 / 状态]
    F[任务音频资源清单与 WAV 文件] --> P
    P --> R[robotd 音频执行与仲裁]
    R --> O[现有 audio-output.py]
    O --> H[ALSA + 功放 GPIO + 扬声器]
    C --> S[仿真后端：只记录音频事件]
```

- **任务模块**只表达播放什么、属于哪个任务以及是否等待，不直接执行 `aplay` 或操作 GPIO。
- **SDK**提供资源名称解析、任务归属、结果查询、定向取消；真机和仿真后端使用同一任务侧接口。支持幂等请求标识，重连重试不会重复播报。
- **robotd 音频执行层**复用 `Sound` / `clip.rs`，补齐有界队列、任务标识、取消、完成通知与失败状态。将纯音频调用从运动控制权判断中独立出来，但保留明确的访问权限；不改变原有运动命令的权限。
- **输出适配**继续复用 `audio-output.py` 的功放使能、预热、锁与清理。音频文件读取、解码和子进程等待在控制循环之外执行。
- **资源清单**把任务使用的名称映射到 `/var/lib/robot/audio-clips/` 下的稳定 WAV 文件。转换时初选 PCM 16-bit / 48 kHz / 单声道，并在真机确认；文件写入采用临时文件与原子替换，部署时检查文件存在、格式、大小、时长与权限。
- **播放调度**首版明确任务之间的排队、替换、忙碌和取消规则；按任务实例和事件去重。系统提示音目前是独立进程，迁移统一仲裁前必须识别其占用，不能把等待锁的任务显示成正在播放。关键系统告警优先级需要显式接入，不能只给 SDK 请求加一个数字就宣称已经抢占成功。

## 分阶段交付

| 阶段 | 交付内容 | 验收标准 |
| --- | --- | --- |
| P0：验证自定义调用 | 部署一段任务 WAV，检查真机协议与 player，验证现有 `robot.soundClip`；复现自主任务时的权限限制 | 确认能按名称播放用户音频，并记录播放失败或拒绝的实际原因 |
| P1：任务音频能力 | 独立音频权限、播放/停止/状态、有界调度、资源清单；Python SDK 与仿真后端；接入一项任务的开始、成功和失败事件 | 任务中能播音且不抢运动控制；取消只停止对应音频；默认不阻塞动作；播放结果可追踪 |
| P2：复用与资源工具 | 音频检查/转换/部署命令；再接第二个任务模块；完善去重、过期、重连与系统提示音协调 | 两个模块使用同一套 API，无需各自启动播放器；重复识别不连播，断连和告警行为明确 |
| P3：可选扩展 | 文字转语音、常用短句、网页资源管理、嘴部联动 | 都复用同一音频服务；新增入口不产生第二套声卡竞争逻辑 |

优先验收「一个任务模块能够调用自定义声音，且不影响任务运行」，随后以第二个任务验证接口可复用。具体工期在真机版本、构建方式与任务运行位置确认后细化。

### P0：验证任务播放路径

1. 以用户已确认的开机出声为起点，核对开机提示音与 robotd 任务播放各自使用的 player、音频设备和配置；重点排查两条路径的差异。
2. 准备一个短 WAV，按设备约定部署到音频目录，验证一次性 `robot.soundClip` 请求。记录「请求拒绝」「已接受但执行失败」「播放完成」分别发生在哪里。
3. 在休息、普通任务、自主跟随/避障和维护状态分别测试。修复任务调用时的音频权限问题，不通过抢运动控制权或全局停止来绕过。
4. 验证系统提示音占用时的播放行为，以及取消后 GPIO、文件锁和子进程是否释放。

下面是实施阶段在**机器人上**执行的只读诊断示例，本次没有运行：

```bash
cat /proc/asound/cards
aplay -l
amixer -c aic3104 cget 'name=Line Playback Switch'
amixer -c aic3104 cget 'name=PCM Playback Volume'
systemctl cat robotd.service webpadd.service wonderduck-prompt-sounds.service
journalctl -u robotd -u webpadd -u wonderduck-prompt-sounds -b -n 100 --no-pager
```

现有命名 WAV 的协议载荷示例：

```json
{"jsonrpc":"2.0","id":1,"method":"robot.soundClip","params":{"name":"welcome"}}
```

前提是 `/var/lib/robot/audio-clips/welcome.wav` 已存在且音频已启用。这是现有 IPC / 受支持控制传输中的 JSON-RPC 消息，不是可直接 POST 给任意 Studio HTTP 路径的请求；调用还受协议握手、会话和控制权限制。返回 accepted 也不能代替播放结果。

### P1：任务调用契约

以下是**建议新增的 SDK 行为**，不代表当前已有可导入的模块：

| SDK 操作 | 行为约定 |
| --- | --- |
| `audio.play(name, task_id=...)` | 提交已部署资源，立即返回播放句柄；请求含稳定幂等标识 |
| `job.status()` | 查询 `queued` / `waiting` / `playing` / `finished` / `cancelled` / `failed`，以及失败原因 |
| `await job.wait(timeout=...)` | 仅在任务协程中等待；超时与播放失败有明确结果 |
| `job.stop()` | 幂等停止这一个播放任务并清理对应进程 |
| `audio.stop_task(task_id)` | 取消该任务实例的当前音频与待播项，不使用机器人全局停止命令 |
| `audio.has_clip(name)` | 检查资源可用性，任务开始前一次检查，不在每个控制周期访问磁盘 |

IPC 层需要扩展带播放 ID 的请求、定向取消和状态查询，并同步协议兼容处理；旧 `robot.soundClip` 可保留作为兼容入口。不能仅在 Python 外面包一层函数、继续把 accepted 当作播完。

队列必须有上限和过期时间，任务阶段已经结束的提示不应很久后才播出。播放失败默认写入任务事件日志；需要声音成功才能继续的流程显式处理 `wait()` 的结果。系统音量先沿用已有配置，避免多个任务不断修改全局音量。

任务取消与正常完成分别处理：取消时清理该任务所属音频；正常完成时允许已提交的完成提示播完，避免统一清理函数马上把完成提示取消。

### 后续：文字播报选择

第一版建议在电脑或选定的服务端生成语音，再通过音频库推给小鸭子，缓存键包含文本、声音、语速和合成版本。这样可以先验证播报体验，机器人侧继续复用已稳定的播放链路。

需要完全离线运行时，再选中文 TTS 引擎并测量 RK3566 上的生成耗时、峰值内存及对运动推理的影响；不能仅凭能安装就认定适合与机器人控制同时运行。云端方案作为可选项，是否允许把文字发送给第三方由用户在实际选择服务时决定。

## 预计修改位置

下表中的「新增」文件名是规划，不代表已实现。

| 所在仓库 | 位置 | 改动 |
| --- | --- | --- |
| 主仓库 | `robotd/src/sound.rs`、`robotd/src/sound/clip.rs` | 播放任务状态、定向停止、清理和仲裁；保留后台文件读取 |
| 主仓库 | `robotd/src/intents.rs`、`robotd/src/main.rs` | 非阻塞音频命令、独立音频授权、状态发布；覆盖休息/自主运动/维护状态 |
| 主仓库 | `duck-ipc-proto/src/lib.rs` | 新的停止/状态消息与任务标识；更新协议兼容处理 |
| 主仓库 | `scripts/audio-output.py`、`scripts/audio_speaker.py` | 主要复用；按真机结果修正配置、预热和退出清理 |
| 主仓库 | 部署配置、systemd drop-in、`scripts/prompt-sounds.py` | 资源目录权限、player 路径、重启持久性与提示音协调 |
| 任务工程 | 新增独立音频客户端模块及 `assets/audio/` 资源清单 | 提供统一 SDK；与 MuJoCo 仿真依赖分离；支持只记录事件的仿真后端 |
| 任务工程 | 任务编排层与结果处理入口 | 在阶段切换和正式结果判定时触发，按任务实例去重和清理 |
| Studio（后续） | 现有音频路由与可选音频库页面 | 通过同一音频服务提供手动调试和资源管理 |


主仓库与任务客户端分别构建和验证；涉及 Studio 时再固定匹配的子模块版本。部署需要保留音频资源、用户设置和设备标定。纯音频播放不要求站立、不自行改变舵机状态；嘴部联动作为后续功能，继续遵守已有运动控制规则。

## 验证与完成标准

桌面自动验证针对真实风险：资源丢失/坏 WAV、任务实例间的取消隔离、重复事件、队列溢出/过期、重试幂等、取消与完成并发、播放器失败，以及 IPC 与 SDK 状态一致性。使用替身播放器验证进程退出和状态传播；仿真后端只记录事件，不因批量训练而在真实设备上发声。

真机验收至少覆盖：

- 短提示音、10 秒语音和较长音频都能完整播放，首字不丢；音量 0 / 低 / 中 / 高档可控。
- 连续播放至少 30 次并交错停止，检查进程、文件描述符、内存、GPIO 与声卡占用是否残留。
- 音频库重启后仍可用；网络断开时已保存资源可以由本地逻辑播放。
- 正常完成、主动停止、文件丢失、解码失败、声卡忙都有可见结果，不出现 accepted 后永远显示正在播放。
- 在休息、普通运动、自主跟随/避障和维护场景分别测试；验证音频调用不抢运动控制权，并比较播放前后的控制周期、延误计数和推理耗时。
- 验证低电量提示、开机/联网提示与用户音频相遇时的行为，以及结束后用户音量能否正确恢复。
- 若采用现有 500 ms 预热，首次可听声音的延迟目标应把预热计入；首版可暂定短文件冷启动不超过约 1 秒、停止不超过约 0.5 秒，实测后修订。这里是验收目标，尚无本次实测结果。

首个可交付演示应是：一个任务开始时播放自定义提示，运行中识别到目标后播放一次音效，完成后根据正式结果播放成功或失败提示；取消任务可以停止该任务的声音，同时运动控制保持原有行为。再让第二个任务复用同一 API，验证这是一项公共能力。
