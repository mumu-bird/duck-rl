# WonderDuck 音频开发与使用

交付日期：2026-11-23。实现位于 `duckaudio/`，用于自定义声音、任务播报、录音和统一播放调度。音频服务独立于 robotd 的运动控制权，继续使用原厂 `audio-output.py` 驱动功放。

本机验收环境使用 `null` 播放后端：音频导入、转码、中文语音合成与存储真实执行；扬声器播放只模拟时间。实机输出、声卡音量、机身麦克风、与运动并行、开机/低电量共存和嘴部动作，必须在设备上另行验收。详见 [逐项验收](acceptance.md)。

## 已实现能力

| 功能 | 实现与入口 |
|---|---|
| 音频库 | WAV、MP3、M4A、OGG、FLAC、浏览器 WebM 导入；转为 PCM16 / 48 kHz / 单声道；列表、改名、稳定资源键、下载、删除、文件校验、空间配额 |
| 播放 | 播放、暂停、继续、跳转、重复、循环、单次增益、状态与错误、停止单个声音/单个任务/全部声音 |
| 调度 | 有界队列、优先级、排队过期、替换本任务、打断、忙碌拒绝、请求幂等、重启后旧任务失败且不重播 |
| 音量 | 0–100、静音、允许播放开关、持久化与重启恢复；真实 mixer 失败会报告 |
| 中文文字播报 | 本地 eSpeak NG，离线合成、语速、语言、缓存、保存或直接播放；默认声音较机械，可上传自己的录音替代 |
| 录音 | 电脑/手机浏览器录音上传；机身 arecord 录音、停止、保存、回放；最长 120 秒 |
| 任务 SDK | 后台提交、Future 错误反馈、事件映射、去重/冷却、按实例取消、日志仿真后端 |
| 任务接入 | 拾取归位：开始/发现目标/成功/失败；任务套件：各回合开始与真实成功/失败判定 |
| 原厂提示音 | 启动、低电量、联网播报通过适配器进入公共队列；原触发条件保持；服务不可用时走原厂路径 |
| 嘴部联动 | 音量包络、网页预览、`MouthFollower` 回调；由现有任务控制器执行嘴部动作，不自行接管舵机 |
| 运维与验收 | 网页、CLI、HTTP/Unix API、事件记录、13 组验收表和导出、systemd、版本回退、保留用户数据的卸载 |

## 本机打开控制台

当前演示服务使用端口 8092，数据目录 `runs/audio-acceptance/`。打开 `http://127.0.0.1:8092`，输入此目录中 `access-token` 文件的内容。令牌不放入 URL，也不会写入验收导出。

从项目根目录启动：

```bash
.venv/bin/python -m duckaudio serve \
  --backend null --data-dir runs/audio-acceptance \
  --host 127.0.0.1 --port 8092 \
  --tts-program "$PWD/vendor/audio-tools/usr/bin/espeak-ng"
```

本机 eSpeak 程序仅用于本机验证，不随板端发布包分发。系统已经安装 eSpeak NG 时可省略 `--tts-program`。输入文件转码需要 FFmpeg；项目虚拟环境也可使用现有 `imageio_ffmpeg` 提供的可执行文件。

导入随包短句：

```bash
.venv/bin/python -m duckaudio.seed \
  --endpoint "unix:$PWD/runs/audio-acceptance/audio.sock" \
  --directory assets/audio
```

默认短句是实际合成的 WAV，校验和在 `assets/audio/manifest.json`。重复导入保留已有同名资源，用户可以替换成自己的声音。

## 板端安装

发布包为 `dist/wonderduck-audio-1.0.0.tar.gz`，旁边有 SHA-256 文件。发布包仅含源码、网页、配置、安装器、文档和默认 WAV，不带开发电脑的二进制程序。需要板端 Python 3.10 或更新版本；3.10 需 `tomli`。当前本机执行验证的 Python 为 3.12。

在机器人空闲、可进行部署验收时，将包复制到机器人并解压，在解压目录执行：

```bash
sudo apt-get install alsa-utils ffmpeg espeak-ng python3-tomli
sudo python3 deploy/audio/install.py check
sudo python3 deploy/audio/install.py install
sudo systemctl status wonderduck-audio --no-pager
sudo duckaudio doctor
```

原厂源码目录默认 `/home/wonderduck/wonderduck`；不同目录用 `--robot-root /实际目录`。安装器检查原厂功放脚本、robot 组、系统依赖和提示音源码版本。提示音适配目前针对提供的源码快照；版本不符时先保留原厂服务，用 `--without-prompt-integration` 安装，系统提示音统一抢占项仍待接入核验。适配器运行时也验证源码摘要，升级固件后不会继续盲用旧适配。

默认监听 `127.0.0.1:8092`。推荐从电脑转发：

```bash
ssh -L 8092:127.0.0.1:8092 用户名@机器鸭地址
```

然后浏览器打开本机端口，使用板端 `/var/lib/wonderduck-audio/access-token`。浏览器录音需要 HTTPS 或本机 localhost 的安全上下文。需要在受信任局域网直接访问时，首次安装可用 `--bind 0.0.0.0`，或修改配置后重启音频服务；普通 HTTP 的局域网网页可以上传和播放，但浏览器通常不允许麦克风录音。服务没有内置公网 TLS，不应直接映射到公网。

安装位置：

| 路径 | 用途 |
|---|---|
| `/opt/wonderduck-audio/current` | 当前源码版本的符号链接 |
| `/opt/wonderduck-audio/previous` | 上一版本的符号链接 |
| `/etc/wonderduck-audio/audio.toml` | 独立音频配置，升级保留 |
| `/var/lib/wonderduck-audio` | 音频库、数据库、设置、令牌和安装备份 |
| `/run/wonderduck-audio/audio.sock` | 板端 SDK 使用的 Unix socket，robot 组可访问 |
| `/etc/systemd/system/wonderduck-audio.service` | 音频服务 |
| 原厂提示音 service 的 `60-duckaudio.conf` | 可移除的适配入口，原厂源码不修改 |

安装器只将原厂 `/etc/robot/robotd.toml` 的 `audio.player` 设置为 `scripts/audio-output.py`，以便原厂声音也遵守功放共享锁；其余 TOML 字段保持，原配置备份。**不会自动重启 robotd 运动服务。** 如果输出 `robotd_restart_required: true`，在机器人安全空闲后正常重启机器人，再验收全部声音共存。首次部署前要核对声卡名称，`device`、`capture_device` 和功放脚本均可配置。

可用 `--without-legacy-player` 保留原厂 player 不变；这时需自行确认它已使用同一功放包装器，否则不能验收与原厂其他声音的互斥。默认安装不会停用原厂运动功能或宠物音频功能。

在电脑生成安装目录而不改变系统：

```bash
# stage 中先准备一份待测 /etc/robot/robotd.toml，或显式跳过原厂 player 改动
python3 deploy/audio/install.py install --root /tmp/audio-stage --without-legacy-player
```

## 任务中使用

本机通过 Unix socket 连接；远程用 HTTP/HTTPS 和 `token_file`。导入 SDK 时，板端任务进程的 `PYTHONPATH` 应包含 `/opt/wonderduck-audio/current`；也可以将这个 Python 包安装到任务自己的虚拟环境。

```python
from duckaudio import AudioClient
from duckaudio.tasks import TaskAudio

client = AudioClient("unix:/run/wonderduck-audio/audio.sock")
audio = TaskAudio(client, task_id="retrieval-20261123-001")
try:
    audio.emit("started")
    # 识别结果由任务模块提供，状态转换时触发：
    audio.emit("target_found")
    # 正式判定成功以后触发：
    future = audio.emit("succeeded")
finally:
    audio.close()  # 正常退出保留已提交的完成提示
```

`emit/play/say` 立即返回 Future，网络传输与 TTS 在后台执行；不要在控制循环调用 `future.result()` 或等待播完。需要等待时在任务编排协程处理：

```python
import asyncio

playback = await asyncio.wrap_future(audio.play("task.target_found", ttl=10))
result = await playback.wait_async(timeout=20)
# 取消任务：audio.cancel()；异常退出：audio.close(cancel=True)
```

`AudioClient.play()` 本身是同步短请求，返回播放句柄；面向实时任务优先使用 `TaskAudio`。`Playback.wait()` 超时会抛出异常，**不会自动停止声音**，需要调用 `stop()`；`failed/expired` 抛出 `AudioError`，`cancelled` 返回取消结果。默认音频失败写日志、保存在 `audio.errors`、传递到 Future，不改变任务的动作成功判定。

默认映射：

| 事件 | 资源键 |
|---|---|
| `started` | `task.started` |
| `target_found` | `task.target_found` |
| `succeeded` | `task.succeeded` |
| `failed` | `task.failed` |
| `needs_help` | `task.needs_help` |

可通过 `TaskAudio(..., events={...})` 自定义映射。默认同一实例每种事件一次；`once=False, cooldown=2` 可重复但防止每帧连播。每个真正的新任务用唯一 `task_id`。取消只影响本实例，仿真默认用 `LogAudioClient` 写入 `audio-events.jsonl`。

`tools/run_retrieval.py` 和 `tools/run_suite.py` 已接入实际事件。仅显式提供 `--audio-endpoint` 才连接外部音频服务；仅训练时不会让真机不断出声。示例和已授权嘴部回调接入见 `examples/audio/`。嘴部回调接收 0–0.65 的归一化开度，由任务控制器根据标定转换成自己的舵机指令；需要嘴部抓取的阶段应令 `allowed()` 返回 False，避免音频动作影响抓取。此回调接口不代表特定真机的舵机标定已经完成。

## 调度、音量与录音规则

- 默认队列最多 32 项；每个音频最长 300 秒，上传最大 20 MiB，音频库默认 512 MiB。循环 `repeat=0` 仍受单任务最长 3600 秒限制。
- `enqueue` 排队；`replace_task` 取消同实例旧声音；`interrupt` 只打断不高于自己优先级的当前声音；`reject` 忙时拒绝。优先级决定下一个待播声音的选择。
- 90–100 的系统优先级仅允许受信任 Unix socket；低电量为 100，开机/联网为 90。低电量可打断服务内较低优先级声音，被打断项为 `cancelled/preempted`，不会自行续播。
- 系统提示音在持有功放锁时临时使用最大音量并解除静音，结束或取消后恢复用户最新设置。网页显示临时覆盖状态。独立服务关闭播放或不可用时，原厂提示音适配回退到原厂逻辑；独立开关不替代原厂的系统告警配置。
- 原厂其他合成音/旧 Studio 播放仍通过共享锁互斥，不属于新队列，无法从新 API 定向取消或抢占这些旧进程。锁占用显示 `waiting`，过期或停止均可退出等待。
- 机身录音与本服务普通播放互斥。系统提示可中止机身录音，再播放告警；已采集有效内容仍入库。原厂程序或其他进程占用麦克风时会报真实声卡错误，不会伪造录音成功。
- `request_id` 重试返回同一个播放任务；同 ID 参数不同返回 409。保留最近 10,000 条终态请求的幂等记录，事件也有长度上限。服务重启将未完成播放标为 `failed/service_restarted`。

## 命令行和 API

```bash
duckaudio import /tmp/my-voice.mp3 --key retrieval.found --name 找到目标
duckaudio play retrieval.found --task-id retrieval-001 --wait
duckaudio play retrieval.found --repeat 0 --gain 0.6 --ttl 20
duckaudio jobs --task-id retrieval-001
duckaudio status 播放ID
duckaudio pause 播放ID
duckaudio seek 播放ID 1.5
duckaudio resume 播放ID
duckaudio stop --task-id retrieval-001
duckaudio settings --volume 50 --no-mute
duckaudio say '我已经找到目标了' --wait
duckaudio record start --seconds 10
duckaudio record stop --playback
duckaudio rename retrieval.found 我的提示音
duckaudio download retrieval.found /tmp/download.wav
duckaudio delete retrieval.found
duckaudio doctor
duckaudio events
duckaudio acceptance
```

HTTP 传输需 `Authorization: Bearer ...`，Unix socket 使用文件权限鉴权。API 返回机器可判定的 `error.code/message`；正常创建播放返回 202，不能将“提交成功”等同于已实际出声。

| 路径 | 方法/功能 |
|---|---|
| `/api/health` | GET 能力、后端、依赖状态 |
| `/api/settings` | GET / PUT 音量、静音、允许播放 |
| `/api/clips` | GET 列表；POST 原始音频体，query 中传 name/key |
| `/api/clips/{key或id}` | GET / PATCH / DELETE |
| `/api/clips/{key或id}/file`、`/envelope` | GET WAV 或嘴型包络 |
| `/api/play` | POST clip、task_id、request_id、repeat、gain、priority、policy、ttl |
| `/api/say` | POST text、voice、speed、key、name、play、task_id |
| `/api/jobs`、`/api/jobs/{id}` | GET 队列历史/单项状态 |
| `/api/jobs/{id}/{pause,resume,seek,stop}` | POST；seek 使用 `{"position": 秒数}` |
| `/api/stop`、`/api/stop-all` | POST；前者使用 task_id 或 job_id |
| `/api/recording`、`/start`、`/stop` | 状态 GET；开始/停止 POST |
| `/api/events?after=序号` | GET 增量事件 |
| `/api/acceptance` | GET / POST 验收记录 |

## 回退和诊断

```bash
sudo python3 deploy/audio/install.py rollback
sudo python3 deploy/audio/install.py uninstall
journalctl -u wonderduck-audio -u wonderduck-prompt-sounds -b -n 120 --no-pager
aplay -l
arecord -l
amixer -c aic3104 cget 'name=Line Playback Switch'
```

回退切换上一源码版本，保留音频和配置。卸载移除本安装器的服务入口并恢复未被其他工具修改的原厂 player 配置，保留音频、令牌和备份；若配置已改动则提示人工合并，不覆盖用户的新设置。原厂 player 的恢复也需要下一次 robotd 启动才生效。没有上一版本时回退会明确拒绝。

`mixer_failed` 检查声卡和 mixer 控件；`player_start_failed/player_failed` 检查功放脚本与 aplay；`recording_failed` 检查麦克风占用；`dependency_missing` 检查 FFmpeg/eSpeak；`clip_not_found` 检查事件资源键；`quota_exceeded` 清理音频或调整配额；`system_priority_denied` 表示从远程请求了系统保留优先级。
