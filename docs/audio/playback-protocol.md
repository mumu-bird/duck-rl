# 小黄鸭自定义音频播放协议

交付日期：2026-11-23。本文是 `duckaudio` 音频服务的**播放协议说明**：任何人（网页控制台、命令行、自己的程序）按本文即可在小黄鸭上播放自己想要的音频。协议以 HTTP over Unix socket / TCP 实现，与运动控制服务 robotd 相互独立。

适用读者：要在机器鸭上播放自有音频的使用者，以及要把播放能力接进自己任务的开发者。总体工程背景见 [音频服务 README](README.md)，逐项验收状态见 [acceptance.md](acceptance.md)。

## 1. 架构与角色

```text
┌─────────────┐   HTTP    ┌──────────────────────────────┐
│ 网页控制台   │ ────────▶ │ duckaudio serve（板端常驻）    │
│ 浏览器       │  带令牌    │  · 音频库（转码 PCM16/48k/单声道）│
├─────────────┤           │  · 播放调度（队列/优先级/策略）   │
│ duckaudio   │ ────────▶ │  · 中文 TTS（eSpeak NG）        │
│ CLI / SDK   │  Unix socket├─────────────┬──────────────────┤
└─────────────┘  文件权限  │             │ 持功放锁调用原厂
                           │             ▼
                           │   audio-output.py → aplay → 功放
                           └──────────────────────────────┘
```

- 服务端：`python3 -m duckaudio serve`，板端由 systemd 单元 `wonderduck-audio` 托管。
- 客户端：`duckaudio` CLI、Python SDK（`duckaudio.client.AudioClient`）、任意能发 HTTP 的程序（curl 等）。
- 播放**不经过** robotd，不抢占运动控制权；与原厂提示音通过功放文件锁互斥。

## 2. 传输与鉴权

### 2.1 服务端点

| 形态 | 地址 | 鉴权方式 |
|---|---|---|
| Unix socket（板端本机默认） | `unix:/run/wonderduck-audio/audio.sock` | 文件系统权限（需在对应用户组） |
| HTTP（局域网/ SSH 转发） | `http://127.0.0.1:8092` | `Authorization: Bearer <token>` |

- 访问令牌在板端 `/var/lib/wonderduck-audio/access-token`（本机验收时在数据目录下），首次安装随机生成，**不要**放进 URL、日志或验收导出。
- 从电脑连接板端的推荐方式（令牌不经过网络监听面缩小）：

```bash
ssh -L 8092:127.0.0.1:8092 用户名@机器鸭地址
```

- 所有请求响应均为 JSON（下载音频文件除外），错误返回结构统一为：

```json
{"error": {"code": "clip_not_found", "message": "音频不存在"}}
```

### 2.2 内容约定

- JSON 请求体使用 `Content-Type: application/json`，UTF-8。
- 上传音频使用原始字节体，`Content-Type: application/octet-stream`，必须带 `Content-Length`。
- 导入的音频统一转码为 **PCM16 / 48 kHz / 单声道 WAV** 后入库；支持 WAV、MP3、M4A、OGG、FLAC 和浏览器 WebM。
- 上传音频必须**非空**；空体返回 `empty_audio`。

## 3. 播放自己音频的三步流程

```text
① 导入 ──▶ ② 播放 ──▶ ③ 监听/控制
```

### 第 1 步：导入音频（获得资源键）

把任意受支持格式的文件导入音频库。导入是可重复的安全操作：同名 key 已存在时保留原资源，可用返回的 `id` 覆盖更新。

```bash
# CLI：导入并指定稳定资源键和显示名
duckaudio import /tmp/my-voice.mp3 --key greeting.hello --name 开机问候
```

```python
from duckaudio import AudioClient

client = AudioClient()  # 默认 unix:/run/wonderduck-audio/audio.sock
clip = client.import_file("/tmp/my-voice.mp3", key="greeting.hello", name="开机问候")
print(clip["key"], clip["id"], clip["duration"])
```

HTTP 等价形式：

```bash
curl -sS --unix-socket /run/wonderduck-audio/audio.sock \
  -X POST "http://localhost/api/clips?key=greeting.hello&name=%E5%BC%80%E6%9C%BA%E9%97%AE%E5%80%99" \
  -H "Content-Type: application/octet-stream" \
  --data-binary @/tmp/my-voice.mp3
```

返回 `201` 与音频对象（字段节选）：

```json
{
  "id": "9f2c…", "key": "greeting.hello", "name": "开机问候",
  "duration": 3.42, "bytes": 328704, "created": 1763900000.0,
  "metadata": {"source": "upload", "sha256": "…"}
}
```

> `key` 是业务使用的稳定标识（建议 `模块.动作` 命名，如 `retrieval.found`）；`id` 是内部 ID。后续播放用 key 或 id 均可。

### 第 2 步：提交播放

```bash
duckaudio play greeting.hello --task-id demo-001 --wait
```

```python
playback = client.play("greeting.hello", task_id="demo-001")
result = playback.wait(timeout=30)   # 阻塞到终态；不等则返回初始任务对象
print(result["state"])               # finished
```

HTTP 等价形式：

```bash
curl -sS --unix-socket /run/wonderduck-audio/audio.sock \
  -X POST http://localhost/api/play \
  -H "Authorization: Bearer $(cat /var/lib/wonderduck-audio/access-token)" \
  -H "Content-Type: application/json" \
  -d '{"clip": "greeting.hello", "task_id": "demo-001"}'
```

返回 `202` 与播放任务对象（详见 [第 5 节](#5-播放任务对象与状态机)）。**202 只代表“已受理”，不等于已经出声**；确认出声需看任务到达 `playing` 或 `finished`。

### 第 3 步：监听与控制

```bash
duckaudio jobs --task-id demo-001      # 队列/历史
duckaudio status 播放ID                 # 单个任务状态
duckaudio pause 播放ID / resume / seek 播放ID 1.5
duckaudio stop --task-id demo-001       # 停止一个任务的全部声音
duckaudio stop --all                    # 停止所有声音
```

```python
status = playback.status()              # GET /api/jobs/{id}
playback.pause(); playback.resume()
playback.seek(1.5)                      # 跳到 1.5 秒
playback.stop()                         # 停止这个播放
```

## 4. HTTP API 协议详表

除特别说明外，均需 `Authorization: Bearer <token>`（Unix socket 本机调用可省略）。

### 4.1 音频库

| 方法与路径 | 功能 | 请求 | 成功响应 |
|---|---|---|---|
| `GET /api/clips` | 音频列表、已用空间与配额 | query：无 | `200 {"clips": [...], "bytes": n, "quota": n}` |
| `POST /api/clips` | 导入音频 | 原始字节体；query 传 `name`、`key` | `201` 音频对象 |
| `GET /api/clips/{key或id}` | 音频详情 | — | `200` 音频对象 |
| `PATCH /api/clips/{key或id}` | 改名/改键 | `{"name": "新名", "key": "新键"}` | `200` 音频对象 |
| `DELETE /api/clips/{key或id}` | 删除音频 | — | `200 {"deleted": id}` |
| `GET /api/clips/{key或id}/file` | 下载 WAV | — | `200 audio/wav` 字节流 |
| `GET /api/clips/{key或id}/envelope` | 嘴部包络数据（0–0.65 归一化开度序列） | — | `200` 包络数组 |

### 4.2 播放

| 方法与路径 | 功能 | 请求体 | 成功响应 |
|---|---|---|---|
| `POST /api/play` | 提交播放 | 见 [4.2.1](#421-播放参数) | `202` 播放任务对象 |
| `GET /api/jobs` | 任务列表 | query：`task_id`、`limit`（默认 100） | `200 {"jobs": [...]}` |
| `GET /api/jobs/{id}` | 单个任务状态 | — | `200` 播放任务对象 |
| `POST /api/jobs/{id}/pause` | 暂停（仅当前播放中任务） | `{}` | `200` 播放任务对象 |
| `POST /api/jobs/{id}/resume` | 继续 | `{}` | `200` 播放任务对象 |
| `POST /api/jobs/{id}/seek` | 跳转 | `{"position": 秒数}` | `200` 播放任务对象 |
| `POST /api/jobs/{id}/stop` | 停止单个播放 | `{}` | `200` 播放任务对象 |
| `POST /api/stop` | 按任务或任务 ID 停止 | `{"job_id": …}` 或 `{"task_id": …}` | `200 {"stopped": n}` |
| `POST /api/stop-all` | 停止全部声音 | `{}` | `200 {"stopped": n}` |

#### 4.2.1 播放参数

`POST /api/play` 请求体字段：

| 字段 | 类型 | 默认 | 约束与语义 |
|---|---|---|---|
| `clip` | string | 必填 | 音频的资源键 `key` 或内部 `id` |
| `task_id` | string | `"manual"` | ≤128 字符。同一任务的声音可整体停止/替换；建议传业务实例 ID |
| `request_id` | string | 随机生成 | ≤128 字符。重试安全：同 ID 同参数返回**同一个**任务；同 ID 不同参数返回 `409 idempotency_conflict` |
| `repeat` | int | `1` | `0–100`；`0` 表示循环播放直到停止 |
| `priority` | int | `10` | `0–100`；越大越先播。`≥90` 为系统保留（低电量 100、开机/联网 90），仅本机受信任 Unix socket 可提交，远程提交返回 `403 system_priority_denied` |
| `policy` | string | `"enqueue"` | `enqueue` 排队；`replace_task` 取消同 `task_id` 旧声音后播放；`interrupt` 只打断不高于自己优先级的当前声音；`reject` 扬声器忙时拒绝（`409 audio_busy`） |
| `ttl` | number | `120` | `1–3600` 秒，排队有效期；超时未播则 `expired/queue_expired` |
| `gain` | number | `1.0` | `0–1` 单次增益，叠加在系统音量之上 |

### 4.3 文字转语音

`POST /api/say`：本地 eSpeak NG 离线合成中文语音（默认声音 `cmn`），可选择直接播放或仅保存。

| 字段 | 类型 | 默认 | 说明 |
|---|---|---|---|
| `text` | string | 必填 | 要播报的文字 |
| `voice` | string | 服务配置 | 声音标识，如 `cmn` |
| `speed` | int | `160` | 语速（词/分钟） |
| `key` / `name` | string | 自动 | 合成结果的保存键/名 |
| `play` | bool | `true` | `false` 时只保存不播放 |
| `task_id` / `request_id` | string | `"speech"` / 随机 | 同 `/api/play` |

返回 `201 {"clip": {...}, "job": {...} 或 null}`。

### 4.4 音量与开关

`GET /api/settings` 返回 `{"volume": 60, "muted": false, "enabled": true}`；
`PUT /api/settings` 接受 `volume`（0–100）、`muted`、`enabled`，持久化并在重启后恢复。真实 mixer 设置失败时报告 `mixer_failed` 事件，不伪造成功。

### 4.5 事件流

`GET /api/events?after=序号` 返回增量事件列表（播放状态迁移、错误、系统提示等）。客户端记录已见到的最大序号，下次用该序号续传，实现轻量监听。

### 4.6 其他

| 方法与路径 | 功能 |
|---|---|
| `GET /api/health` | 能力与依赖状态：版本、后端、ffmpeg/tts 可用性、限额、当前设置 |
| `POST /api/recording/start`、`POST /api/recording/stop`、`GET /api/recording` | 机身/浏览器录音的开始、停止与状态 |
| `GET /api/acceptance`、`POST /api/acceptance` | 验收清单读取与记录 |

## 5. 播放任务对象与状态机

### 5.1 任务对象字段（节选）

```json
{
  "id": "3ab7…", "request_id": "…", "clip_key": "greeting.hello",
  "name": "开机问候", "task_id": "demo-001",
  "state": "playing", "position": 1.20, "duration": 3.42,
  "priority": 10, "repeat": 1, "iteration": 1, "gain": 1.0,
  "created": 1763900000.0, "updated": 1763900001.2, "expires": 1763900120.0,
  "backend": "board", "audible_verified": false,
  "error": null, "error_code": null
}
```

### 5.2 状态机

```text
                 ┌─────────────── 取消/抢占/停止 ───────────────┐
                 ▼                                            │
queued ──▶ waiting ──▶ playing ──▶ finished                  cancelled
(排队等ttl)  (等功放锁)   │  ▲        │                          ▲
   │                     │  └─ paused ┘（pause/resume/seek）    │
   │ ttl 超时             │ 失败                                │
   ▼                     ▼                                    │
expired ◀──────────── failed ◀─────────────────────────────────┘
```

| 状态 | 含义 |
|---|---|
| `queued` | 已受理，在队列等待 |
| `waiting` | 等待原厂功放锁（原厂提示音等占用时临时出现） |
| `playing` | 正在出声 |
| `paused` | 已暂停 |
| `finished` | 正常播完（终态） |
| `cancelled` | 被取消/抢占/停止（终态，`error_code` 如 `preempted`、`cancelled`、`service_restarted`） |
| `failed` | 失败（终态，见 `error_code`） |
| `expired` | 排队超时未播（终态，`queue_expired`） |

只有 `finished` 代表播完；`waiting`/`queued` 都不代表已出声。服务重启会把未完成任务标为 `failed/service_restarted`，**不会**自动重播。

## 6. CLI 快速参考

```text
duckaudio [--endpoint unix:/run/wonderduck-audio/audio.sock] [--token-file 路径] <命令>

import  文件 [--key 键] [--name 名]     导入音频
play    音频键 [--task-id ID] [--repeat 次数|0循环] [--priority 0-100]
                 [--policy enqueue|replace_task|interrupt|reject]
                 [--gain 0-1] [--ttl 秒] [--request-id ID] [--wait]
say     文字 [--voice cmn] [--speed 160] [--key 键] [--save-only] [--wait]
jobs    [--task-id ID]                  任务列表
status  播放ID                          任务状态
pause   播放ID / resume 播放ID
seek    播放ID 秒数
stop    --task-id ID | --job-id ID | --all
rename  音频键 新名 [--key 新键]
download 音频键 本地文件
delete  音频键
settings [--volume 0-100] [--mute|--no-mute] [--enabled|--no-enabled]
record  start [--seconds 秒] [--name 名] [--playback] | stop [--playback] | status
list / doctor / events / acceptance
serve   [--config 文件] [--data-dir 目录] [--backend board|alsa|null]
        [--host 地址] [--port 端口] [--socket 路径]
```

## 7. Python SDK 示例

```python
from duckaudio import AudioClient

# 板端本机默认走 Unix socket；跨机用 http://地址 并传 token
client = AudioClient("unix:/run/wonderduck-audio/audio.sock")

# 任务内播放：同一实例的声音可整体替换、取消
def on_target_found(task):
    if not client.has_clip("retrieval.found"):
        client.import_file("assets/audio/found.wav", key="retrieval.found")
    job = client.play("retrieval.found", task_id=task, policy="replace_task",
                      request_id=f"{task}:found")
    return job.wait(timeout=10, raise_on_error=False)

# 中文播报（合成失败会抛 AudioError，code=decode_failed）
client.say("我已经找到目标了", task_id="speech", play=True)

# 循环播放背景音乐直到显式停止
bg = client.play("bgm.loop", task_id="ambient", repeat=0, gain=0.5)
# ... 需要时 client.stop_task("ambient")
```

## 8. 错误码

所有错误均返回 `{"error": {"code": "...", "message": "..."}}`，HTTP 状态码如下表。

### 连接与鉴权

| code | HTTP | 场景与处理 |
|---|---|---|
| `connection_failed` | 503 | 连不上音频服务：检查服务状态与端点 |
| `unauthorized` | 401 | 令牌缺失或不匹配：使用数据目录下 `access-token` 文件内容 |
| `origin_denied` | 403 | 网页来源不受信任：将来源加入 `trusted_origins` |
| `not_found` | 404 | 接口路径不存在 |
| `already_running` | 409 | 同一数据目录已有音频服务在运行 |

### 音频库与导入

| code | HTTP | 场景与处理 |
|---|---|---|
| `clip_not_found` | 404 | 音频键/ID 不存在：先导入或核对 key |
| `clip_file_missing` | 404 | 音频文件丢失：重新导入 |
| `clip_in_use` | 409 | 音频正在播放中，不能删除：先停止 |
| `duplicate_key` | 409 | 资源键已被占用：换 key 或先删除/改名 |
| `length_required` | 411 | 上传未带 `Content-Length` |
| `upload_too_large` | 413 | 超过单文件上限（默认 20 MiB） |
| `audio_too_long` | 413 | 超过单音频时长上限（默认 300 秒） |
| `quota_exceeded` | 507 | 超出音频库总容量：清理或调大 `max_library_bytes` |
| `empty_audio` | 400 | 上传为空或音频无有效采样 |
| `truncated_upload` | 400 | 上传数据不完整：重传 |
| `invalid_audio` | 422 | 文件不是有效音频或无法解析 |
| `invalid_file` | 400 | 不是普通音频文件（如目录、设备文件） |
| `decode_failed` | 422 | 转码或语音合成失败：检查 FFmpeg / eSpeak NG 及文件本身 |
| `processing_timeout` | 408 | 转码/合成超时（默认 90 秒） |
| `dependency_missing` | 503 | 缺 FFmpeg/eSpeak NG 或其无法启动 |
| `invalid_key` / `invalid_text` | 400 | 资源键或名称不合法（长度/字符集） |

### 播放与调度

| code | HTTP | 场景与处理 |
|---|---|---|
| `job_not_found` | 404 | 播放任务 ID 不存在 |
| `invalid_repeat` / `invalid_priority` / `invalid_policy` / `invalid_ttl` / `invalid_gain` | 400 | 播放参数越界，按 [4.2.1](#421-播放参数) 修正 |
| `invalid_position` | 400 | seek 位置非法（超出音频时长） |
| `audio_busy` | 409 | `reject` 策略下扬声器忙：改用 `enqueue` 或稍后重试 |
| `recording_busy` | 409 | 机身录音中：先停止录音 |
| `audio_disabled` | 409 | 播放开关被关闭：`duckaudio settings --enabled` |
| `not_playing` | 409 | 暂停/跳转的目标不是当前播放任务 |
| `not_active` | 409 | 操作与任务当前状态不兼容 |
| `queue_full` | 429 | 队列已满：降低并发或停止旧任务 |
| `task_queue_full` | 429 | 单任务待播数量超限 |
| `processing_busy` | 429 | 正在合成其他语音，稍后重试 |
| `queue_expired` | 408 | 排队超时未播（任务终态 `expired`） |
| `idempotency_conflict` | 409 | 同一 `request_id` 用了不同参数：换 ID 或统一参数 |
| `system_priority_denied` | 403 | 远程提交系统保留优先级（≥90） |
| `playback_timeout` | 408 | 播放进程超时无输出 |
| `player_start_failed` | 503 | 原厂功放脚本或 aplay 启动失败 |
| `player_failed` | 503 | 播放进程异常退出 |
| `player_drain_timeout` | 503 | 播放收尾超时 |
| `mixer_failed` | 503 | 声卡/mixer 控件异常，检查 `amixer -c aic3104` |
| `hardware_unavailable` | 503 | 后端不可用（声卡/设备缺失） |
| `hardware_unverified` | 409 | 硬件未验收，操作被拒绝 |
| `playback_failed` | 503 | 播放失败（SDK `wait()` 默认抛出），查看事件与 journalctl |
| `service_restarted` | — | 服务重启导致任务失败，需重新提交 |

### 录音与语音合成

| code | HTTP | 场景与处理 |
|---|---|---|
| `recording_failed` | 503 | 麦克风被占用或录音失败 |
| `recording_timeout` | 408 | 录音超时 |
| `invalid_voice` / `invalid_speed` | 400 | 声音名或语速（80–350）不合法 |
| `invalid_volume` | 400 | 音量超出 0–100 |

## 9. 限额与行为约定

| 项 | 默认值 | 说明 |
|---|---|---|
| 上传大小 | 20 MiB | `max_upload_bytes` |
| 单个音频时长 | 300 秒 | `max_duration_seconds` |
| 音频库总容量 | 512 MiB | `max_library_bytes` |
| 队列长度 | 32 | `max_queue`；`repeat=0` 循环仍受单任务 3600 秒上限 |
| 排队有效期 | 120 秒 | `ttl`，范围 1–3600 |
| 机身录音 | 120 秒 | `max_record_seconds` |
| 幂等记录 | 10,000 条 | 终态请求的 `request_id` 记录 |

行为约定：

- 机身录音与普通播放互斥；系统提示（低电量等）可中止录音转播告警。
- 原厂其他合成音/旧 Studio 播放**不在**新队列内，只能通过共享锁互斥，无法用新 API 定向取消。
- `null` 后端（本机模拟）只模拟播放耗时，**不代表真机已验证出声**；`/api/health` 的 `hardware_verified` 反映硬件验收状态。
- 浏览器控制台与录音需要 HTTPS 或 localhost 安全上下文；局域网纯 HTTP 可上传播放但不能用麦克风。

## 10. 相关文档

- [音频服务 README](README.md)：能力总览、板端安装、运维与回退
- [验收清单](acceptance.md)：13 组验收项与当前状态
- 源码：`duckaudio/`（`client.py` SDK、`server.py` HTTP 服务、`engine.py` 调度、`media.py` 转码、`cli.py` 命令行）
