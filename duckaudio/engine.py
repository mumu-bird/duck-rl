from __future__ import annotations

import array
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave

from .errors import AudioError
from .media import RATE, terminate_process, validate_text
from .store import TERMINAL


class Cancelled(Exception):
    pass


class Reposition(Exception):
    pass


class PlaybackEngine:
    """One scheduler and one PCM owner; no calls into the motor-control daemon."""
    def __init__(self, config, store, library):
        self.config, self.store, self.library = config, store, library
        self.cv = threading.Condition(threading.RLock())
        self.jobs = {}
        self.queue = []
        self.current = None
        self.closed = False
        self.capture_interrupt = None
        self.mixer_error = None
        self.system_prompt_override = False
        self.enabled = store.get_setting("enabled", True)
        self.volume = store.get_setting("volume", 60)
        self.muted = store.get_setting("muted", False)
        self.thread = threading.Thread(target=self._worker, name="duckaudio-playback", daemon=True)
        self.thread.start()

    def _save(self, job, event=False):
        job["updated"] = time.time()
        public = self._public(job)
        self.store.save_job(public)
        if event:
            self.store.event("playback", {"id": job["id"], "task_id": job["task_id"], "state": job["state"],
                                          "error": job.get("error"), "position": job["position"]})

    @staticmethod
    def _public(job):
        return {k: v for k, v in job.items() if not k.startswith("_")}

    def _state(self, job, state, error=None, code=None):
        with self.cv:
            job.update(state=state, error=error, error_code=code)
            self._save(job, event=True)
            self.cv.notify_all()

    def submit(self, clip, *, task_id="manual", request_id=None, repeat=1, priority=10,
               policy="enqueue", ttl=120, gain=1.0):
        task_id = validate_text(task_id, "任务 ID", 128)
        if request_id is not None:
            request_id = validate_text(request_id, "请求 ID", 128)
        if type(repeat) is not int or not 0 <= repeat <= 100:
            raise AudioError("重复次数必须为 0–100；0 表示循环直到停止", "invalid_repeat")
        if type(priority) is not int or not 0 <= priority <= 100:
            raise AudioError("优先级必须在 0–100 之间", "invalid_priority")
        if policy not in ("enqueue", "replace_task", "interrupt", "reject"):
            raise AudioError("播放策略无效", "invalid_policy")
        if not isinstance(ttl, (int, float)) or isinstance(ttl, bool) or not math.isfinite(ttl) or not 1 <= ttl <= 3600:
            raise AudioError("排队有效期必须在 1–3600 秒", "invalid_ttl")
        if not isinstance(gain, (int, float)) or isinstance(gain, bool) or not math.isfinite(gain) or not 0 <= gain <= 1:
            raise AudioError("音频增益必须在 0–1 之间", "invalid_gain")
        payload = dict(clip=clip, task_id=task_id, repeat=repeat, priority=priority, policy=policy, ttl=ttl, gain=gain)
        fingerprint = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        with self.cv:
            previous = self.store.find_request(request_id)
            if previous:
                if previous["fingerprint"] != fingerprint:
                    raise AudioError("同一请求 ID 不能用于不同播放参数", "idempotency_conflict", 409)
                return self.status(previous["id"])
            if self.closed or not self.enabled:
                raise AudioError("音频服务已关闭播放", "audio_disabled", 409)
            if self.capture_interrupt is not None and priority < 90:
                raise AudioError("机身正在录音，请先停止录音", "recording_busy", 409)
            resource = self.store.get_clip(clip)
            self.library.path(resource)
            active = [j for j in self.jobs.values() if j["state"] not in TERMINAL]
            if policy == "reject" and active:
                raise AudioError("扬声器已有播放任务", "audio_busy", 409)
            if policy == "replace_task":
                self.stop_task(task_id)
                active = [j for j in self.jobs.values() if j["state"] not in TERMINAL and not j["_cancel"]]
            if len(active) >= self.config.max_queue + 1 and priority < 90:
                raise AudioError("播放队列已满", "queue_full", 429)
            if priority >= 90 and len(active) >= self.config.max_queue + 1:
                lowest = min(active, key=lambda j: (j["priority"], j["created"]))
                if lowest["priority"] >= priority:
                    raise AudioError("告警队列已满", "queue_full", 429)
                self._cancel(lowest, "被更高优先级提示替换", "preempted")
            now = time.time()
            job = dict(id=uuid.uuid4().hex, request_id=request_id, fingerprint=fingerprint,
                       clip_id=resource["id"], clip_key=resource["key"], name=resource["name"], task_id=task_id,
                       state="queued", created=now, updated=now, expires=now + ttl, priority=priority,
                       repeat=repeat, iteration=1, duration=resource["duration"], position=0.0, gain=float(gain),
                       error=None, error_code=None, backend=self.config.backend, audible_verified=False,
                       _clip=resource, _cancel=False, _paused=False, _seek=None, _process=None)
            if self.capture_interrupt is not None:
                self.capture_interrupt.set()
            self.jobs[job["id"]] = job
            self.queue.append(job["id"])
            self._save(job, event=True)
            if self.current:
                current = self.jobs[self.current]
                if (policy == "interrupt" and priority >= current["priority"]) or (priority >= 90 and priority > current["priority"]):
                    self._cancel(current, "被新播放任务打断", "preempted")
            self.cv.notify_all()
            return self._public(job)

    def status(self, identifier):
        with self.cv:
            return self._public(self.jobs[identifier]) if identifier in self.jobs else self.store.get_job(identifier)

    def list_jobs(self, task_id=None, limit=100):
        with self.cv:
            persisted = self.store.list_jobs(task_id=task_id, limit=limit)
            return [self._public(self.jobs[j["id"]]) if j["id"] in self.jobs else j for j in persisted]

    def _cancel(self, job, reason="用户停止", code="cancelled"):
        if job["state"] in TERMINAL:
            return
        job["_cancel"] = True
        job["_cancel_reason"], job["_cancel_code"] = reason, code
        if job["id"] != self.current:
            self._state(job, "cancelled", reason, code)
        self.cv.notify_all()

    def stop(self, identifier):
        with self.cv:
            job = self.jobs.get(identifier)
            if job:
                self._cancel(job)
                return self._public(job)
            return self.store.get_job(identifier)

    def stop_task(self, task_id):
        with self.cv:
            count = 0
            for job in self.jobs.values():
                if job["task_id"] == task_id and job["state"] not in TERMINAL:
                    self._cancel(job)
                    count += 1
            return {"task_id": task_id, "cancelled": count}

    def stop_all(self):
        with self.cv:
            for job in self.jobs.values():
                self._cancel(job)
            return {"stopped": True}

    def control(self, identifier, action, position=None):
        with self.cv:
            job = self.jobs.get(identifier)
            if not job or job["state"] in TERMINAL:
                raise AudioError("该播放任务已经结束", "not_active", 409)
            if action == "stop":
                return self.stop(identifier)
            if identifier != self.current:
                raise AudioError("只能暂停或跳转当前播放任务", "not_playing", 409)
            if action == "pause":
                job["_paused"] = True
                self._state(job, "paused")
            elif action == "resume":
                job["_paused"] = False
                self._state(job, "playing" if job.get("_process") or self.config.backend == "null" else "waiting")
            elif action == "seek":
                if isinstance(position, bool) or not isinstance(position, (int, float)) or not math.isfinite(position) or not 0 <= position < job["duration"]:
                    raise AudioError("跳转位置超出音频范围", "invalid_position")
                job["_seek"] = float(position)
            else:
                raise AudioError("未知播放控制操作", "invalid_action")
            self.cv.notify_all()
            return self._public(job)

    def delete_clip(self, identifier):
        with self.cv:
            clip = self.store.get_clip(identifier)
            if any(j["clip_id"] == clip["id"] and j["state"] not in TERMINAL for j in self.jobs.values()):
                raise AudioError("该音频正在播放或排队，请先停止对应任务", "clip_in_use", 409)
            self.store.delete_clip(clip["id"])
            return {"deleted": clip["id"]}

    def reserve_capture(self, interrupted):
        with self.cv:
            if self.capture_interrupt is not None or any(j["state"] not in TERMINAL for j in self.jobs.values()):
                raise AudioError("请先停止播放或录音，再开始机身录音", "playback_busy", 409)
            self.capture_interrupt = interrupted

    def release_capture(self, interrupted):
        with self.cv:
            if self.capture_interrupt is interrupted:
                self.capture_interrupt = None
                self.cv.notify_all()

    def settings(self):
        return dict(enabled=self.enabled, volume=self.volume, muted=self.muted,
                    backend=self.config.backend, mixer_error=self.mixer_error,
                    system_prompt_override=self.system_prompt_override)

    def _apply_mixer(self, new_volume, new_muted):
        card = self.config.device.split(":", 1)[-1].split(",", 1)[0].removeprefix("CARD=")
        operations = []
        if new_muted or new_volume == 0:
            operations.append(["cset", "name=Line Playback Switch", "off,off"])
        else:
            operations += [["cset", "name=PCM Playback Volume", f"{new_volume}%,{new_volume}%"],
                           ["cset", "name=Line Playback Switch", "on,on"]]
        for args in operations:
            try:
                result = subprocess.run([self.config.mixer_program, "-c", card, *args], capture_output=True, timeout=5)
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise AudioError(f"无法设置声卡音量: {exc}", "mixer_failed", 503) from exc
            if result.returncode:
                raise AudioError("声卡音量设置失败: " + result.stderr.decode("utf-8", "replace")[:1000], "mixer_failed", 503)

    def set_settings(self, *, volume=None, muted=None, enabled=None):
        if volume is not None and (type(volume) is not int or not 0 <= volume <= 100):
            raise AudioError("音量必须为 0–100 的整数", "invalid_volume")
        if muted is not None and type(muted) is not bool:
            raise AudioError("muted 必须为布尔值")
        if enabled is not None and type(enabled) is not bool:
            raise AudioError("enabled 必须为布尔值")
        with self.cv:
            new_volume = self.volume if volume is None else volume
            new_muted = self.muted if muted is None else muted
            if self.config.backend != "null" and (volume is not None or muted is not None):
                if not self.system_prompt_override:
                    self._apply_mixer(new_volume, new_muted)
            self.mixer_error = None
            self.volume, self.muted = new_volume, new_muted
            for key, value in (("volume", self.volume), ("muted", self.muted)):
                self.store.set_setting(key, value)
            if enabled is not None:
                self.enabled = enabled
                self.store.set_setting("enabled", enabled)
                if not enabled:
                    self.stop_all()
            self.store.event("settings", self.settings())
            return self.settings()

    def _check(self, job):
        if self.closed or job["_cancel"]:
            raise Cancelled()
        if not job.get("_started_playback") and time.time() >= job["expires"]:
            raise AudioError("等待扬声器时提示已过期", "queue_expired", 408)
        if time.monotonic() > job["_deadline"]:
            raise AudioError("播放任务超过最长运行时间", "playback_timeout", 408)
        if job["_seek"] is not None:
            raise Reposition()

    def _pace(self, job, seconds):
        end = time.monotonic() + seconds
        with self.cv:
            while True:
                self._check(job)
                if job["_paused"]:
                    process = job.get("_process")
                    if process and process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGSTOP)
                        except ProcessLookupError:
                            pass
                    paused_at = time.monotonic()
                    while job["_paused"]:
                        self.cv.wait(.05)
                        self._check(job)
                    end += time.monotonic() - paused_at
                    if process and process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGCONT)
                        except ProcessLookupError:
                            pass
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return
                self.cv.wait(min(.02, remaining))

    def _lock_output(self, job):
        if self.config.backend == "null":
            return None
        path = Path(self.config.lock_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(path, os.O_RDONLY | os.O_CREAT | os.O_NOFOLLOW, 0o666)
        lock = os.fdopen(fd, "rb")
        try:
            while True:
                self._check(job)
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return lock
                except BlockingIOError:
                    self._pace(job, .05)
        except Exception:
            lock.close()
            raise

    def _start_player(self, job, lock, diagnostics):
        if self.config.backend == "null":
            return None
        player = self.config.player if self.config.backend == "board" else (self.config.player if self.config.player != "/home/wonderduck/wonderduck/scripts/audio-output.py" else "aplay")
        args = [sys.executable, player] if player.endswith(".py") else [player]
        if self.config.backend == "board":
            args += ["--lock-fd", str(lock.fileno()), "--warmup-prompt-ms", "0"]
        args += ["-q", "-D", self.config.device, "-t", "raw", "-f", "S16_LE", "-c", "1", "-r", str(RATE)]
        try:
            process = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=diagnostics, start_new_session=True, pass_fds=(lock.fileno(),), bufsize=0)
            os.set_blocking(process.stdin.fileno(), False)
            job["_process"] = process
            return process
        except OSError as exc:
            raise AudioError(f"无法启动实际播放器: {exc}", "player_start_failed", 503) from exc

    def _write(self, job, data):
        process = job.get("_process")
        if process is None:
            return
        pending = memoryview(data)
        while pending:
            self._pace(job, 0)
            if process.poll() is not None:
                raise AudioError("播放器提前退出", "player_failed", 503)
            if select.select([], [process.stdin.fileno()], [], .05)[1]:
                try:
                    written = os.write(process.stdin.fileno(), pending)
                    pending = pending[written:]
                except BlockingIOError:
                    continue
                except BrokenPipeError as exc:
                    raise AudioError("播放器输入已关闭", "player_failed", 503) from exc

    def _play_segment(self, job, offset):
        lock = None
        process = None
        override = False
        with tempfile.TemporaryFile() as diagnostics:
            try:
                self._state(job, "paused" if job["_paused"] else "waiting")
                lock = self._lock_output(job)
                if self.config.backend != "null" and job["priority"] >= 90:
                    with self.cv:
                        override = self.system_prompt_override = True
                        self._apply_mixer(100, False)
                process = self._start_player(job, lock, diagnostics)
                if self.config.backend == "board":
                    warmup_frames = int(RATE * self.config.warmup_ms / 1000)
                    while warmup_frames:
                        frames = min(960, warmup_frames)
                        self._write(job, bytes(frames * 2))
                        self._pace(job, frames / RATE)
                        warmup_frames -= frames
                job["_started_playback"] = True
                self._state(job, "paused" if job["_paused"] else "playing")
                with wave.open(str(self.library.path(job["_clip"])), "rb") as source:
                    source.setpos(min(int(offset * RATE), source.getnframes()))
                    job["position"] = offset
                    saved_at = time.monotonic()
                    while data := source.readframes(960):
                        self._pace(job, 0)
                        if job["gain"] != 1:
                            values = array.array("h", data)
                            if sys.byteorder != "little":
                                values.byteswap()
                            values = array.array("h", (int(x * job["gain"]) for x in values))
                            if sys.byteorder != "little":
                                values.byteswap()
                            data = values.tobytes()
                        self._write(job, data)
                        self._pace(job, len(data) / 2 / RATE)
                        with self.cv:
                            job["position"] = min(job["duration"], source.tell() / RATE)
                            if time.monotonic() - saved_at >= .5:
                                self._save(job)
                                saved_at = time.monotonic()
                if process:
                    process.stdin.close()
                    drain_deadline = time.monotonic() + 3
                    while process.poll() is None:
                        self._pace(job, .02)
                        if time.monotonic() > drain_deadline:
                            raise AudioError("播放器未在预期时间结束", "player_drain_timeout", 503)
                    if process.returncode:
                        diagnostics.seek(0)
                        raise AudioError("播放器执行失败: " + diagnostics.read(2048).decode("utf-8", "replace"), "player_failed", 503)
            finally:
                terminate_process(process)
                if process and process.stdin and not process.stdin.closed:
                    process.stdin.close()
                job["_process"] = None
                if override:
                    with self.cv:
                        self.system_prompt_override = False
                        try:
                            self._apply_mixer(self.volume, self.muted)
                        except AudioError as exc:
                            self.mixer_error = str(exc)
                            self.store.event("mixer_error", {"error": str(exc)})
                if lock:
                    lock.close()

    def _run(self, job):
        job["_deadline"] = time.monotonic() + self.config.max_job_seconds
        offset = 0.0
        while True:
            try:
                self._play_segment(job, offset)
                self._check(job)
            except Reposition:
                with self.cv:
                    offset, job["_seek"] = job["_seek"], None
                    job["position"] = offset
                continue
            if job["repeat"] and job["iteration"] >= job["repeat"]:
                return
            job["iteration"] += 1
            offset = 0.0

    def _worker(self):
        while True:
            with self.cv:
                self.queue = [i for i in self.queue if self.jobs[i]["state"] not in TERMINAL]
                if self.closed:
                    return
                if not self.queue or self.capture_interrupt is not None:
                    self.cv.wait(.2)
                    continue
                for identifier in list(self.queue):
                    job = self.jobs[identifier]
                    if time.time() >= job["expires"]:
                        self._state(job, "expired", "排队提示已过期", "queue_expired")
                        self.queue.remove(identifier)
                if not self.queue:
                    continue
                identifier = min(self.queue, key=lambda i: (-self.jobs[i]["priority"], self.jobs[i]["created"]))
                self.queue.remove(identifier)
                self.current = identifier
                job = self.jobs[identifier]
            try:
                self._run(job)
                self._state(job, "finished")
            except Cancelled:
                self._state(job, "cancelled", job.get("_cancel_reason", "服务停止"), job.get("_cancel_code", "cancelled"))
            except AudioError as exc:
                self._state(job, "expired" if exc.code == "queue_expired" else "failed", str(exc), exc.code)
            except Exception as exc:
                self._state(job, "failed", f"播放失败: {exc}", "playback_error")
            finally:
                with self.cv:
                    self.current = None
                    # Bound in-memory history; persistent history supports later status queries.
                    terminals = [i for i, j in self.jobs.items() if j["state"] in TERMINAL]
                    for old in terminals[:-200]:
                        self.jobs.pop(old, None)
                    self.cv.notify_all()

    def close(self):
        with self.cv:
            self.stop_all()
            self.closed = True
            self.cv.notify_all()
        self.thread.join(timeout=6)
        if self.thread.is_alive():
            raise RuntimeError("音频线程未停止，不能关闭存储")
