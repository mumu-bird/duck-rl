from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
import uuid

from .errors import AudioError
from .media import terminate_process, validate_key, validate_text


class Recorder:
    def __init__(self, config, store, library, engine):
        self.config, self.store, self.library, self.engine = config, store, library, engine
        self.lock = threading.RLock()
        self.value = {"state": "idle", "clip_id": None, "error": None}
        self.process = None
        self.thread = None
        self.started = 0
        self.stopping = False
        self.play_after = False
        self.interrupted = threading.Event()

    def status(self):
        with self.lock:
            result = dict(self.value)
            result["elapsed"] = round(time.monotonic() - self.started, 2) if result["state"] == "recording" else None
            return result

    def start(self, name="机身录音", key=None, seconds=120, playback=False):
        if self.config.backend == "null":
            raise AudioError("当前为模拟播放模式，机身麦克风需在真机验收", "hardware_unavailable", 503)
        name = validate_text(name)
        key = validate_key(key or "recording_" + uuid.uuid4().hex[:12])
        if type(seconds) is not int or not 1 <= seconds <= self.config.max_record_seconds:
            raise AudioError(f"录音时长必须在 1–{self.config.max_record_seconds} 秒之间")
        with self.lock:
            if self.value["state"] in ("recording", "saving"):
                raise AudioError("已有录音正在进行", "recording_busy", 409)
            self.interrupted.clear()
            self.engine.reserve_capture(self.interrupted)
            path = self.store.tmp / (uuid.uuid4().hex + ".wav")
            log = path.with_suffix(".log")
            try:
                diagnostic = log.open("wb")
            except OSError:
                self.engine.release_capture(self.interrupted)
                raise
            try:
                self.process = subprocess.Popen([self.config.capture_program, "-q", "-D", self.config.capture_device,
                    "-t", "wav", "-f", "S16_LE", "-r", "48000", "-c", "2", "-d", str(seconds), str(path)],
                    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=diagnostic, start_new_session=True)
            except OSError as exc:
                diagnostic.close()
                self.engine.release_capture(self.interrupted)
                path.unlink(missing_ok=True)
                log.unlink(missing_ok=True)
                raise AudioError(f"无法启动机身录音: {exc}", "recording_failed", 503) from exc
            diagnostic.close()
            self.started = time.monotonic()
            self.stopping, self.play_after = False, bool(playback)
            self.value = {"state": "recording", "clip_id": None, "error": None}
            self.thread = threading.Thread(target=self._monitor, args=(path, log, name, key, seconds), daemon=True, name="duckaudio-recording")
            self.thread.start()
            return self.status()

    def stop(self, playback=None):
        with self.lock:
            if playback is not None:
                self.play_after = bool(playback)
            if self.process and self.process.poll() is None:
                self.stopping = True
                try:
                    os.killpg(self.process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
            return self.status()

    def _monitor(self, path, log, name, key, seconds):
        process = self.process
        try:
            stop_deadline = None
            while process.poll() is None:
                if self.interrupted.is_set() and not self.stopping:
                    self.stop(playback=False)
                if self.stopping:
                    stop_deadline = stop_deadline or time.monotonic() + 3
                if time.monotonic() - self.started > seconds + 5 or (stop_deadline and time.monotonic() > stop_deadline):
                    terminate_process(process)
                    raise AudioError("录音进程未及时停止", "recording_timeout")
                time.sleep(.05)
            if process.returncode and not self.stopping:
                raise AudioError("机身录音失败: " + log.read_text(errors="replace")[:2000], "recording_failed")
            self.engine.release_capture(self.interrupted)
            with self.lock:
                self.value["state"] = "saving"
            clip = self.library.import_file(path, name=name, key=key, source_type="recording")
            job = self.engine.submit(clip["id"], task_id="recorder") if self.play_after else None
            with self.lock:
                self.value = {"state": "idle", "clip_id": clip["id"], "job_id": job["id"] if job else None, "error": None}
        except Exception as exc:
            with self.lock:
                self.value = {"state": "failed", "clip_id": None, "error": str(exc)}
        finally:
            terminate_process(process)
            self.engine.release_capture(self.interrupted)
            path.unlink(missing_ok=True)
            log.unlink(missing_ok=True)
            with self.lock:
                self.process = None

    def close(self):
        self.stop(playback=False)
        if self.thread:
            self.thread.join(timeout=self.config.decode_timeout_seconds + 8)
            if self.thread.is_alive():
                raise RuntimeError("录音线程未停止，不能关闭存储")
