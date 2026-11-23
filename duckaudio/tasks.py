"""Non-blocking task events and opt-in animation of an already-authorized mouth."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import json
import logging
from pathlib import Path
import threading
import time
import uuid

from .errors import AudioError
from .store import TERMINAL

DEFAULT_EVENTS = {"started": "task.started", "target_found": "task.target_found",
                  "succeeded": "task.succeeded", "failed": "task.failed", "needs_help": "task.needs_help"}


class LoggedPlayback:
    def __init__(self, data):
        self.id, self.data = data["id"], data

    def status(self): return dict(self.data)
    def stop(self): return self.status()
    def wait(self, timeout=30, raise_on_error=True): return self.status()
    async def wait_async(self, timeout=30, raise_on_error=True): return self.status()


class LogAudioClient:
    """Simulation never contacts a robot or plays on the development machine."""
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.requests = {}

    def _write(self, data):
        with self.lock, self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(data, ensure_ascii=False) + "\n")

    def play(self, clip, *, task_id="simulation", request_id=None, **options):
        if request_id and request_id in self.requests:
            return self.requests[request_id]
        data = {"id": uuid.uuid4().hex, "state": "finished", "clip_key": clip, "task_id": task_id,
                "backend": "log", "audible_verified": False, "time": time.time(), **options}
        self._write(data)
        result = LoggedPlayback(data)
        if request_id:
            self.requests[request_id] = result
            if len(self.requests) > 10000:
                self.requests.pop(next(iter(self.requests)))
        return result

    def stop_task(self, task_id):
        self._write({"event": "cancel", "task_id": task_id, "time": time.time()})
        return {"task_id": task_id, "cancelled": 0}


class TaskAudio:
    """Task orchestration API. emit/play/say enqueue locally and return a Future immediately."""
    def __init__(self, client, *, task_id=None, events=None, logger=None, pending_limit=16):
        self.client = client
        self.task_id = task_id or uuid.uuid4().hex
        self.events = dict(DEFAULT_EVENTS if events is None else events)
        self.logger = logger or logging.getLogger("duckaudio.tasks")
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="task-audio")
        self.capacity = threading.BoundedSemaphore(pending_limit)
        self.lock = threading.Lock()
        self.seen = {}
        self.cancelled = False
        self.closed = False
        self.errors = []

    def _submit(self, action):
        with self.lock:
            if self.closed or self.cancelled:
                future = Future()
                future.set_exception(AudioError("任务音频已关闭", "task_closed", 409))
                return future
            if not self.capacity.acquire(False):
                future = Future()
                future.set_exception(AudioError("任务待发送音频已满", "task_queue_full", 429))
                return future
            def execute():
                try:
                    if self.cancelled:
                        return None
                    result = action()
                    if self.cancelled:
                        self.client.stop_task(self.task_id)
                    return result
                except Exception as exc:
                    self.errors.append(str(exc))
                    del self.errors[:-50]
                    self.logger.warning("任务 %s 音频失败: %s", self.task_id, exc)
                    raise
                finally:
                    self.capacity.release()
            return self.pool.submit(execute)

    def play(self, clip, **options):
        return self._submit(lambda: self.client.play(clip, task_id=self.task_id, **options))

    def emit(self, event, *, once=True, cooldown=2, **options):
        if event not in self.events:
            raise KeyError(f"未配置音频事件 {event}")
        now = time.monotonic()
        with self.lock:
            previous = self.seen.get(event)
            if previous and (once or now - previous[0] < cooldown):
                return previous[1]
            # Create a stable request ID for each actual event occurrence.
            placeholder = Future()
            self.seen[event] = (now, placeholder)
        actual = self.play(self.events[event], request_id=uuid.uuid4().hex, **options)
        def done(future):
            try:
                placeholder.set_result(future.result())
            except Exception as exc:
                placeholder.set_exception(exc)
        actual.add_done_callback(done)
        return placeholder

    def say(self, text, **options):
        return self._submit(lambda: self.client.say(text, task_id=self.task_id, **options))

    def cancel(self):
        with self.lock:
            self.cancelled = True
        # Enqueued work observes cancelled; after the in-flight submission, cancel its output too.
        if not self.closed:
            return self.pool.submit(self.client.stop_task, self.task_id)
        return None

    def close(self, *, cancel=False, wait=True):
        if cancel:
            self.cancel()
        with self.lock:
            self.closed = True
        self.pool.shutdown(wait=wait)


class MouthFollower:
    """Drive a callback supplied by the existing motor owner, never seize motor ownership."""
    def __init__(self, playback, envelope, set_mouth, *, allowed=lambda: True, interval=.05, maximum=.65):
        self.playback, self.envelope, self.set_mouth = playback, envelope, set_mouth
        self.allowed, self.interval, self.maximum = allowed, interval, maximum
        self.stop_event = threading.Event()
        self.error = None
        self.thread = threading.Thread(target=self._run, daemon=True, name="audio-mouth")

    def start(self):
        self.thread.start()
        return self

    def _run(self):
        try:
            while not self.stop_event.is_set():
                state = self.playback.status()
                if not self.allowed():
                    return
                if state["state"] in TERMINAL:
                    break
                index = int(state.get("position", 0) * 1000 / self.envelope["step_ms"])
                values = self.envelope["values"]
                amount = values[min(index, len(values)-1)] if values and state["state"] == "playing" else 0
                self.set_mouth(min(self.maximum, max(0, amount)))
                self.stop_event.wait(self.interval)
        except Exception as exc:
            self.error = str(exc)
        finally:
            if self.allowed():
                try:
                    self.set_mouth(0.0)
                except Exception as exc:
                    self.error = str(exc)

    def close(self):
        self.stop_event.set()
        self.thread.join(timeout=6)
