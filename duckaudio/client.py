from __future__ import annotations

import asyncio
import http.client
import json
from pathlib import Path
import socket
import time
from urllib.parse import urlencode, urlsplit, quote
import uuid

from .errors import AudioError
from .store import TERMINAL


class UnixConnection(http.client.HTTPConnection):
    def __init__(self, path, timeout):
        super().__init__("localhost", timeout=timeout)
        self.path = path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.path)


class Playback:
    def __init__(self, client, data):
        self.client, self.id = client, data["id"]
        self.initial = data

    def status(self):
        return self.client.request("GET", "/api/jobs/" + self.id)

    def stop(self):
        return self.client.request("POST", "/api/stop", {"job_id": self.id})

    def pause(self):
        return self.client.request("POST", f"/api/jobs/{self.id}/pause", {})

    def resume(self):
        return self.client.request("POST", f"/api/jobs/{self.id}/resume", {})

    def seek(self, seconds):
        return self.client.request("POST", f"/api/jobs/{self.id}/seek", {"position": seconds})

    def wait(self, timeout=30, raise_on_error=True):
        deadline = time.monotonic() + timeout
        while True:
            result = self.status()
            if result["state"] in TERMINAL:
                if raise_on_error and result["state"] in ("failed", "expired"):
                    raise AudioError(result.get("error") or "播放失败", result.get("error_code") or "playback_failed", 503)
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError("等待播放完成超时；任务未自动取消")
            time.sleep(min(.05, max(0, deadline - time.monotonic())))

    async def wait_async(self, timeout=30, raise_on_error=True):
        return await asyncio.to_thread(self.wait, timeout, raise_on_error)


class AudioClient:
    def __init__(self, endpoint="unix:/run/wonderduck-audio/audio.sock", *, token=None, token_file=None, timeout=5):
        self.endpoint, self.timeout = endpoint, timeout
        self.token = Path(token_file).read_text().strip() if token_file else token
        if not endpoint.startswith("unix:") and urlsplit(endpoint).scheme not in ("http", "https"):
            raise ValueError("音频地址须使用 unix:、http:// 或 https://")

    def _connection(self, timeout):
        if self.endpoint.startswith("unix:"):
            return UnixConnection(self.endpoint[5:], timeout), ""
        parsed = urlsplit(self.endpoint)
        cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
        return cls(parsed.hostname, parsed.port, timeout=timeout), parsed.path.rstrip("/")

    def request(self, method, path, data=None, *, body=None, content_length=None, timeout=None, binary=False):
        connection, prefix = self._connection(timeout or self.timeout)
        headers = {}
        if self.token:
            headers["Authorization"] = "Bearer " + self.token
        if data is not None:
            body = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
            headers["Content-Type"] = "application/json"
        if content_length is not None:
            headers["Content-Length"] = str(content_length)
            headers["Content-Type"] = "application/octet-stream"
        try:
            connection.request(method, prefix + path, body=body, headers=headers)
            response = connection.getresponse()
            raw = response.read()
            if response.status >= 400:
                try:
                    error = json.loads(raw)["error"]
                except (ValueError, KeyError, TypeError):
                    error = {"message": raw.decode("utf-8", "replace")[:1000], "code": "http_error"}
                raise AudioError(error["message"], error.get("code", "http_error"), response.status)
            return raw if binary else json.loads(raw)
        except (OSError, http.client.HTTPException) as exc:
            raise AudioError(f"无法连接音频服务: {exc}", "connection_failed", 503) from exc
        finally:
            connection.close()

    def health(self):
        return self.request("GET", "/api/health")

    def clips(self):
        return self.request("GET", "/api/clips")["clips"]

    def has_clip(self, key):
        try:
            self.request("GET", "/api/clips/" + quote(key, safe=""))
            return True
        except AudioError as exc:
            if exc.status == 404:
                return False
            raise

    def import_file(self, path, *, key=None, name=None):
        path = Path(path)
        query = urlencode({k: v for k, v in {"key": key, "name": name or path.stem}.items() if v is not None})
        with path.open("rb") as handle:
            return self.request("POST", "/api/clips?" + query, body=handle, content_length=path.stat().st_size, timeout=120)

    def play(self, clip, *, task_id="manual", request_id=None, **options):
        data = self.request("POST", "/api/play", dict(clip=clip, task_id=task_id, request_id=request_id or uuid.uuid4().hex, **options))
        return Playback(self, data)

    def say(self, text, *, task_id="speech", play=True, **options):
        result = self.request("POST", "/api/say", dict(text=text, task_id=task_id, play=play, **options), timeout=180)
        if result["job"]:
            result["playback"] = Playback(self, result["job"])
        return result

    def stop_task(self, task_id):
        return self.request("POST", "/api/stop", {"task_id": task_id})

    def settings(self, **changes):
        return self.request("PUT", "/api/settings", changes) if changes else self.request("GET", "/api/settings")

    def delete(self, identifier):
        return self.request("DELETE", "/api/clips/" + quote(identifier, safe=""))

    def rename(self, identifier, name, key=None):
        return self.request("PATCH", "/api/clips/" + quote(identifier, safe=""), {"name": name, **({"key": key} if key else {})})

    def download(self, identifier, path):
        data = self.request("GET", "/api/clips/" + quote(identifier, safe="") + "/file", binary=True)
        Path(path).write_bytes(data)
        return Path(path)
