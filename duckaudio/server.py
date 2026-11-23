from __future__ import annotations

import fcntl
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import socketserver
import stat
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit
import uuid

from . import __version__
from .engine import PlaybackEngine
from .errors import AudioError
from .media import MediaLibrary, validate_key, validate_text
from .recording import Recorder
from .store import Store

FEATURES = [
 ("library", "上传、转换、列表、重命名、删除与下载", "software"),
 ("playback", "播放、停止、暂停、继续与跳转", "software"),
 ("queue", "队列、重复、过期与优先级", "software"),
 ("task", "任务隔离、去重、取消和失败反馈", "software"),
 ("tts", "中文文字播报与缓存", "software"),
 ("browser_recording", "浏览器录音、保存与播放", "browser"),
 ("speaker", "真机发声、首字完整和音质", "hardware"),
 ("volume", "真机音量、静音和设置恢复", "hardware"),
 ("microphone", "机身录音、保存和回放", "hardware"),
 ("concurrent_motion", "自主任务播放不影响运动控制", "hardware"),
 ("system_prompts", "开机及低电量提示与任务音频协调", "hardware"),
 ("mouth", "已有授权任务控制器的嘴部联动", "hardware"),
 ("restart", "服务重启与资源保留、旧任务清理", "software"),
]


class Application:
    def __init__(self, config):
        self.config = config.validate()
        directory = Path(config.data_dir)
        directory.mkdir(parents=True, exist_ok=True, mode=0o750)
        self.instance = (directory / ".service.lock").open("a+")
        try:
            fcntl.flock(self.instance, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self.instance.close()
            raise AudioError("该数据目录已有音频服务运行", "already_running", 409) from exc
        token_path = Path(config.token_file)
        token_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            fd = os.open(token_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, "w") as handle:
                handle.write(secrets.token_urlsafe(32) + "\n")
        except FileExistsError:
            if token_path.is_symlink() or not token_path.is_file():
                raise AudioError("访问令牌文件无效", "invalid_token_file")
        os.chmod(token_path, 0o600)
        self.token = token_path.read_text().strip()
        if len(self.token) < 24:
            raise AudioError("访问令牌至少需要 24 个字符", "invalid_token_file")
        self.store = Store(directory)
        self.library = MediaLibrary(config, self.store)
        self.engine = PlaybackEngine(config, self.store, self.library)
        self.recorder = Recorder(config, self.store, self.library, self.engine)
        if config.backend != "null":
            try:
                self.engine.set_settings(volume=self.engine.volume, muted=self.engine.muted)
            except AudioError as exc:
                self.engine.mixer_error = str(exc)
                self.store.event("mixer_error", {"error": str(exc)})
        self.http = self.unix = None
        self.threads = []
        self.active_servers = []
        self.closed = False

    def capabilities(self):
        def available(program):
            return bool(shutil.which(program)) or (Path(program).is_file() and os.access(program, os.X_OK))
        return {"version": __version__, "backend": self.config.backend,
                "simulated": self.config.backend == "null", "hardware_verified": False,
                "ffmpeg": available(self.config.ffmpeg), "tts": available(self.config.tts_program),
                "tts_voice": self.config.tts_voice, "recording": self.config.backend != "null" and available(self.config.capture_program),
                "player_present": self.config.backend == "null" or Path(self.config.player).is_file() or bool(shutil.which(self.config.player)),
                "limits": {"upload_bytes": self.config.max_upload_bytes, "duration_seconds": self.config.max_duration_seconds,
                           "record_seconds": self.config.max_record_seconds, "queue": self.config.max_queue},
                "settings": self.engine.settings(), "recording_status": self.recorder.status()}

    def acceptance(self):
        saved = self.store.get_setting("acceptance", {})
        return [{"id": key, "name": name, "scope": scope, **saved.get(key, {"result": "pending", "notes": ""})}
                for key, name, scope in FEATURES]

    def save_acceptance(self, data):
        key = data.get("id")
        feature = next((f for f in FEATURES if f[0] == key), None)
        if not feature or data.get("result") not in ("pass", "fail", "pending"):
            raise AudioError("验收项目或结果无效")
        if feature[2] == "hardware" and self.config.backend == "null" and data["result"] == "pass":
            raise AudioError("模拟模式不能将真机项目标记为通过", "hardware_unverified", 409)
        saved = self.store.get_setting("acceptance", {})
        saved[key] = {"result": data["result"], "notes": str(data.get("notes", ""))[:2000], "time": time.time(), "backend": self.config.backend}
        self.store.set_setting("acceptance", saved)
        return self.acceptance()

    def start(self):
        try:
            self.http = LimitedHTTPServer((self.config.host, self.config.port), Handler)
            self.http.application, self.http.local = self, False
            path = Path(self.config.socket)
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o750)
            if path.exists() or path.is_symlink():
                if path.is_symlink() or not stat.S_ISSOCK(path.stat().st_mode):
                    raise AudioError("Unix socket 路径已被其他文件占用")
                with socket.socket(socket.AF_UNIX) as probe:
                    try:
                        probe.connect(str(path))
                    except OSError:
                        path.unlink()
                    else:
                        raise AudioError("Unix socket 已有服务监听", "already_running", 409)
            self.unix = LocalHTTPServer(str(path), Handler)
            self.unix.application, self.unix.local = self, True
            os.chmod(path, 0o660)
            for server in (self.http, self.unix):
                thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .1}, daemon=True)
                thread.start()
                self.threads.append(thread)
                self.active_servers.append(server)
            return self
        except Exception:
            self.close()
            raise

    def close(self):
        if self.closed:
            return
        self.closed = True
        for server in self.active_servers:
            server.shutdown()
        for server in (self.http, self.unix):
            if server:
                server.disconnect_clients()
                server.server_close()
        self.recorder.close()
        self.engine.close()
        self.store.close()
        if self.unix:
            Path(self.config.socket).unlink(missing_ok=True)
        self.instance.close()


class LimitedHTTPServer(ThreadingHTTPServer):
    daemon_threads = False
    block_on_close = True
    allow_reuse_address = True

    def __init__(self, *args, **kwargs):
        self.slots = threading.BoundedSemaphore(24)
        self.clients = set()
        self.clients_lock = threading.Lock()
        super().__init__(*args, **kwargs)

    def process_request(self, request, client_address):
        if not self.slots.acquire(False):
            try:
                request.sendall(b"HTTP/1.1 503 Service Unavailable\r\nConnection: close\r\nContent-Length: 0\r\n\r\n")
            finally:
                self.shutdown_request(request)
            return
        with self.clients_lock:
            self.clients.add(request)
        try:
            super().process_request(request, client_address)
        except Exception:
            with self.clients_lock:
                self.clients.discard(request)
            self.slots.release()
            raise

    def disconnect_clients(self):
        with self.clients_lock:
            for request in tuple(self.clients):
                try:
                    request.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            with self.clients_lock:
                self.clients.discard(request)
            self.slots.release()


class LocalHTTPServer(LimitedHTTPServer):
    address_family = socket.AF_UNIX

    def server_bind(self):
        socketserver.TCPServer.server_bind(self)
        self.server_name, self.server_port = "localhost", 0


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "WonderDuckAudio/1.0"

    def setup(self):
        super().setup()
        self.connection.settimeout(30)

    def log_message(self, *_):
        pass

    @property
    def app(self):
        return self.server.application

    def _respond(self, data, status=200, content_type="application/json; charset=utf-8"):
        try:
            self._send_response(data, status, content_type)
        except (ConnectionError, TimeoutError):
            self.close_connection = True

    def _send_response(self, data, status=200, content_type="application/json; charset=utf-8"):
        if isinstance(data, (dict, list)):
            data = json.dumps(data, ensure_ascii=False, allow_nan=False).encode()
        elif isinstance(data, str):
            data = data.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Connection", "close")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'self'")
        origin = self.headers.get("Origin")
        if origin in self.app.config.trusted_origins:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)
        self.close_connection = True

    def _authenticate(self):
        if self.server.local:
            return
        token = self.headers.get("Authorization", "")
        if not hmac.compare_digest(token, "Bearer " + self.app.token):
            raise AudioError("请输入音频服务访问令牌", "unauthorized", 401)
        origin = self.headers.get("Origin")
        if origin and origin not in self.app.config.trusted_origins:
            parsed = urlsplit(origin)
            if parsed.netloc != self.headers.get("Host") or parsed.scheme not in ("http", "https"):
                raise AudioError("请求来源不允许", "origin_denied", 403)

    def _length(self, limit):
        if self.headers.get("Transfer-Encoding"):
            raise AudioError("请使用带 Content-Length 的上传", "length_required", 411)
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise AudioError("Content-Length 无效") from exc
        if length < 0 or length > limit:
            raise AudioError("请求体超过限制", "upload_too_large", 413)
        return length

    def _json(self):
        size = self._length(64 * 1024)
        data = self.rfile.read(size)
        if len(data) != size:
            raise AudioError("请求数据不完整")
        try:
            value = json.loads(data or b"{}", parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        except (ValueError, UnicodeDecodeError) as exc:
            raise AudioError("JSON 请求无效", "invalid_json") from exc
        if not isinstance(value, dict):
            raise AudioError("请求内容必须是 JSON 对象")
        return value

    def do_OPTIONS(self):
        origin = self.headers.get("Origin")
        if origin not in self.app.config.trusted_origins:
            self._respond({"error": "origin denied"}, 403)
            return
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Headers", "Authorization, Content-Type")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, PATCH, DELETE, OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_GET(self): self._dispatch()
    def do_HEAD(self): self._dispatch()
    def do_POST(self): self._dispatch()
    def do_PUT(self): self._dispatch()
    def do_PATCH(self): self._dispatch()
    def do_DELETE(self): self._dispatch()

    def _dispatch(self):
        try:
            parts = urlsplit(self.path)
            path = unquote(parts.path)
            query = {k: v[-1] for k, v in parse_qs(parts.query).items()}
            method = "GET" if self.command == "HEAD" else self.command
            static = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"), "/style.css": ("style.css", "text/css; charset=utf-8")}
            if path in static and method == "GET":
                name, content_type = static[path]
                return self._respond((Path(__file__).parent / "web" / name).read_bytes(), content_type=content_type)
            self._authenticate()
            result, status = self._api(method, path, query)
            if result is not _SENT:
                return self._respond(result, status)
        except AudioError as exc:
            self._respond(exc.as_dict(), exc.status)
        except (TypeError, ValueError, KeyError) as exc:
            self._respond(AudioError("请求参数无效: " + str(exc), "invalid_request").as_dict(), 400)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            self.close_connection = True
        except Exception as exc:
            self.app.store.event("api_error", {"type": type(exc).__name__, "message": str(exc)[:500]})
            self._respond(AudioError("音频服务处理失败，请查看事件日志", "internal_error", 500).as_dict(), 500)

    def _api(self, method, path, query):
        app = self.app
        if path == "/api/health" and method == "GET":
            return app.capabilities(), 200
        if path == "/api/settings" and method in ("GET", "PUT"):
            return (app.engine.settings(), 200) if method == "GET" else (app.engine.set_settings(**self._json()), 200)
        if path == "/api/clips" and method == "GET":
            clips = app.store.list_clips()
            # Envelope is fetched separately to keep listing small.
            for clip in clips:
                clip["metadata"].pop("envelope", None)
            return {"clips": clips, "bytes": app.store.total_bytes(), "quota": app.config.max_library_bytes}, 200
        if path == "/api/clips" and method == "POST":
            size = self._length(app.config.max_upload_bytes)
            if not size:
                raise AudioError("上传音频为空", "empty_audio")
            temp = app.store.tmp / (uuid.uuid4().hex + ".upload")
            try:
                with temp.open("xb") as output:
                    remaining = size
                    while remaining:
                        block = self.rfile.read(min(65536, remaining))
                        if not block:
                            raise AudioError("上传数据不完整", "truncated_upload")
                        output.write(block)
                        remaining -= len(block)
                clip = app.library.import_file(temp, key=query.get("key"), name=query.get("name"), source_type="upload")
                return clip, 201
            finally:
                temp.unlink(missing_ok=True)
        if path.startswith("/api/clips/"):
            suffix = path[len("/api/clips/"):].split("/")
            clip = app.store.get_clip(suffix[0])
            if len(suffix) == 2 and suffix[1] == "file" and method == "GET":
                data = app.library.path(clip).read_bytes()
                self._respond(data, content_type="audio/wav")
                return _SENT, 200
            if len(suffix) == 2 and suffix[1] == "envelope" and method == "GET":
                return clip["metadata"]["envelope"], 200
            if len(suffix) != 1:
                raise AudioError("接口不存在", "not_found", 404)
            if method == "GET":
                return clip, 200
            if method == "PATCH":
                data = self._json()
                name = validate_text(data.get("name", clip["name"]))
                key = validate_key(data["key"]) if "key" in data else None
                return app.store.rename_clip(clip["id"], name, key), 200
            if method == "DELETE":
                return app.engine.delete_clip(clip["id"]), 200
        if path == "/api/play" and method == "POST":
            data = self._json()
            if not self.server.local and data.get("priority", 10) >= 90:
                raise AudioError("系统优先级仅允许本机受信任调用", "system_priority_denied", 403)
            return app.engine.submit(**data), 202
        if path == "/api/say" and method == "POST":
            data = self._json()
            play = data.pop("play", True)
            task_id = data.pop("task_id", "speech")
            request_id = data.pop("request_id", None)
            clip = app.library.speak(**data)
            job = app.engine.submit(clip["id"], task_id=task_id, request_id=request_id) if play else None
            return {"clip": clip, "job": job}, 201
        if path == "/api/jobs" and method == "GET":
            return {"jobs": app.engine.list_jobs(task_id=query.get("task_id"), limit=int(query.get("limit", 100)))}, 200
        if path.startswith("/api/jobs/"):
            parts = path[len("/api/jobs/"):].split("/")
            if len(parts) == 1 and method == "GET":
                return app.engine.status(parts[0]), 200
            if len(parts) == 2 and method == "POST":
                data = self._json()
                return app.engine.control(parts[0], parts[1], **data), 200
        if path == "/api/stop" and method == "POST":
            data = self._json()
            if "job_id" in data:
                return app.engine.stop(data["job_id"]), 200
            if "task_id" in data:
                return app.engine.stop_task(data["task_id"]), 200
            raise AudioError("请指定 job_id 或 task_id")
        if path == "/api/stop-all" and method == "POST":
            return app.engine.stop_all(), 200
        if path == "/api/recording" and method == "GET":
            return app.recorder.status(), 200
        if path == "/api/recording/start" and method == "POST":
            return app.recorder.start(**self._json()), 202
        if path == "/api/recording/stop" and method == "POST":
            return app.recorder.stop(**self._json()), 202
        if path == "/api/events" and method == "GET":
            return {"events": app.store.events(after=int(query.get("after", 0)))}, 200
        if path == "/api/acceptance":
            if method == "GET":
                return {"items": app.acceptance(), "backend": app.config.backend}, 200
            if method == "POST":
                return {"items": app.save_acceptance(self._json())}, 200
        raise AudioError("接口不存在", "not_found", 404)


_SENT = object()
