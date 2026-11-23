from __future__ import annotations

import dataclasses
import os
from pathlib import Path
import shutil
try:
    import tomllib
except ModuleNotFoundError:  # Board images using Python 3.10.
    import tomli as tomllib

from .errors import AudioError


def executable(name):
    if name == "ffmpeg":
        override = os.environ.get("DUCKAUDIO_FFMPEG")
        if override:
            return override
        found = shutil.which(name)
        if found:
            return found
        try:
            import imageio_ffmpeg
            return imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    return shutil.which(name) or name


@dataclasses.dataclass
class Config:
    data_dir: str = "/var/lib/wonderduck-audio"
    backend: str = "board"
    player: str = "/home/wonderduck/wonderduck/scripts/audio-output.py"
    device: str = "plughw:aic3104"
    capture_device: str = "plughw:aic3104,0"
    capture_program: str = "arecord"
    mixer_program: str = "amixer"
    lock_file: str = "/run/lock/audio-output.lock"
    socket: str = "/run/wonderduck-audio/audio.sock"
    host: str = "127.0.0.1"
    port: int = 8092
    token_file: str = ""
    ffmpeg: str = ""
    tts_program: str = ""
    tts_voice: str = "cmn"
    tts_data: str = ""
    warmup_ms: int = 500
    max_upload_bytes: int = 20 * 1024 * 1024
    max_duration_seconds: int = 300
    max_library_bytes: int = 512 * 1024 * 1024
    max_queue: int = 32
    max_job_seconds: int = 3600
    decode_timeout_seconds: int = 90
    max_record_seconds: int = 120
    trusted_origins: list[str] = dataclasses.field(default_factory=list)

    def validate(self):
        if self.backend not in ("board", "alsa", "null"):
            raise AudioError("backend 必须是 board、alsa 或 null")
        for name in ("data_dir", "socket", "lock_file"):
            if not Path(getattr(self, name)).is_absolute():
                raise AudioError(f"{name} 必须使用绝对路径")
        if not 0 <= self.port <= 65535:
            raise AudioError("端口超出范围")
        for name in ("max_upload_bytes", "max_duration_seconds", "max_library_bytes", "max_queue",
                     "max_job_seconds", "decode_timeout_seconds", "max_record_seconds"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise AudioError(f"{name} 必须是正整数")
        if not 0 <= self.warmup_ms <= 3000:
            raise AudioError("预热时长必须在 0–3000 ms")
        self.ffmpeg = self.ffmpeg or executable("ffmpeg")
        self.tts_program = self.tts_program or shutil.which("espeak-ng") or shutil.which("espeak") or "espeak-ng"
        self.token_file = self.token_file or str(Path(self.data_dir) / "access-token")
        return self

    @classmethod
    def load(cls, path=None, **overrides):
        data = {}
        if path:
            with open(path, "rb") as handle:
                data = tomllib.load(handle).get("audio", {})
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise AudioError("未知配置项: " + ", ".join(sorted(unknown)))
        data.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**data).validate()
