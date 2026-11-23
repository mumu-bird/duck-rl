from __future__ import annotations

import array
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import wave

from .errors import AudioError

RATE = 48000
KEY = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,95}$")


def validate_key(value):
    if not isinstance(value, str) or not KEY.fullmatch(value) or value in (".", ".."):
        raise AudioError("资源键须为 1–96 位字母、数字、点、下划线或短横线", "invalid_key")
    return value


def validate_text(value, label="名称", maximum=160):
    if not isinstance(value, str) or not value.strip() or len(value) > maximum or any(ord(c) < 32 and c not in "\n\t" for c in value):
        raise AudioError(f"{label}为空、过长或包含控制字符", "invalid_text")
    return value.strip()


def terminate_process(process, timeout=1.0):
    if process is None:
        return
    deadline = time.monotonic() + timeout
    try:
        os.killpg(process.pid, signal.SIGCONT)
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    # The wrapper can exit before its child; inspect the group, not just its leader.
    while True:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        if time.monotonic() >= deadline:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            break
        time.sleep(.02)
    if process.poll() is None:
        process.wait(timeout=2)


def run_process(args, timeout, *, env=None):
    # Diagnostics go to a bounded-read temporary file instead of an unbounded PIPE.
    with tempfile.TemporaryFile() as log:
        try:
            process = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=log, start_new_session=True, env=env)
        except OSError as exc:
            raise AudioError(f"无法启动 {Path(args[0]).name}: {exc}", "dependency_missing", 503) from exc
        try:
            status = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise AudioError("音频处理超时", "processing_timeout", 408) from exc
        finally:
            terminate_process(process)
        if status:
            log.seek(0)
            detail = log.read(4096).decode("utf-8", "replace").strip()
            raise AudioError("音频处理失败: " + detail, "decode_failed", 422)


def wav_info(path, *, normalized=False):
    try:
        with wave.open(str(path), "rb") as handle:
            frames, rate = handle.getnframes(), handle.getframerate()
            if not frames or not rate:
                raise AudioError("音频没有可播放的采样", "empty_audio")
            if normalized and (handle.getnchannels(), handle.getsampwidth(), rate) != (1, 2, RATE):
                raise AudioError("内部音频格式不符合 PCM16/48kHz/单声道", "invalid_format")
            expected = frames * handle.getnchannels() * handle.getsampwidth()
            actual = 0
            while block := handle.readframes(32768):
                actual += len(block)
            if actual != expected:
                raise AudioError("音频数据被截断", "truncated_audio")
            return {"duration": frames / rate, "frames": frames, "sample_rate": rate,
                    "channels": handle.getnchannels(), "sample_width": handle.getsampwidth()}
    except (wave.Error, EOFError, OSError) as exc:
        raise AudioError(f"无法读取 WAV: {exc}", "invalid_audio", 422) from exc


def envelope(path, step_ms=40):
    values = []
    with wave.open(str(path), "rb") as handle:
        count = max(1, int(handle.getframerate() * step_ms / 1000))
        while data := handle.readframes(count):
            samples = array.array("h", data)
            if sys.byteorder != "little":
                samples.byteswap()
            rms = math.sqrt(sum(x*x for x in samples) / max(1, len(samples))) / 32768
            values.append(round(min(1.0, max(0.0, (rms - 0.012) * 6)), 3))
    return {"step_ms": step_ms, "values": values}


class MediaLibrary:
    def __init__(self, config, store):
        self.config, self.store = config, store
        self.processing = threading.BoundedSemaphore(2)
        self.commit_lock = threading.RLock()
        self.tts_lock = threading.Lock()

    def path(self, clip):
        path = self.store.clips / clip["filename"]
        if path.is_symlink() or not path.is_file() or path.resolve().parent != self.store.clips.resolve():
            raise AudioError("资源文件丢失或路径无效", "clip_file_missing", 404)
        return path

    def import_file(self, source, *, key=None, name=None, source_type="upload", metadata=None):
        source = Path(source)
        if source.is_symlink() or not source.is_file():
            raise AudioError("只接受普通音频文件", "invalid_file")
        if source_type not in ("recording", "tts") and source.stat().st_size > self.config.max_upload_bytes:
            raise AudioError("上传文件超过大小限制", "upload_too_large", 413)
        identifier = uuid.uuid4().hex
        key = validate_key(key or "clip_" + identifier[:12])
        name = validate_text(name or key)
        if not self.processing.acquire(blocking=False):
            raise AudioError("正在处理其他音频，请稍后重试", "processing_busy", 429)
        temp = self.store.tmp / (identifier + ".wav")
        try:
            # Even canonical WAVs are fully scanned; malformed/truncated input is rejected.
            try:
                info = wav_info(source, normalized=True)
            except AudioError as exc:
                if exc.code in ("truncated_audio", "empty_audio"):
                    raise
                info = None
            if info:
                if info["duration"] > self.config.max_duration_seconds:
                    raise AudioError("音频时长超过限制", "audio_too_long", 413)
                with source.open("rb") as inp, temp.open("xb") as out:
                    while block := inp.read(65536):
                        out.write(block)
            else:
                run_process([self.config.ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-y",
                    "-protocol_whitelist", "file,pipe", "-format_whitelist", "wav,mp3,ogg,flac,aac,mov,matroska,webm",
                    "-i", str(source), "-map", "0:a:0", "-vn", "-sn", "-dn", "-t", str(self.config.max_duration_seconds + 1),
                    "-ac", "1", "-ar", str(RATE), "-c:a", "pcm_s16le", "-f", "wav", str(temp)], self.config.decode_timeout_seconds)
                info = wav_info(temp, normalized=True)
                if info["duration"] > self.config.max_duration_seconds + .001:
                    raise AudioError("音频时长超过限制，未保存截断版本", "audio_too_long", 413)
            size = temp.stat().st_size
            metadata = dict(metadata or {})
            metadata["format"] = "PCM16/48000/mono"
            metadata["envelope"] = envelope(temp)
            with temp.open("rb") as handle:
                hasher = hashlib.sha256()
                while block := handle.read(65536):
                    hasher.update(block)
                digest = hasher.hexdigest()
            clip = {"id": identifier, "key": key, "name": name, "filename": identifier + ".wav",
                    "sha256": digest, "duration": info["duration"], "bytes": size,
                    "source": source_type, "created": time.time(), "metadata": metadata}
            with self.commit_lock:
                if self.store.total_bytes() + size > self.config.max_library_bytes:
                    raise AudioError("音频库空间配额不足", "quota_exceeded", 507)
                target = self.store.clips / clip["filename"]
                os.replace(temp, target)
                try:
                    self.store.add_clip(clip)
                except Exception:
                    target.unlink(missing_ok=True)
                    raise
            self.store.event("clip_imported", {"id": identifier, "key": key})
            return clip
        finally:
            temp.unlink(missing_ok=True)
            self.processing.release()

    def speak(self, text, *, voice=None, speed=160, key=None, name=None):
        text = validate_text(text, "播报文字", 2000)
        voice = voice or self.config.tts_voice
        if not isinstance(voice, str) or not re.fullmatch(r"[a-zA-Z0-9_+/-]{1,48}", voice) or voice.startswith("-"):
            raise AudioError("语音名称无效", "invalid_voice")
        if type(speed) is not int or not 80 <= speed <= 350:
            raise AudioError("语速应在 80–350 之间", "invalid_speed")
        signature = hashlib.sha256(json.dumps([text, voice, speed, self.config.tts_program, self.config.tts_data], ensure_ascii=False).encode()).hexdigest()
        cache_key = "tts_" + signature[:24]
        if not self.tts_lock.acquire(blocking=False):
            raise AudioError("正在合成其他语音，请稍后重试", "processing_busy", 429)
        try:
            if key is None:
                try:
                    cached = self.store.get_clip(cache_key)
                    self.path(cached)
                    return cached
                except AudioError as exc:
                    if exc.status != 404:
                        raise
            text_file = self.store.tmp / (uuid.uuid4().hex + ".txt")
            wav_file = text_file.with_suffix(".wav")
            try:
                text_file.write_text(text, encoding="utf-8")
                args = [self.config.tts_program, "-v", voice, "-s", str(speed), "-f", str(text_file), "-w", str(wav_file)]
                if self.config.tts_data:
                    args += ["--path=" + self.config.tts_data]
                run_process(args, self.config.decode_timeout_seconds)
                return self.import_file(wav_file, key=key or cache_key, name=name or text[:50], source_type="tts",
                                        metadata={"text": text, "voice": voice, "speed": speed, "cache": signature})
            finally:
                text_file.unlink(missing_ok=True)
                wav_file.unlink(missing_ok=True)
        finally:
            self.tts_lock.release()
