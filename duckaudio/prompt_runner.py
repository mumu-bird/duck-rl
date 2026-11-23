"""Run the unchanged manufacturer prompt daemon with a reversible scheduler adapter."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time

from .client import AudioClient
from .errors import AudioError

# Filled from the reviewed manufacturer snapshot; changed firmware falls back unchanged.
SUPPORTED_SOURCE_SHA256 = "6bac64c15de5f2f5999e94f24912e73e874c1f2effd1b56b0ce428974749716e"


def attach(module, endpoint):
    original = module.PromptService.launch
    original_sync = getattr(module.PromptService, "sync_volume", None)

    def ready(self):
        now = time.monotonic()
        if now >= getattr(self, "_audio_probe_after", 0):
            try:
                health = AudioClient(endpoint, timeout=.5).health()
                self._audio_ready = health["settings"]["enabled"]
            except (AudioError, OSError):
                self._audio_ready = False
            self._audio_probe_after = now + .5
        return self._audio_ready

    def sync_volume(self, now):
        if not ready(self) or (self.child is not None and not getattr(self, "_shared_child", False)):
            return original_sync(self, now)
        # On recovery from the legacy fallback, restore its saved mixer snapshot once.
        if self.volume is not None and (self.volume.saved is not None or self.volume.error):
            try:
                self.volume.restore()
                client = AudioClient(endpoint, timeout=3)
                settings = client.settings()
                client.settings(volume=settings["volume"], muted=settings["muted"])
            except (AudioError, OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
                self.warn(f"Cannot restore legacy prompt volume: {exc}", now)
        self.low_volume_asserted = False

    def launch(self, kind, now):
        if not ready(self):
            self._shared_child = False
            return original(self, kind, now)
        source = None
        try:
            arguments = [sys.executable, "-m", "duckaudio.prompt_bridge", "--endpoint", endpoint, "--kind", kind]
            timeout = module.PLAYER_TIMEOUT + 12
            if kind == "network":
                data, _automatic = self.pending_network
                source = tempfile.TemporaryFile()
                source.write(data)
                source.seek(0)
                timeout = max(timeout, len(data) / 48000.0 + 17)
            else:
                sound = self.sound_dir / ("startup.wav" if kind == "startup" else "low_battery.wav")
                arguments += ["--file", str(sound)]
            # The scheduler raises and restores volume while holding the output lock.
            self.child = subprocess.Popen(arguments, stdin=source or subprocess.DEVNULL,
                                          stdout=subprocess.DEVNULL, stderr=None)
            if kind == "low_battery":
                self.last_warning_attempt = now
                self.sync_timing_marker(now)
            self._shared_child = True
            self.child_kind, self.child_started, self.child_timeout = kind, now, timeout
        except OSError as exc:
            self.warn(f"Cannot start shared audio bridge: {exc}", now)
            self._shared_child = False
            return original(self, kind, now)
        finally:
            if source:
                source.close()
    module.PromptService.launch = launch
    if original_sync is not None:
        module.PromptService.sync_volume = sync_volume


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--endpoint", default="unix:/run/wonderduck-audio/audio.sock")
    args = parser.parse_args(argv)
    source = args.source.resolve()
    sys.path.insert(0, str(source.parent))
    spec = importlib.util.spec_from_file_location("wonderduck_original_prompt", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hashlib.sha256(source.read_bytes()).hexdigest() == SUPPORTED_SOURCE_SHA256:
        attach(module, args.endpoint)
    else:
        print("原厂提示音源码版本变化：保持原厂路径，暂不接入新音频队列。", file=sys.stderr, flush=True)
    return module.main()


if __name__ == "__main__":
    raise SystemExit(main())
