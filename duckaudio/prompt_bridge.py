"""Finite manufacturer WAV prompts enter the shared queue without holding the PCM lock."""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import signal
import sys
import tempfile
import threading
import time

from .client import AudioClient
from .errors import AudioError
from .store import TERMINAL


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--kind", choices=("startup", "low_battery", "network"), required=True)
    parser.add_argument("--file", type=Path)
    args = parser.parse_args(argv)
    stopped = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stopped.set())
    client = AudioClient(args.endpoint, timeout=3)
    job = None
    ephemeral = False
    clip = None
    try:
        with tempfile.TemporaryDirectory(prefix="duckaudio-prompt-") as directory:
            path = args.file
            if path is None:
                path = Path(directory) / "prompt.wav"
                data = sys.stdin.buffer.read(20 * 1024 * 1024 + 1)
                if len(data) > 20 * 1024 * 1024:
                    raise AudioError("原厂提示音超出限制")
                path.write_bytes(data)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()[:24]
            key = f"system.{args.kind}.{digest}"
            if not client.has_clip(key):
                try:
                    clip = client.import_file(path, key=key, name=f"原厂提示音 {args.kind}")
                    ephemeral = args.kind == "network"
                except AudioError as exc:
                    if exc.code != "duplicate_key":
                        raise
            if stopped.is_set():
                return 75
            job = client.play(key, task_id="system." + args.kind,
                              priority=100 if args.kind == "low_battery" else 90,
                              policy="replace_task", ttl=10)
            while not stopped.wait(.08):
                state = job.status()
                if state["state"] in TERMINAL:
                    return 0 if state["state"] == "finished" else 75
            return 75
    except (AudioError, OSError) as exc:
        print(f"音频提示接入失败: {exc}", file=sys.stderr)
        return 75  # Manufacturer retries, falling back to its player if service is unavailable.
    finally:
        if job:
            try:
                job.stop()
                if ephemeral:
                    job.wait(timeout=3, raise_on_error=False)
            except (AudioError, TimeoutError):
                pass
        if ephemeral and clip:
            try:
                client.delete(clip["id"])
            except AudioError:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
