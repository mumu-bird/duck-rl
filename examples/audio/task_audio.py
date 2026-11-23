"""Run a bounded audio-only task demo; this script never sends motor commands."""
from __future__ import annotations

import argparse
import time

from duckaudio import AudioClient
from duckaudio.tasks import LogAudioClient, TaskAudio


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", help="Omit to log only; e.g. unix:/run/wonderduck-audio/audio.sock")
    parser.add_argument("--token-file")
    parser.add_argument("--cancel", action="store_true")
    args = parser.parse_args()
    client = AudioClient(args.endpoint, token_file=args.token_file) if args.endpoint else LogAudioClient("audio-demo.jsonl")
    audio = TaskAudio(client)
    try:
        audio.emit("started").result(timeout=5)  # Await submission here, never in a control loop.
        time.sleep(.5)
        audio.emit("target_found")
        if args.cancel:
            audio.cancel().result(timeout=5)
        else:
            audio.emit("succeeded").result(timeout=5)
    finally:
        audio.close()

if __name__ == "__main__":
    main()
