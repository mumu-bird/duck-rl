#!/usr/bin/env python3
"""Paired held-out mobile retrieval trials. Same tosses in all grasp modes."""

import argparse
import hashlib
import math
import json
import os
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
import sys
import uuid

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from duckaudio import AudioClient
from duckaudio.tasks import LogAudioClient, TaskAudio
import mujoco
from ducklab.assets import ROOT
from ducklab.retrieval.episode import Episode
from ducklab.retrieval.grasp import load_profiles
from ducklab.retrieval.scene import BIN, HEIGHT, WIDTH, OBJECTS, make_scene
from ducklab.video import Video


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audio-endpoint", help="explicit real-device audio endpoint; omitted logs simulation events")
    p.add_argument("--audio-token-file", type=Path)
    p.add_argument(
        "--split", choices=sorted({i["split"] for i in OBJECTS}), default="test"
    )
    p.add_argument(
        "--preset",
        choices=("v1", "reviewed"),
        default="v1",
        help="reviewed uses the validation-selected V6, physical jaw and records every episode",
    )
    p.add_argument(
        "--mode",
        choices=("assisted", "contact", "jaw"),
        nargs="+",
        default=None,
    )
    p.add_argument("--trials", type=int, default=10)
    p.add_argument("--mass-grams", type=float, default=4.0, help="target mass in grams")
    p.add_argument("--seed", type=int, default=66310)
    p.add_argument("--bin", type=float, nargs=2, default=BIN, metavar=("X", "Y"))
    p.add_argument("--profiles", type=Path)
    p.add_argument("--weights", type=Path)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", type=Path)
    recording = p.add_mutually_exclusive_group()
    recording.add_argument("--video-seed", type=int)
    recording.add_argument(
        "--video-all", action="store_true", help="record every trial including failures"
    )
    p.add_argument("--no-toss", action="store_true")
    p.add_argument("--object")
    a = p.parse_args()
    if a.preset == "reviewed":
        a.profiles = a.profiles or ROOT / "configs/retrieval-grasps-v6.json"
        a.weights = a.weights or ROOT / "runs/fetch/round3-vision/change.pt"
        a.mode = a.mode or ["jaw"]
        a.video_all = a.video_seed is None
        if a.out is None:
            stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
            a.out = ROOT / "runs/fetch" / f"reviewed-{stamp}"
        if (a.out / "episodes.json").exists():
            p.error("reviewed output already has episodes; use a new --out directory")
    else:
        a.profiles = a.profiles or ROOT / "configs/retrieval-grasps.json"
        a.weights = a.weights or ROOT / "runs/fetch/vision/change.pt"
        a.mode = a.mode or ["assisted", "jaw", "contact"]
        a.out = a.out or ROOT / "runs/fetch/test"
    print(f"Results: {a.out}", flush=True)
    if not math.isfinite(a.mass_grams) or a.mass_grams <= 0:
        p.error("mass must be finite and positive")
    if a.trials < 1 or not all(map(math.isfinite, a.bin)):
        p.error("trials must be positive and bin coordinates must be finite")
    from ducklab.retrieval.vision import Detector
    import torch

    torch.set_num_threads(2)
    detector = Detector(a.weights, a.device)
    profiles = load_profiles(a.profiles)
    a.out.mkdir(parents=True, exist_ok=True)
    audio_client = AudioClient(a.audio_endpoint, token_file=a.audio_token_file) if a.audio_endpoint else LogAudioClient(a.out / "audio-events.jsonl")
    audio_run = uuid.uuid4().hex[:8]
    rows = []
    items = [
        i
        for i in OBJECTS
        if i["split"] == a.split and (not a.object or a.object == i["id"])
    ]
    if not items:
        p.error("object not in requested split")
    for item in items:
        for mode in a.mode:
            model, half = make_scene(
                {**item, "mass_kg": a.mass_grams / 1000},
                a.bin,
                moving_jaw=mode == "jaw",
            )
            with ExitStack() as handles:
                camera = handles.enter_context(mujoco.Renderer(model, HEIGHT, WIDTH))
                if a.video_all or a.video_seed is not None:
                    wide = handles.enter_context(mujoco.Renderer(model, 320, 536))
                for seed in range(a.seed, a.seed + a.trials):
                    video = None
                    if a.video_all or seed == a.video_seed:
                        video = Video(
                            a.out / f"{item['id']}-{mode}-{seed}.mp4",
                            wide,
                            camera,
                            f"拾取归位 · {item['id']} · {mode}",
                            seed,
                            arena_view=(
                                [a.bin[0] / 2, a.bin[1] / 2, 0.078],
                                max(0.90, 1.9 * max(map(abs, a.bin))),
                            ),
                        )
                    audio = TaskAudio(audio_client, task_id=f"retrieval:{audio_run}:{item['id']}:{mode}:{seed}")
                    audio.emit("started")
                    try:
                        row = Episode(
                            model,
                            half,
                            camera,
                            detector,
                            mode,
                            seed,
                            a.bin,
                            a.no_toss,
                            video,
                            profiles=profiles,
                            audio=audio,
                        ).run()
                        audio.emit("succeeded" if row["success"] else "failed")
                    finally:
                        audio.close(cancel=sys.exc_info()[0] is not None)
                        if video and sys.exc_info()[0] is not None:
                            video.finish()
                    if video:
                        video.finish("完整通过" if row["success"] else "本轮未通过")
                    row.update(object=item["id"], family=item["family"])
                    rows.append(row)
                    print(
                        item["id"], mode, seed, row["stop"], row["success"], flush=True
                    )
            save(a, rows)
    save(a, rows)


def save(a, rows):
    result = {
        "split": a.split,
        "seed": a.seed,
        "trials_per_object_mode": a.trials,
        "bin_m": a.bin,
        "mass_grams": a.mass_grams,
        "no_toss": a.no_toss,
        "profiles": json.loads(a.profiles.read_text()),
        "weights_sha256": hashlib.sha256(a.weights.read_bytes()).hexdigest(),
        "rows": rows,
    }
    path = a.out / "episodes.json"
    tmp = path.with_suffix(".partial.json")
    tmp.write_text(json.dumps(result, indent=2) + "\n")
    tmp.replace(path)


if __name__ == "__main__":
    main()
