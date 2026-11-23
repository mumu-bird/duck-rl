#!/usr/bin/env python3
"""Run a staged one-duck-at-a-time collection batch with fixed paired seeds."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from ducklab.assets import ROOT
from ducklab.retrieval.duck import load_duck_variants, sample_duck
from ducklab.retrieval.episode import Episode
from ducklab.retrieval.grasp import load_profiles
from ducklab.retrieval.scene import BIN, HEIGHT, OBJECTS, WIDTH, make_scene
from ducklab.retrieval.vision import Detector
from ducklab.video import Video


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--profiles", type=Path, default=ROOT / "configs/linlee-duck-profiles.json"
    )
    parser.add_argument(
        "--weights", type=Path, default=ROOT / "runs/fetch/round3-vision/change.pt"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=203000)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--variant", action="append", dest="variants")
    parser.add_argument("--include-reference", action="store_true")
    parser.add_argument("--mass-grams", type=float, default=12.0)
    parser.add_argument("--bin", type=float, nargs=2, default=BIN, metavar=("X", "Y"))
    parser.add_argument("--video-all", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.trials < 1 or args.mass_grams <= 0 or not np.isfinite(args.mass_grams):
        parser.error("trials and mass-grams must be positive and finite")
    if (args.out / "episodes.json").exists() and not args.force:
        parser.error("output already has episodes.json; choose a new directory or use --force")
    return args


def result_summary(rows: list[dict]) -> dict:
    count = len(rows)
    return {
        "episodes": count,
        "success": sum(bool(row["success"]) for row in rows),
        "lifted": sum(bool(row["lifted"]) for row in rows),
        "held_at_release": sum(bool(row["held_at_release"]) for row in rows),
        "in_tray_at_end": sum(bool(row["in_tray_at_end"]) for row in rows),
        "pose_restored": sum(bool(row["pose_restored"]) for row in rows),
        "falls": sum(row["stop"] == "fall" for row in rows),
        "success_rate": round(
            sum(bool(row["success"]) for row in rows) / count, 4
        )
        if count
        else 0.0,
    }


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    profiles = load_profiles(args.profiles)
    detector = Detector(args.weights, args.device)
    templates = {item["id"]: item for item in load_duck_variants()}
    selected = args.variants or list(templates)
    missing = sorted(set(selected) - set(templates))
    if missing:
        raise SystemExit(f"unknown duck variants: {', '.join(missing)}")

    jobs = []
    for variant_index, name in enumerate(selected):
        template = templates[name]
        for trial in range(args.trials):
            seed = args.seed + variant_index * 1000 + trial
            jobs.append(
                {
                    "variant": name,
                    "trial": trial,
                    "seed": seed,
                    "item": sample_duck(template, np.random.default_rng(seed)),
                }
            )

    if args.include_reference:
        for item_index, item in enumerate(OBJECTS):
            if item["id"] not in {"round_bird", "sony_duck"}:
                continue
            for trial in range(args.trials):
                seed = args.seed + 90000 + item_index * 1000 + trial
                jobs.append(
                    {
                        "variant": item["id"],
                        "trial": trial,
                        "seed": seed,
                        "item": {**item, "mass_kg": args.mass_grams / 1000.0},
                    }
                )

    rows = []
    for job_index, job in enumerate(jobs):
        item = job["item"]
        model, half = make_scene(item, args.bin, moving_jaw=True)
        with ExitStack() as handles:
            camera = handles.enter_context(mujoco.Renderer(model, HEIGHT, WIDTH))
            wide = None
            video = None
            if args.video_all:
                wide = handles.enter_context(mujoco.Renderer(model, 320, 536))
                video = Video(
                    args.out / f"{job['variant']}-{job['seed']}.mp4",
                    wide,
                    camera,
                    f"鸭鸭集合 · {job['variant']}",
                    job["seed"],
                    arena_view=(
                        [args.bin[0] / 2, args.bin[1] / 2, 0.078],
                        max(0.90, 1.9 * max(map(abs, args.bin))),
                    ),
                )
            try:
                row = Episode(
                    model,
                    half,
                    camera,
                    detector,
                    "jaw",
                    job["seed"],
                    args.bin,
                    video=video,
                    profiles=profiles,
                ).run()
            finally:
                if video and sys.exc_info()[0] is not None:
                    video.finish()
            if video:
                video.finish("完整通过" if row["success"] else "本轮未通过")
        row.update(
            {
                "variant": job["variant"],
                "trial": job["trial"],
                "mass_grams": round(float(item.get("mass_kg", 0.0)) * 1000, 3),
                "sample_scale": item.get("sample_scale"),
            }
        )
        rows.append(row)
        print(
            f"{job_index + 1}/{len(jobs)} {job['variant']} seed={job['seed']} "
            f"success={row['success']} stop={row['stop']}",
            flush=True,
        )

    result = {
        "task": "staged_duck_collection",
        "scope": "one duck per scene; physical multi-duck contact and multi-target vision are intentionally deferred",
        "seed": args.seed,
        "trials_per_variant": args.trials,
        "bin_m": list(args.bin),
        "profiles": json.loads(args.profiles.read_text()),
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "collection_order": selected,
        "rows": rows,
        "summary": result_summary(rows),
        "next_phases": [
            "calibrate mass, size, friction and beak clearance from real LINLEE samples",
            "capture before/after frames with all ducks present and train a target-selection model",
            "add a shared-scene multi-duck collision episode after single-duck success is stable",
        ],
    }
    temp = args.out / "episodes.partial.json"
    temp.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    temp.replace(args.out / "episodes.json")
    print(json.dumps(result["summary"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
