#!/usr/bin/env python3
"""Train a first duck-specific grasp profile with paired randomized simulations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np

from ducklab.assets import ROOT
from ducklab.retrieval.duck import (
    duck_profile_candidates,
    load_duck_variants,
    profiles_for,
    sample_duck,
    serializable_profiles,
)
from ducklab.retrieval.episode import Episode
from ducklab.retrieval.scene import BIN, HEIGHT, OBJECTS, WIDTH, make_scene
from ducklab.retrieval.vision import Detector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=ROOT / "runs/fetch/linlee-duck-training"
    )
    parser.add_argument(
        "--weights", type=Path, default=ROOT / "runs/fetch/round3-vision/change.pt"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--train-trials", type=int, default=1)
    parser.add_argument("--validation-trials", type=int, default=2)
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=9,
        help="number of coordinated reach/close/rise candidates; use 27 for the full grid",
    )
    parser.add_argument("--mass-grams", type=float, default=12.0)
    parser.add_argument("--bin", type=float, nargs=2, default=BIN, metavar=("X", "Y"))
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.train_trials < 1 or args.validation_trials < 1:
        parser.error("trial counts must be positive")
    if args.candidate_limit < 1 or args.candidate_limit > 27:
        parser.error("candidate-limit must be between 1 and 27")
    if args.mass_grams <= 0 or not np.isfinite(args.mass_grams):
        parser.error("mass-grams must be finite and positive")
    return args


def candidate_subset(limit: int):
    candidates = duck_profile_candidates()
    if limit >= len(candidates):
        return candidates
    # Keep the search spread across approach distance, closure timing and rise timing.
    indices = np.linspace(0, len(candidates) - 1, limit, dtype=int)
    return [candidates[int(index)] for index in dict.fromkeys(indices)]


def randomized_jobs(args: argparse.Namespace) -> list[dict]:
    jobs = []
    for variant_index, template in enumerate(load_duck_variants()):
        for trial in range(args.train_trials):
            seed = args.seed + variant_index * 1000 + trial
            rng = np.random.default_rng(seed)
            jobs.append(
                {
                    "split": "train",
                    "template": template["id"],
                    "seed": seed,
                    "item": sample_duck(template, rng),
                }
            )
    return jobs


def reference_jobs(args: argparse.Namespace) -> list[dict]:
    wanted = {"round_bird", "sony_duck", "bath_bird"}
    jobs = []
    index = 0
    for item in OBJECTS:
        if item["id"] not in wanted:
            continue
        for trial in range(args.validation_trials):
            seed = args.seed + 50000 + index * 1000 + trial
            jobs.append(
                {
                    "split": "validation",
                    "template": item["id"],
                    "seed": seed,
                    "item": {**item, "mass_kg": args.mass_grams / 1000.0},
                }
            )
        index += 1
    return jobs


def run_job(job: dict, profiles: dict, detector: Detector, bin_xy) -> dict:
    item = job["item"]
    model, half = make_scene(item, bin_xy, moving_jaw=True)
    with mujoco.Renderer(model, HEIGHT, WIDTH) as camera:
        result = Episode(
            model,
            half,
            camera,
            detector,
            "jaw",
            job["seed"],
            bin_xy,
            profiles=profiles,
        ).run()
    result.update(
        {
            "split": job["split"],
            "template": job["template"],
            "object": item["id"],
            "mass_grams": round(float(item.get("mass_kg", 0.0)) * 1000, 3),
            "sample_scale": item.get("sample_scale"),
        }
    )
    return result


def summary(rows: list[dict]) -> dict:
    count = len(rows)
    return {
        "episodes": count,
        "success": sum(bool(row["success"]) for row in rows),
        "lifted": sum(bool(row["lifted"]) for row in rows),
        "held_at_release": sum(bool(row["held_at_release"]) for row in rows),
        "in_tray_at_end": sum(bool(row["in_tray_at_end"]) for row in rows),
        "falls": sum(row["stop"] == "fall" for row in rows),
        "success_rate": round(
            sum(bool(row["success"]) for row in rows) / count, 4
        )
        if count
        else 0.0,
    }


def rank(measures: dict) -> tuple:
    return (
        measures["success_rate"],
        measures["held_at_release"] / measures["episodes"]
        if measures["episodes"]
        else 0.0,
        measures["lifted"] / measures["episodes"] if measures["episodes"] else 0.0,
        -measures["falls"] / measures["episodes"] if measures["episodes"] else 0.0,
    )


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    if (args.out / "selection.json").exists() and not args.force:
        raise SystemExit(f"{args.out} already contains a selection; choose a new --out or use --force")
    args.out.mkdir(parents=True, exist_ok=True)

    candidates = candidate_subset(args.candidate_limit)
    train_jobs = randomized_jobs(args)
    val_jobs = reference_jobs(args)
    detector = Detector(args.weights, args.device)
    candidate_rows = []
    best = None

    for index, grasp in enumerate(candidates):
        profiles = profiles_for(grasp)
        train_rows = []
        for job in train_jobs:
            row = run_job(job, profiles, detector, args.bin)
            train_rows.append(row)
        val_rows = []
        for job in val_jobs:
            row = run_job(job, profiles, detector, args.bin)
            val_rows.append(row)
        train_summary = summary(train_rows)
        val_summary = summary(val_rows)
        record = {
            "candidate": index,
            "grasp": vars(grasp),
            "train": train_summary,
            "validation": val_summary,
            "train_rows": train_rows,
            "validation_rows": val_rows,
        }
        candidate_rows.append(record)
        # Synthetic LINLEE-like variants are the deployment target. Existing duck meshes
        # remain a useful regression check, but should not dominate selection.
        key = tuple(
            2.0 * train_value + val_value
            for train_value, val_value in zip(rank(train_summary), rank(val_summary))
        )
        if best is None or key > best[0]:
            best = (key, index, profiles)
        print(
            f"candidate={index:02d} train={train_summary['success']}/{train_summary['episodes']} "
            f"val={val_summary['success']}/{val_summary['episodes']} "
            f"held={val_summary['held_at_release']} lifted={val_summary['lifted']}",
            flush=True,
        )

    assert best is not None
    _, selected_index, selected_profiles = best
    selected = candidate_rows[selected_index]
    save_json(args.out / "candidates.json", candidate_rows)
    save_json(
        args.out / "selection.json",
        {
            "selected_candidate": selected_index,
            "selected_key": list(best[0]),
            "selected_grasp": selected["grasp"],
            "train": selected["train"],
            "validation": selected["validation"],
        },
    )
    save_json(args.out / "frozen.json", serializable_profiles(selected_profiles))
    save_json(
        args.out / "summary.json",
        {
            "task": "duck_pick_carry_drop_return",
            "scope": "randomized synthetic duck shapes plus round_bird/sony_duck/bath_bird validation",
            "device": args.device,
            "weights": str(args.weights),
            "seed": args.seed,
            "candidate_count": len(candidates),
            "train_jobs": len(train_jobs),
            "validation_jobs": len(val_jobs),
            "selected_candidate": selected_index,
            "selected_train": selected["train"],
            "selected_validation": selected["validation"],
            "profile_file": str(args.out / "frozen.json"),
        },
    )
    print(json.dumps({"selected": selected_index, "validation": selected["validation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
