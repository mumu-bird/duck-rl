#!/usr/bin/env python3
"""Compare production and GPU-compatible physics profiles on paired CPU tasks."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import tools.train_retrieval_ppo as trainer
from ducklab.assets import ROOT as ASSET_ROOT
from ducklab.retrieval.grasp import load_profiles
from ducklab.retrieval.scene import BIN, OBJECTS
from ducklab.retrieval.vision import Detector


EVENTS = (
    "lifted",
    "held_at_release",
    "in_tray_at_end",
    "pose_restored",
    "success",
)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def candidate_scene_builder(base_builder):
    def build(*args, **kwargs):
        model, half = base_builder(*args, **kwargs)
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
        model.opt.noslip_iterations = 3
        model.opt.tolerance = 1e-6
        return model, half

    return build


def main():
    default_out = ROOT / "runs/fetch/retrieval-integrator-paired-cpu-20261123"
    default_items = (
        "felt_pen",
        "capped_crayon",
        "056_tennis_ball",
        "012_strawberry",
        "061_foam_brick",
        "bath_bird",
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", nargs="+", default=default_items)
    parser.add_argument("--trials", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--weights", type=Path, default=ASSET_ROOT / "runs/fetch/round3-vision/change.pt"
    )
    parser.add_argument("--out", type=Path, default=default_out)
    args = parser.parse_args()

    if args.trials < 1 or not args.items or len(set(args.items)) != len(args.items):
        parser.error("trials must be positive and item ids must be unique")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")
    if not args.weights.is_file():
        parser.error(f"vision weights not found: {args.weights}")
    if (args.out / "summary.json").exists():
        parser.error(f"output already has a summary; choose a new --out: {args.out}")

    items_by_id = {item["id"]: item for item in OBJECTS}
    unknown = [item_id for item_id in args.items if item_id not in items_by_id]
    if unknown:
        parser.error(f"unknown item ids: {unknown}")
    non_train = [item_id for item_id in args.items if items_by_id[item_id]["split"] != "train"]
    if non_train:
        parser.error(f"paired physics probe only accepts train items: {non_train}")

    args.out.mkdir(parents=True, exist_ok=True)
    detector = Detector(args.weights, args.device)
    profiles = load_profiles(ASSET_ROOT / "configs/retrieval-grasps-v6.json")
    base_builder = trainer.make_scene
    candidate_builder = candidate_scene_builder(base_builder)
    pairs = []

    for item_index, item_id in enumerate(args.items):
        item = items_by_id[item_id]
        for trial in range(args.trials):
            seed = args.seed + item_index * 101 + trial
            rows = {}
            for profile_name, builder, label in (
                ("production", base_builder, "production-cpu"),
                ("implicit_noslip", candidate_builder, "implicit-noslip-cpu"),
            ):
                trainer.make_scene = builder
                video_path = (
                    args.out
                    / "videos"
                    / profile_name
                    / f"{item_id}-{seed}.mp4"
                )
                rows[profile_name] = trainer.run_episode(
                    item,
                    seed,
                    BIN,
                    detector,
                    profiles,
                    policy=None,
                    video_path=video_path,
                    label=label,
                )
            trainer.make_scene = base_builder
            pairs.append({"item_id": item_id, "seed": seed, **rows})
            save_json(
                args.out / "progress.json",
                {
                    "completed_pairs": len(pairs),
                    "total_pairs": len(args.items) * args.trials,
                    "pairs": pairs,
                },
            )
            print(
                json.dumps(
                    {
                        "item_id": item_id,
                        "seed": seed,
                        "production": rows["production"]["success"],
                        "implicit_noslip": rows["implicit_noslip"]["success"],
                    }
                ),
                flush=True,
            )

    trainer.make_scene = base_builder
    disagreements = {
        event: sum(
            bool(pair["production"][event])
            != bool(pair["implicit_noslip"][event])
            for pair in pairs
        )
        for event in EVENTS
    }
    disagreements["stop"] = sum(
        pair["production"]["stop"] != pair["implicit_noslip"]["stop"]
        for pair in pairs
    )
    summary = {
        "purpose": "paired CPU task-level physics comparison; no PPO updates and no GPU physics",
        "mujoco_version": mujoco.__version__,
        "torch": torch.__version__,
        "vision_device": args.device,
        "trials_per_item": args.trials,
        "items": args.items,
        "seed_base": args.seed,
        "profiles": {
            "production": {
                "integrator": "implicitfast",
                "noslip_iterations": 3,
                "solver_tolerance": 1e-10,
            },
            "implicit_noslip": {
                "integrator": "implicit",
                "noslip_iterations": 3,
                "solver_tolerance": 1e-6,
            },
        },
        "episodes_per_profile": len(pairs),
        "production_summary": trainer.summarize(
            [pair["production"] for pair in pairs]
        ),
        "implicit_noslip_summary": trainer.summarize(
            [pair["implicit_noslip"] for pair in pairs]
        ),
        "event_disagreements": disagreements,
        "pairs": pairs,
    }
    save_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
