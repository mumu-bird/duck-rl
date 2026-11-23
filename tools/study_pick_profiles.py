#!/usr/bin/env python3
"""Fit the existing 2 grasp schedules on paired, independently tossed training objects."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = {
    "baseline": ((0.072, 47, 82), (0.091, 60, 98)),
    "early": ((0.072, 46, 62), (0.091, 53, 77)),
    "forward": ((0.083, 49, 66), (0.097, 55, 80)),
    "near": ((0.065, 43, 65), (0.084, 52, 80)),
    "medium": ((0.077, 53, 76), (0.088, 57, 88)),
    "quick": ((0.075, 42, 55), (0.093, 48, 68)),
    "delayed": ((0.078, 56, 90), (0.095, 64, 98)),
    "reach": ((0.089, 48, 72), (0.103, 58, 87)),
}


SMOOTH = {
    "baseline": ((0.072, 47, 82, 0), (0.091, 60, 98, 0)),
    "ramp10": ((0.072, 47, 82, 10), (0.091, 60, 98, 10)),
    "ramp24": ((0.072, 47, 82, 24), (0.091, 60, 98, 24)),
    "ramp45": ((0.072, 47, 82, 45), (0.091, 60, 98, 45)),
    "early_smooth": ((0.072, 47, 65, 30), (0.091, 60, 82, 30)),
    "mid_smooth": ((0.072, 47, 75, 18), (0.091, 60, 91, 18)),
    "gentle": ((0.072, 47, 98, 24), (0.091, 60, 103, 24)),
    "uniform": ((0.072, 47, 107, 0), (0.091, 60, 107, 0)),
    "smooth_near": ((0.070, 47, 82, 18), (0.089, 60, 98, 18)),
    "smooth_ahead": ((0.074, 47, 82, 18), (0.093, 60, 98, 18)),
}


RETRIES = {
    name: ((0.072, 47, 82, 45, rad, yaw, reach), (0.091, 60, 98, 45, rad, yaw, reach))
    for name, rad, yaw, reach in [
        ("baseline", 0, 0, 0),
        ("local", 0.10, 0, 0),
        ("near", 0.10, 0, -0.007),
        ("far", 0.10, 0, 0.007),
        ("side", 0.12, 0.25, 0),
        ("other_side", 0.12, -0.25, 0),
        ("angled_near", 0.12, 0.32, -0.005),
        ("angled_far", 0.12, -0.32, 0.005),
    ]
}


EDGES = {
    name: (
        (0.072, 47, 82, 45, 0.12, yaw, reach, side),
        (0.091, 60, 98, 45, 0.12, yaw, reach, side),
    )
    for name, yaw, reach, side in [
        ("baseline", 0.32, -0.005, 0),
        ("left", 0.32, -0.005, 0.009),
        ("right", 0.32, -0.005, -0.009),
        ("left_far", 0.35, 0.009, 0.011),
        ("right_far", -0.35, 0.009, -0.011),
        ("left_near", 0.5, -0.012, 0.012),
        ("right_near", -0.5, -0.012, -0.012),
        ("cross", 0.75, 0.004, 0.006),
        ("shallow", 0.18, 0.016, -0.003),
    ]
}


def run(args, name, split, seed, trials):
    directory = args.out / split / name
    directory.mkdir(parents=True, exist_ok=True)
    profile = args.out / (name + ".json")
    general, thin = args.bank[name]
    params = {
        key: dict(
            zip(
                (
                    "reach",
                    "close",
                    "rise",
                    "ramp",
                    "retry_radius",
                    "retry_yaw",
                    "retry_reach",
                    "retry_side",
                )[: len(values)],
                values,
                strict=True,
            )
        )
        for key, values in [("general", general), ("thin", thin)]
    }
    profile.write_text(json.dumps(params, indent=2) + "\n")
    with (directory / "run.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "tools/run_retrieval.py"),
                "--mode",
                "jaw",
                "--split",
                split,
                "--seed",
                str(seed),
                "--trials",
                str(trials),
                "--profiles",
                str(profile),
                "--weights",
                str(args.weights),
                "--out",
                str(directory),
            ],
            check=True,
            env={**os.environ, "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"},
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    rows = json.loads((directory / "episodes.json").read_text())["rows"]
    result = dict(
        name=name,
        total=len(rows),
        success=sum(r["success"] for r in rows),
        falls=sum(r["stop"] == "fall" for r in rows),
        held=sum(r["held_at_release"] for r in rows),
        lifted=sum(r["lifted"] for r in rows),
    )
    print(split, result, flush=True)
    return result


def rank(row):
    return (-row["success"], row["falls"], -row["held"], -row["lifted"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument(
        "--round", choices=("coarse", "smooth", "retry", "edge"), default="coarse"
    )
    p.add_argument("--weights", type=Path, default=ROOT / "runs/fetch/vision/change.pt")
    a = p.parse_args()
    a.bank = {"coarse": CANDIDATES, "smooth": SMOOTH, "retry": RETRIES, "edge": EDGES}[
        a.round
    ]
    a.out.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(a.workers) as executor:

        def evaluate(names, split, seed, trials):
            futures = [executor.submit(run, a, n, split, seed, trials) for n in names]
            return sorted((f.result() for f in futures), key=rank)

        seed, trials = {
            "coarse": (110100, 5),
            "smooth": (110700, 8),
            "retry": (111100, 8),
            "edge": (113100, 10),
        }[a.round]
        train = evaluate(
            a.bank, "flat_train" if a.round == "edge" else "train", seed, trials
        )
        candidates = list(dict.fromkeys([r["name"] for r in train[:3]] + ["baseline"]))
        val_split, val_seed, val_trials = {
            "coarse": ("val", 110300, 10),
            "smooth": ("val", 110300, 10),
            "retry": ("val", 111300, 12),
            "edge": ("flat_val", 113300, 15),
        }[a.round]
        val = evaluate(candidates, val_split, val_seed, val_trials)
    (a.out / "selection.json").write_text(
        json.dumps(dict(train=train, val=val, selected=val[0]["name"]), indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
