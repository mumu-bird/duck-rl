#!/usr/bin/env python3
"""Fit tray approach and release on training objects, then select using different identities."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--weights", required=True, type=Path)
    p.add_argument(
        "--profiles", type=Path, default=ROOT / "configs/retrieval-grasps-v6.json"
    )
    p.add_argument(
        "--stroke",
        action="store_true",
        help="keep base-to-tray geometry fixed and learn the stationary release",
    )
    p.add_argument(
        "--lower",
        action="store_true",
        help="test crouching over the tray to reduce drop height",
    )
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    base = json.loads(a.profiles.read_text())

    def save(name, reach, side, neck, pitch, **stroke):
        path = a.out / (name + ".json")
        value = dict(
            reach=round(reach, 5), side=round(side, 5), neck=neck, pitch=pitch, **stroke
        )
        path.write_text(json.dumps({**base, "delivery": value}, indent=2) + "\n")
        return name

    def evaluate(name, split, seed, count):
        out = a.out / split / name
        out.mkdir(parents=True, exist_ok=True)
        with (out / "run.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/run_retrieval.py"),
                    "--split",
                    split,
                    "--mode",
                    "jaw",
                    "--profiles",
                    str(a.out / (name + ".json")),
                    "--weights",
                    str(a.weights),
                    "--trials",
                    str(count),
                    "--seed",
                    str(seed),
                    "--out",
                    str(out),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        rows = json.loads((out / "episodes.json").read_text())["rows"]
        positions = [r["release_from_tray_m"] for r in rows if r["held_at_release"]]
        return dict(
            name=name,
            n=len(rows),
            success=sum(r["success"] for r in rows),
            held=len(positions),
            stored=sum(r["in_tray_at_end"] for r in rows),
            falls=sum(r["stop"] == "fall" for r in rows),
            release_median_m=np.median(positions, axis=0).tolist()
            if positions
            else None,
        )

    baseline = save("baseline", 0.106, 0.0, -0.28, -0.12)
    train_seed, val_seed = (
        (119100, 119300)
        if a.lower
        else ((117100, 117300) if a.stroke else (116100, 116300))
    )
    old = evaluate(baseline, "train", train_seed, 6)
    dx, dy, _ = old["release_median_m"]
    center_x = 0.106 + np.clip(dx, -0.035, 0.035)
    center_y = np.clip(dy, -0.035, 0.035)
    if a.lower:
        options = [
            save(
                f"dip-{release}-{int(reach * 1000)}",
                reach,
                0.0,
                0.0,
                0.0,
                release_tick=release,
                open_ticks=10,
                retain_until_open=True,
                lower_to_tray=True,
            )
            for reach in (0.096, 0.106, 0.119)
            for release in (35, 47, 61)
        ]
    elif a.stroke:
        options = [
            save(
                name,
                0.106,
                0.0,
                neck,
                pitch,
                release_tick=release,
                open_ticks=speed,
                retain_until_open=retain,
            )
            for name, neck, pitch, release, speed, retain in [
                ("retained", -0.28, -0.12, 28, 10, True),
                ("gradual", -0.28, -0.12, 38, 18, True),
                ("early", -0.28, -0.12, 17, 8, True),
                ("level", 0.0, 0.0, 33, 12, True),
                ("tilt", -0.45, -0.25, 36, 12, True),
                ("lower", 0.22, 0.18, 38, 14, True),
                ("lower_early", 0.4, 0.30, 21, 8, True),
                ("lower_late", 0.4, 0.30, 45, 15, True),
                ("dip", 0.52, 0.42, 42, 12, False),
                ("quiet", -0.10, -0.06, 42, 18, False),
            ]
        ]
    else:
        options = [
            save("centred", center_x, center_y, -0.28, -0.12),
            save("closer", center_x - 0.017, center_y, -0.28, -0.12),
            save("farther", center_x + 0.017, center_y, -0.28, -0.12),
            save("left", center_x, center_y + 0.016, -0.28, -0.12),
            save("right", center_x, center_y - 0.016, -0.28, -0.12),
            save("steeper", center_x, center_y, -0.38, -0.25),
            save("upright", center_x, center_y, -0.08, -0.04),
            save("level", center_x, center_y, 0.0, 0.0),
        ]

    def rank(r):
        return (-r["success"], r["falls"], -r["stored"])

    with ThreadPoolExecutor(5) as pool:
        train = sorted(
            [
                old,
                *pool.map(lambda name: evaluate(name, "train", train_seed, 6), options),
            ],
            key=rank,
        )
        candidates = list(dict.fromkeys([r["name"] for r in train[:3]] + [baseline]))
        val = sorted(
            pool.map(lambda name: evaluate(name, "val", val_seed, 12), candidates),
            key=rank,
        )
    winner = val[0]["name"]
    (a.out / "selection.json").write_text(
        json.dumps(dict(selected=winner, train=train, val=val), indent=2) + "\n"
    )
    shutil.copyfile(a.out / (winner + ".json"), a.out / "frozen.json")
    print("Selected", winner, val)


if __name__ == "__main__":
    main()
