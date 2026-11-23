#!/usr/bin/env python3
"""Search real-contact trajectories on TRAIN identities, using controlled pickup placements."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from itertools import product
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
from ducklab.assets import ROOT
from ducklab.policy import Controller
from ducklab.retrieval.grasp import GraspProfile
from ducklab.retrieval.scene import OBJECTS, make_scene, place


def score_item(args):
    item, profiles, seed, repeats, thin = args
    model, half = make_scene(item, moving_jaw=True)
    results = []
    for param in profiles:
        profile = GraspProfile(*param)
        heights = []
        held = []
        rng = np.random.default_rng(seed)
        for _ in range(repeats):
            data = mujoco.MjData(model)
            ctrl = Controller(model, data)
            # Reset placements are supervision for fitting, never robot observations.
            xy = rng.uniform([-0.005, -0.008], [0.005, 0.008]) + [profile.reach, 0]
            place(
                data,
                "parcel",
                (*xy, half[2] + 0.001),
                (
                    np.pi / 2 + rng.uniform(-0.18, 0.18)
                    if thin
                    else rng.uniform(-np.pi, np.pi)
                ),
            )
            mujoco.mj_forward(model, data)
            cmd = np.zeros(13, np.float32)
            for _ in range(65):
                ctrl.step(ctrl.act("stand", cmd), "stand")
            initial_z = data.body("parcel").xpos[2]
            max_z = 0.0
            for tick in range(168):
                pair, opening = profile.command(tick)
                cmd[:2] = pair
                data.actuator("bill_servo").ctrl[0] = opening
                ctrl.step(ctrl.act("groundpick", cmd), "groundpick")
                max_z = max(max_z, data.body("parcel").xpos[2] - initial_z)
            heights.append(float(max_z))
            held.append(int(data.body("parcel").xpos[2] - initial_z > 0.10))
        results.append(
            dict(
                params=asdict(profile),
                held=sum(held),
                trials=repeats,
                height_sum=sum(heights),
            )
        )
    return item["id"], results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--thin",
        action="store_true",
        help="fit transverse approaches to elongated objects",
    )
    parser.add_argument("--repeats", type=int, default=4)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", type=Path, default=ROOT / "runs/fetch/grasp-fit.json")
    a = parser.parse_args()
    profiles = list(product((0.072, 0.082, 0.092, 0.102), (47, 53, 59), (60, 70, 82)))
    if a.thin:
        profiles = list(
            product((0.083, 0.091, 0.099, 0.106), (50, 55, 60, 68), (82, 98, 107))
        )
    items = [
        i
        for i in OBJECTS
        if i["split"] == "train" and (not a.thin or i["family"] == "pen")
    ]
    tasks = [
        (item, profiles, 2047 + n, a.repeats, a.thin) for n, item in enumerate(items)
    ]
    with ProcessPoolExecutor(a.workers) as pool:
        scores = dict(pool.map(score_item, tasks))
    summary = []
    for idx, params in enumerate(profiles):
        rows = [r[idx] for r in scores.values()]
        summary.append(
            dict(
                params=asdict(GraspProfile(*params)),
                held=sum(r["held"] for r in rows),
                trials=sum(r["trials"] for r in rows),
                summed_peak_m=sum(r["height_sum"] for r in rows),
            )
        )
    summary.sort(key=lambda r: (r["held"], r["summed_peak_m"]), reverse=True)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(
        json.dumps(
            dict(selected=summary[0], ranking=summary, per_object=scores), indent=2
        )
        + "\n"
    )
    print(summary[:3])


if __name__ == "__main__":
    main()
