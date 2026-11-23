#!/usr/bin/env python3
"""Fit a second-attempt scooping motion with the actual bill on flat training pieces."""

import argparse
from concurrent.futures import ProcessPoolExecutor
from itertools import product
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ducklab.assets import ROOT
from ducklab.policy import Controller
from ducklab.retrieval.grasp import GraspProfile
from ducklab.retrieval.scene import OBJECTS, make_scene, place


def score(job):
    item, batch, repeats, seed = job
    model, half = make_scene(item, moving_jaw=True)
    pitch_ids = [model.joint(name).id for name in ("neck_pitch", "head_pitch")]
    limits = model.jnt_range[pitch_ids].T
    parcel = model.body("parcel").id
    lips = {model.geom(name).id for name in ("lower_lip", "upper_lip")}
    out = []
    for params in batch:
        profile = GraspProfile(**params)
        rng = np.random.default_rng(seed)
        held = lifted = falls = 0
        for _ in range(repeats):
            data = mujoco.MjData(model)
            ctrl = Controller(model, data)
            x, y = rng.uniform(-0.005, 0.005, 2) + (profile.reach, 0)
            place(data, "parcel", (x, y, half[2] + 0.001), rng.uniform(-np.pi, np.pi))
            mujoco.mj_forward(model, data)
            cmd = np.zeros(13, np.float32)
            for t in range(75):
                ctrl.step(ctrl.act("stand", cmd), "stand")
            z0 = data.body("parcel").xpos[2]
            high = 0.0
            stable = 0
            for t in range(192):
                cmd[:2], angle = profile.command(t)
                data.actuator("bill_servo").ctrl[0] = angle
                joints = ctrl.act("groundpick", cmd)
                joints[5:7] = np.clip(joints[5:7] + profile.stroke(t), *limits)
                ctrl.step(joints, "groundpick")
                dz = data.body("parcel").xpos[2] - z0
                high = max(high, dz)
                touch = dz > 0.05 and any(
                    (c.geom1 in lips and model.geom_bodyid[c.geom2] == parcel)
                    or (c.geom2 in lips and model.geom_bodyid[c.geom1] == parcel)
                    for c in data.contact
                )
                stable = stable + 1 if touch else 0
                if data.qpos[2] < 0.045 or data.body("trunk_base").xmat[8] < 0.1:
                    falls += 1
                    stable = 0
                    break
            held += int(stable >= 15)
            lifted += int(high >= 0.035)
        out.append(
            dict(
                params=params,
                object=item["id"],
                held=held,
                lifted=lifted,
                falls=falls,
                trials=repeats,
            )
        )
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    # Train on three thickness/shape variants. All candidates see the same placements.
    choices = [
        dict(reach=r, close=c, rise=82, ramp=45, scoop_neck=n, scoop_pitch=p)
        for r, c, n, p in product(
            (0.057, 0.069, 0.080, 0.093),
            (42, 52, 64, 76),
            (-0.22, 0.0, 0.22),
            (-0.32, 0.0, 0.32),
        )
    ]
    train = [i for i in OBJECTS if i["split"] == "flat_train"]
    jobs = [
        (i, choices[k : k + 12], 3, 114100 + idx)
        for idx, i in enumerate(train)
        for k in range(0, len(choices), 12)
    ]
    with ProcessPoolExecutor(args.workers) as pool:
        groups = list(pool.map(score, jobs))
    rows = [row for group in groups for row in group]
    ranking = []
    for params in choices:
        group = [row for row in rows if row["params"] == params]
        ranking.append(
            dict(
                params=params,
                **{
                    k: sum(r[k] for r in group)
                    for k in ("held", "lifted", "falls", "trials")
                },
            )
        )
    ranking.sort(key=lambda r: (-r["held"], -r["lifted"], r["falls"]))
    (args.out / "fit.json").write_text(
        json.dumps(dict(ranking=ranking, rows=rows), indent=2) + "\n"
    )
    # Distinct candidates, all selected before any second-panel evaluation.
    base = json.loads((ROOT / "configs/retrieval-grasps-v4.json").read_text())
    for idx, candidate in enumerate(ranking[:3]):
        profile = {
            **base,
            "retry": {**candidate["params"], "retry_radius": 0.12, "retry_yaw": 0.32},
        }
        (args.out / f"candidate-{idx}.json").write_text(
            json.dumps(profile, indent=2) + "\n"
        )
    print(ranking[:3], flush=True)


if __name__ == "__main__":
    main()
