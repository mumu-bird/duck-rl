#!/usr/bin/env python3
"""Generate supervised before/after pairs from the robot's head camera."""

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
from ducklab.policy import Controller
from ducklab.retrieval.scene import (
    CLUTTER,
    HEIGHT,
    WIDTH,
    OBJECTS,
    camera_frame,
    make_scene,
    place,
    recolor,
    target_mask,
)


def generate(item, count, seed, out):
    m, half = make_scene(item)
    d = mujoco.MjData(m)
    c = Controller(m, d)
    for _ in range(75):
        c.step(c.act("stand", np.zeros(13, np.float32)), "stand")
    posture = d.qpos.copy()
    rng = np.random.default_rng(seed)
    examples = []
    with mujoco.Renderer(m, HEIGHT, WIDTH) as camera:
        for i in range(count):
            d.qpos[:] = posture
            d.qpos[c.q[5]] += rng.uniform(-0.06, 0.06)
            d.qpos[c.q[7]] += rng.uniform(-0.10, 0.10)
            for old, z in zip(CLUTTER, (0.013, 0.006)):
                place(
                    d,
                    old,
                    (*rng.uniform([0.15, -0.25], [0.55, 0.25]), z),
                    rng.uniform(-3.14, 3.14),
                )
            place(d, "parcel", (-2.0, -2.0, 0.02))
            recolor(m, rng)
            exposure = rng.uniform(0.80, 1.05)
            mujoco.mj_forward(m, d)
            before = camera_frame(camera, d)
            # Static clutter is present in both views. Tiny pose and exposure changes
            # also occur in no-toss negatives and should not trigger retrieval.
            d.qpos[c.q[5]] += rng.normal(0, 0.003)
            d.qpos[c.q[7]] += rng.normal(0, 0.003)
            d.qpos[:2] += rng.normal(0, 0.0006, 2)
            if i % 7:
                place(
                    d,
                    "parcel",
                    (*rng.uniform([0.16, -0.18], [0.47, 0.15]), half[2]),
                    rng.uniform(-3.14, 3.14),
                )
            mujoco.mj_forward(m, d)
            after = camera_frame(camera, d)
            mask = target_mask(camera, m, d)

            def photometric(frame, gain):
                return np.clip(
                    frame.astype(np.float32) * gain
                    + rng.normal(0, rng.uniform(0, 1.4), frame.shape),
                    0,
                    255,
                ).astype(np.uint8)

            examples.append(
                (
                    photometric(before, exposure),
                    photometric(after, exposure * rng.uniform(0.97, 1.03)),
                    mask,
                )
            )
    before, after, mask = map(np.stack, zip(*examples))
    name = out / (item["id"] + ".npz")
    temp = name.with_suffix(".partial.npz")
    np.savez_compressed(temp, before=before, after=after, mask=mask)
    temp.replace(name)
    print(
        item["id"],
        item["split"],
        len(mask),
        "visible",
        int(np.any(mask, axis=(1, 2)).sum()),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=ROOT / "runs/fetch/frames")
    p.add_argument("--train", type=int, default=120)
    p.add_argument("--eval", type=int, default=56)
    p.add_argument("--seed", type=int, default=2401)
    p.add_argument(
        "--include-challenge",
        action="store_true",
        help="also render post-training novel categories",
    )
    a = p.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    for idx, item in enumerate(OBJECTS):
        if item["split"] not in ("train", "val", "test") and not (
            item["split"] == "challenge" and a.include_challenge
        ):
            continue
        generate(
            item,
            a.train if item["split"] == "train" else a.eval,
            a.seed + idx * 4129,
            a.out,
        )
    (a.out / "generation.json").write_text(
        json.dumps(
            dict(
                seed=a.seed,
                train_per_id=a.train,
                eval_per_id=a.eval,
                height=HEIGHT,
                width=WIDTH,
            ),
            indent=2,
        )
    )
