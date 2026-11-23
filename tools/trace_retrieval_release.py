#!/usr/bin/env python3
"""Read-only 200 Hz contact referee for recorded release failures and controls."""

import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mujoco
import numpy as np
import torch
from ducklab.assets import ROOT
from ducklab.retrieval.scene import BIN, HEIGHT, WIDTH, OBJECTS, make_scene
from ducklab.retrieval.episode import Episode
from ducklab.retrieval.grasp import load_profiles
from ducklab.retrieval.vision import Detector


def observe_episode(sim):
    model, data = sim.m, sim.d
    original_step = sim.c.step
    parcel_id = model.body("parcel").id
    support_ids = {
        g
        for g in range(model.ngeom)
        if model.geom(g).name.startswith("tray_")
        or model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE
    }
    lip_ids = {model.geom(n).id for n in ("upper_lip", "lower_lip")}
    trace, support = [], []
    last_lip = None
    t0 = None

    def wrapper(targets, slot, observer=None):
        def observe():
            nonlocal last_lip, t0
            if observer is not None:
                observer()
            if sim.release_tick is None:
                return
            if t0 is None:
                t0 = data.time
            t = data.time - t0
            if t > 2.0:
                return
            pos = data.body("parcel").xpos - np.array([*sim.bin_xy, 0.0])
            contacts = {
                other
                for c in data.contact
                for a, other in ((c.geom1, c.geom2), (c.geom2, c.geom1))
                if model.geom_bodyid[a] == parcel_id
            }
            if contacts & lip_ids:
                last_lip = round(t, 3)
            for g in sorted(contacts & support_ids):
                name = model.geom(g).name
                if not any(e["name"] == name for e in support):
                    support.append(
                        dict(
                            name=name,
                            t=round(t, 3),
                            xyz_cm=(100 * pos).round(2).tolist(),
                        )
                    )
            if int(round(t / 0.005)) % 4 == 0:
                trace.append(dict(t=round(t, 3), xyz_cm=(100 * pos).round(2).tolist()))

        return original_step(targets, slot, observe)

    sim.c.step = wrapper
    row = sim.run()
    return row, dict(
        trace=trace,
        first_contact_per_surface=support,
        last_lip_contact_s=last_lip,
        final_from_tray_cm=(
            100 * (data.body("parcel").xpos - np.array([*sim.bin_xy, 0.0]))
        )
        .round(2)
        .tolist(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT / "runs/fetch/retrieval-review")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    detector = Detector(ROOT / "runs/fetch/round3-vision/change.pt", "cuda")
    profiles = load_profiles(ROOT / "configs/retrieval-grasps-v6.json")
    records = []
    for panel in ("test", "blind", "blind2", "blind3"):
        run = json.loads((args.root / f"v6-{panel}" / "episodes.json").read_text())
        for item in (i for i in OBJECTS if i["split"] == panel):
            held = [
                r
                for r in run["rows"]
                if r["object"] == item["id"] and r["held_at_release"]
            ]
            cases = [r for r in held if not r["in_tray_at_end"]] + [
                r for r in held if r["success"]
            ][:1]
            if not cases:
                continue
            model, half = make_scene(item, BIN, moving_jaw=True)
            with mujoco.Renderer(model, HEIGHT, WIDTH) as cam:
                for old in cases:
                    row, trace = observe_episode(
                        Episode(
                            model,
                            half,
                            cam,
                            detector,
                            "jaw",
                            old["seed"],
                            BIN,
                            profiles=profiles,
                        )
                    )
                    row.update(object=item["id"], family=item["family"])
                    if row != {k: v for k, v in old.items() if k != "video"}:
                        raise ValueError(
                            f"referee replay diverged: {item['id']} {old['seed']}"
                        )
                    records.append(
                        dict(
                            panel=panel,
                            object=item["id"],
                            seed=row["seed"],
                            passed=row["success"],
                            **trace,
                        )
                    )
    args.out.write_text(json.dumps(records, indent=2) + "\n")
    print(f"{len(records)} trajectories reproduced")


if __name__ == "__main__":
    main()
