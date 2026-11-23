#!/usr/bin/env python3
"""Summarize a complete paired cohort; reject missing or duplicate tosses."""

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import median

FIELDS = (
    "detected",
    "target_contact",
    "lifted",
    "held_at_release",
    "in_tray_at_end",
    "pose_restored",
    "success",
    "wrong_contact",
    "fixture_activated",
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    data = json.loads(args.episodes.read_text())
    rows = data["rows"]
    groups = sorted({(r["object"], r["mode"]) for r in rows})
    per_group = data["trials_per_object_mode"]
    results = {}
    expected = list(range(data["seed"], data["seed"] + per_group))
    objects = {key[0] for key in groups}
    modes = {key[1] for key in groups}
    if len(groups) != len(objects) * len(modes):
        raise ValueError("incomplete object-mode grid")
    for obj, mode in groups:
        trials = [r for r in rows if (r["object"], r["mode"]) == (obj, mode)]
        successes = [r for r in trials if r["success"]]
        if sorted(t["seed"] for t in trials) != expected:
            raise ValueError(f"incomplete or duplicate seeds for {obj}, {mode}")
        if mode != "assisted" and any(t["fixture_activated"] for t in trials):
            raise ValueError(f"unexpected attachment in {mode}")
        summary = {key: sum(int(t[key]) for t in trials) for key in FIELDS}
        summary.update(
            trials=len(trials),
            success_seeds=[t["seed"] for t in successes],
            stops=dict(Counter(t["stop"] for t in trials)),
        )
        for k in ("home_cm", "yaw_deg", "joint_rms_rad"):
            summary["successful_median_" + k] = (
                round(median(t[k] for t in successes), 4) if successes else None
            )
        results.setdefault(obj, {})[mode] = summary
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {k: v for k, v in data.items() if k != "rows"}
            | {"total": len(rows), "results": results},
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
