#!/usr/bin/env python3
"""Supplement the six TRAIN archives with procedural size/shape variation. Keep eval pairs fixed."""

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
from render_retrieval import OBJECTS, ROOT, generate

# Each added object is stored in its own train bucket before concatenation. No test asset
# or scan is used to make variants. Shape and albedo are visible in the RGB pairs.
VARIANTS = {
    "felt_pen": [("thin_rod", "pen", 0.104), ("stubby_rod", "pen", 0.047)],
    "capped_crayon": [("short_chalk", "pen", 0.055), ("long_chalk", "pen", 0.094)],
    "056_tennis_ball": [
        ("small_pebble", "irregular", 0.022),
        ("long_pebble", "irregular", 0.041),
    ],
    "012_strawberry": [
        ("tiny_oval", "irregular", 0.018),
        ("large_oval", "irregular", 0.052),
    ],
    "061_foam_brick": [
        ("small_eraser", "brick", 0.022),
        ("long_eraser", "brick", 0.055),
    ],
    "bath_bird": [("tiny_bird", "duck", 0.026), ("large_bird", "duck", 0.048)],
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, default=ROOT / "runs/fetch/frames")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--count", type=int, default=80, help="additional pairs per training variant"
    )
    args = parser.parse_args()
    train_ids = {item["id"] for item in OBJECTS if item["split"] == "train"}
    if set(VARIANTS) != train_ids:
        raise ValueError("variant buckets do not match the training identities")
    extra = args.out / "variants"
    extra.mkdir(parents=True, exist_ok=True)
    for i, (identity, variants) in enumerate(VARIANTS.items()):
        paths = [args.original / (identity + ".npz")]
        for j, (name, family, length) in enumerate(variants):
            generate(
                dict(id=name, family=family, length=length, split="train"),
                args.count,
                110450 + i * 500 + j * 199,
                extra,
            )
            paths.append(extra / (name + ".npz"))
        parts = {key: [] for key in ("before", "after", "mask")}
        for path in paths:
            with np.load(path) as data:
                for key in parts:
                    parts[key].append(data[key])
        np.savez_compressed(
            args.out / (identity + ".npz"),
            **{k: np.concatenate(v) for k, v in parts.items()},
        )
    for item in OBJECTS:
        if item["split"] in ("val", "test"):
            filename = item["id"] + ".npz"
            shutil.copyfile(args.original / filename, args.out / filename)
    (args.out / "generation.json").write_text(
        json.dumps(dict(variants=VARIANTS, pairs_per_variant=args.count), indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
