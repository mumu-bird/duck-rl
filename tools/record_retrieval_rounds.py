#!/usr/bin/env python3
"""Archive a matching success and failure from each candidate's measured rows."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
FETCH = ROOT / "runs/fetch"
# Panel identifiers, action profile, measured evaluation followed by training fallback.
ROUNDS = {
    2: ("round2-fit", "delayed", ["val", "train"]),
    3: ("round3-fit", "ramp45", ["val", "train"]),
    4: ("round4-fit", "angled_near", ["val", "train"]),
    5: ("round5-fit", "right_far", ["flat_val", "flat_train"]),
    6: ("round6-fit", "candidate-1", ["validation/flat_val", "validation/val"]),
    7: ("round7-fit", "farther", ["val", "train"]),
    8: ("round8-fit", "tilt", ["val", "train"]),
    9: ("round9-fit", "dip-35-119", ["val", "train"]),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    jobs = []
    for rnd, (folder, name, splits) in ROUNDS.items():
        selected = {}
        for split in splits:
            path = FETCH / folder / split / name / "episodes.json"
            run = json.loads(path.read_text())
            for row in run["rows"]:
                selected.setdefault(
                    row["success"], (rnd, folder, name, run["split"], row, path)
                )
        jobs.extend(selected.values())

    def record(job):
        rnd, folder, name, split, row, reference = job
        seed = row["seed"]
        obj = row["object"]
        out = args.out / f"r{rnd}" / f"{obj}-{seed}"
        out.mkdir(parents=True, exist_ok=True)
        weights = FETCH / ("vision" if rnd == 2 else "round3-vision") / "change.pt"
        command = [
            sys.executable,
            str(ROOT / "tools/run_retrieval.py"),
            "--mode",
            "jaw",
            "--split",
            split,
            "--object",
            obj,
            "--seed",
            str(seed),
            "--trials",
            "1",
            "--video-all",
            "--profiles",
            str(FETCH / folder / (name + ".json")),
            "--weights",
            str(weights),
            "--out",
            str(out),
        ]
        with (out / "record.log").open("w") as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        replay = json.loads((out / "episodes.json").read_text())["rows"][0]
        if replay != row:
            raise ValueError(f"round {rnd} differed from {reference}: {obj} {seed}")
        return dict(
            round=rnd,
            candidate=name,
            split=split,
            row=row,
            reference=str(reference.relative_to(ROOT)),
            video=str((out / f"{obj}-jaw-{seed}.mp4").relative_to(args.out)),
        )

    with ThreadPoolExecutor(4) as pool:
        records = list(pool.map(record, jobs))
    (args.out / "clips.json").write_text(json.dumps(records, indent=2) + "\n")
    print(f"{len(records)} verified candidate replays")


if __name__ == "__main__":
    main()
