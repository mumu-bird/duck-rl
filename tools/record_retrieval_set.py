#!/usr/bin/env python3
"""Record every simulated toss, keeping per-object outputs and a single benchmark file."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--profiles", type=Path, default=ROOT / "configs/retrieval-grasps.json"
    )
    parser.add_argument(
        "--weights", type=Path, default=ROOT / "runs/fetch/vision/change.pt"
    )
    parser.add_argument(
        "--split",
        required=True,
        choices=("test", "blind", "val", "challenge", "blind2", "blind3"),
    )
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--trials", required=True, type=int)
    parser.add_argument("--mass-grams", type=float, default=4.0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument(
        "--reference", type=Path, help="verify every jaw row against an independent run"
    )
    parser.add_argument("--bin", nargs=2, type=float, metavar=("X", "Y"))
    parser.add_argument("--no-toss", action="store_true")
    args = parser.parse_args()
    items = [
        i["id"]
        for i in json.loads((ROOT / "configs/retrieval-objects.json").read_text())[
            "objects"
        ]
        if i["split"] == args.split
    ]
    args.out.mkdir(parents=True, exist_ok=True)

    def record(identity):
        directory = args.out / identity
        directory.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(ROOT / "tools/run_retrieval.py"),
            "--mode",
            "jaw",
            "--object",
            identity,
            "--split",
            args.split,
            "--seed",
            str(args.seed),
            "--trials",
            str(args.trials),
            "--profiles",
            str(args.profiles),
            "--weights",
            str(args.weights),
            "--out",
            str(directory),
            "--mass-grams",
            str(args.mass_grams),
            "--video-all",
        ]
        if args.bin:
            command.extend(["--bin", *map(str, args.bin)])
        if args.no_toss:
            command.append("--no-toss")
        with (directory / "record.log").open("w") as log:
            subprocess.run(
                command,
                check=True,
                env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        return json.loads((directory / "episodes.json").read_text())

    with ThreadPoolExecutor(args.workers) as executor:
        batches = list(executor.map(record, items))
    rows = [r for batch in batches for r in batch["rows"]]
    if len(rows) != args.trials * len(items):
        raise ValueError("incomplete capture")
    if args.reference:
        reference = {
            (r["object"], r["seed"]): r
            for r in json.loads(args.reference.read_text())["rows"]
            if r["mode"] == "jaw"
        }
        for r in rows:
            if r != reference[(r["object"], r["seed"])]:
                raise ValueError(f"replay diverged: {r['object']} seed {r['seed']}")
    for row in rows:
        row["video"] = str(
            Path(row["object"]) / f"{row['object']}-jaw-{row['seed']}.mp4"
        )
    result = {**batches[0], "rows": rows}
    temp = args.out / "episodes.partial.json"
    temp.write_text(json.dumps(result, indent=2) + "\n")
    temp.replace(args.out / "episodes.json")
    print(
        f"{args.out}: {sum(r['success'] for r in rows)}/{len(rows)} full passes, {len(rows)} recordings",
        flush=True,
    )


if __name__ == "__main__":
    main()
