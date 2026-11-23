#!/usr/bin/env python3
"""Validate shortlisted scoop motions in full-length trials without using either blind panel."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    profiles = {
        "v4": ROOT / "configs/retrieval-grasps-v4.json",
        **{p.stem: p for p in sorted(args.fit.glob("candidate-*.json"))},
    }
    if len(profiles) != 4:
        parser.error("expected three candidates and one control")

    def run(job):
        name, split = job
        out = args.fit / "validation" / split / name
        out.mkdir(parents=True, exist_ok=True)
        with (out / "run.log").open("w") as log:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/run_retrieval.py"),
                    "--mode",
                    "jaw",
                    "--split",
                    split,
                    "--seed",
                    "114300" if split == "flat_val" else "114500",
                    "--trials",
                    "12",
                    "--weights",
                    str(args.weights),
                    "--profiles",
                    str(profiles[name]),
                    "--out",
                    str(out),
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )
        rows = json.loads((out / "episodes.json").read_text())["rows"]
        return (
            name,
            split,
            dict(
                n=len(rows),
                success=sum(r["success"] for r in rows),
                lifted=sum(r["lifted"] for r in rows),
                falls=sum(r["stop"] == "fall" for r in rows),
            ),
        )

    summaries = {k: {} for k in profiles}
    with ThreadPoolExecutor(5) as pool:
        for name, split, measure in pool.map(
            run, [(n, s) for n in profiles for s in ("flat_val", "val")]
        ):
            summaries[name][split] = measure

    def score(name):
        flat, general = summaries[name]["flat_val"], summaries[name]["val"]
        return (
            flat["success"] + general["success"],
            flat["success"],
            -flat["falls"] - general["falls"],
            flat["lifted"],
        )

    winner = max(profiles, key=score)
    (args.fit / "selection.json").write_text(
        json.dumps(dict(selected=winner, validation=summaries), indent=2) + "\n"
    )
    shutil.copyfile(profiles[winner], args.fit / "frozen.json")
    print(winner, summaries)


if __name__ == "__main__":
    main()
