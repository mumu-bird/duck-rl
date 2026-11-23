#!/usr/bin/env python3
"""Evaluate normalized Wonder Duck episode results from a CSV file."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import tomllib
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "task",
    "episode",
    "seed",
    "success",
    "fell",
    "joint_limit_violations",
    "nonfinite",
}
DEFAULT_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "acceptance.toml"


def _parse_flag(value: str | None, column: str, line: int) -> bool:
    if value is None or not value.strip():
        raise ValueError(f"line {line}: {column} is empty")
    normalized = value.strip().lower()
    if normalized in {"1", "true"}:
        return True
    if normalized in {"0", "false"}:
        return False
    raise ValueError(f"line {line}: {column} must be 0/1 or true/false")


def load_results(path: Path, expected_task: str) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = set(reader.fieldnames or ())
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"{path}: missing CSV columns: {', '.join(sorted(missing))}")
        rows: list[dict[str, Any]] = []
        seen_episodes: set[int] = set()
        for line, raw in enumerate(reader, start=2):
            try:
                task = (raw["task"] or "").strip()
                episode = int(raw["episode"])
                seed = int(raw["seed"])
                violations = int(raw["joint_limit_violations"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}: line {line}: invalid task, episode, seed, or violation count") from exc
            if task != expected_task:
                raise ValueError(f"{path}: line {line}: expected task {expected_task!r}, got {task!r}")
            if episode in seen_episodes:
                raise ValueError(f"{path}: line {line}: duplicate episode id {episode}")
            if violations < 0:
                raise ValueError(f"{path}: line {line}: joint_limit_violations cannot be negative")
            seen_episodes.add(episode)
            rows.append(
                {
                    "episode": episode,
                    "seed": seed,
                    "success": _parse_flag(raw["success"], "success", line),
                    "fell": _parse_flag(raw["fell"], "fell", line),
                    "joint_limit_violations": violations,
                    "nonfinite": _parse_flag(raw["nonfinite"], "nonfinite", line),
                }
            )
    if not rows:
        raise ValueError(f"{path}: CSV contains no episodes")
    return rows


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    count = len(rows)
    return {
        "episodes": count,
        "successes": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / count,
        "terminal_falls": sum(row["fell"] for row in rows),
        "terminal_fall_rate": sum(row["fell"] for row in rows) / count,
        "joint_limit_violation_episodes": sum(row["joint_limit_violations"] > 0 for row in rows),
        "joint_limit_violations": sum(row["joint_limit_violations"] for row in rows),
        "nonfinite_episodes": sum(row["nonfinite"] for row in rows),
    }


def evaluate(metrics: dict[str, Any], criteria: dict[str, Any]) -> dict[str, bool]:
    return {
        "minimum_episodes": metrics["episodes"] >= criteria["minimum_episodes"],
        "minimum_success_rate": metrics["success_rate"] >= criteria["minimum_success_rate"],
        "terminal_fall_rate": metrics["terminal_fall_rate"]
        < criteria["maximum_terminal_fall_rate_exclusive"],
        "joint_limit_violations": metrics["joint_limit_violation_episodes"]
        <= criteria["maximum_joint_limit_violation_episodes"],
        "nonfinite_episodes": metrics["nonfinite_episodes"] <= criteria["maximum_nonfinite_episodes"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("penalty_kick", "push_recovery"))
    parser.add_argument("candidate", type=Path, help="normalized candidate evaluation CSV")
    parser.add_argument("--baseline", type=Path, help="optional normalized baseline CSV for comparison")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="acceptance TOML")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        with args.config.open("rb") as handle:
            config = tomllib.load(handle)
        criteria = config[args.task]
        candidate_rows = load_results(args.candidate, args.task)
        candidate = summarize(candidate_rows)
        checks = evaluate(candidate, criteria)
        report: dict[str, Any] = {
            "task": args.task,
            "candidate": candidate,
            "criteria": criteria,
            "checks": checks,
            "passed": all(checks.values()),
        }
        if args.baseline:
            baseline = summarize(load_results(args.baseline, args.task))
            report["baseline"] = baseline
            report["success_rate_delta"] = candidate["success_rate"] - baseline["success_rate"]
    except (OSError, KeyError, tomllib.TOMLDecodeError, ValueError) as exc:
        print(f"evaluation error: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
