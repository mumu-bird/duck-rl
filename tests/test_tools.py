from __future__ import annotations

import csv
import json
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from tools.evaluate import evaluate, load_results, summarize
from tools.preflight import build_report


CSV_COLUMNS = [
    "task",
    "episode",
    "seed",
    "success",
    "fell",
    "joint_limit_violations",
    "nonfinite",
]


class EvaluateTests(unittest.TestCase):
    def write_results(self, path: Path, rows: list[list[object]]) -> None:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(CSV_COLUMNS)
            writer.writerows(rows)

    def test_penalty_kick_passes_at_thresholds(self) -> None:
        rows = [
            {
                "success": i < 70,
                "fell": 70 <= i < 74,
                "joint_limit_violations": 0,
                "nonfinite": False,
            }
            for i in range(100)
        ]
        checks = evaluate(
            summarize(rows),
            {
                "minimum_episodes": 100,
                "minimum_success_rate": 0.70,
                "maximum_terminal_fall_rate_exclusive": 0.05,
                "maximum_joint_limit_violation_episodes": 0,
                "maximum_nonfinite_episodes": 0,
            },
        )
        self.assertTrue(all(checks.values()))

    def test_strict_fall_rate_limit_fails_at_five_percent(self) -> None:
        rows = [
            {
                "success": True,
                "fell": i < 5,
                "joint_limit_violations": 0,
                "nonfinite": False,
            }
            for i in range(100)
        ]
        checks = evaluate(
            summarize(rows),
            {
                "minimum_episodes": 100,
                "minimum_success_rate": 0.70,
                "maximum_terminal_fall_rate_exclusive": 0.05,
                "maximum_joint_limit_violation_episodes": 0,
                "maximum_nonfinite_episodes": 0,
            },
        )
        self.assertFalse(checks["terminal_fall_rate"])

    def test_csv_rejects_duplicate_episode_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "episodes.csv"
            self.write_results(
                path,
                [
                    ["penalty_kick", 1, 11, 1, 0, 0, 0],
                    ["penalty_kick", 1, 12, 1, 0, 0, 0],
                ],
            )
            with self.assertRaisesRegex(ValueError, "duplicate episode"):
                load_results(path, "penalty_kick")

    def test_csv_rejects_task_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "episodes.csv"
            self.write_results(path, [["push_recovery", 0, 9, 1, 0, 0, 0]])
            with self.assertRaisesRegex(ValueError, "expected task"):
                load_results(path, "penalty_kick")

    def test_csv_rejects_empty_flag(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "episodes.csv"
            self.write_results(path, [["penalty_kick", 0, 9, "", 0, 0, 0]])
            with self.assertRaisesRegex(ValueError, "success is empty"):
                load_results(path, "penalty_kick")

    def test_cli_reports_candidate_and_baseline(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            candidate = Path(temp_dir) / "candidate.csv"
            baseline = Path(temp_dir) / "baseline.csv"
            self.write_results(
                candidate,
                [["penalty_kick", i, 1000 + i, int(i < 70), int(70 <= i < 74), 0, 0] for i in range(100)],
            )
            self.write_results(
                baseline,
                [["penalty_kick", i, 2000 + i, int(i < 50), int(50 <= i < 53), 0, 0] for i in range(100)],
            )
            script = Path(__file__).resolve().parents[1] / "tools" / "evaluate.py"
            result = subprocess.run(
                [sys.executable, str(script), "penalty_kick", str(candidate), "--baseline", str(baseline)],
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(result.stdout)
            self.assertTrue(report["passed"])
            self.assertAlmostEqual(report["success_rate_delta"], 0.20)


class PreflightTests(unittest.TestCase):
    def test_missing_assets_are_reported_as_blocked(self) -> None:
        args = Namespace(
            wonderrl_root=None,
            robot_mjcf=None,
            training_project=None,
            baseline_policy=None,
        )
        report = build_report(args)
        self.assertEqual(report["status"], "BLOCKED_MISSING_ASSETS")
        self.assertEqual(len(report["assets"]), 4)

    def test_valid_xml_and_asset_paths_are_reported_ready(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            wonderrl = root / "wonderrl"
            wonderrl.mkdir()
            model = root / "duck.xml"
            model.write_text('<mujoco model="test"><worldbody/></mujoco>', encoding="utf-8")
            project = root / "training"
            project.mkdir()
            policy = root / "policy.onnx"
            policy.write_bytes(b"test")
            args = Namespace(
                wonderrl_root=str(wonderrl),
                robot_mjcf=str(model),
                training_project=str(project),
                baseline_policy=str(policy),
            )
            report = build_report(args)
            self.assertEqual(report["status"], "ASSETS_READY_PLATFORM_UNVERIFIED")

    def test_nonexistent_training_project_is_invalid(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            wonderrl = root / "wonderrl"
            wonderrl.mkdir()
            model = root / "duck.xml"
            model.write_text('<mujoco model="test"><worldbody/></mujoco>', encoding="utf-8")
            policy = root / "policy.onnx"
            policy.write_bytes(b"test")
            args = Namespace(
                wonderrl_root=str(wonderrl),
                robot_mjcf=str(model),
                training_project=str(root / "missing-project"),
                baseline_policy=str(policy),
            )
            report = build_report(args)
            self.assertEqual(report["status"], "BLOCKED_MISSING_ASSETS")
            self.assertEqual(report["assets"][2]["status"], "invalid")


if __name__ == "__main__":
    unittest.main()
