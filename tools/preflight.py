#!/usr/bin/env python3
"""Check whether official Wonder Duck training assets are available."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def _resolve_argument(value: str | None, env_name: str) -> Path | None:
    raw = value or os.environ.get(env_name)
    return Path(raw).expanduser().resolve() if raw else None


def _check_asset(label: str, path: Path | None, expected: str) -> dict[str, Any]:
    if path is None:
        return {"label": label, "status": "missing", "detail": f"provide {expected}"}
    if expected == "directory" and not path.is_dir():
        return {"label": label, "status": "invalid", "path": str(path), "detail": "not a directory"}
    if expected == "file" and not path.is_file():
        return {"label": label, "status": "invalid", "path": str(path), "detail": "not a file"}
    if expected == "path" and not path.exists():
        return {"label": label, "status": "invalid", "path": str(path), "detail": "path does not exist"}
    result: dict[str, Any] = {"label": label, "status": "present", "path": str(path)}
    if label == "robot_mjcf" and path.suffix.lower() in {".xml", ".mjcf"}:
        try:
            root = ET.parse(path).getroot()
        except (ET.ParseError, OSError) as exc:
            return {**result, "status": "invalid", "detail": f"MJCF XML cannot be parsed: {exc}"}
        if root.tag != "mujoco":
            return {**result, "status": "invalid", "detail": f"expected <mujoco> root, found <{root.tag}>"}
        result["xml_root"] = root.tag
    return result


def _host_probe() -> dict[str, Any]:
    executable = shutil.which("nvidia-smi")
    gpu = {"nvidia_smi_found": bool(executable), "usable": False, "devices": []}
    if executable:
        try:
            probe = subprocess.run(
                [executable, "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                timeout=4,
                check=False,
            )
            gpu["usable"] = probe.returncode == 0
            if gpu["usable"]:
                gpu["devices"] = [line.strip() for line in probe.stdout.splitlines() if line.strip()]
        except (OSError, subprocess.TimeoutExpired):
            pass
    return {
        "scope": "process running this script; it may be isolated from the physical training host",
        "os": platform.platform(),
        "python": platform.python_version(),
        "mujoco_python_module_visible": importlib.util.find_spec("mujoco") is not None,
        "nvidia": gpu,
        "note": "informational only; WonderRL may bundle dependencies or run in another environment",
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    assets = [
        _check_asset("wonderrl_root", _resolve_argument(args.wonderrl_root, "WONDERRL_ROOT"), "a directory"),
        _check_asset("robot_mjcf", _resolve_argument(args.robot_mjcf, "WONDERDUCK_MJCF"), "an MJCF file"),
        _check_asset(
            "training_project",
            _resolve_argument(args.training_project, "WONDERDUCK_TRAINING_PROJECT"),
            "path",
        ),
        _check_asset(
            "baseline_policy",
            _resolve_argument(args.baseline_policy, "WONDERDUCK_BASELINE_POLICY"),
            "the official baseline policy file",
        ),
    ]
    assets_ready = all(item["status"] == "present" for item in assets)
    return {
        "status": "ASSETS_READY_PLATFORM_UNVERIFIED" if assets_ready else "BLOCKED_MISSING_ASSETS",
        "assets": assets,
        "host_probe": _host_probe(),
        "next_step": (
            "verify this WonderRL release's documented OS/GPU requirements and run its official smoke test"
            if assets_ready
            else "obtain the exact official WonderRL package, Wonder Duck MJCF, training project, and baseline policy"
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wonderrl-root", help="directory containing the official WonderRL release")
    parser.add_argument("--robot-mjcf", help="exact Wonder Duck MJCF file")
    parser.add_argument("--training-project", help="official WonderRL training project file or directory")
    parser.add_argument("--baseline-policy", help="official baseline checkpoint or policy file")
    return parser.parse_args()


def main() -> int:
    report = build_report(parse_args())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "ASSETS_READY_PLATFORM_UNVERIFIED" else 2


if __name__ == "__main__":
    sys.exit(main())
