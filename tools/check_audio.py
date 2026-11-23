#!/usr/bin/env python3
"""Run audio + affected task regressions and write a machine-readable local report."""
from pathlib import Path
import datetime
import importlib
import json
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))


def main():
    loader = unittest.TestLoader()
    suite = loader.discover(str(ROOT / "tests/audio"), pattern="test_*.py")
    for name in ("test_tools", "test_retrieval", "test_duck_retrieval", "test_self_catch", "test_straight"):
        suite.addTests(loader.loadTestsFromModule(importlib.import_module(name)))
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    directory = ROOT / "runs/audio-acceptance"
    directory.mkdir(parents=True, exist_ok=True)
    report = {"generated_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),
        "python": sys.version, "run": result.testsRun,
        "passed": result.testsRun-len(result.failures)-len(result.errors)-len(result.skipped),
        "success": result.wasSuccessful(), "failures": [(str(test),message) for test,message in result.failures],
        "errors": [(str(test),message) for test,message in result.errors],
        "skipped": [(str(test),reason) for test,reason in result.skipped],
        "scope": "Audio service, deployment and affected task regressions; fake PCM/capture subprocesses, real FFmpeg and eSpeak; no robot hardware",
        "full_repository_limit": "Earlier complete discovery: test_selfcatch_field import failed because torch is absent; 11 PPO tests skipped. These training-only tests are outside this audio regression."}
    (directory / "software-test-report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    raise SystemExit(main())
