"""Paired offline ablation of two command features absent in stock ground-pick.

This is a command-semantics proxy, not a complete robotd emulator. It retains
simulated contact sensing, the lab jaw servo and the existing low-level ONNX.
"""
import argparse
import hashlib
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from ducklab.policy import Controller
from ducklab.retrieval.grasp import load_profiles
from ducklab.retrieval.ppo import PPOPolicy
from ducklab.retrieval.scene import BIN, OBJECTS
from ducklab.retrieval.vision import Detector
from tools.train_retrieval_ppo import run_episode, summarize, save_json


class StockCommandProxy(Controller):
    """Preserve original ONNX; use fixed pick phase and remove post-ONNX overrides."""
    def __init__(self, model, data):
        super().__init__(model, data)
        self.pick_tick = 0

    def act(self, slot, command):
        command = command.copy()
        if slot == "groundpick":
            command[:] = 0
            angle = self.pick_tick * .02 / 4.0 * np.pi * 2
            command[:2] = np.cos(angle), np.sin(angle)
            self.pick_tick += 1
        else:
            self.pick_tick = 0
        self.unmodified = super().act(slot, command).copy()
        return self.unmodified.copy()

    def step(self, targets, slot, observer=None):
        # The official runtime consumes high-level intents; these post-policy
        # target writes do not have an equivalent in the inspected public API.
        super().step(self.unmodified, slot, observer)


class TracePolicy:
    def __init__(self, policy):
        self.policy, self.trace = policy, []

    def act(self, stage, obs):
        action = self.policy.act(stage, obs)
        self.trace.append(dict(stage=stage, obs=np.asarray(obs).tolist(),
                               actions=np.asarray(action).tolist()))
        return action

    def resolve(self, *args, **kwargs):
        return self.policy.resolve(*args, **kwargs)

    def finish_episode(self, *args, **kwargs):
        return self.policy.finish_episode(*args, **kwargs)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--trials", type=int, default=2)
    p.add_argument("--masses", nargs="+", type=float, default=[4.0, 12.0])
    p.add_argument("--seed", type=int, default=2026112302)
    args = p.parse_args()
    if args.trials < 1 or not all(np.isfinite(m) and m > 0 for m in args.masses):
        p.error("positive trial count and finite positive masses required")
    if (args.out / "summary.json").exists():
        p.error("output already completed; choose a new directory")
    torch.set_num_threads(1)
    policy = PPOPolicy(device="cpu", train=False)
    policy.load(args.checkpoint, optimizer=False)
    detector = Detector(ROOT / "runs/fetch/round3-vision/change.pt", "cuda")
    profiles = load_profiles(ROOT / "configs/retrieval-grasps-v6.json")
    items = [i for i in OBJECTS if i["id"] in ("felt_pen", "056_tennis_ball", "bath_bird")]
    rows, traces = [], []
    for idx, item in enumerate(items):
        for trial in range(args.trials):
            seed = args.seed + idx * 101 + trial
            for mass in args.masses:
                for name, control in (("simulation", Controller), ("stock_command_proxy", StockCommandProxy)):
                    recorded = TracePolicy(policy)
                    video = args.out / "videos" / name / f"{item['id']}-{seed}-{mass:g}g.mp4"
                    with patch("ducklab.retrieval.episode.Controller", control):
                        result = run_episode(item, seed, BIN, detector, profiles, recorded,
                                             video_path=video, label=name, mass_grams=mass)
                    result.update(object=item["id"], seed=seed, mode=name, video=str(video))
                    rows.append(result)
                    traces.extend(dict(object=item["id"], seed=seed, mode=name, mass_grams=mass,
                                       **t) for t in recorded.trace)
                    save_json(args.out / "episodes.json", rows)
                    save_json(args.out / "decision-traces.json", traces)
                    print(name, item["id"], mass, result["success"], result["stop"], flush=True)
    summary = dict(task="retrieval", kind="offline_command_semantics_ablation",
                   checkpoint=str(args.checkpoint.resolve()),
                   checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                   hardware_tested=False, motion_permitted=False,
                   limitations=["same existing simulator ONNX; not factory model equivalence",
                                "retains simulated contact oracle and uncalibrated jaw physics",
                                "does not emulate full robotd filters, ownership or timing"],
                   summary={mode: summarize([r for r in rows if r["mode"] == mode])
                            for mode in ("simulation", "stock_command_proxy")},
                   decision_trace_count=len(traces), rows=rows)
    save_json(args.out / "summary.json", summary)
    print(summary["summary"], flush=True)


if __name__ == "__main__":
    main()
