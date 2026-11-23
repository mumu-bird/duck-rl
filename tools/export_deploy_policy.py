#!/usr/bin/env python3
"""Export a trained deployable policy checkpoint to the on-robot ONNX contract.

Reads a ``policy-*.pt`` produced by tools/train_deploy.py and writes:

* ``policy.onnx``        obs[1,61] float32 -> actions[1,14] float32, opset 18 —
                         loads into any robotd policy slot (Studio 模型管理);
* ``deploy-manifest.json`` everything the robot side needs: joint order, home
                         pose, action scale, servo gain, delay, mouth handling
                         and the command-block mapping, plus file checksums.

Usage:
    python3 tools/export_deploy_policy.py runs/deploy/toss_catch-*/policy-best.pt
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from ducklab.deploy import contract as C
from ducklab.deploy.ppo import DeployPPO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--task", required=True, choices=["duck_lift", "toss_catch"])
    ap.add_argument("--out", default=None, help="output directory (default: alongside)")
    args = ap.parse_args()

    ckpt = Path(args.checkpoint)
    out = Path(args.out or ckpt.parent)
    out.mkdir(parents=True, exist_ok=True)

    policy = DeployPPO()
    policy.load(ckpt)
    onnx_path = out / "policy.onnx"
    policy.export_onnx(onnx_path)

    # parity check: torch vs onnxruntime
    import onnxruntime as ort
    rng = np.random.default_rng(0)
    probe = rng.normal(0, 1, (8, 61)).astype(np.float32)
    torch_out, _, _ = policy.act_batch(probe, deterministic=True)
    sess = ort.InferenceSession(str(onnx_path))
    onnx_out = sess.run(None, {"obs": probe})[0]
    max_err = float(np.abs(torch_out - onnx_out).max())
    assert max_err < 2e-4, f"ONNX parity failed: {max_err}"

    digest = hashlib.sha256(onnx_path.read_bytes()).hexdigest()
    manifest = dict(
        task=args.task,
        contract=dict(
            obs_len=C.OBS_LEN, action_len=C.ACTION_LEN,
            control_hz=C.CONTROL_HZ, physics_hz=C.PHYSICS_HZ,
            physics_dt=C.PHYSICS_DT, substeps=C.SUBSTEPS,
            action_scale=C.ACTION_SCALE, kp_fw=C.KP_FW,
            deadband_counts=C.DEADBAND_COUNTS,
            delay_steps_official=list(C.DELAY_STEPS_OFFICIAL),
            head_lowpass=C.HEAD_LOWPASS, legs_lowpass=C.LEGS_LOWPASS,
            voltage_adapt=C.VOLTAGE_ADAPT,
        ),
        joints=dict(
            names=C.JOINT_NAMES, ids=C.JOINT_IDS, mouth_index=C.MOUTH_INDEX,
            home_pose=C.DEFAULT_POSITION.tolist(),
            limits=[list(map(float, pair)) for pair in C.JOINT_LIMITS],
        ),
        mouth=dict(
            bus_id=34, closed_rad=C.MOUTH_CLOSED, open_rad=C.MOUTH_OPEN,
            encoder_closed=C.MOUTH_ENCODER_CLOSED, encoder_open=C.MOUTH_ENCODER_OPEN,
            note="policy never drives the mouth; use the rule controller in "
                 "ducklab/deploy/env.py (BeakController) or an equivalent board-side "
                 "controller writing GOAL_POSITION on bus servo 34",
        ),
        command_block=dict(
            twist="vx, vy, vyaw (m/s, m/s, rad/s), trunk frame",
            head="neck_pitch, head_pitch, head_yaw, head_roll absolute radians",
            body="x,y always zero; z, roll, pitch nominal zero; yaw zero",
        ),
        onnx=dict(file=onnx_path.name, sha256=digest, opset=18,
                  inputs={"obs": [1, 61]}, outputs={"actions": [1, 14]},
                  torch_onnx_max_abs_error=max_err),
    )
    (out / "deploy-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"ONNX: {onnx_path}  sha256={digest[:16]}...  parity={max_err:.2e}")
    print(f"清单: {out / 'deploy-manifest.json'}")


if __name__ == "__main__":
    main()
