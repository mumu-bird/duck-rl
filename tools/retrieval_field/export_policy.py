"""Export a frozen high-level pickup actor; verify stdlib and ONNX replay."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from ducklab.retrieval.ppo import PPOPolicy, OBS_SIZE, STAGES
from tools.retrieval_field.shadow import SCHEMA, verify_cases
from tools.train_retrieval_ppo import save_json


class DeterministicActor(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, observations, stages):
        features = self.net.body(observations)
        means = torch.stack([head(features) for head in self.net.actors], dim=1)
        indices = stages.reshape(-1, 1, 1).expand(-1, 1, 6)
        return means.gather(1, indices).squeeze(1).tanh()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--traces", type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--cases", type=int, default=1200)
    args = p.parse_args()
    if args.cases < 2 or (args.out / "manifest.json").exists():
        p.error("at least two cases and a fresh output directory required")
    torch.set_num_threads(1)
    policy = PPOPolicy(device="cpu", train=False)
    policy.load(args.checkpoint, optimizer=False)
    policy.net.eval()
    weights = {name: value.detach().tolist() for name, value in policy.net.state_dict().items()
               if name.startswith(("body.", "actors."))}
    model = dict(schema=SCHEMA, task="retrieval", weights=weights,
                 checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
                 scope="six high-level grasp/delivery residuals; not servo commands",
                 motion_permitted=False, hardware_tested=False)
    rng = np.random.default_rng(20261123)
    obs = rng.uniform(-1, 1, (args.cases, OBS_SIZE)).astype(np.float32)
    stages = ["pick" if i % 2 == 0 else "delivery" for i in range(args.cases)]
    for i, stage in enumerate(stages):
        obs[i, :2] = [stage == "pick", stage == "delivery"]
    actions, _ = policy.act_batch(stages, obs)
    cases = [dict(stage=stage, obs=o.tolist(), actions=a.tolist(), source="synthetic_normalized")
             for stage, o, a in zip(stages, obs, actions)]
    traces = json.loads(args.traces.read_text()) if args.traces else []
    for trace in traces:
        action, _ = policy.act_batch([trace["stage"]], [trace["obs"]])
        cases.append(dict(stage=trace["stage"], obs=trace["obs"], actions=action[0].tolist(),
                          source="simulation_decision"))
    started = time.monotonic()
    verified = verify_cases(model, cases)
    verified.update(host_elapsed_seconds=time.monotonic()-started, synthetic_cases=args.cases,
                    simulated_cases=len(traces), real_sensor_cases=0)
    if verified["code"] != "SHADOW_REPLAY_PASS":
        raise ValueError(verified)
    args.out.mkdir(parents=True, exist_ok=True)
    save_json(args.out / "actor.json", model)
    save_json(args.out / "replay-cases.json", cases)
    actor = DeterministicActor(policy.net).eval()
    ids = torch.tensor([STAGES[s] for s in stages], dtype=torch.long)
    onnx_path = args.out / "high-level-actor.onnx"
    torch.onnx.export(actor, (torch.from_numpy(obs[:2]), ids[:2]), str(onnx_path),
                      input_names=["observations", "stages"], output_names=["residuals"],
                      dynamic_axes={"observations": {0: "batch"}, "stages": {0: "batch"},
                                    "residuals": {0: "batch"}}, opset_version=17, dynamo=False)
    import onnxruntime as ort
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(onnx_path), options, providers=["CPUExecutionProvider"])
    all_obs = np.asarray([c["obs"] for c in cases], dtype=np.float32)
    all_stages = np.asarray([STAGES[c["stage"]] for c in cases], dtype=np.int64)
    onnx_actions = session.run(None, {"observations": all_obs, "stages": all_stages})[0]
    delta = float(np.max(np.abs(onnx_actions - np.asarray([c["actions"] for c in cases]))))
    if delta > 2e-6:
        raise ValueError(f"ONNX equivalence failed: {delta}")
    batch_errors = {}
    for count in (1, 2, 17):
        actual = session.run(None, {"observations": all_obs[:count], "stages": all_stages[:count]})[0]
        expected = np.asarray([c["actions"] for c in cases[:count]])
        batch_errors[str(count)] = float(np.max(np.abs(actual - expected)))
        if batch_errors[str(count)] > 2e-6:
            raise ValueError(f"ONNX batch {count} equivalence failed")
    verified["onnx_batch_errors"] = batch_errors
    verified["onnx_max_abs_error"] = delta
    verified["onnx_provider"] = "CPUExecutionProvider on development host"
    save_json(args.out / "verification.json", verified)
    manifest = dict(schema=SCHEMA, task="retrieval", hardware_tested=False, motion_permitted=False,
                    checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=model["checkpoint_sha256"],
                    files={name: hashlib.sha256((args.out / name).read_bytes()).hexdigest()
                           for name in ("actor.json", "replay-cases.json", "high-level-actor.onnx", "verification.json")})
    save_json(args.out / "manifest.json", manifest)
    print(json.dumps(verified, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
