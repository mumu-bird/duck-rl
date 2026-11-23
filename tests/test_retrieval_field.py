import copy
import hashlib
import json
import math
import unittest

from tools.retrieval_field import shadow
from tools.retrieval_field.field_check import replay_bundle


def small_model():
    return dict(schema=shadow.SCHEMA, motion_permitted=False, checkpoint_sha256="fixture",
                weights={"body.0.weight": [[.01]*29, [-.02]*29], "body.0.bias": [.1, -.1],
                         "body.2.weight": [[.2, .3], [.4, -.5]], "body.2.bias": [.01, .02],
                         "actors.0.weight": [[.1, .2]]*6, "actors.0.bias": [0.0]*6,
                         "actors.1.weight": [[-.2, .3]]*6, "actors.1.bias": [.01]*6})


class FieldReplayTest(unittest.TestCase):
    def test_rejects_wrong_stage_nonfinite_and_out_of_range_inputs(self):
        good = [1.0, 0.0] + [0.0]*27
        for stage, obs in [("drop", good), ("delivery", good), ("pick", good[:-1]),
                           ("pick", good[:3]+[math.nan]+good[4:]),
                           ("pick", good[:3]+[1.1]+good[4:]),
                           ("pick", good[:3]+[True]+good[4:])]:
            with self.assertRaises(ValueError):
                shadow.infer(small_model(), stage, obs)

    def test_mismatch_fails_replay_without_granting_motion(self):
        case = dict(stage="pick", obs=[1., 0.] + [0.]*27, actions=[.99]*6)
        result = shadow.verify_cases(small_model(), [case])
        self.assertEqual(result["code"], "SHADOW_REPLAY_FAIL")
        self.assertFalse(result["motion_permitted"])
        self.assertFalse(result["hardware_tested"])

    def test_bundle_integrity_rejects_modified_actor(self):
        model = small_model()
        obs = [1., 0.] + [0.]*27
        cases = [dict(stage="pick", obs=obs, actions=shadow.infer(model, "pick", obs))]
        files = {"actor.json": json.dumps(model).encode(),
                 "replay-cases.json": json.dumps(cases).encode(),
                 "verification.json": b"{}", "high-level-actor.onnx": b"test-only-placeholder"}
        manifest = dict(schema=shadow.SCHEMA, task="retrieval", motion_permitted=False,
                        checkpoint_sha256="fixture",
                        files={k:hashlib.sha256(v).hexdigest() for k,v in files.items()})
        files["manifest.json"] = json.dumps(manifest).encode()
        result = replay_bundle(files.__getitem__)
        self.assertEqual(result["code"], "SHADOW_REPLAY_PASS")
        self.assertFalse(result["motion_permitted"])
        files["actor.json"] += b" "
        with self.assertRaisesRegex(ValueError, "checksum"):
            replay_bundle(files.__getitem__)

    def test_reference_actions_must_be_finite_and_complete(self):
        for values in ([0.]*5, [float("nan")]*6):
            with self.assertRaises(ValueError):
                shadow.verify_cases(small_model(), [dict(stage="pick", obs=[1.,0.]+[0.]*27, actions=values)])

    def test_stdlib_and_export_wrapper_match_pytorch_actor(self):
        try:
            import torch
            import numpy as np
            from ducklab.retrieval.ppo import PPOPolicy
            from tools.retrieval_field.export_policy import DeterministicActor
        except ImportError:
            self.skipTest("requires training environment")
        torch.manual_seed(17)
        policy = PPOPolicy(device="cpu", hidden=32, train=False)
        model = dict(schema=shadow.SCHEMA, motion_permitted=False,
                     weights={k:v.tolist() for k,v in policy.net.state_dict().items()
                              if k.startswith(("body.","actors."))})
        rng = np.random.default_rng(7)
        for count in (1, 7, 19):
            obs = rng.uniform(-1, 1, (count,29)).astype(np.float32)
            stages = ["pick" if i%2 == 0 else "delivery" for i in range(count)]
            ids = [shadow.STAGES[s] for s in stages]
            for i, stage in enumerate(stages):
                obs[i,:2] = [stage == "pick", stage == "delivery"]
            reference, _ = policy.act_batch(stages, obs)
            actual = DeterministicActor(policy.net)(torch.from_numpy(obs), torch.tensor(ids)).detach().numpy()
            np.testing.assert_allclose(actual, reference, atol=2e-6, rtol=0)
            for stage, row, expected in zip(stages,obs,reference):
                np.testing.assert_allclose(shadow.infer(model,stage,row.tolist()), expected,atol=2e-6,rtol=0)


if __name__ == "__main__":
    unittest.main()
