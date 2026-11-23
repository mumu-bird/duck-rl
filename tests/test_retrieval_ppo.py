import unittest
import tempfile
from pathlib import Path

try:
    import numpy as np
    import torch

    from ducklab.retrieval.ppo import ACTION_SIZE, OBS_SIZE, PPOPolicy
    from ducklab.retrieval.vision import Detector, silhouette_features
    from tools.train_retrieval_ppo import (
        _RemotePolicy,
        training_curriculum,
    )
except ImportError:
    np = torch = PPOPolicy = silhouette_features = training_curriculum = None
    Detector = _RemotePolicy = None


@unittest.skipIf(torch is None, "requires PyTorch")
class PPOTest(unittest.TestCase):
    def test_replay_curriculum_retains_prior_difficulty_levels(self):
        rng = np.random.default_rng(13)
        early = [training_curriculum(0.1, rng, "replay")[3] for _ in range(100)]
        middle = [training_curriculum(0.35, rng, "replay")[3] for _ in range(2000)]
        late = [training_curriculum(0.85, rng, "replay")[3] for _ in range(5000)]

        self.assertEqual(set(early), {"easy"})
        middle_rates = {name: middle.count(name) / len(middle) for name in set(middle)}
        self.assertGreater(middle_rates["easy"], 0.60)
        self.assertGreater(middle_rates["mixed"], 0.30)
        self.assertNotIn("randomized", middle_rates)
        late_rates = {name: late.count(name) / len(late) for name in set(late)}
        self.assertAlmostEqual(late_rates["easy"], 0.45, delta=0.03)
        self.assertAlmostEqual(late_rates["mixed"], 0.35, delta=0.03)
        self.assertAlmostEqual(late_rates["randomized"], 0.20, delta=0.03)

    def test_challenge_objects_enter_only_the_expanded_curriculum(self):
        rng = np.random.default_rng(19)
        easy = [
            training_curriculum(0.1, rng, "replay", include_challenge=True)[0]
            for _ in range(40)
        ]
        expanded = [
            training_curriculum(0.8, rng, "sequential", include_challenge=True)[0]
            for _ in range(2000)
        ]
        self.assertTrue(all(item["split"] != "challenge" for item in easy))
        challenge_rate = sum(item["split"] == "challenge" for item in expanded) / len(
            expanded
        )
        self.assertAlmostEqual(challenge_rate, 0.25, delta=0.04)

    def test_silhouette_features_encode_elongation_and_fill(self):
        yy, xx = np.mgrid[:4, :8]
        elongation, fill = silhouette_features(yy.ravel(), xx.ravel())
        self.assertAlmostEqual(elongation, 1 / 3, places=6)
        self.assertAlmostEqual(fill, 1.0, places=6)
        sparse_yy = np.array([0, 0, 3, 3])
        sparse_xx = np.array([0, 7, 0, 7])
        sparse_elongation, sparse_fill = silhouette_features(
            sparse_yy, sparse_xx
        )
        self.assertAlmostEqual(sparse_elongation, elongation, places=6)
        self.assertAlmostEqual(sparse_fill, 0.125, places=6)

    def test_phase_actions_are_bounded_and_update(self):
        torch.manual_seed(8)
        policy = PPOPolicy(device="cpu", minibatch_size=4, update_epochs=2)
        for index in range(8):
            obs = np.zeros(OBS_SIZE, dtype=np.float32)
            obs[index % 2] = 1.0
            action = policy.act("pick" if index % 2 == 0 else "delivery", obs)
            self.assertEqual(action.shape, (6,))
            self.assertTrue(np.isfinite(action).all())
            self.assertLessEqual(float(np.max(np.abs(action))), 1.0)
            stage = "pick" if index % 2 == 0 else "delivery"
            policy.resolve(stage, index / 4 - 1, done=True)
            policy.finish_episode()
        metrics = policy.update()
        self.assertEqual(metrics["decisions"], 8)
        self.assertTrue(np.isfinite(list(metrics.values())).all())
        self.assertGreaterEqual(metrics["approx_kl"], 0.0)
        self.assertGreaterEqual(float(policy.net.log_std.min().item()), -1.351)
        self.assertLessEqual(float(policy.net.log_std.max().item()), -0.799)

    def test_batched_rollouts_can_be_committed_to_the_cuda_learner_buffer(self):
        torch.manual_seed(18)
        policy = PPOPolicy(device="cpu", minibatch_size=4, update_epochs=2)
        stages = ["pick", "delivery"] * 4
        observations = np.zeros((8, OBS_SIZE), dtype=np.float32)
        observations[:, 0] = np.arange(8) / 8
        actions, records = policy.act_batch(stages, observations)
        self.assertEqual(actions.shape, (8, ACTION_SIZE))
        self.assertEqual(len(records), 8)
        self.assertTrue(np.isfinite(actions).all())
        for index, record in enumerate(records):
            policy.finish_external_episode(
                [
                    {
                        **record,
                        "reward": float(index) / 4 - 0.5,
                        "done": True,
                    }
                ]
            )
        metrics = policy.update()
        self.assertEqual(metrics["decisions"], 8)
        self.assertTrue(np.isfinite(list(metrics.values())).all())

    def test_remote_policy_preserves_action_records_until_reward_resolution(self):
        record = {
            "obs": np.zeros(OBS_SIZE, dtype=np.float32),
            "latent": np.ones(ACTION_SIZE, dtype=np.float32),
            "log_prob": -1.0,
            "value": 0.25,
            "stage": 0,
        }

        class Services:
            def request(self, kind, payload):
                self.kind = kind
                self.payload = payload
                return np.zeros(ACTION_SIZE, dtype=np.float32), record

        remote = _RemotePolicy(Services())
        action = remote.act("pick", np.zeros(OBS_SIZE, dtype=np.float32))
        self.assertEqual(action.shape, (ACTION_SIZE,))
        remote.resolve("pick", 1.5, done=True)
        remote.finish_episode(0.2)
        self.assertEqual(len(remote.path), 1)
        self.assertEqual(remote.path[0]["reward"], 1.5)
        self.assertTrue(remote.path[0]["done"])
        self.assertAlmostEqual(remote.terminal_reward, 0.2)

    def test_batched_detector_matches_single_scene_results(self):
        class FixedNet(torch.nn.Module):
            def forward(self, frames):
                logits = torch.full(
                    (
                        len(frames),
                        frames.shape[-2],
                        frames.shape[-1],
                    ),
                    -10.0,
                    device=frames.device,
                )
                logits[:, 10:14, 20:26] = 10.0
                logits[:, 40:48, 80:86] = 10.0
                return logits

        detector = Detector.__new__(Detector)
        detector.device = "cpu"
        detector.net = FixedNet().eval()
        before = np.zeros((64, 112, 3), dtype=np.uint8)
        after = before.copy()
        top_roi = np.zeros((64, 112), dtype=bool)
        top_roi[8:16, 18:29] = True
        bottom_roi = np.zeros((64, 112), dtype=bool)
        bottom_roi[38:50, 77:90] = True
        empty_roi = np.zeros((64, 112), dtype=bool)
        samples = [
            (before, after, top_roi),
            (before, after, bottom_roi),
            (before, after, empty_roi),
        ]
        batch_results = detector.locate_batch(samples)
        single_results = [detector.locate(*sample) for sample in samples]
        self.assertEqual(len(batch_results), len(single_results))
        for batch, single in zip(batch_results, single_results, strict=True):
            self.assertEqual(batch is None, single is None)
            if batch is None:
                continue
            for left, right in zip(batch, single, strict=True):
                if left is None:
                    self.assertIsNone(right)
                elif isinstance(left, np.ndarray):
                    np.testing.assert_array_equal(left, right)
                else:
                    self.assertAlmostEqual(left, right)

    def test_tiny_rollout_is_not_overfit(self):
        policy = PPOPolicy(device="cpu")
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        policy.act("pick", obs)
        policy.resolve("pick", 1.0, done=True)
        policy.finish_episode()
        self.assertIsNone(policy.update())
        self.assertEqual(len(policy.buffer), 1)

    def test_pick_retry_can_make_a_second_delivery_decision(self):
        policy = PPOPolicy(device="cpu")
        obs = np.zeros(OBS_SIZE, dtype=np.float32)
        for stage, reward, done in (
            ("pick", 0.8, False),
            ("delivery", -0.8, False),
            ("pick", 0.6, False),
            ("delivery", 2.0, True),
        ):
            policy.act(stage, obs)
            policy.resolve(stage, reward, done=done)
        policy.finish_episode()
        self.assertEqual(len(policy.buffer), 4)
        self.assertFalse(policy.pending)
        self.assertFalse(policy.path)

    def test_loads_first_round_checkpoint_with_new_features(self):
        old = {
            "body.0.weight": torch.ones(256, 26),
            "body.0.bias": torch.zeros(256),
            "body.2.weight": torch.ones(256, 256),
            "body.2.bias": torch.zeros(256),
            "actor.weight": torch.ones(6, 256),
            "actor.bias": torch.zeros(6),
            "critic.weight": torch.ones(1, 256),
            "critic.bias": torch.zeros(1),
            "log_std": torch.zeros(6),
        }
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "legacy.pt"
            torch.save(
                {"obs_size": 26, "action_size": 6, "model": old}, checkpoint
            )
            policy = PPOPolicy(device="cpu")
            policy.load(checkpoint)
            weight = policy.net.body[0].weight.detach()
            self.assertTrue(torch.all(weight[:, :4] == 1))
            self.assertTrue(torch.all(weight[:, 4:7] == 0))
            self.assertTrue(torch.all(weight[:, 7:] == 1))
            self.assertTrue(torch.all(policy.net.log_std == 0))

    def test_migrates_round_three_checkpoint_without_changing_old_features(self):
        source = PPOPolicy(device="cpu")
        old_model = {
            key: value.detach().clone()
            for key, value in source.net.state_dict().items()
        }
        old_model["body.0.weight"] = torch.arange(27, dtype=torch.float32).repeat(
            256, 1
        )
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "round3.pt"
            torch.save(
                {"obs_size": 27, "action_size": 6, "model": old_model},
                checkpoint,
            )
            policy = PPOPolicy(device="cpu")
            policy.load(checkpoint)
            weight = policy.net.body[0].weight.detach()
            self.assertTrue(torch.all(weight[:, :5] == torch.arange(5)))
            self.assertTrue(torch.all(weight[:, 5:7] == 0))
            self.assertTrue(torch.all(weight[:, 7:] == torch.arange(5, 27)))


if __name__ == "__main__":
    unittest.main()
