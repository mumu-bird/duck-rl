"""Contract tests for the deployable training stack.

These pin the numbers that must match the robot exactly; a silent drift here
produces a plausible-looking robot that falls over on hardware.
"""

from pathlib import Path
import unittest

import numpy as np

from ducklab.deploy import contract as C
from ducklab.deploy.env import DeployEnv
from ducklab.deploy.scene import ROBOT_XML


class TestContract(unittest.TestCase):
    def test_joint_layout(self):
        self.assertEqual(C.NUM_JOINTS, 15)
        self.assertEqual(C.MOUTH_INDEX, 9)
        self.assertEqual(C.ACTION_LEN, 14)
        self.assertEqual(C.OBS_LEN, 61)
        self.assertEqual(C.JOINT_IDS[5:10], [30, 31, 32, 33, 34])
        self.assertEqual(C.JOINT_IDS[10:], [10, 11, 12, 13, 14])
        self.assertEqual(C.JOINT_NAMES[C.MOUTH_INDEX], "mouth")

    def test_observation_layout(self):
        action = np.arange(14, dtype=float)
        obs = C.build_observation(
            [1, 2, 3], [4, 5, 6], np.zeros(14), np.zeros(14), action,
            C.build_command([.1, .2, .3], [.4, .5, .6, .7]),
        )
        self.assertEqual(obs.shape, (61,))
        self.assertEqual(list(obs[0:3]), [1, 2, 3])
        self.assertEqual(list(obs[3:6]), [4, 5, 6])
        self.assertTrue(np.allclose(obs[6:20], -C.DEFAULT_POSITION[C.POLICY_JOINTS]))
        self.assertTrue(np.allclose(obs[20:34], 0))
        self.assertTrue(np.allclose(obs[34:48], action))
        self.assertTrue(np.allclose(obs[48:51], [.1, .2, .3]))
        self.assertTrue(np.allclose(obs[51:55], [.4, .5, .6, .7]))
        # body x, y and yaw are hardcoded zero on the robot
        self.assertEqual((obs[55], obs[56], obs[60]), (0.0, 0.0, 0.0))

    def test_targets_are_home_plus_action(self):
        action = np.linspace(-0.5, 0.5, 14)
        targets = C.targets_from_action(action)
        self.assertEqual(targets.shape, (15,))
        self.assertEqual(targets[C.MOUTH_INDEX], 0.0, "mouth is never policy-driven")
        for slot, joint in enumerate(C.POLICY_JOINTS):
            self.assertAlmostEqual(targets[joint], C.DEFAULT_POSITION[joint] + action[slot])

    def test_mouth_encoder_contract(self):
        self.assertEqual(C.mouth_angle_to_encoder(C.MOUTH_CLOSED), 2048)
        self.assertEqual(C.mouth_angle_to_encoder(C.MOUTH_OPEN), 1600)
        self.assertAlmostEqual(C.encoder_to_mouth_angle(1600), C.MOUTH_OPEN, places=6)
        # opening lowers the count (HX direction is negative)
        self.assertLess(C.MOUTH_ENCODER_OPEN, C.MOUTH_ENCODER_CLOSED)

    def test_home_encoder_targets(self):
        for joint in C.POLICY_JOINTS:
            self.assertEqual(
                C.joint_angle_to_encoder(joint, C.DEFAULT_POSITION[joint]),
                int(C.PHYSICAL_HOME_ENCODER_TARGETS[joint]),
            )

    def test_rates_and_scale(self):
        self.assertEqual(C.SUBSTEPS, 4)
        self.assertAlmostEqual(C.PHYSICS_DT, 0.005)
        self.assertEqual(C.CONTROL_HZ, 50)
        self.assertEqual(C.ACTION_SCALE, 1.0)
        self.assertEqual(C.KP_FW, 15.0)
        self.assertEqual(C.DELAY_STEPS_OFFICIAL, (3, 6))


class TestEnvironment(unittest.TestCase):
    def test_identified_assets_present(self):
        self.assertTrue(ROBOT_XML.is_file(), f"missing {ROBOT_XML}")

    def test_model_compiles_with_contract(self):
        env = DeployEnv("toss_catch", seed=1)
        self.assertEqual(env.model.nu, 15)      # 14 policy joints + mouth
        self.assertEqual(env.model.opt.timestep, C.PHYSICS_DT)
        obs = env.reset()
        self.assertEqual(obs.shape, (61,))
        self.assertTrue(np.isfinite(obs).all())

    def test_step_is_stable_and_finite(self):
        env = DeployEnv("duck_lift", seed=2)
        obs = env.reset()
        for _ in range(60):
            obs, reward, done, info = env.step(np.zeros(14))
            self.assertTrue(np.isfinite(obs).all())
            self.assertTrue(np.isfinite(reward))
            if done:
                obs = env.reset()

    def test_mouth_outside_policy(self):
        env = DeployEnv("duck_lift", seed=3)
        env.reset()
        # even a saturating action must not move the mouth through the policy
        before = env.data.qpos[env.L["mouth_qpos"]]
        for _ in range(20):
            env.step(np.ones(14))
        after = env.data.qpos[env.L["mouth_qpos"]]
        self.assertLessEqual(abs(after - before), C.MOUTH_OPEN)


if __name__ == "__main__":
    unittest.main()
