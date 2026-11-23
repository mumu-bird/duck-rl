"""Guard against identity leakage and false passes for a moving, attached or overhanging object."""

import json
import math
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SplitTest(unittest.TestCase):
    def test_no_identity_reused(self):
        items = json.loads((ROOT / "configs/retrieval-objects.json").read_text())[
            "objects"
        ]
        ids = [item["id"] for item in items]
        self.assertEqual(len(ids), len(set(ids)))
        held = {i["family"] for i in items if i["split"] == "test"}
        self.assertEqual(held, {"pen", "ball", "duck"})
        sources = {i["id"] for i in items if i.get("source")}
        self.assertEqual(
            sources,
            set(json.loads((ROOT / "configs/retrieval-sha256.json").read_text())),
        )


try:
    import mujoco
    from ducklab.retrieval.episode import BeakFixture, settled
    from ducklab.retrieval.referee import at_home, in_tray
    from ducklab.retrieval.scene import (
        BIN,
        BIN_HALF,
        BASE_Z,
        OBJECTS,
        make_scene,
        place,
    )
except ImportError:
    mujoco = None


@unittest.skipUnless(
    mujoco is not None
    and (ROOT / "vendor/wonderduck/simulator/robot/wonderduck/robot.xml").exists(),
    "requires lab simulator",
)
class PhysicsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model, cls.half = make_scene(OBJECTS[0])

    def setUp(self):
        self.data = mujoco.MjData(self.model)

    def test_latch_cannot_reach_through_air(self):
        place(self.data, "parcel", (0.25, 0, 0.22))
        mujoco.mj_forward(self.model, self.data)
        grip = BeakFixture(self.model, self.data, True)
        self.assertFalse(grip.sense(close=True))
        self.assertFalse(self.data.eq_active.any())

    def test_bin_requires_rest_free_contact_and_full_extent(self):
        def reset(x, y, z):
            place(self.data, "parcel", (x, y, z))
            mujoco.mj_forward(self.model, self.data)

        x, y = BIN
        z = BASE_Z + self.half[2] - 0.00015
        reset(x, y, z)
        self.assertTrue(in_tray(self.model, self.data, self.half, BIN))
        self.data.joint("parcel_free").qvel[0] = 0.12
        self.assertFalse(in_tray(self.model, self.data, self.half, BIN))
        reset(x, y, z)
        self.data.eq_active[self.model.equality("hold_parcel").id] = True
        self.assertFalse(in_tray(self.model, self.data, self.half, BIN))
        self.data.eq_active[:] = False
        reset(x, y, 0.12)
        self.assertFalse(in_tray(self.model, self.data, self.half, BIN))
        reset(x + BIN_HALF[0] - 0.006, y, z)
        self.assertFalse(in_tray(self.model, self.data, self.half, BIN))

    def test_return_requires_matching_pose(self):
        e = dict(
            home_cm=1.0,
            yaw_deg=5.0,
            joint_rms_rad=0.07,
            max_joint_rad=0.14,
            height_cm=0.2,
            tilt_deg=1.0,
        )
        self.assertTrue(at_home(e))
        for key, bad in [
            ("home_cm", 4.0),
            ("yaw_deg", 8.0),
            ("max_joint_rad", 0.3),
            ("tilt_deg", 10.0),
        ]:
            self.assertFalse(at_home({**e, key: bad}))

    def test_motion_does_not_trigger_a_pick(self):
        from collections import deque

        history = deque([(0.0, 0.0)] * 7, maxlen=7)
        self.assertTrue(settled(history))
        history.append((0.015, 0.0))
        self.assertFalse(settled(history))
        history.clear()
        self.assertFalse(settled(history))


class GraspProfileTest(unittest.TestCase):
    def test_legacy_phase_and_opt_in_smoothing(self):
        from ducklab.retrieval.grasp import GraspProfile

        legacy = GraspProfile(rise=82)
        smooth = GraspProfile(rise=82, ramp=45)
        previous = None
        for tick in range(198):
            phase = (tick if tick < 82 else min(198, tick + 25)) * math.pi / 100
            self.assertEqual(
                legacy.command(tick)[0], (math.cos(phase), math.sin(phase))
            )
            direction, opening = smooth.command(tick)
            self.assertEqual(opening, legacy.command(tick)[1])
            if previous is not None:
                self.assertLess(math.dist(previous, direction), 0.060)
            previous = direction

    def test_frozen_retry_and_release_timing(self):
        from ducklab.retrieval.grasp import DeliveryProfile, load_profiles

        profiles = load_profiles(ROOT / "configs/retrieval-grasps-v6.json")
        self.assertEqual(profiles["general"].stroke(60), (0.0, 0.0))
        self.assertLess(profiles["retry"].stroke(60)[1], 0.0)
        self.assertEqual(profiles["retry"].stroke(150), (0.0, 0.0))
        for invalid in ({"open_ticks": 0}, {"release_tick": 78}):
            with self.assertRaises(ValueError):
                DeliveryProfile(**invalid)

    def test_angle_stays_within_servo_range(self):
        from ducklab.retrieval.grasp import load_profiles

        for profile in load_profiles().values():
            for tick in range(200):
                pair, angle = profile.command(tick)
                self.assertAlmostEqual(sum(v * v for v in pair), 1.0)
                self.assertTrue(0.0 <= angle <= math.radians(39.375))


if __name__ == "__main__":
    unittest.main()
