"""Tests for the small-duck domain and staged retrieval workflow."""

from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class DuckDomainTest(unittest.TestCase):
    def test_variants_sample_inside_declared_ranges(self):
        from ducklab.retrieval.duck import load_duck_variants, sample_duck

        rng = np.random.default_rng(7)
        for template in load_duck_variants():
            sample = sample_duck(template, rng)
            self.assertEqual(sample["family"], "duck")
            self.assertEqual(len(sample["half_m"]), 3)
            self.assertGreater(sample["mass_kg"], 0.0)
            self.assertGreaterEqual(sample["sample_scale"], template["scale"][0])
            self.assertLessEqual(sample["sample_scale"], template["scale"][1])
            self.assertIn("contact_friction", sample)

    def test_profile_overlay_preserves_delivery(self):
        from ducklab.retrieval.duck import duck_profile_candidates, profiles_for

        profile = duck_profile_candidates()[0]
        profiles = profiles_for(profile)
        self.assertIs(profiles["general"], profile)
        self.assertEqual(profiles["delivery"].release_tick, 28)
        self.assertGreater(profiles["retry"].retry_radius, 0.0)


try:
    import mujoco
    from ducklab.retrieval.duck import load_duck_variants, sample_duck
    from ducklab.retrieval.scene import make_scene
except ImportError:
    mujoco = None


@unittest.skipUnless(
    mujoco is not None
    and (ROOT / "vendor/wonderduck/simulator/robot/wonderduck/robot.xml").exists(),
    "requires lab simulator",
)
class DuckSceneTest(unittest.TestCase):
    def test_parametric_duck_has_visual_features_and_soft_contact(self):
        template = load_duck_variants()[0]
        item = sample_duck(template, np.random.default_rng(11))
        model, half = make_scene(item, moving_jaw=True)
        self.assertTrue(np.all(half > 0))
        self.assertGreaterEqual(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bird_head"), 0
        )
        self.assertGreaterEqual(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "eye_left"), 0
        )
        hull = model.geom("parcel_hull")
        self.assertLess(float(hull.solref[0]), 0.06)
        self.assertGreater(float(hull.friction[0]), 0.5)


if __name__ == "__main__":
    unittest.main()
