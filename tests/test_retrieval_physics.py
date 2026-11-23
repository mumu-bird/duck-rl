import unittest

try:
    import numpy as np
    from tools.train_retrieval_ppo import make_episode_scene
    from ducklab.retrieval.scene import OBJECTS, BIN
except ImportError:
    make_episode_scene = None


@unittest.skipIf(make_episode_scene is None, "requires training environment")
class RetrievalPhysicsTest(unittest.TestCase):
    def test_payload_mass_and_explicit_bill_friction_are_applied(self):
        item = next(x for x in OBJECTS if x["id"] == "felt_pen")
        base, _ = make_episode_scene(item, BIN)
        model, _ = make_episode_scene(item, BIN, 12.0, 0.37)
        self.assertAlmostEqual(float(model.body("parcel").mass[0]), .012, places=8)
        parcel = model.geom_bodyid == model.body("parcel").id
        pairs = parcel[model.pair_geom1] | parcel[model.pair_geom2]
        self.assertTrue(pairs.any(), "test must exercise explicit bill contact pairs")
        np.testing.assert_allclose(model.geom_friction[parcel, 0], .37)
        np.testing.assert_allclose(model.pair_friction[pairs, :2], .37)
        np.testing.assert_array_equal(model.pair_friction[~pairs], base.pair_friction[~pairs])
        np.testing.assert_array_equal(model.pair_friction[:, 2:], base.pair_friction[:, 2:])

    def test_invalid_physics_is_rejected_before_loading_a_model(self):
        for mass, friction in [(0, None), (-1, None), (float("nan"), None),
                               (4, 0), (4, float("inf"))]:
            with self.assertRaises(ValueError):
                make_episode_scene({}, BIN, mass, friction)


if __name__ == "__main__":
    unittest.main()
