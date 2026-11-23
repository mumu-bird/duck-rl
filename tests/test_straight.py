import math
import unittest

from ducklab.straight import LAB_SLIP_CURVE, Pose2D, StraightAssist


class StraightAssistTests(unittest.TestCase):
    def setUp(self):
        self.ctrl = StraightAssist()
        self.forward = (.12, 0., 0.)
        self.start = Pose2D(2., -1., math.pi/2)

    def test_start_frame_and_manual_release(self):
        self.assertEqual(self.ctrl.command(self.forward, self.start, .02), self.forward)
        # Facing +Y, a displacement towards -X is left of the starting line.
        displaced = Pose2D(1.8, 0., math.pi/2 + .1)
        vx, vy, wz = self.ctrl.command(self.forward, displaced, .02)
        self.assertEqual(vx, .12)
        self.assertLess(vy, 0.)
        self.assertLess(wz, 0.)
        for manual in ((.12, 0., .08), (.12, .04, 0.), (0., 0., 0.)):
            self.assertEqual(self.ctrl.command(manual, displaced, .02), manual)
            self.assertIsNone(self.ctrl.origin)
            self.assertEqual(self.ctrl.command(self.forward, displaced, .02), self.forward)

    def test_stale_feedback_releases_and_heading_wraps(self):
        ctrl = StraightAssist(heading_only=True)
        ctrl.command(self.forward, Pose2D(0., 0., math.pi-.01), .02)
        self.assertAlmostEqual(ctrl.command(self.forward, Pose2D(0., 0., -math.pi+.01), .02)[2], -.04)
        self.assertEqual(ctrl.command(self.forward, None, .02), self.forward)
        self.assertIsNone(ctrl.origin)
        self.assertEqual(ctrl.command((math.nan,0.,0.), self.start, .02), (0.,0.,0.))

    def test_speed_change_does_not_reprice_travel(self):
        ctrl = StraightAssist(slip_curve=LAB_SLIP_CURVE)
        ctrl.command((.08, 0., 0.), Pose2D(0., 0., 0.), .02)
        ctrl.command((.08, 0., 0.), Pose2D(1., 0., 0.), .02)
        ctrl.command((.18, 0., 0.), Pose2D(1., 0., 0.), .02)
        self.assertAlmostEqual(ctrl.slip, .036)


if __name__ == '__main__':
    unittest.main()
