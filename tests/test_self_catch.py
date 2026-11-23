"""The new self-catch referee must require a real, airborne release and return."""

import unittest

import numpy as np

from ducklab.self_catch import (
    BOTH,
    BASE,
    BASEZ,
    FLOOR,
    LIP,
    MZ,
    MX,
    MY,
    T,
    UP,
    X,
    Z,
    judge,
)


def flight(ground=False):
    before = np.zeros((20, 14), dtype=float)
    before[:, T] = np.linspace(-0.10, -0.005, len(before))
    before[:, BOTH] = 1
    before[:, LIP] = 1
    before[:, UP] = 1
    before[:, BASEZ] = 0.11
    after = np.zeros((140, 14), dtype=float)
    t = np.arange(len(after)) * 0.005
    release, catch = 12, 38
    u = np.clip(t - release * 0.005, 0, (catch - release) * 0.005)
    ball_x = 0.04 + 0.17 * u
    z = 0.16 + 0.60 * u - 4.905 * u**2
    after[:, T] = t
    after[:, X], after[:, Z] = ball_x, z
    after[:, BASE] = 0.09 * u
    after[:, MX] = ball_x
    after[:, MY] = 0
    after[:, MZ] = z
    after[:, UP] = 1
    after[:, BASEZ] = 0.11
    after[:release, LIP] = 1
    after[:release, BOTH] = 1
    after[catch:, LIP] = 1
    after[catch:, BOTH] = 1
    if ground:
        after[25, FLOOR] = 1
    return np.vstack((before, after))


class RefereeTests(unittest.TestCase):
    def test_counts_airborne_chase_and_sustained_catch(self):
        result = judge(flight())
        self.assertTrue(result["success"], result)
        self.assertTrue(result["released"])
        self.assertTrue(result["recaught"])
        self.assertTrue(result["held"])

    def test_rejects_ball_that_touched_floor_before_returning(self):
        result = judge(flight(ground=True))
        self.assertFalse(result["success"], result)
        self.assertFalse(result["recaught"])

    def test_closed_mouth_and_stationary_base_do_not_pass(self):
        tr = flight()
        tr[:, BASE] = 0
        self.assertFalse(judge(tr)["success"])
        tr[:, LIP] = tr[:, BOTH] = 1
        self.assertFalse(judge(tr)["released"])

    def test_one_frame_contact_is_not_a_hold(self):
        tr = flight()
        tr[-49:, BOTH] = 0
        self.assertFalse(judge(tr)["success"])


if __name__ == "__main__":
    unittest.main()
