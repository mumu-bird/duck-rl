"""Optional straight-line assistance for forward joystick commands (SI units)."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float


# Measured error per forward metre for the lab's default flat floor and foot odometry.
# Use an empty curve with independently localized odometry. Calibrate for a new setup.
LAB_SLIP_CURVE = ((.08, .036), (.12, .021), (.18, -.003))


def _slip_rate(curve, speed):
    if not curve:
        return 0.0
    for (lo, a), (hi, b) in zip(curve, curve[1:]):
        if speed <= hi:
            return a + (b - a) * max(0.0, (speed - lo) / (hi - lo))
    return curve[-1][1]


def _limit(value, cap):
    return max(-cap, min(value, cap))


class StraightAssist:
    """Latch a line on forward input; explicit lateral, turn and stop take priority."""

    def __init__(self, *, heading_only=False, slip_curve=()):
        self.heading_only = heading_only
        self.slip_curve = tuple(slip_curve)
        self.reset()

    def reset(self):
        self.origin = None
        self.filtered_side = self.integral = self.slip = self.previous_forward = 0.0

    def command(self, twist, pose, dt):
        """Return (forward m/s, left m/s, CCW rad/s). Supply None for stale odometry."""
        vx, vy, turn = twist
        if not all(map(math.isfinite, twist)):
            self.reset()
            return 0.0, 0.0, 0.0
        if not (.08 - 1e-7 <= vx <= .18 + 1e-7 and abs(vy) <= .008 and abs(turn) <= .025):
            self.reset()
            return vx, vy, turn
        if (pose is None or not all(map(math.isfinite, (pose.x, pose.y, pose.yaw)))
                or not 0 < dt <= .1):
            self.reset()
            return vx, vy, turn
        if self.origin is None:
            self.origin = pose
        yaw_error = math.remainder(pose.yaw - self.origin.yaw, 2 * math.pi)
        if self.heading_only:
            return vx, 0.0, _limit(-2 * yaw_error, .45)

        cos0, sin0 = math.cos(self.origin.yaw), math.sin(self.origin.yaw)
        dx, dy = pose.x - self.origin.x, pose.y - self.origin.y
        forward = dx * cos0 + dy * sin0
        self.slip += _slip_rate(self.slip_curve, vx) * (forward - self.previous_forward)
        self.previous_forward = forward
        side = -dx * sin0 + dy * cos0 + self.slip
        # Filter the side-to-side sway of individual footsteps before steering.
        alpha = 1 - .86 ** (dt / .02)
        self.filtered_side += alpha * (side - self.filtered_side)
        self.integral = _limit(self.integral + dt * self.filtered_side, .15)
        return (vx, _limit(-self.filtered_side - .25 * self.integral, .055),
                _limit(-2 * yaw_error - 1.4 * self.filtered_side, .45))
