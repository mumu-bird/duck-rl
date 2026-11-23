"""Local frame anchored on waiting pose. Only IMU and encoders enter the estimator."""

import math
import mujoco
import numpy as np

from ..bench import plane_xy
from ..foot_odometry import FootOdometry, _rotation
from .scene import HEIGHT, WIDTH


def rotate(yaw, xy):
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array((c * xy[0] - s * xy[1], s * xy[0] + c * xy[1]))


class Navigation:
    def __init__(self, controller):
        self.c = controller
        self.odom = FootOdometry(controller.m, controller.q)
        self.fk = mujoco.MjData(controller.m)
        self.filtered = None
        self.velocity = np.zeros(2)
        self.previous_filtered = None
        self.heading_origin = None

    def imu(self):
        data = self.c.d
        self.odom.observe_imu(
            data.sensor("orientation").data,
            data.sensor("imu_ang_vel").data,
            self.c.m.opt.timestep,
        )

    def update(self):
        # Use the AHRS relative heading when upright. Pure rate integration loses
        # accuracy through the near-vertical ground-pick posture.
        if self.odom.up[2] > 0.90:
            attitude = (
                _rotation(self.c.d.sensor("orientation").data) @ self.odom.mount.T
            )
            heading = math.atan2(attitude[1, 0], attitude[0, 0])
            if self.heading_origin is None:
                self.heading_origin = heading - self.odom.yaw
            delta = math.remainder(
                heading - self.heading_origin - self.odom.yaw, math.tau
            )
            self.odom.yaw += 0.045 * delta
        self.pose = self.odom.observe_encoders(self.c.d.qpos[self.c.q])
        xy = np.array([self.pose.x, self.pose.y])
        self.filtered = (
            xy if self.filtered is None else 0.78 * self.filtered + 0.22 * xy
        )
        if self.previous_filtered is not None:
            dt = 4 * self.c.m.opt.timestep
            measured = (self.filtered - self.previous_filtered) / dt
            self.velocity = 0.72 * self.velocity + 0.28 * measured
        self.previous_filtered = self.filtered.copy()

    def camera_pose(self):
        # Reconstruct the head-camera extrinsics from encoders, gravity, yaw and
        # the lowest support vertex. No simulated base or object position is read.
        pose = self.pose
        up = self.odom.up
        cy, sy = math.cos(pose.yaw), math.sin(pose.yaw)
        pitch = math.atan2(-up[0], math.hypot(up[1], up[2]))
        roll = math.atan2(up[1], up[2])
        cp, sp, cr, sr = (
            math.cos(pitch),
            math.sin(pitch),
            math.cos(roll),
            math.sin(roll),
        )
        mat = np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
        )
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, mat.ravel())
        self.fk.qpos[:7] = (pose.x, pose.y, 0.0, *quat)
        self.fk.qpos[self.c.q] = self.c.d.qpos[self.c.q]
        mujoco.mj_kinematics(self.c.m, self.fk)
        min_z = min(
            float(
                (verts @ self.fk.geom_xmat[g].reshape(3, 3)[2]).min()
                + self.fk.geom_xpos[g, 2]
            )
            for g, verts in zip(self.odom.geoms, self.odom.solids)
        )
        self.fk.qpos[2] = -min_z
        mujoco.mj_kinematics(self.c.m, self.fk)
        mujoco.mj_comPos(self.c.m, self.fk)
        mujoco.mj_camlight(self.c.m, self.fk)
        return self.fk.cam("head_rgb").xpos.copy(), self.fk.cam(
            "head_rgb"
        ).xmat.reshape(3, 3).copy()

    def floor_point(self, pixel):
        self.camera_pose()
        return plane_xy(self.c.m, self.fk, pixel, (HEIGHT, WIDTH), 0.001)

    def drive(self, goal, heading, tolerance):
        error = rotate(-self.pose.yaw, goal - self.filtered)
        yaw = math.remainder(heading - self.pose.yaw, math.tau)
        arrived = np.linalg.norm(error) < tolerance and abs(yaw) < 0.05
        cmd = np.zeros(13, np.float32)
        # The walking policy hardly moves on commands below roughly 0.02 m/s.
        speed = np.where(
            abs(error) > 0.004,
            np.sign(error) * np.maximum(abs(error) * 2.4, 0.047),
            0.0,
        )
        cmd[:3] = (
            *np.clip(speed, [-0.12, -0.10], [0.15, 0.10]),
            np.clip(2 * yaw, -0.65, 0.65),
        )
        return cmd, arrived


class FloorWarp:
    """Reproject a clean ground reference into the new camera pose using pinhole rays."""

    def __init__(self, model, reference, pose):
        self.reference, self.pose = reference, pose
        self.focal = (
            0.5 * HEIGHT / np.tan(np.deg2rad(model.cam("head_rgb").fovy[0]) / 2)
        )
        yy, xx = np.indices((HEIGHT, WIDTH))
        self.rays = np.stack(
            (
                (xx + 0.5 - WIDTH / 2) / self.focal,
                -(yy + 0.5 - HEIGHT / 2) / self.focal,
                -np.ones_like(xx),
            ),
            axis=-1,
        )

    def near(self, live, pose, goal, radius=0.060):
        current, rot = pose
        ray = self.rays @ rot.T
        z = ray[:, :, 2]
        scale = -current[2] / np.minimum(z, -1.0e-5)
        points = current + ray * scale[:, :, None]
        source = (points - self.pose[0]) @ self.pose[1]
        depth = np.maximum(-source[:, :, 2], 0.001)
        u = WIDTH / 2 - 0.5 + self.focal * source[:, :, 0] / depth
        v = HEIGHT / 2 - 0.5 - self.focal * source[:, :, 1] / depth
        valid = (
            (z < 0)
            & (source[:, :, 2] < 0)
            & (u >= 0)
            & (v >= 0)
            & (u < WIDTH - 1)
            & (v < HEIGHT - 1)
        )
        roi = valid & (np.linalg.norm(points[:, :, :2] - goal, axis=2) < radius)
        u = np.clip(u, 0, WIDTH - 1.001)
        v = np.clip(v, 0, HEIGHT - 1.001)
        x = u.astype(int)
        y = v.astype(int)
        fx = (u - x)[:, :, None]
        fy = (v - y)[:, :, None]
        ref = self.reference
        image = (1 - fy) * ((1 - fx) * ref[y, x] + fx * ref[y, x + 1]) + fy * (
            (1 - fx) * ref[y + 1, x] + fx * ref[y + 1, x + 1]
        )
        return np.where(valid[:, :, None], image, live).astype(np.uint8), roi


def tray_anchor(rgb, pose, focal, expected):
    """Estimate centre and orientation of the known teal rim from head RGB."""
    r, g, b = rgb.astype(float).transpose(2, 0, 1)
    yy, xx = np.where((g > 1.45 * r) & (g > 0.98 * b) & (b > 1.35 * r) & (g > 55))
    if len(xx) < 40:
        return None
    origin, rotation = pose
    rays = (
        np.column_stack(
            (
                (xx + 0.5 - WIDTH / 2) / focal,
                -(yy + 0.5 - HEIGHT / 2) / focal,
                -np.ones(len(xx)),
            )
        )
        @ rotation.T
    )
    downward = rays[:, 2] < -0.12
    rays = rays[downward]
    pts = origin[:2] + rays[:, :2] * ((0.014 - origin[2]) / rays[:, 2:3])
    pts = pts[np.linalg.norm(pts - expected, axis=1) < 0.165]
    if len(pts) < 40:
        return None
    # Restrict the fit to near-forward views; the square tray has a 90° ambiguity.
    grid = np.deg2rad(np.arange(-26, 27, 2))
    best = None
    for theta in grid:
        axes = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        rotated = pts @ axes
        lo, hi = np.quantile(rotated, (0.008, 0.992), axis=0)
        size = hi - lo
        area = float(np.prod(size))
        if best is None or area < best[0]:
            best = (area, (lo + hi) @ axes.T / 2, theta, size)
    if np.any(best[3] < 0.134) or np.any(best[3] > 0.210):
        return None
    return best[1], best[2]
