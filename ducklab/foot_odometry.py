"""Planar odometry from simulated IMU tilt, rate and joint encoders on WonderDuck."""
import math

import mujoco
import numpy as np

from .straight import Pose2D


def _rotation(quat):
    result = np.empty(9)
    mujoco.mju_quat2Mat(result, quat)
    return result.reshape(3, 3)


class FootOdometry:
    def __init__(self, model, joint_addresses):
        self.model = model
        self.joints = joint_addresses
        self.fk = mujoco.MjData(model)
        root = model.joint('root_freejoint').qposadr[0]
        self.fk.qpos[root:root+7] = (0, 0, 0, 1, 0, 0, 0)
        self.mount = (_rotation(model.body_quat[model.body('imu').id])
                      @ _rotation(model.site_quat[model.site('imu').id]))
        self.geoms = [model.geom(s + '_foot_collision').id for s in ('left', 'right')]
        self.solids = []
        for geom in self.geoms:
            mesh = model.geom_dataid[geom]
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            self.solids.append(model.mesh_vert[start:start+count].astype(np.float64))
        self.up = np.array([0., 0., 1.])
        self.yaw = 0.
        self.xy = np.zeros(2)
        self.previous_rate = self.previous_feet = None

    def observe_imu(self, site_quat, gyro, dt, yaw_rate_bias=0.):
        # Gravity in the trunk frame is invariant to the simulated AHRS's absolute yaw.
        self.up = (_rotation(site_quat) @ self.mount.T)[2]
        omega = self.mount @ gyro
        rate = (omega[1] * self.up[1] + omega[2] * self.up[2]) / max(.2, 1-self.up[0]**2)
        rate += yaw_rate_bias
        if self.previous_rate is not None:
            self.yaw += .5 * (rate + self.previous_rate) * dt
        self.previous_rate = rate

    def observe_encoders(self, angles):
        roll = math.atan2(self.up[1], self.up[2])
        pitch = math.atan2(-self.up[0], math.hypot(self.up[1], self.up[2]))
        sr, cr = math.sin(roll), math.cos(roll)
        sp, cp = math.sin(pitch), math.cos(pitch)
        sy, cy = math.sin(self.yaw), math.cos(self.yaw)
        attitude = np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                             [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
                             [-sp, cp*sr, cp*cr]])
        self.fk.qpos[self.joints] = angles
        mujoco.mj_kinematics(self.model, self.fk)
        positions = self.fk.geom_xpos[self.geoms] @ attitude.T
        rotations = attitude @ self.fk.geom_xmat[self.geoms].reshape(2, 3, 3)
        if self.previous_feet is not None:
            old_positions, old_rotations = self.previous_feet
            # Choose the lowest sole vertex halfway through this encoder interval.
            # A stationary support point moves backwards relative to an advancing trunk.
            candidates = []
            for foot, vertices in enumerate(self.solids):
                vertical = vertices @ (.5 * (rotations[foot, 2] + old_rotations[foot, 2]))
                index = vertical.argmin()
                height = vertical[index] + .5 * (positions[foot, 2] + old_positions[foot, 2])
                candidates.append((height, foot, index))
            _, foot, index = min(candidates)
            vertex = self.solids[foot][index]
            delta = (old_positions[foot] - positions[foot]
                     + (old_rotations[foot] - rotations[foot]) @ vertex)
            self.xy += delta[:2]
        self.previous_feet = positions, rotations
        return Pose2D(*self.xy, self.yaw)
