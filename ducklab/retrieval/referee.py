"""Independent world-state checks, never used for driving the simulated robot."""

import math
import mujoco
import numpy as np

from .scene import BASE_Z, BIN_HALF


def heading(data):
    forward = data.body("trunk_base").xmat.reshape(3, 3)[:, 0]
    return math.atan2(forward[1], forward[0])


def captured_pose(controller):
    data = controller.d
    joints = data.qpos[controller.q].copy()
    if mujoco.mj_name2id(controller.m, mujoco.mjtObj.mjOBJ_JOINT, "bill_open") >= 0:
        joints = np.append(joints, data.joint("bill_open").qpos[0])
    return (
        data.body("trunk_base").xpos.copy(),
        heading(data),
        joints,
        data.body("trunk_base").xmat.reshape(3, 3)[:, 2].copy(),
    )


def home_errors(controller, start):
    xyz, yaw, q, up = captured_pose(controller)
    src, src_yaw, src_q, src_up = start
    return dict(
        home_cm=float(100 * np.linalg.norm(xyz[:2] - src[:2])),
        yaw_deg=float(abs(math.degrees(math.remainder(yaw - src_yaw, math.tau)))),
        joint_rms_rad=float(np.sqrt(np.mean((q - src_q) ** 2))),
        max_joint_rad=float(np.max(abs(q - src_q))),
        height_cm=float(abs(xyz[2] - src[2]) * 100),
        tilt_deg=float(np.degrees(np.arccos(np.clip(up @ src_up, -1, 1)))),
    )


def at_home(errors):
    return (
        errors["home_cm"] <= 3.5
        and errors["yaw_deg"] <= 7.0
        and errors["joint_rms_rad"] <= 0.10
        and errors["max_joint_rad"] <= 0.22
        and errors["height_cm"] <= 1.2
        and errors["tilt_deg"] <= 6.0
    )


def in_tray(model, data, half, bin_xy):
    parcel = data.body("parcel")
    extent = np.abs(parcel.xmat.reshape(3, 3)) @ half
    contained = np.all(np.abs(parcel.xpos[:2] - bin_xy) + extent[:2] <= BIN_HALF)
    tray_floor = model.geom("tray_base").id
    parcel_id = model.body("parcel").id
    supported = any(
        (c.geom1 == tray_floor and model.geom_bodyid[c.geom2] == parcel_id)
        or (c.geom2 == tray_floor and model.geom_bodyid[c.geom1] == parcel_id)
        for c in data.contact
    )
    free = data.joint("parcel_free").qvel
    return bool(
        contained
        and supported
        and parcel.xpos[2] > BASE_Z
        and np.linalg.norm(free[:3]) < 0.018
        and np.linalg.norm(free[3:]) < 1.2
        and not np.any(data.eq_active)
    )
