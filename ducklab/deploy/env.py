"""Deployable task environments on the official identified robot model.

Two tasks, both expressed strictly through the on-robot policy contract
(obs[61] -> actions[14], 50 Hz over 200 Hz physics):

``duck_lift``   — a LINLEE duck proxy sits on the ground in front; the duck must
                  walk up, put the bill over it, close the beak (separate rule
                  controller, bus servo 34), lift and hold.

``toss_catch``  — a ball or cube is tossed from 0.9..1.6 m away toward a point
                  just in front of the duck; the duck must walk forward to meet
                  it and catch it in the beak before it touches the ground,
                  then hold.

Perception is *oracle with tracker noise and latency*, exactly mirroring how
the real robot works: an external tracker (the factory colour-tracking app, or
a custom vision node) converts the object pose into the 13-D command block
(twist + head targets); the policy itself never sees the object. The beak is
never policy-driven — it is a rule controller that maps one-to-one onto bus
servo 34 on the robot.
"""

from __future__ import annotations

import mujoco
import numpy as np

from . import contract as C
from . import scene as S
from .servo import FittedServo

GRAVITY = np.array([0.0, 0.0, -9.81])


def _quat_to_mat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class Tracker:
    """External object tracker: oracle pose + gaussian noise + fixed latency."""

    def __init__(self, latency_ticks=2, pos_noise=0.015, vel_noise=0.06, seed=None):
        self.rng = np.random.default_rng(seed)
        self.latency = latency_ticks
        self.pos_noise = pos_noise
        self.vel_noise = vel_noise
        self.buffer = []

    def reset(self, pos, vel):
        self.buffer = [(pos.copy(), vel.copy()) for _ in range(self.latency + 1)]

    def push(self, pos, vel):
        noisy_p = pos + self.rng.normal(0, self.pos_noise, 3)
        noisy_v = vel + self.rng.normal(0, self.vel_noise, 3)
        self.buffer.append((noisy_p, noisy_v))
        self.buffer.pop(0)

    def read(self):
        return self.buffer[0]


class BeakController:
    """Rule controller for bus servo 34 (the mouth), outside the policy.

    The same rules run on the robot: open while approaching, close when the
    bill is over the target and the intercept window opens, hold while carrying.
    """

    OPEN, CLOSED = C.MOUTH_OPEN, C.MOUTH_CLOSED

    def __init__(self, task):
        self.task = task
        self.state = "open"
        self.hold_time = 0.0
        self.lost_time = 0.0

    def reset(self):
        self.state = "open"
        self.hold_time = 0.0
        self.lost_time = 0.0

    def update(self, dt, ctx):
        """ctx: dict with tip/obj poses, contact flags and task-specific data."""
        if self.state == "closed":
            if ctx["contact"]:
                self.hold_time += dt
                self.lost_time = 0.0
                return self.CLOSED
            self.lost_time += dt
            if self.lost_time > 0.35:
                self.state = "open"
                self.hold_time = 0.0
                return self.OPEN
            return self.CLOSED
        if self.task == "duck_lift":
            horiz = ctx["tip_to_obj_horiz"]
            above = ctx["tip_above_obj"]
            if horiz < 0.055 and above > 0.012 and ctx["descending_or_stable"]:
                self.state = "closed"
                self.hold_time = 0.0
                return self.CLOSED
        else:  # toss_catch
            if ctx["intercept_now"]:
                self.state = "closed"
                self.hold_time = 0.0
                return self.CLOSED
        return self.OPEN


class DeployEnv:
    """One task instance. Step is one 20 ms policy tick (4 x 5 ms physics)."""

    TASKS = {
        "duck_lift": dict(object="linlee_duck", seconds=12.0),
        "toss_catch": dict(object="ball", seconds=5.0),
    }

    def __init__(self, task="duck_lift", seed=None, object_kind=None, record=False,
                 shared=None):
        assert task in self.TASKS
        self.task = task
        cfg = self.TASKS[task]
        self.object_kind = object_kind or cfg["object"]
        self.seconds = cfg["seconds"]
        self.max_ticks = int(self.seconds * C.CONTROL_HZ)
        self.rng = np.random.default_rng(seed)
        if shared is None:
            self.model, _, self.L = S.build(object_kind=self.object_kind)
        else:  # one immutable mjModel shared by every worker (mjData is per-env)
            self.model, self.L = shared
        self.data = mujoco.MjData(self.model)
        self.servo = FittedServo(self.model.nu, seed=seed)
        self.dof_all = np.append(self.L["dof_joints"], self.L["mouth_dof"]) \
            if self.L["mouth_dof"] >= 0 else self.L["dof_joints"]
        self.servo.setup_friction(self.model, self.dof_all)  # one-time, constant
        self.tracker = Tracker(seed=None if seed is None else seed + 1)
        self.beak = BeakController(task)
        self.record = record
        self.trace = None
        self.ticks = 0
        self.last_action = np.zeros(C.ACTION_LEN)
        self.obj_rest_z = 0.0
        self._lift_start = None
        self._hold_ticks = 0
        self._caught = False
        self._ground_touch_after_toss = False
        self._episode_seed = seed

    # ------------------------------------------------------------- helpers

    def _object_pose(self):
        q = self.data.qpos[self.L["object_qpos"]:self.L["object_qpos"] + 7]
        dq = self.data.qvel[self.L["object_dof"]:self.L["object_dof"] + 6]
        return q[:3], q[3:7], dq[:3], dq[3:6]

    def _trunk_state(self):
        q = self.data.qpos
        base_q = q[:7]
        R = _quat_to_mat(base_q[3:7])
        gyro = R.T @ self.data.qvel[3:6]          # trunk frame
        gravity = R.T @ np.array([0, 0, -1.0])    # projected gravity
        return gyro, gravity, base_q[:3], R

    def _full_state(self):
        """(q, dq) over the 14 policy joints plus the mouth when modelled."""
        d = self.data
        q = d.qpos[self.L["qpos_joints"]]
        dq = d.qvel[self.L["dof_joints"]]
        if self.L["mouth_qpos"] >= 0:
            q = np.append(q, d.qpos[self.L["mouth_qpos"]])
            dq = np.append(dq, d.qvel[self.L["mouth_dof"]])
        return q, dq

    def _tip_pose(self):
        self.data.site_xpos[self.L["mouth_tip"]]  # refreshed by mj_forward
        return self.data.site_xpos[self.L["mouth_tip"]].copy()

    def _contacts(self):
        obj, ground = self.L["object_geom"], self.L["ground_geom"]
        pu, pl = self.L["pad_upper"], self.L["pad_lower"]
        beak_contact = False
        ground_contact = False
        for i in range(self.data.ncon):
            g1, g2 = self.data.contact[i].geom1, self.data.contact[i].geom2
            pair = {g1, g2}
            if obj in pair and (pu in pair or pl in pair):
                beak_contact = True
            if obj in pair and ground in pair:
                ground_contact = True
        return beak_contact, ground_contact

    def _head_command(self, obj_pos):
        tip = self._tip_pose()
        head = self.data.xpos[self.L["head"]]
        d = obj_pos - head
        horiz = float(np.hypot(d[0], d[1]))
        dyaw = float(np.arctan2(d[1], max(d[0], 1e-6)))
        drop = float(np.arctan2(max(head[2] - obj_pos[2], -0.5), max(horiz, 1e-6)))
        neck = C.DEFAULT_POSITION[5] - 0.6 * drop
        pitch = C.DEFAULT_POSITION[6] - 0.4 * drop
        yaw = C.DEFAULT_POSITION[7] + dyaw
        roll = C.DEFAULT_POSITION[8]
        return np.clip([neck, pitch, yaw, roll],
                       C.JOINT_LIMITS[[5, 6, 7, 8], 0], C.JOINT_LIMITS[[5, 6, 7, 8], 1])

    def _twist_command(self, obj_pos, obj_vel):
        if self.task == "duck_lift":
            # stand off point: 0.24 m in front of the object along +x
            goal = np.array([obj_pos[0] - 0.24, obj_pos[1], 0.0])
        else:
            # pursue the predicted intercept point ahead of the ball
            goal = obj_pos + obj_vel * 0.35
        base = self.data.qpos[:3]
        d = goal - base
        dist = float(np.hypot(d[0], d[1]))
        heading = float(np.arctan2(d[1], max(d[0], 1e-6)))
        vx = float(np.clip(1.6 * dist, 0.0, C.MAX_LINEAR))
        vyaw = float(np.clip(1.8 * heading, -C.MAX_ANGULAR, C.MAX_ANGULAR))
        return np.array([vx, 0.0, vyaw])

    # ------------------------------------------------------------- reset

    def reset(self, seed=None):
        if seed is not None:
            self._episode_seed = seed
            self.rng = np.random.default_rng(seed)
        mujoco = __import__("mujoco")
        mujoco.mj_resetData(self.model, self.data)
        m, d = self.model, self.data
        mujoco.mj_forward(m, d)
        q0 = d.qpos.copy()
        q0[:7] = [0, 0, C.BASE_HOME_HEIGHT, 1, 0, 0, 0]
        q0[self.L["qpos_joints"]] = C.DEFAULT_POSITION[
            [i for i in range(C.NUM_JOINTS) if i != C.MOUTH_INDEX]
        ]
        if self.L["mouth_qpos"] >= 0:
            q0[self.L["mouth_qpos"]] = C.MOUTH_CLOSED
        d.qpos[:] = q0
        d.ctrl[:] = 0.0
        mujoco.mj_forward(m, d)

        # object placement / toss
        if self.task == "duck_lift":
            x = self.rng.uniform(0.30, 0.55)
            y = self.rng.uniform(-0.10, 0.10)
            z = 0.024 if self.object_kind == "linlee_duck" else 0.015
            yaw = self.rng.uniform(-np.pi, np.pi)
            pos = np.array([x, y, z])
            quat = np.array([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
            vel = np.zeros(3)
            ang = np.zeros(3)
        else:
            launch = np.array([
                self.rng.uniform(0.9, 1.6),
                self.rng.uniform(-0.25, 0.25),
                self.rng.uniform(0.5, 0.9),
            ])
            aim = np.array([
                self.rng.uniform(0.25, 0.55),
                self.rng.uniform(-0.08, 0.08),
                self.rng.uniform(0.15, 0.32),
            ])
            flight = self.rng.uniform(0.8, 1.3)
            vel = (aim - launch - 0.5 * GRAVITY * flight ** 2) / flight
            pos = launch
            quat = np.array([1.0, 0, 0, 0])
            ang = np.zeros(3)
        oq = self.L["object_qpos"]
        d.qpos[oq:oq + 3] = pos
        d.qpos[oq + 3:oq + 7] = quat
        od = self.L["object_dof"]
        d.qvel[od:od + 3] = vel
        d.qvel[od + 3:od + 6] = ang
        self.obj_rest_z = pos[2]
        self._toss_time = 0.0

        # settle at home with the servo seeded on the home pose
        self.servo.reset(q0[self.L["qpos_joints"]].copy() if self.L["mouth_qpos"] < 0
                         else np.append(q0[self.L["qpos_joints"]], q0[self.L["mouth_qpos"]]))
        targets = C.targets_from_action(np.zeros(C.ACTION_LEN))
        if self.L["mouth_qpos"] >= 0:
            targets[C.MOUTH_INDEX] = C.MOUTH_CLOSED
        for _ in range(30):
            full_q, full_dq = self._full_state()
            d.ctrl[:] = self.servo.command(targets, full_q, full_dq, C.PHYSICS_DT)
            mujoco.mj_step(m, d)
        self.tracker.reset(pos, vel)
        self.beak.reset()
        self.ticks = 0
        self.last_action = np.zeros(C.ACTION_LEN)
        self._hold_ticks = 0
        self._caught = False
        self._ground_touch_after_toss = False
        self.trace = [] if self.record else None
        return self.observe()

    # ------------------------------------------------------------- observe

    def observe(self):
        m, d = self.model, self.data
        mujoco = __import__("mujoco")
        gyro, gravity, _, _ = self._trunk_state()
        q = d.qpos[self.L["qpos_joints"]]
        dq = d.qvel[self.L["dof_joints"]]
        obj_pos, _, obj_vel, _ = self._object_pose()
        head_cmd = self._head_command(obj_pos)
        twist = self._twist_command(obj_pos, obj_vel)
        command = C.build_command(twist, head_cmd, body=(0.0, 0.0, 0.0))
        return C.build_observation(gyro, gravity, q, dq, self.last_action, command)

    # ------------------------------------------------------------- step

    def step(self, action):
        m, d = self.model, self.data
        mujoco = __import__("mujoco")
        action = np.asarray(action, dtype=np.float64).reshape(C.ACTION_LEN)
        targets = C.clip_targets(C.targets_from_action(action))
        if self.L["mouth_qpos"] < 0:
            targets = targets[:C.MOUTH_INDEX]

        reward = 0.0
        info = {"success": False, "fall": False, "timeout": False}
        dt_tick = C.SUBSTEPS * C.PHYSICS_DT

        for sub in range(C.SUBSTEPS):
            tip = self._tip_pose()
            obj_pos, _, obj_vel, _ = self._object_pose()
            beak_contact, ground_contact = self._contacts()
            tip_to_obj = obj_pos - tip
            horiz = float(np.hypot(tip_to_obj[0], tip_to_obj[1]))
            prev_tip_z = tip[2]
            tip_v = (tip[2] - getattr(self, "_prev_tip_z", tip[2])) / C.PHYSICS_DT
            self._prev_tip_z = prev_tip_z
            ctx = dict(
                contact=beak_contact,
                tip_to_obj_horiz=horiz,
                tip_above_obj=tip[2] - obj_pos[2],
                descending_or_stable=tip_v < 0.05,
                intercept_now=self._intercept_now(tip, obj_pos, obj_vel, ground_contact),
            )
            if self.L["mouth_qpos"] >= 0:
                mouth = self.beak.update(C.PHYSICS_DT, ctx)
                targets = np.concatenate(
                    [targets[:C.MOUTH_INDEX], [mouth], targets[C.MOUTH_INDEX + 1:]]
                )
            ctrl_targets = targets
            full_q, full_dq = self._full_state()
            d.ctrl[:] = self.servo.command(ctrl_targets, full_q, full_dq, C.PHYSICS_DT)
            mujoco.mj_step(m, d)

        self.last_action = action
        self.ticks += 1
        self._toss_time += dt_tick

        # ---- world-state judging (blind to policy actions)
        tip = self._tip_pose()
        obj_pos, _, obj_vel, _ = self._object_pose()
        beak_contact, ground_contact = self._contacts()
        gyro, gravity, base_pos, R = self._trunk_state()
        tilt = float(np.arccos(np.clip(-gravity[2], -1, 1)))
        fallen = (base_pos[2] < 0.075) or (tilt > 1.1) or not np.isfinite(d.qpos).all()

        # tracker feeds the NEXT observation (latency modelled here)
        self.tracker.push(obj_pos, obj_vel)

        if self.task == "duck_lift":
            lifted = obj_pos[2] - self.obj_rest_z
            if beak_contact and lifted > 0.012:
                self._hold_ticks += 1
            else:
                self._hold_ticks = 0
            success = self._hold_ticks >= int(1.0 * C.CONTROL_HZ) and not fallen
            # reward
            horiz = float(np.hypot(obj_pos[0] - tip[0], obj_pos[1] - tip[1]))
            reward += -1.2 * min(horiz, 0.5)
            reward += 2.0 * min(max(tip[2] - obj_pos[2], 0.0), 0.10)
            if beak_contact:
                reward += 1.0
            if self._hold_ticks > 0:
                reward += 3.0 * min(lifted, 0.05) + 0.5
        else:
            if ground_contact and not self._caught:
                self._ground_touch_after_toss = True
            if beak_contact and not self._ground_touch_after_toss:
                self._caught = True
            if self._caught and beak_contact:
                self._hold_ticks += 1
            else:
                self._hold_ticks = max(0, self._hold_ticks - 2)
            success = self._caught and self._hold_ticks >= int(1.0 * C.CONTROL_HZ) and not fallen
            # reward: forward progress toward the predicted intercept + tracking
            intercept = obj_pos + obj_vel * 0.35
            progress = float(np.hypot(intercept[0] - base_pos[0], intercept[1] - base_pos[1]))
            reward += 2.0 * max(0.0, 0.6 - progress)
            if self._caught:
                reward += 8.0 + 0.5
            if self._ground_touch_after_toss:
                reward -= 5.0

        # common terms
        if not fallen:
            reward += 0.15
        reward -= 0.004 * float(np.square(action).sum())
        if fallen:
            reward -= 10.0
            info["fall"] = True

        done = fallen or success or self.ticks >= self.max_ticks
        if done and not success:
            info["timeout"] = self.ticks >= self.max_ticks
        info["success"] = success
        info["hold_ticks"] = self._hold_ticks
        info["caught"] = self._caught
        info["obj_z"] = float(obj_pos[2])
        info["tilt"] = tilt

        if self.record:
            self.trace.append(dict(
                t=self.ticks, obj=obj_pos.tolist(), tip=tip.tolist(),
                base=base_pos.tolist(), beak=bool(beak_contact),
                ground=bool(ground_contact), success=success,
            ))

        return self.observe(), reward, done, info

    def _intercept_now(self, tip, obj_pos, obj_vel, ground_contact):
        if self.task != "toss_catch" or ground_contact:
            return False
        rel = obj_pos - tip
        dist = float(np.linalg.norm(rel))
        if dist > 0.30:
            return False
        closing = float(rel @ obj_vel) < 0.0
        t_hit = dist / max(float(np.linalg.norm(obj_vel)), 1e-3)
        return closing and t_hit < 0.14
