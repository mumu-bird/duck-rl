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
from .servo import CommandConditioner

GRAVITY = np.array([0.0, 0.0, -9.81])
GAMMA = 0.99


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
            if horiz < 0.09 and 0.004 < above < 0.035 and ctx["descending_or_stable"]:
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
        "duck_lift": dict(object="linlee_duck", seconds=20.0),
        "toss_catch": dict(object="ball", seconds=5.0),
    }

    def __init__(self, task="duck_lift", seed=None, object_kind=None, record=False,
                 shared=None, curriculum=None, freeze_body=False, force_head=False):
        assert task in self.TASKS
        self.task = task
        self.curriculum = curriculum
        self._curriculum_progress = 0.0
        # freeze_body: 第一阶段解耦——身体目标钳制在 home 附近，只放开头部
        # 关节，让策略先学「低头对准」而不被平衡问题干扰。
        self.freeze_body = freeze_body
        # force_head: 头部关节强制为「瞄准物体」，策略只学腿部补偿。
        # 「边低边稳」是最难的耦合（头部链占体重 38%，低头必前移重心），
        # 直接学需同时发现「低头+补偿」，联合概率极低；强制低头把它拆成
        # 可学的单技能：给定低头姿态，学腿部怎么补。
        self.force_head = force_head
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
        self.servo = CommandConditioner(self.model.nu, seed=seed)
        self.dof_all = np.append(self.L["dof_joints"], self.L["mouth_dof"]) \
            if self.L["mouth_dof"] >= 0 else self.L["dof_joints"]
        self.L["gyro_slice"] = slice(
            int(self.model.sensor_adr[self.L["gyro"]]),
            int(self.model.sensor_adr[self.L["gyro"]] + 3),
        )
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
        self._phi = 0.0

    # ------------------------------------------------------------- helpers

    def _object_pose(self):
        q = self.data.qpos[self.L["object_qpos"]:self.L["object_qpos"] + 7]
        dq = self.data.qvel[self.L["object_dof"]:self.L["object_dof"] + 6]
        return q[:3], q[3:7], dq[:3], dq[3:6]

    def _trunk_state(self):
        q = self.data.qpos
        base_q = q[:7]
        R = _quat_to_mat(base_q[3:7])
        # gyro: the official model's IMU-frame gyro sensor (the imu body is
        # mounted rotated 90 deg about Y relative to the trunk, and the
        # policies were trained against exactly this sensor reading).
        gyro = self.data.sensordata[self.L["gyro_slice"]].copy()
        gravity = R.T @ np.array([0, 0, -1.0])    # projected gravity, trunk frame
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

    def task_ctx(self):
        """Live task geometry for expert selection during warm start."""
        tip = self._tip_pose()
        obj_pos = self._object_pose()[0]
        return dict(
            horiz=float(np.hypot(obj_pos[0] - tip[0], obj_pos[1] - tip[1])),
            tip_dist=float(np.linalg.norm(obj_pos - tip)),
            above=float(tip[2] - obj_pos[2]),
        )

    def set_force_head(self, value):
        self.force_head = bool(value)

    def set_curriculum_progress(self, progress):
        self._curriculum_progress = float(np.clip(progress, 0.0, 1.0))

    def _tip_pose(self):
        # 官方模型的 mouth_tip 站点（喙尖）；模型重建时由 layout 提供
        return self.data.site_xpos[self.L["mouth_tip_site"]].copy()

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
        """Go-to-goal in the trunk frame.

        The old world-frame heading (atan2(dy, dx)) flips to +-pi the moment
        the duck overshoots the goal, so vyaw oscillated between +-1 and the
        duck never settled — the expert mix therefore never reached the pick
        policy. Body-frame errors with a clamped forward denominator and an
        angle deadband keep it stable.
        """
        if self.task == "duck_lift":
            # 实测：喙尖水平偏移恒为「躯干+7.2~8.6cm」（低头只降高度、
            # 不前移）。行走实际比目标滞后 3-5cm，故目标取 obj-6cm：
            # 躯干停在 obj-10cm 左右时，喙尖正好压在物体上。
            goal = np.array([obj_pos[0] - 0.06, obj_pos[1], 0.0])
            # 停留点在物体后 12cm，但朝向要对准物体——喙尖沿躯干前轴啄出，
            # 躯干朝向偏 10° 就会让啄点横偏 2cm+。
            aim = obj_pos
        else:
            goal = obj_pos + obj_vel * 0.35
            aim = goal
        base = self.data.qpos[:3]
        q = self.data.qpos[3:7]
        yaw = float(np.arctan2(2.0 * (q[0] * q[3] + q[1] * q[2]),
                               1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])))
        dx, dy = float(goal[0] - base[0]), float(goal[1] - base[1])
        fwd = float(np.cos(yaw) * dx + np.sin(yaw) * dy)
        lat = float(-np.sin(yaw) * dx + np.cos(yaw) * dy)
        # 到位滞回：进入到位态后只做朝向修正，不再走路——否则边瞄准边移动
        # 会震荡，永远进不了 stand/pick 分支（实测 stand=0、喙尖只到 0.15）。
        pos_err = float(np.hypot(fwd, lat))
        if pos_err < 0.07:
            self._arrived = True
        elif pos_err > 0.16:
            self._arrived = False
        if self._arrived:
            ax, ay = float(aim[0] - base[0]), float(aim[1] - base[1])
            aim_angle = float(np.arctan2(-np.sin(yaw) * ax + np.cos(yaw) * ay,
                                         np.cos(yaw) * ax + np.sin(yaw) * ay))
            if abs(aim_angle) > 0.12:
                return np.array([0.0, 0.0, float(np.clip(
                    1.5 * aim_angle, -C.MAX_ANGULAR, C.MAX_ANGULAR))])
            return np.zeros(3)
        angle = float(np.arctan2(lat, max(fwd, 0.05)))
        if abs(angle) > 0.15:
            vyaw = float(np.clip(1.5 * angle, -C.MAX_ANGULAR, C.MAX_ANGULAR))
            vx = float(np.clip(0.8 * max(fwd, 0.0), 0.0, C.MAX_LINEAR))
        else:
            vyaw = 0.0
            vx = float(np.clip(1.6 * max(fwd, 0.0), 0.0, C.MAX_LINEAR))
        vy = float(np.clip(0.5 * lat, -0.05, 0.05))
        return np.array([vx, vy, vyaw])

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
        q0[:7] = [0, 0, self.L["home_height"], 1, 0, 0, 0]
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
            if self.curriculum is not None:
                far, near = self.curriculum
                span = far + (near - far) * self._curriculum_progress
                x = self.rng.uniform(max(0.10, span - 0.07), span)
            else:
                x = self.rng.uniform(0.15, 0.26)
            y = self.rng.uniform(-0.03, 0.03)
            z = 0.025 if self.object_kind == "linlee_duck" else 0.015
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
        targets = C.targets_from_action(np.zeros(C.ACTION_LEN))
        if self.L["mouth_qpos"] < 0:
            targets = np.concatenate(
                [targets[:C.MOUTH_INDEX], targets[C.MOUTH_INDEX + 1:]]
            )
        self.servo.reset(targets)
        for _ in range(30):
            self._write_ctrl(self.servo.step(targets, C.PHYSICS_DT))
            mujoco.mj_step(m, d)
        self.tracker.reset(pos, vel)
        self.beak.reset()
        self.ticks = 0
        self.last_action = np.zeros(C.ACTION_LEN)
        self._hold_ticks = 0
        self._caught = False
        self._ground_touch_after_toss = False
        self._phi = 0.0
        self._arrived = False
        self._min_horiz = 9.0
        self._min_tip = 9.0
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

    def _write_ctrl(self, targets):
        """Scatter JOINT_NAMES-ordered targets into data.ctrl by actuator id.

        The official XML declares the 14 policy actuators in joint order and this
        module appends m_mouth last, so ctrl is NOT in JOINT_NAMES order (the
        mouth sits at the end, not at index 9).
        """
        ctrl = np.zeros(self.model.nu)
        policy_targets = np.concatenate(
            [targets[:C.MOUTH_INDEX], targets[C.MOUTH_INDEX + 1:]]
        )
        if self.freeze_body:
            # 腿部 10 个关节钳回 home（JOINT_NAMES 顺序：左腿 5 + 右腿 5），
            # 只放开 4 个头部关节（neck/head pitch/yaw/roll，在向量中段）。
            policy_targets[:5] = C.DEFAULT_POSITION[:5]
            policy_targets[9:] = C.DEFAULT_POSITION[10:]
        ctrl[self.L["policy_act"]] = policy_targets
        if self.L["mouth_act"] >= 0:
            ctrl[self.L["mouth_act"]] = targets[C.MOUTH_INDEX]
        self.data.ctrl[:] = ctrl

    # ------------------------------------------------------------- step

    def step(self, action):
        m, d = self.model, self.data
        mujoco = __import__("mujoco")
        action = np.asarray(action, dtype=np.float64).reshape(C.ACTION_LEN)
        if self.force_head:
            obj_pos = self._object_pose()[0]
            head_cmd = self._head_command(obj_pos)
            action[4:8] = head_cmd - C.DEFAULT_POSITION[5:9]
        targets = C.clip_targets(C.targets_from_action(action))
        if self.L["mouth_qpos"] < 0:
            targets = np.concatenate(
                [targets[:C.MOUTH_INDEX], targets[C.MOUTH_INDEX + 1:]]
            )

        reward = 0.0
        info = {"success": False, "fall": False, "timeout": False}
        dt_tick = C.SUBSTEPS * C.PHYSICS_DT

        for sub in range(C.SUBSTEPS):
            self._write_ctrl(self.servo.step(targets, C.PHYSICS_DT))
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
            # 稠密接近奖励：身体到物体的距离每 tick 直接给分。势能整形只奖赏
            # 「进度」，不动则零分——学生于是满足于站着（每 tick +0.15 白拿），
            # 实测 8000 回合后仍停在物体 15-20cm 外。稠密奖励让「走近并待住」
            # 的累积（0.8 x 0.3 x 1000 = 240）远超站着的 150。
            body_dist = float(np.hypot(obj_pos[0] - base_pos[0],
                                       obj_pos[1] - base_pos[1]))
            reward += 0.8 * max(0.0, 0.40 - body_dist)
            # 互斥阶段奖励（同一 tick 只取最高阶段）。
            # 早期版本分量叠加（每 tick ~2.1），600 tick 回报 ~1200 让 critic
            # 崩掉（value_loss 500+）；且「站着不动」曾是局部最优（喙尖恒定
            # 0.20 不低头）——3D 对准的稠密奖励解决了后者。
            horiz = float(np.hypot(obj_pos[0] - tip[0], obj_pos[1] - tip[1]))
            tip_to_obj = float(np.linalg.norm(obj_pos - tip))
            # 阶段间差距加大：低位(0.8→1.5)与贴近(1.0→2.5)，因为「低头会
            # 前移重心」必须靠腿部补偿——这是最难的耦合技能，奖励差距不够
            # 时策略满足于站着（喙尖 0.19 不动）。
            # 温和配置：过大的阶段差距 + 递减跌倒惩罚在课程拉近后会让策略
            # 跳进破坏性区域（实测 KL 7.1、存活 21）。数值回到已验证稳定的
            # 量级，靠训练量而不是奖励锐度推进。
            if self._hold_ticks > 0:                      # 6 夹住保持
                reward += 3.0 * min(lifted, 0.04) + 0.5
            elif beak_contact:                            # 5 喙部接触
                reward += 2.0
            elif tip_to_obj < 0.06:                       # 4 喙尖贴近
                reward += 1.5
            # 极准大奖：喙尖 3cm 内且水平 2cm 内—— pads 能夹住物体的区域
            if tip_to_obj < 0.03 and horiz < 0.02:
                reward += 3.0

            elif tip[2] < 0.08:                           # 3 低位稳定
                reward += 1.0
            elif tip[2] < 0.16:                           # 2 中度低头（过渡）
                reward += 0.6
            else:                                         # 1 站立/接近
                reward += 0.15
            # 3D 对准稠密奖励：让「更近」每 tick 严格更优
            # 对准奖励权重 3.0：喙尖从 6cm 移到 3cm 每 tick 多拿 0.1，900 tick
            # 累积 90——足以压过「站着不动」的吸引力。实测学生对准停在 6cm
            # （ pads 在喙尖 ±8mm，够不到 6cm 外的物体）。
            reward += 3.0 * max(0.0, 0.20 - tip_to_obj)
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
            # 跌倒惩罚随课程进度递减：低头会前移重心，早期策略需要空间
            # 练习「边低边稳」——一摔就重罚会把策略逼回「站着不动」。
            reward -= 10.0
            info["fall"] = True

        if success:
            reward += 20.0
        # 仿真爆炸瞬间 tip/obj 可能是 NaN，奖励随之 NaN 并污染 GAE。
        if not np.isfinite(reward):
            reward = -10.0
        done = fallen or success or self.ticks >= self.max_ticks
        if done and not success:
            info["timeout"] = self.ticks >= self.max_ticks
        info["success"] = success
        info["hold_ticks"] = self._hold_ticks
        info["caught"] = self._caught
        info["obj_z"] = float(obj_pos[2])
        info["tilt"] = tilt
        info["tip_horiz"] = float(np.hypot(obj_pos[0] - tip[0], obj_pos[1] - tip[1]))
        info["tip_dist"] = float(np.linalg.norm(obj_pos - tip))
        info["mouth_closed"] = bool(self.data.qpos[self.L["mouth_qpos"]] < 0.1) \
            if self.L["mouth_qpos"] >= 0 else False
        self._min_horiz = min(self._min_horiz, info["tip_horiz"])
        self._min_tip = min(self._min_tip, info["tip_dist"])

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
