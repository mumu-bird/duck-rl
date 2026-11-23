"""Free-standing toss -> settled target -> pickup -> tray -> starting pose."""

from collections import deque
from dataclasses import replace
import math

import mujoco
import numpy as np

from ..policy import Controller, HOME
from .navigation import FloorWarp, Navigation, rotate, tray_anchor
from .grasp import DeliveryProfile, load_profiles
from .referee import at_home, captured_pose, heading, home_errors, in_tray
from .scene import CLUTTER, camera_frame, place, recolor


PHASE_NAMES = {
    "wait": "等新物体停稳",
    "approach": "对准目标",
    "stand_to_pick": "近距离复查",
    "pick": "低头夹取",
    "carry": "携物行走",
    "stand_to_drop": "箱沿站稳",
    "drop": "松嘴投放",
    "return": "返回起点",
    "brake": "回位制动",
    "restore": "恢复初始姿势",
}
GRIP_NAMES = {
    "jaw": "开合鸭嘴 · 摩擦接触",
    "contact": "固定鸭嘴 · 仅接触",
    "assisted": "接触辅助夹持",
}


class BeakFixture:
    """Only an actual mouth-mesh contact can close the optional simulated latch."""

    def __init__(self, model, data, assisted):
        self.m, self.d, self.assisted = model, data, assisted
        self.items = {model.body(name).id: name for name in ("parcel", *CLUTTER)}
        self.mouth = model.geom("mouth_ground_collision").id
        self.upper = model.geom("head_bottom_ground_collision").id
        self.bill_geoms = [
            self.mouth,
            self.upper,
            model.geom("head_roll_ground_collision").id,
        ]
        if model.geom("parcel_hull").contype[0] == 8:
            self.bill_geoms += [model.geom(n).id for n in ("upper_lip", "lower_lip")]
        self.touched = self.attached = None
        self.ever_attached = False
        self.distance = None

    def sense(self, close=False):
        tip = self.d.site("mouth_tip").xpos
        for contact in self.d.contact:
            bill = next(
                (g for g in self.bill_geoms if g in (contact.geom1, contact.geom2)),
                None,
            )
            if bill is None:
                continue
            other = contact.geom2 if bill == contact.geom1 else contact.geom1
            name = self.items.get(self.m.geom_bodyid[other])
            distance = np.linalg.norm(tip - contact.pos)
            if name is None or (distance > 0.022 and bill not in self.bill_geoms[3:]):
                continue
            if close:
                if self.touched is None:
                    self.touched, self.distance = name, float(distance)
                if self.assisted and self.attached is None:
                    eq = self.m.equality("hold_" + name).id
                    for offset, body_name in ((0, "head_roll"), (3, name)):
                        body = self.d.body(body_name)
                        self.m.eq_data[eq, offset : offset + 3] = body.xmat.reshape(
                            3, 3
                        ).T @ (contact.pos - body.xpos)
                    self.d.eq_active[eq] = True
                    self.attached, self.ever_attached = eq, True
            return True
        return False

    def release(self):
        if self.attached is not None:
            self.d.eq_active[self.attached] = False
            self.attached = None


def settled(track):
    if len(track) < track.maxlen:
        return False
    points = np.array(track)
    return np.linalg.norm(np.ptp(points, axis=0)) <= 0.006


def seed_scene(model, data, rng):
    recolor(model, rng)
    for name, x, y, z in [
        (CLUTTER[0], 0.46, -0.11, 0.013),
        (CLUTTER[1], 0.23, -0.16, 0.006),
    ]:
        xy = rng.uniform(-0.035, 0.035, 2) + (x, y)
        place(data, name, (*xy, z), rng.uniform(-math.pi, math.pi))
    place(data, "parcel", (-2.0, -2.0, 0.02))
    mujoco.mj_forward(model, data)


def toss(data, rng):
    place(
        data,
        "parcel",
        (rng.uniform(0.17, 0.27), rng.uniform(-0.095, 0.05), rng.uniform(0.065, 0.115)),
        rng.uniform(-math.pi, math.pi),
    )
    data.joint("parcel_free").qvel[:] = (
        rng.uniform(0.14, 0.31),
        rng.uniform(-0.09, 0.09),
        rng.uniform(0.10, 0.27),
        *rng.uniform(-2.0, 2.0, 3),
    )


class Episode:
    def __init__(
        self,
        model,
        half,
        renderer,
        detector,
        mode,
        seed,
        bin_xy,
        no_toss=False,
        video=None,
        profiles=None,
        policy=None,
        audio=None,
    ):
        self.m = model
        self.d = mujoco.MjData(model)
        self.half = half
        self.c = Controller(self.m, self.d)
        self.nav = Navigation(self.c)
        self.grip = BeakFixture(model, self.d, mode == "assisted")
        self.mode = mode
        self.renderer, self.detector, self.video = renderer, detector, video
        self.audio = audio
        self.policy = policy
        self.policy_pending = set()
        self.policy_actions = {"pick": [], "delivery": []}
        self.policy_rewards = {"pick": [], "delivery": []}
        self.profiles = profiles or load_profiles()
        self.delivery = self.profiles.get("delivery", DeliveryProfile())
        self.release_relative = None
        self.generic_profile = self.profiles["general"]
        self.profile = self.generic_profile
        self.reach = self.profile.reach if mode == "jaw" else 0.097
        self.pick_offset = np.zeros(2)
        self.fine_target = None
        self.jac = np.zeros((3, model.nv))
        self.neck_dofs = self.c.v[5:8]
        self.seed, self.no_toss = seed, no_toss
        self.bin_xy = np.asarray(bin_xy)
        self.rng = np.random.default_rng(seed)
        seed_scene(model, self.d, self.rng)
        self.stage = "stand"
        self.age = 0
        self.events = []
        self.track = deque(maxlen=7)
        self.target = None
        self.target_axis_angle = 0.0
        self.target_scale = 0.0
        self.target_shape = np.zeros(2, dtype=np.float32)
        self.loc_error = self.approach_error = None
        self.carry_release = False
        self.lost_ticks = 0
        self.grip_angle = None
        self.raised = False
        self.release_tick = None
        self.tray_ticks = self.pose_ticks = 0
        self.verified_tray = False
        self.landing_height = None
        self.frames = 0
        self.floor_still = None
        self.last_lift = 0.0
        self.tick = 0
        self.stop = "timeout"
        self.waypoint = None
        self.pick_tip_distance = None
        self.failed_pick = False
        self.retry_center = None
        self.retry_offset = np.zeros(2)
        self.attempts = 0
        self.home_adjustments = 0
        self.stop_xy_bias = np.zeros(2)
        self.stop_heading_bias = 0.0
        self.ever_contact = False
        self.ever_wrong = False
        self.visual_fixes = 0

    def transition(self, stage):
        self.events.append([int(self.tick), stage])
        self.stage = stage
        self.age = 0

    def _snapshot(self, tag):
        if self.video is None:
            return
        self.video.snapshot(
            self.d,
            tag,
            [
                PHASE_NAMES.get(self.stage, self.stage),
                GRIP_NAMES[self.mode],
                f"抬起 {self.last_lift * 100:.1f} cm",
                f"留在箱内 {self.tray_ticks * 0.02:.1f} s",
            ],
        )

    def policy_observation(self, stage):
        yaw = self.nav.pose.yaw

        def local(point):
            return np.clip(rotate(-yaw, point - self.nav.filtered) / 0.5, -1.0, 1.0)

        target = np.zeros(2) if self.target is None else local(self.target)
        bin_local = local(self.bin_goal)
        home_local = local(self.home)
        axis = 2.0 * self.target_axis_angle
        heading_error = 2.0 * math.remainder(self.pick_heading - yaw, math.tau)
        velocity = rotate(-yaw, self.nav.velocity) / 0.25
        angular = self.d.sensor("imu_ang_vel").data / 3.0
        joints = (self.d.qpos[self.c.q][5:8] - HOME[5:8]) / 1.2
        bill_id = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_JOINT, "bill_open"
        )
        bill = (
            self.d.qpos[self.m.jnt_qposadr[bill_id]] / math.radians(39.375)
            if bill_id >= 0
            else 0.0
        )
        return np.asarray(
            [
                float(stage == "pick"),
                float(stage == "delivery"),
                *target,
                float(self.target_scale),
                *self.target_shape,
                *bin_local,
                *home_local,
                math.sin(axis),
                math.cos(axis),
                math.sin(heading_error),
                math.cos(heading_error),
                *np.clip(velocity, -1.0, 1.0),
                *np.clip(angular, -1.0, 1.0),
                *np.clip(self.nav.odom.up, -1.0, 1.0),
                *np.clip(joints, -1.0, 1.0),
                float(np.clip(bill, 0.0, 1.0)),
                float(np.clip(self.age / 480.0, 0.0, 1.0)),
                float(min(self.attempts, 2) / 2.0),
            ],
            dtype=np.float32,
        )

    def _decide(self, stage):
        if self.policy is None:
            return np.zeros(6, dtype=np.float32)
        action = np.asarray(
            self.policy.act(stage, self.policy_observation(stage)), dtype=np.float32
        ).reshape(6)
        action = np.clip(action, -1.0, 1.0)
        self.policy_actions[stage].append(action.tolist())
        self.policy_pending.add(stage)
        return action

    def _resolve(self, stage, reward, done=False):
        if self.policy is not None and stage in self.policy_pending:
            reward = float(reward)
            self.policy.resolve(stage, reward, done=done)
            self.policy_rewards[stage].append(reward)
            self.policy_pending.remove(stage)

    def _choose_pick_profile(self):
        if self.policy is None:
            return
        action = self._decide("pick")
        self.reach = float(np.clip(self.profile.reach + 0.014 * action[0], 0.05, 0.11))
        self.profile = replace(
            self.profile,
            reach=self.reach,
            close=int(np.clip(round(self.profile.close + 14 * action[1]), 28, 78)),
            rise=int(np.clip(round(self.profile.rise + 16 * action[2]), 62, 112)),
            scoop_pitch=float(
                np.clip(self.profile.scoop_pitch + 0.18 * action[3], -0.48, 0.16)
            ),
            scoop_neck=float(
                np.clip(self.profile.scoop_neck + 0.12 * action[4], -0.20, 0.20)
            ),
        )
        self.pick_heading += 0.12 * float(action[5])

    def _choose_delivery_profile(self):
        action = self._decide("delivery")
        self.delivery = replace(
            self.delivery,
            reach=float(np.clip(self.delivery.reach + 0.026 * action[0], 0.07, 0.15)),
            side=float(np.clip(self.delivery.side + 0.035 * action[1], -0.08, 0.08)),
            neck=float(np.clip(self.delivery.neck + 0.16 * action[2], -0.52, 0.02)),
            pitch=float(np.clip(self.delivery.pitch + 0.14 * action[3], -0.38, 0.18)),
            release_tick=int(
                np.clip(round(self.delivery.release_tick + 16 * action[4]), 8, 76)
            ),
            open_ticks=int(
                np.clip(round(self.delivery.open_ticks + 6 * action[5]), 3, 20)
            ),
        )

    def _pick_reward(self, held):
        gap = 0.16 if self.pick_tip_distance is None else self.pick_tip_distance
        close = max(0.0, 1.0 - gap / 0.16)
        lifted = min(1.0, self.last_lift / 0.035)
        return (
            -0.45
            + 0.25 * close
            + 0.45 * float(self.ever_contact)
            + 1.35 * lifted
            + 0.35 * float(held)
            - 0.55 * float(self.ever_wrong)
        )

    @staticmethod
    def _delivery_reward(result):
        return (
            -0.45
            + 0.65 * float(result["held_at_release"])
            + 2.0 * float(result["in_tray_at_end"])
            + 0.55 * float(result["pose_restored"])
            + 2.5 * float(result["success"])
            - 0.7 * float(result["wrong_contact"])
            - 1.0 * float(result["stop"] == "fall")
        )

    def initialize(self):
        cmd = np.zeros(13, np.float32)
        for _ in range(75):
            self.nav.update()
            self.c.step(self.c.act("stand", cmd), "stand", self.nav.imu)
        self.nav.update()
        self.home = self.nav.filtered.copy()
        self.yaw = self.nav.pose.yaw
        self.start = captured_pose(self.c)
        self.reference = camera_frame(self.renderer, self.d)
        self.warp = FloorWarp(self.m, self.reference, self.nav.camera_pose())
        self.pick_heading = self.yaw
        self.bin_goal = self.home + rotate(self.yaw, self.bin_xy)
        self.anchor = tray_anchor(
            self.reference, self.nav.camera_pose(), self.warp.focal, self.bin_goal
        )
        self.transition("wait")
        if not self.no_toss:
            toss(self.d, self.rng)
            mujoco.mj_forward(self.m, self.d)
        self._snapshot("initial")

    def find_target(self):
        frame = camera_frame(self.renderer, self.d)
        self.frames += 1
        if self.attempts and self.profile.retry_radius > 0:
            # Warp the original empty floor into the actual return camera pose.
            # A missed pick leaves the target nearby, a lost carry leaves it near the bill.
            center = self.target if self.retry_center is None else self.retry_center
            background, roi = self.warp.near(
                frame, self.nav.camera_pose(), center, self.profile.retry_radius
            )
            found = self.detector.locate(background, frame, roi=roi)
        else:
            found = self.detector.locate(self.reference, frame)
        point = self.nav.floor_point(found[0]) if found is not None else None
        if point is None or not 0.08 < np.linalg.norm(point - self.home) < 0.8:
            self.track.clear()
            return
        self.track.append(point)
        if not settled(self.track):
            return
        self.attempts += 1
        self.failed_pick = False
        self.target = np.median(self.track, axis=0)
        self.pick_heading = self.yaw
        self.target_axis_angle = 0.0
        self.target_scale = (
            float(np.clip(np.sqrt(np.prod(found[3])) / 64.0, 0.0, 1.0))
            if len(found) > 3
            else 0.0
        )
        self.target_shape = (
            np.clip(np.asarray(found[4:6], dtype=np.float32), 0.0, 1.0)
            if len(found) > 5
            else np.zeros(2, dtype=np.float32)
        )
        self.profile = self.generic_profile
        self.reach = self.profile.reach if self.mode == "jaw" else 0.097
        if self.mode == "jaw" and found[2] is not None:
            left = self.nav.floor_point(found[1] - found[2])
            right = self.nav.floor_point(found[1] + found[2])
            if left is not None and right is not None:
                shaft = right - left
                self.target_axis_angle = math.atan2(float(shaft[1]), float(shaft[0]))
                normal = math.atan2(shaft[1], shaft[0]) + math.pi / 2
                self.pick_heading = self.yaw + math.remainder(
                    normal - self.yaw, math.pi
                )
                self.profile = self.profiles["thin"]
                self.reach = self.profile.reach
        if self.attempts > 1 and self.mode == "jaw":
            if "retry" in self.profiles:
                self.profile = self.profiles["retry"]
                self.reach = self.profile.reach
            self.pick_heading += self.profile.retry_yaw
            self.reach += self.profile.retry_reach
            self.retry_offset = rotate(
                self.pick_heading, (0.0, self.profile.retry_side)
            )
        self.retry_center = None
        self._choose_pick_profile()
        self.waypoint = (
            self.target
            + self.retry_offset
            - rotate(self.pick_heading, (self.reach, 0.0))
        )
        # Referee fields are logged only. They never feed back to the task policy.
        object_xy = self.d.body("parcel").xpos[:2]
        truth_in_map = self.home + rotate(
            self.yaw - self.start[1], object_xy - self.start[0][:2]
        )
        self.loc_error = float(np.linalg.norm(self.target - truth_in_map))
        self.floor_still = float(np.linalg.norm(self.d.joint("parcel_free").qvel[:3]))
        self.transition("approach")
        self._snapshot("target-found")
        if self.audio is not None:
            self.audio.emit("target_found")

    def correct_pose(self):
        if self.anchor is None:
            return
        live = camera_frame(self.renderer, self.d)
        observed = tray_anchor(
            live, self.nav.camera_pose(), self.warp.focal, self.bin_goal
        )
        if observed is None:
            return
        centre, angle = observed
        dyaw = np.clip(self.anchor[1] - angle, -0.08, 0.08) * 0.12
        xy = self.nav.odom.xy.copy()
        fixed = self.anchor[0] + rotate(dyaw, xy - centre)
        displacement = np.clip(fixed - xy, -0.035, 0.035) * 0.7
        self.nav.odom.xy += displacement
        self.nav.odom.yaw += dyaw
        self.nav.odom.previous_feet = None
        self.nav.filtered += displacement
        self.nav.update()
        self.visual_fixes += 1

    def drive(self, cmd):
        if (
            self.stage == "return"
            and self.age % 9 == 0
            and np.linalg.norm(self.nav.filtered - self.home) < 0.18
        ):
            self.correct_pose()
        goal, threshold = {
            "approach": (self.waypoint, 0.009),
            "carry": (
                self.bin_goal
                - rotate(self.yaw, (self.delivery.reach, self.delivery.side)),
                0.012,
            ),
            "return": (self.home, 0.011),
        }[self.stage]
        direction = self.pick_heading if self.stage == "approach" else self.yaw
        if self.stage == "return":
            goal = goal - self.stop_xy_bias
            direction -= self.stop_heading_bias
        cmd[:], arrived = self.nav.drive(goal, direction, threshold)
        if self.stage == "carry" and self.mode == "jaw":
            cmd[0] = np.clip(cmd[0], -0.085, 0.085)
            cmd[1] = np.clip(cmd[1], -0.060, 0.060)
        if not arrived:
            if self.age >= 480:
                self.stop = self.stage + "_timeout"
                self.transition("done")
            return "walk"
        if self.stage == "approach":
            self.transition("stand_to_pick")
        elif self.stage == "carry":
            self.transition("stand_to_drop")
        else:
            self.transition("brake")
            cmd[:] = 0.0
            return "walk"
        cmd[:] = 0.0
        return "stand"

    def plan(self):
        cmd = np.zeros(13, np.float32)
        slot = "stand"
        if self.stage == "wait":
            if self.age % 5 == 0:
                self.find_target()
            if self.stage == "wait" and self.age >= 140:
                self.stop = "no_new_settled_object"
                self.transition("done")
        elif self.stage in ("approach", "carry", "return"):
            slot = self.drive(cmd)
        elif self.stage == "stand_to_pick" and self.age >= 28:
            live = camera_frame(self.renderer, self.d)
            reference, roi = self.warp.near(live, self.nav.camera_pose(), self.target)
            found = self.detector.locate(reference, live, roi=roi)
            if found is not None:
                near = self.nav.floor_point(found[0])
                if near is not None and np.linalg.norm(near - self.target) < 0.042:
                    self.fine_target = near
                    if len(found) > 3:
                        self.target_scale = float(
                            np.clip(np.sqrt(np.prod(found[3])) / 64.0, 0.0, 1.0)
                        )
                    if len(found) > 5:
                        self.target_shape = np.clip(
                            np.asarray(found[4:6], dtype=np.float32), 0.0, 1.0
                        )
            target = self.target if self.fine_target is None else self.fine_target
            target = target + self.retry_offset
            body = np.array((self.nav.pose.x, self.nav.pose.y))
            self.pick_offset = np.clip(
                target - body - rotate(self.nav.pose.yaw, (self.reach, 0.0)),
                -0.022,
                0.022,
            )
            self.landing_height = float(self.d.body("parcel").xpos[2])
            self.transition("pick")
            base = self.d.body("trunk_base").xpos[:2]
            self.approach_error = float(
                np.linalg.norm(
                    base
                    + rotate(heading(self.d), (self.reach, 0.0))
                    - self.d.body("parcel").xpos[:2]
                )
            )
        elif self.stage in ("pick", "drop"):
            phase = (self.age - 1) * math.pi / 100
            cmd[:2] = (math.cos(phase), math.sin(phase))
            slot = "groundpick"
            if self.stage == "pick" and self.mode == "jaw":
                cmd[:2], _ = self.profile.command(self.age - 1)
            if self.stage == "drop" and self.age == (
                self.delivery.release_tick if self.mode == "jaw" else 35
            ):
                # Referee samples location at actual initiation of opening.
                parcel = self.d.body("parcel").xpos
                tip = self.d.site("mouth_tip").xpos
                self.release_relative = (
                    parcel - np.array([*self.bin_xy, 0.0])
                ).tolist()
                self.carry_release = (
                    np.linalg.norm(parcel - tip) < 0.075
                    and parcel[2] > 0.03
                    and self.grip.touched == "parcel"
                )
                self.grip.release()
                self.release_tick = self.tick
                self._snapshot("release")
            if (
                self.stage == "drop"
                and self.mode == "jaw"
                and not self.delivery.lower_to_tray
            ):
                # A stationary release avoids brushing objects out of a shallow bin.
                slot = "stand"
                cmd[:] = 0
                if self.age >= 78:
                    self.transition("return")
            if self.age >= 198:
                if self.stage == "pick":
                    held = self.grip.attached is not None or self.grip.sense()
                    self._resolve("pick", self._pick_reward(held))
                    if held:
                        if self.mode == "jaw":
                            self.grip_angle = float(self.d.joint("bill_open").qpos[0])
                        if self.policy is not None:
                            self._choose_delivery_profile()
                        self.transition("carry")
                        self._snapshot("picked")
                    else:
                        self.failed_pick = True
                        self.transition("return")
                else:
                    self.transition("return")
        elif self.stage == "stand_to_drop" and self.age >= 24:
            self.transition("drop")
        elif self.stage == "brake":
            # Let the walking policy bring both feet to rest before switching to
            # the stand policy. The IMU closes the heading loop during braking.
            slot = "walk"
            miss = math.remainder(
                self.yaw - self.stop_heading_bias - self.nav.pose.yaw, math.tau
            )
            cmd[2] = np.clip(2.0 * miss, -0.30, 0.30) if self.age <= 28 else 0.0
            if self.age >= 46:
                self.transition("restore")
                slot = "stand"
        elif self.stage == "restore":
            if self.age in (18, 32, 45):
                self.correct_pose()
            if self.age == 46 and self.home_adjustments < 2:
                drift = np.linalg.norm(self.nav.filtered - self.home)
                if (
                    drift > 0.021
                    or abs(math.remainder(self.nav.pose.yaw - self.yaw, math.tau))
                    > 0.070
                ):
                    # Re-measure after the walk-to-stand transition. The final
                    # loaded footstep may rotate the trunk after odometry says stop.
                    self.stop_xy_bias = np.clip(
                        self.stop_xy_bias + 0.8 * (self.nav.filtered - self.home),
                        -0.035,
                        0.035,
                    )
                    miss = math.remainder(self.nav.pose.yaw - self.yaw, math.tau)
                    self.stop_heading_bias = float(
                        np.clip(self.stop_heading_bias + 0.95 * miss, -0.28, 0.28)
                    )
                    self.home_adjustments += 1
                    self.pose_ticks = 0
                    self.transition("return")
            if self.stage == "restore" and self.age >= 74:
                if self.failed_pick and self.attempts < 2:
                    self.grip.touched = None
                    self.track.clear()
                    self.fine_target = None
                    self.pose_ticks = 0
                    self.transition("wait")
                else:
                    self.stop = (
                        "no_hold_after_lift"
                        if self.failed_pick
                        else "sequence_finished"
                    )
                    self.transition("done")
        return slot, cmd

    def run(self):
        self.initialize()
        for self.tick in range(2800):
            self.nav.update()
            self.age += 1
            slot, cmd = self.plan()
            if self.stage == "done":
                break

            def contact_and_imu():
                self.nav.imu()
                if self.stage == "pick" and 28 <= self.age <= 112:
                    self.grip.sense(close=True)

            if self.mode == "jaw":
                opening = 0.0
                if self.stage == "pick":
                    _, angle = self.profile.command(self.age - 1)
                    opening = angle / math.radians(39.375)
                elif (
                    self.stage in ("carry", "stand_to_drop")
                    and self.grip_angle is not None
                ):
                    opening = max(0.0, self.grip_angle - 0.055) / math.radians(39.375)
                elif self.stage == "drop":
                    held = 0.0
                    if self.delivery.retain_until_open and self.grip_angle is not None:
                        held = max(0.0, self.grip_angle - 0.055) / math.radians(39.375)
                    elapsed = self.age - self.delivery.release_tick + 1
                    if elapsed <= 0:
                        opening = held
                    elif self.age <= 103:
                        opening = min(
                            1.0,
                            held + (1.0 - held) * elapsed / self.delivery.open_ticks,
                            (104 - self.age) / 10,
                        )
                self.d.actuator("bill_servo").ctrl[0] = max(
                    0.0, opening
                ) * math.radians(39.375)
            joints = self.c.act(slot, cmd)
            if self.mode == "jaw" and self.stage in ("carry", "stand_to_drop"):
                # Tilt the bill up while walking, and damp policy head swings.
                gain = min(1.0, self.age / 22.0) if self.stage == "carry" else 1.0
                joints[6] = (1 - 0.8 * gain) * joints[6] + 0.8 * gain * (-0.075)
                joints[7] = (1 - 0.70 * gain) * joints[7]
            if self.mode == "jaw" and self.stage == "drop":
                gain = max(0.0, min(1.0, self.age / 24.0, (78 - self.age) / 20.0))
                joints[5] += self.delivery.neck * gain
                joints[6] += self.delivery.pitch * gain
            if self.stage == "pick" and 18 <= self.age <= 122:
                gain = max(
                    0.0, min(1.0, (self.age - 18) / 23.0, (122 - self.age) / 18.0)
                )
                self.nav.camera_pose()
                mujoco.mj_jacSite(
                    self.m, self.nav.fk, self.jac, None, self.m.site("mouth_tip").id
                )
                j = self.jac[:, self.neck_dofs]
                correction = j.T @ np.linalg.solve(
                    j @ j.T + 0.00025 * np.eye(3), [*self.pick_offset, 0.0]
                )
                joints[5:8] += gain * np.clip(correction, -0.23, 0.23)
            if self.stage == "pick" and (
                self.profile.scoop_neck or self.profile.scoop_pitch
            ):
                joints[5:7] += self.profile.stroke(self.age - 1)
                ids = [self.m.joint(name).id for name in ("neck_pitch", "head_pitch")]
                joints[5:7] = np.clip(joints[5:7], *self.m.jnt_range[ids].T)
            self.c.step(joints, slot, contact_and_imu)
            self.ever_contact |= self.grip.touched == "parcel"
            self.ever_wrong |= self.grip.touched in CLUTTER
            if self.stage == "carry" and self.mode == "jaw":
                self.lost_ticks = 0 if self.grip.sense() else self.lost_ticks + 1
                if self.lost_ticks >= 8:
                    if self.profile.retry_radius > 0:
                        self.retry_center = self.nav.filtered + rotate(
                            self.nav.pose.yaw, (0.11, 0.0)
                        )
                    self.failed_pick = True
                    self.lost_ticks = 0
                    self._resolve("delivery", -0.8)
                    self.transition("return")
            if self.stage == "pick" and 26 <= self.age <= 113:
                gap = float(
                    np.linalg.norm(
                        self.d.site("mouth_tip").xpos - self.d.body("parcel").xpos
                    )
                )
                self.pick_tip_distance = (
                    gap
                    if self.pick_tip_distance is None
                    else min(gap, self.pick_tip_distance)
                )
            if (
                self.landing_height is not None
                and self.grip.touched == "parcel"
                and self.stage == "pick"
            ):
                self.last_lift = max(
                    self.last_lift,
                    float(self.d.body("parcel").xpos[2] - self.landing_height),
                )
                self.raised |= self.last_lift >= 0.035
            if self.release_tick is not None:
                self.tray_ticks = (
                    self.tray_ticks + 1
                    if in_tray(self.m, self.d, self.half, self.bin_xy)
                    else 0
                )
                self.verified_tray |= self.tray_ticks >= 25
            if self.stage == "restore":
                quiet = np.linalg.norm(self.d.qvel[:3]) < 0.035
                self.pose_ticks = (
                    self.pose_ticks + 1
                    if quiet and at_home(home_errors(self.c, self.start))
                    else 0
                )
            if self.video is not None and self.tick % 5 == 0:
                self.video.capture(
                    self.d,
                    [
                        PHASE_NAMES.get(self.stage, self.stage),
                        GRIP_NAMES[self.mode],
                        f"抬起 {self.last_lift * 100:.1f} cm",
                        f"留在箱内 {self.tray_ticks * 0.02:.1f} s",
                    ],
                )
            if self.d.qpos[2] < 0.045 or self.d.body("trunk_base").xmat[8] < 0.10:
                self.stop = "fall"
                break
        errors = home_errors(self.c, self.start)
        stored = self.tray_ticks >= 25 and self.release_tick is not None
        home = self.pose_ticks >= 25
        success = bool(
            self.raised
            and self.carry_release
            and stored
            and home
            and not self.ever_wrong
            and self.stop == "sequence_finished"
            and not self.no_toss
        )
        result = dict(
            seed=self.seed,
            mode=self.mode,
            no_toss=self.no_toss,
            success=success,
            stop=self.stop,
            detected=self.target is not None,
            localization_cm=None if self.loc_error is None else self.loc_error * 100,
            speed_at_decision_m_s=self.floor_still,
            approach_cm=None
            if self.approach_error is None
            else self.approach_error * 100,
            min_tip_to_center_cm=None
            if self.pick_tip_distance is None
            else self.pick_tip_distance * 100,
            target_contact=bool(self.ever_contact),
            wrong_contact=bool(self.ever_wrong),
            fixture_activated=self.grip.ever_attached,
            lifted=bool(self.raised),
            max_lift_cm=self.last_lift * 100,
            released=self.release_tick is not None,
            held_at_release=bool(self.carry_release),
            in_tray_after_release=bool(self.verified_tray),
            in_tray_at_end=bool(stored),
            pose_restored=bool(home),
            events=self.events,
            frames=self.frames,
            pick_heading_rad=self.pick_heading,
            attempts=self.attempts,
            visual_home_fixes=self.visual_fixes,
            physics_steps=int(self.tick),
            fine_rgb=self.fine_target is not None,
            release_relative_m=self.release_relative,
            target_visual_features={
                "area_scale": float(self.target_scale),
                "elongation": float(self.target_shape[0]),
                "silhouette_fill": float(self.target_shape[1]),
            },
            **errors,
        )
        if self.policy is not None:
            result["policy_actions"] = self.policy_actions
            result["policy_rewards"] = self.policy_rewards
        if "pick" in self.policy_pending:
            self._resolve("pick", self._pick_reward(False) - 1.0)
        if "delivery" in self.policy_pending:
            self._resolve("delivery", self._delivery_reward(result), done=True)
        if self.policy is not None:
            terminal_bonus = (
                0.0
                if "delivery" in self.policy_actions
                else (-1.0 if result["stop"] == "fall" else -0.45)
            )
            self.policy.finish_episode(terminal_bonus)
        return result
