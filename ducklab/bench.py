"""Head RGB + calibrated tabletop plane + neck IK, on a supported robot."""
import mujoco
import numpy as np

from .assets import TABLE_Z
from .policy import HOME, MODELS


def view_options():
    options = mujoco.MjvOption()
    options.geomgroup[3] = 0  # collision meshes overlap the visual rig
    options.sitegroup[:] = 0
    return options


def centroid(rgb, color):
    r, g, b = rgb.astype('int16').transpose(2, 0, 1)
    if color == 'red':
        mask = (r > 100) & (r > 3*g) & (r > 3*b)
    elif color == 'green':
        mask = (g > 100) & (g > 2*r) & (g > 1.7*b)
    elif color == 'blue':
        mask = (b > 115) & (b > 3*g) & (b > 3*r)
    else:
        raise ValueError(color)
    yy, xx = np.nonzero(mask)
    if len(xx) >= 8:
        return np.array([np.median(xx), np.median(yy)])


def plane_xy(model, data, pixel, shape, height):
    h, w = shape[:2]
    f = .5*h / np.tan(np.deg2rad(model.cam('head_rgb').fovy[0]) / 2)
    view = data.cam('head_rgb')
    ray = view.xmat.reshape(3, 3) @ np.array([(pixel[0] + .5 - w/2)/f,
                                               -(pixel[1] + .5 - h/2)/f, -1.])
    if ray[2] >= -1e-5:
        return None
    depth = (height - view.xpos[2]) / ray[2]
    if depth > 0:
        return (view.xpos + depth * ray)[:2]


class Bench:
    def __init__(self, controller, rng, color, camera, mode='fixture'):
        self.c = controller
        self.m, self.d = controller.m, controller.d
        self.camera, self.options = camera, view_options()
        self.color = color
        self.mode = mode
        self.target = 'parcel' if color == 'red' else 'spare'
        self.other = 'spare' if color == 'red' else 'parcel'
        sign = rng.choice([-1, 1])
        xy = np.array([rng.uniform(.083,.092), sign*rng.uniform(.014,.026)])
        pad = np.array([rng.uniform(.081,.090), -sign*rng.uniform(.022,.033)])
        self.m.geom('pad').pos[:2] = pad
        self.d.joint(self.target+'_free').qpos[:3] = [*xy, TABLE_Z + .0062]
        self.d.joint(self.other+'_free').qpos[:3] = [rng.uniform(.123,.148),
                                                      rng.uniform(-.055,.055), TABLE_Z + .0062]
        mujoco.mj_forward(self.m, self.d)
        self.initial = self.d.body(self.target).xpos.copy()
        self.other_start = self.d.body(self.other).xpos.copy()
        self.tip = self.m.site('mouth_tip').id
        self.held = None
        self.jac = np.zeros((3, self.m.nv))
        self.ik_state = mujoco.MjData(self.m)
        self.head_q = controller.q[5:8]
        self.head_v = controller.v[5:8]
        self.limits = self.m.jnt_range[[self.m.joint(k).id for k in ('neck_pitch','head_pitch','head_yaw')]]
        self.angles = HOME[5:8].copy()
        self.trim = np.zeros(3)
        motor = MODELS['stand']['calibration']
        self.kp = .166 * motor['firmwareP'] * motor['error_gain_ratio'] * 7.7 * motor['kt'] / motor['R']
        self.stage, self.age = 'scan', 0
        self.events = []
        self.seen = {}
        self.grip_xy = None
        self.max_height, self.other_drift = 0., 0.
        self.seen_frames, self.image = 0, None
        self.initial_error = None
        self.final_seen = None
        self.contact_target = None
        self.hold_ids = [self.m.equality('hold_'+name).id for name in ('parcel','spare')]
        self.used_hold = False

    def advance(self, name):
        self.stage, self.age = name, 0
        self.events.append([name, round(self.d.time, 2)])

    def observe(self):
        self.camera.update_scene(self.d, camera='head_rgb', scene_option=self.options)
        self.image = self.camera.render()
        for color, height in [(self.color, TABLE_Z+.009), ('green', TABLE_Z+.0003)]:
            px = centroid(self.image, color)
            if px is not None:
                xy = plane_xy(self.m, self.d, px, self.image.shape, height)
                if xy is not None and .065 < xy[0] < .16 and abs(xy[1]) < .085:
                    self.seen[color] = xy
        self.seen_frames += 1

    def inverse_kinematics(self, goal):
        # Only the supported trunk and the measured joint configuration enter FK.
        s = self.ik_state
        s.qpos[:] = self.d.qpos
        s.qpos[self.head_q] = self.angles
        for _ in range(7):
            mujoco.mj_kinematics(self.m, s)
            mujoco.mj_comPos(self.m, s)
            delta = goal - s.site_xpos[self.tip]
            if np.linalg.norm(delta) < .00015:
                break
            mujoco.mj_jacSite(self.m, s, self.jac, None, self.tip)
            jac = self.jac[:, self.head_v]
            correction = jac.T @ np.linalg.solve(jac@jac.T + .000018*np.eye(3), delta)
            s.qpos[self.head_q] = np.clip(s.qpos[self.head_q] + np.clip(correction,-.13,.13),
                                          self.limits[:,0]+.008, self.limits[:,1]-.008)
        self.angles += np.clip(s.qpos[self.head_q]-self.angles, -.035,.035)

    def latch_on_contact(self):
        # The fixture emulates a closing beak. No object pose is set during transport.
        tip = self.d.site_xpos[self.tip]
        blocks = {self.m.geom(name).id:name for name in ('parcel','spare')}
        bill = {self.m.geom(n).id for n in ('tip_pad','mouth_ground_collision')}
        for contact in self.d.contact:
            ids = {contact.geom1, contact.geom2}
            hit = ids & blocks.keys()
            if hit and ids & bill and np.linalg.norm(tip-contact.pos) < .012:
                self.contact_target = blocks[hit.pop()]
                self.held = self.m.equality('hold_'+self.contact_target).id
                for index, name in [(0,'head_roll'), (3,self.contact_target)]:
                    body = self.d.body(name)
                    self.m.eq_data[self.held,index:index+3] = body.xmat.reshape(3,3).T @ (contact.pos-body.xpos)
                self.d.eq_active[self.held] = self.mode == 'fixture'
                self.grip_xy = self.seen[self.color] - tip[:2]
                return True
        return False

    def plan(self):
        self.age += 1
        tip = self.d.site_xpos[self.tip]
        if self.stage == 'scan':
            if self.age % 5 == 0:
                self.observe()
            if self.age >= 20 and {self.color,'green'} <= self.seen.keys():
                self.initial_error = float(np.linalg.norm(self.seen[self.color]-self.initial[:2]))
                self.advance({'locate':'identified','push':'push_setup'}.get(self.mode, 'approach'))
        elif self.stage.startswith('push_'):
            pick, dest = self.seen[self.color], self.seen['green']
            direction = (dest-pick)/np.linalg.norm(dest-pick)
            start, stop = pick-.024*direction, dest-.012*direction
            xy, height = start, .031
            if self.stage == 'push_setup' and self.age >= 55:
                self.advance('push_lower')
            elif self.stage == 'push_lower':
                height = .012
                if self.age >= 45:
                    self.advance('push_slide')
            elif self.stage == 'push_slide':
                height = .012
                xy = start + (stop-start)*min(1.,self.age/90)
                if self.age >= 102:
                    self.advance('release')
            self.inverse_kinematics(np.array([*xy,TABLE_Z+height]))
        elif self.stage in ('approach','touch','lift','carry','lower'):
            pick = self.seen[self.color]
            dest = self.seen['green'] - (self.grip_xy if self.grip_xy is not None else 0)
            xy = pick if self.stage in ('approach','touch','lift') else dest
            if self.stage in ('lift','carry'):
                xy = xy - np.array([.012,0])
            height = {'approach':.026,'touch':.004,'lift':.045,'carry':.045,'lower':.017}[self.stage]
            goal = np.array([*xy, TABLE_Z+height])
            self.inverse_kinematics(goal)
            near = np.linalg.norm(tip-goal) < .005
            if self.stage == 'approach' and (near or self.age > 65):
                self.advance('touch')
            elif self.stage == 'touch' and self.latch_on_contact():
                self.advance('lift')
            elif self.stage == 'lift' and self.age >= 38 and tip[2] > TABLE_Z+.035:
                self.advance('carry')
            elif self.stage == 'carry' and self.age >= 40 and np.linalg.norm(tip[:2]-xy) < .006:
                self.advance('lower')
            elif self.stage == 'lower' and self.age >= 30 and near:
                self.d.eq_active[self.held] = False
                self.advance('release')
        elif self.stage == 'release':
            self.angles += np.clip(HOME[5:8]-self.angles,-.03,.03)
            if self.age == 40:
                self.seen.clear()
                self.observe()
                if {self.color,'green'} <= self.seen.keys():
                    self.final_seen = float(np.linalg.norm(self.seen[self.color]-self.seen['green']))
    def step(self):
        self.plan()
        self.used_hold = self.used_hold or bool(self.d.eq_active[self.hold_ids].any())
        target = HOME.copy()
        # Feed forward the calibrated rig's gravity torque. Integrate encoder error
        # slowly to counter servo deadband under load, with bounded anti-windup.
        self.trim = np.clip(self.trim + .018*(self.angles-self.d.qpos[self.head_q]), -.16, .16)
        target[5:8] = np.clip(self.angles + self.d.qfrc_bias[self.head_v]/self.kp + self.trim,
                              self.limits[:,0], self.limits[:,1])
        self.c.step(target, 'stand')
        self.max_height = max(self.max_height, self.d.body(self.target).xpos[2]-self.initial[2])
        self.other_drift = max(self.other_drift, np.linalg.norm((self.d.body(self.other).xpos-self.other_start)[:2]))

    def result(self):
        position = self.d.body(self.target).xpos
        error = float(np.linalg.norm(position[:2]-self.d.geom('pad').xpos[:2]))
        on_table = abs(position[2] - TABLE_Z - .006) < .003
        speed = float(np.linalg.norm(self.d.joint(self.target+'_free').qvel[:3]))
        settled = self.stage == 'release' and self.age >= 25 and on_table and speed < .01
        passed = settled and error <= .015 and self.other_drift < .004
        if self.mode == 'locate':
            passed = self.initial_error is not None and self.initial_error <= .005
        elif self.mode not in ('push', 'push_feedback'):
            passed = passed and self.max_height >= .015 and self.contact_target == self.target
        return dict(success=bool(passed), fixture=self.mode, initial_error_mm=None if self.initial_error is None else round(self.initial_error*1000,2),
                    final_rgb_pad_error_mm=None if self.final_seen is None else round(self.final_seen*1000,2),
                    phase=self.stage, events=self.events, rgb_frames=self.seen_frames, color=self.color,
                    lift_mm=round(self.max_height*1000,2), pad_error_mm=round(error*1000,2),
                    other_drift_mm=round(self.other_drift*1000,2), released_settled=bool(settled),
                    on_table=bool(on_table), end_z_mm=round(float(position[2]-TABLE_Z)*1000,2),
                    grasp_constraint_ever_active=self.used_hold)
