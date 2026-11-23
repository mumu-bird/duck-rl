"""Inference adapter for the published Wonder Duck simulator policies (50 Hz)."""
from functools import lru_cache
import json
import mujoco
import numpy as np
import onnxruntime as ort
from .assets import ROOT

POLICIES = ROOT / 'vendor/wonderduck'
MODELS = json.loads((POLICIES/'simulator/active-models.json').read_text())['models']
HOME_ANGLES = dict(zip(('left_hip_yaw','left_hip_roll','left_hip_pitch','left_knee','left_ankle',
                        'neck_pitch','head_pitch','head_yaw','head_roll','right_hip_yaw',
                        'right_hip_roll','right_hip_pitch','right_knee','right_ankle'),
                       (-.0107378652,-.073631078,-.386563152,0,.386563152,.411106855,.349747628,
                        0,0,.0107378652,.073631078,.386563152,0,-.386563152), strict=True))
HOME = np.asarray(list(HOME_ANGLES.values()))


@lru_cache
def session(slot):
    opts = ort.SessionOptions(); opts.intra_op_num_threads = opts.inter_op_num_threads = 1
    return ort.InferenceSession(str(POLICIES/'policies'/MODELS[slot]['file']), opts,
                                providers=['CPUExecutionProvider'])


class Controller:
    def __init__(self, m, d):
        self.m, self.d = m, d
        names = list(HOME_ANGLES)
        self.q = [m.joint(n).qposadr[0] for n in names]
        self.v = [m.joint(n).dofadr[0] for n in names]
        self.a = [m.actuator(n).id for n in names]
        root = m.joint('root_freejoint').qposadr[0]
        d.qpos[root:root+7] = (0, 0, .111848115018488, 1, 0, 0, 0)
        self.d.qpos[self.q] = HOME
        self.last = np.zeros(14, np.float32)
        self.delay = np.tile(HOME, (5, 1)); self.filtered = HOME.copy(); self.tick = 0
        self.memory = {k: np.zeros((1,1,5),np.float32) for k in ('h_in','c_in')}
        self.finished = False
        mujoco.mj_forward(m,d)

    def act(self, slot, command):
        session_ = session(slot); d = self.d
        gx, gy, gz = d.sensor('imu_ang_vel').data
        gravity = -d.body('trunk_base').xmat.reshape(3,3)[2]
        obs = np.concatenate(([-gz,gy,gx], gravity, d.qpos[self.q]-HOME, d.qvel[self.v], self.last, command))
        feeds = {a.name: (self.memory[a.name] if a.name in self.memory else obs[None].astype(np.float32))
                 for a in session_.get_inputs()}
        out = dict(zip([x.name for x in session_.get_outputs()], session_.run(None, feeds),strict=True))
        if 'h_out' in out:
            self.memory = {'h_in': out.pop('h_out'), 'c_in':out.pop('c_out')}
            if not all(np.isfinite(value).all() for value in self.memory.values()):
                raise FloatingPointError('non-finite recurrent state')
            self.finished = self.memory['h_in'].flat[4] == 1
        self.last[:] = next(iter(out.values())).reshape(14)
        if not np.isfinite(self.last).all():
            raise FloatingPointError('non-finite action')
        return HOME + self.last * MODELS[slot]['actionScale']

    def step(self, targets, slot, observer=None):
        p = MODELS[slot]['calibration']; kt = p['kt']
        for _ in range(4):
            self.delay[self.tick] = targets; self.tick = (self.tick+1)%5
            self.filtered += np.clip(self.delay[self.tick]-self.filtered, -.5, .5)
            err = self.filtered - self.d.qpos[self.q]
            voltage = np.where(abs(err) <= .003067961575771,0,np.clip(.166*p['firmwareP']*p['error_gain_ratio']*err,-1,1))*7.7
            self.d.ctrl[self.a] = kt*(voltage - kt*self.d.qvel[self.v])/p['R']
            mujoco.mj_step(self.m,self.d)
            if observer is not None:
                observer()
