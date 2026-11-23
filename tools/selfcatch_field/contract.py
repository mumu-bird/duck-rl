"""Offline ablations of the frozen self-catch actor; never opens a robot connection."""

import argparse
from functools import lru_cache
import hashlib
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from ducklab.policy import HOME, Controller  # noqa: E402
from ducklab.self_catch import OPEN, SelfCatchEnv  # noqa: E402
from tools.train_self_catch import Policy, torch  # noqa: E402

FROZEN = ROOT / 'runs/self-catch-ppo/short-hop-04/selected.pt'
SHA = '0f8a47cc5dde3d3e56f2001d78dcb007f20147f89db2d55b9c2f37ae4af37a17'
HANDOFF = Path.home() / 'WonderDuck机器鸭/WonderDuck机器鸭/3 源码资料/wonderduck-simulator-handoff-20261113-144034.tar.gz'
GATES = {'walk': 'f93d22be84b47a74496f45348179232cbabff8880f766ecb9476e026e8052a66',
         'stand': 'da2ac6be8a86ef89ea2aa00a68398c0819d6cfa612b0c3ef54737fe333c4ad36'}
MODES = ('trained_5ms', 'direct_20ms', 'stock_intent_20ms', 'handoff_intent_20ms')


@lru_cache(maxsize=1)
def factory_bundle():
    import onnxruntime as ort
    sessions = {}
    with tarfile.open(HANDOFF) as tar:
        prefix = HANDOFF.name.removesuffix('.tar.gz') + '/resources/policies/'
        for m in json.load(tar.extractfile(prefix + 'manifest.json'))['models']:
            if (slot := m['slot']) not in GATES:
                continue
            raw = tar.extractfile(prefix + m['file']).read()
            if hashlib.sha256(raw).hexdigest() != GATES[slot]:
                raise ValueError('unexpected handoff gait weights')
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = opts.inter_op_num_threads = 1
            sess = ort.InferenceSession(raw, opts, providers=['CPUExecutionProvider'])
            meta = sess.get_modelmeta().custom_metadata_map
            if not np.allclose(np.fromstring(meta['default_joint_pos'], sep=','), HOME):
                raise ValueError('incompatible HOME')
            sessions[slot] = sess
    if sessions.keys() != GATES.keys():
        raise ValueError('incomplete gait set')
    return sessions


class IntentController(Controller):
    """Translate post-policy absolute neck/head targets into pre-policy HOME offsets.

    The parent environment edits the returned array before step(). Preview the neutral
    output to recover its preparation interpolation; advance the last-action history
    exactly once using the actual command, never using the preview as a previous action.
    """

    def __init__(self, m, d):
        super().__init__(m, d)
        self.filtered_cmd = np.zeros(13, np.float32)

    def act(self, slot, command):
        self.before = self.last.copy()
        self.intent = command.copy()
        self.preview = self.predict(slot, command).copy()
        return self.preview.copy()

    def predict(self, slot, command):
        if not getattr(self, 'factory', False):
            return super().act(slot, command)
        d = self.d
        gx, gy, gz = d.sensor('imu_ang_vel').data
        obs = np.r_[[-gz, gy, gx], -d.body('trunk_base').xmat.reshape(3, 3)[2],
                    d.qpos[self.q]-HOME, d.qvel[self.v], self.last, command]
        self.last[:] = factory_bundle()[slot].run(['actions'], {'obs': obs[None].astype(np.float32)})[0][0]
        return HOME + self.last

    def step(self, targets, slot, observer=None):
        self.last[:] = self.before
        desired = self.intent.copy()
        # Initial ten idle ticks don't command a head pose at all.
        if not np.array_equal(targets[5:7], self.preview[5:7]):
            desired[3:5] = targets[5:7] - HOME[5:7]
        self.filtered_cmd += .2 * (desired - self.filtered_cmd)
        slot = 'stand' if np.linalg.norm(self.filtered_cmd[:3]) <= .05 else 'walk'
        absolute = self.predict(slot, self.filtered_cmd)
        super().step(absolute, slot, observer)


class TransferEnv(SelfCatchEnv):
    """Keep actor scales unchanged while varying the actuator delivery path only."""

    def __init__(self, mode):
        if mode not in MODES:
            raise ValueError(f'unknown proxy mode: {mode}')
        super().__init__(fine_jaw=True)
        self.mode = mode

    def reset(self, seed, record=False):
        obs = super().reset(seed, record)
        if self.mode in ('stock_intent_20ms', 'handoff_intent_20ms'):
            # Controller.__init__ sets the home pose. Keep the randomized state exact.
            q = self.d.qpos.copy()
            self.c = IntentController(self.m, self.d)
            self.c.factory = self.mode == 'handoff_intent_20ms'
            self.d.qpos[:] = q
            import mujoco
            mujoco.mj_forward(self.m, self.d)
        return obs

    def tick(self, k, nc=0., hc=0., recover=99, close=99, catch_speed=None):
        if self.mode == 'trained_5ms':
            return super().tick(k, nc, hc, recover, close, catch_speed)
        # Latch the exact same opening command on the next 50 Hz tick, not in a substep.
        try:
            self.fine_jaw = False
            return super().tick(k, nc, hc, recover, close, catch_speed)
        finally:
            self.fine_jaw = True


def load_actor():
    if hashlib.sha256(FROZEN.read_bytes()).hexdigest() != SHA:
        raise ValueError('frozen checkpoint hash mismatch')
    policy = Policy().eval()
    policy.load_state_dict(torch.load(FROZEN, weights_only=True, map_location='cpu'))
    return policy


def evaluate(policy, mode, seeds):
    env = TransferEnv(mode)
    rows = []
    for seed in seeds:
        obs = env.reset(seed)
        for phase in range(2):
            with torch.inference_mode():
                u = policy.actors[phase](torch.from_numpy(obs)).tanh().numpy()
            obs, _, _, info = env.step(u)
        rows.append(dict(seed=seed, **info))
    return dict(n=len(rows), **{k: sum(r[k] for r in rows) for k in
                ('armed', 'upright', 'released', 'recaught', 'success', 'ground')}, rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--episodes', type=int, default=100)
    p.add_argument('--seed', type=int, default=5200000)
    a = p.parse_args()
    if a.episodes < 1:
        p.error('need at least one seed')
    a.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    policy = load_actor()
    report = dict(hardware=False, sha256=SHA, fine_jaw_rad=OPEN, low_level={
        f: hashlib.sha256((ROOT / 'vendor/wonderduck/policies' / f).read_bytes()).hexdigest()
        for f in ('stand.onnx', 'velocity.onnx')}, modes={})
    report['handoff_gaits'] = GATES
    for mode in MODES:
        r = evaluate(policy, mode, range(a.seed, a.seed + a.episodes))
        report['modes'][mode] = r
        print(mode, {k: v for k, v in r.items() if k != 'rows'}, flush=True)
    (a.out / 'transfer.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
