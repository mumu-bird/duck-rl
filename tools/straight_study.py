"""Compare joystick straight walking over fixed forward distances in a clear flat arena."""
import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ducklab.assets import ROBOT
from ducklab.foot_odometry import FootOdometry
from ducklab.policy import Controller, MODELS, POLICIES
from ducklab.straight import LAB_SLIP_CURVE, StraightAssist

MODES = ('official', 'heading', 'straight')
# Side-specific friction is applied to both foot pads, which have priority over the floor.
CASES = [('nominal', v, 1., 1.) for v in (.08, .12, .18)] + [
    ('left_slick', .12, .88, 1.08), ('right_slick', .12, 1.08, .88),
    ('gyro_bias', .12, 1., 1.)]
CHECKPOINTS = (1, 2, 5)


@lru_cache(maxsize=3)
def flat_model(left, right):
    root = ET.parse(ROBOT / 'robot.xml').getroot()
    compiler = root.find('compiler')
    compiler.set('meshdir', str((ROBOT / compiler.get('meshdir', '.')).resolve()))
    ET.SubElement(root.find('worldbody'), 'geom', name='floor', type='plane',
                  size='0 0 .04', conaffinity='5', friction='1 .005 .0001')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    for side, friction in [('left', left), ('right', right)]:
        for pad in ('foot_collision', 'foot_upper_collision'):
            geom = model.geom(f'{side}_{pad}').id
            model.geom_friction[geom, 0] = friction
    return model


def body_yaw(q):
    w, x, y, z = q
    return math.atan2(2*(w*z + x*y), 1-2*(y*y + z*z))


def evaluate(spec):
    (profile, speed, left_mu, right_mu), seed, mode, keep_trace = spec
    model = flat_model(left_mu, right_mu)
    data = mujoco.MjData(model)
    walk = Controller(model, data)
    data.qpos[walk.q] += np.random.default_rng(seed).uniform(-.001, .001, len(walk.q))
    mujoco.mj_forward(model, data)
    command = np.zeros(13, np.float32)
    for _ in range(50):
        walk.step(walk.act('stand', command), 'stand')
    root = model.joint('root_freejoint').qposadr[0]
    origin = data.qpos[root:root+2].copy()
    initial_yaw = body_yaw(data.qpos[root+3:root+7])
    forward = np.array([math.cos(initial_yaw), math.sin(initial_yaw)])
    left = np.array([-forward[1], forward[0]])
    limits = model.jnt_range[model.dof_jntid[walk.v]]
    dt = 4*model.opt.timestep
    bias = math.radians(.12 if seed % 2 else -.12) if profile == 'gyro_bias' else 0.
    estimate = None if mode == 'official' else FootOdometry(model, walk.q)
    assist = StraightAssist(heading_only=mode == 'heading', slip_curve=LAB_SLIP_CURVE)

    def read_imu():
        estimate.observe_imu(data.sensor('orientation').data, data.sensor('imu_ang_vel').data,
                             model.opt.timestep, yaw_rate_bias=bias)

    if estimate is not None:
        read_imu()
    marks, trace = {}, []
    peak = max_excess = 0.
    last_x = last_y = last_heading = 0.
    x = y = yaw_deg = 0.
    state = 'timeout'
    timeout = 3 + 5 / (.6 * speed)
    for tick in range(math.ceil(timeout / dt)):
        twist = (speed, 0., 0.)
        if estimate is not None:
            pose = estimate.observe_encoders(data.qpos[walk.q])
            twist = assist.command(twist, pose, dt)
        command[:3] = twist
        walk.step(walk.act('walk', command), 'walk', read_imu if estimate is not None else None)
        if (not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all() or any(
                data.warning[w].number for w in (mujoco.mjtWarning.mjWARN_BADQPOS,
                mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC))):
            state = 'invalid'
            break
        pos = data.qpos[root:root+2] - origin
        x, y = float(pos @ forward), float(pos @ left)
        quat = data.qpos[root+3:root+7]
        yaw_deg = math.degrees(math.remainder(body_yaw(quat) - initial_yaw, 2*math.pi))
        joints = data.qpos[walk.q]
        max_excess = max(max_excess, float(np.max(limits[:,0] - joints)), float(np.max(joints-limits[:,1])))
        for mark in CHECKPOINTS:
            if mark not in marks and last_x < mark <= x:
                weight = (mark-last_x)/(x-last_x)
                cross = last_y + weight*(y-last_y)
                heading = last_heading + weight*math.remainder(yaw_deg-last_heading, 360.)
                marks[mark] = dict(lateral_cm=100*cross, yaw_deg=heading,
                                   peak_cm=100*max(peak, abs(cross)), time_s=(tick+weight)*dt)
        peak = max(peak, abs(y))
        if keep_trace and tick % 10 == 0:
            trace.append([round((tick+1)*dt, 3), round(x,5), round(y,5)])
        if data.qpos[root+2] < .07 or 1-2*(quat[1]**2 + quat[2]**2) < .55:
            state = 'fell'
            break
        if 5 in marks:
            state = 'reached'
            break
        last_x, last_y, last_heading = x, y, yaw_deg
    row = dict(profile=profile, speed=speed, seed=seed, mode=mode, state=state,
               duration_s=round((tick+1)*dt,3), final_x_m=x, final_y_m=y, peak_attempt_cm=100*peak,
               max_joint_excess_rad=max_excess, scope_ok=max_excess<=1e-4,
               pass_5m=state=='reached' and marks[5]['peak_cm'] <= 5 and max_excess<=1e-4)
    for mark in CHECKPOINTS:
        for key in ('lateral_cm','yaw_deg','peak_cm','time_s'):
            row[f'{key}_{mark}m'] = marks.get(mark, {}).get(key)
    if keep_trace:
        trace.append([row['duration_s'], round(x,5), round(y,5)])
    return row, trace


def summarize(rows):
    groups = []
    for profile, speed, *_ in CASES:
        for mode in MODES:
            group = [r for r in rows if (r['profile'],r['speed'],r['mode'])==(profile,speed,mode)]
            if not group:
                continue
            stats = dict(profile=profile, speed=speed, mode=mode, episodes=len(group),
                         fell=sum(r['state']=='fell' for r in group), timeout=sum(r['state']=='timeout' for r in group),
                         invalid=sum(r['state']=='invalid' for r in group), limits=sum(not r['scope_ok'] for r in group),
                         passed_5m=sum(r['pass_5m'] for r in group))
            for mark in CHECKPOINTS:
                reached = [r for r in group if r[f'lateral_cm_{mark}m'] is not None]
                stats[f'reached_{mark}m'] = len(reached)
                for key in ('lateral_cm','yaw_deg','peak_cm'):
                    vals = [abs(r[f'{key}_{mark}m']) for r in reached]
                    stats[f'median_abs_{key}_{mark}m'] = round(float(np.median(vals)),2) if vals else None
                    stats[f'max_abs_{key}_{mark}m'] = round(max(vals),2) if vals else None
            groups.append(stats)
    return groups


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, default=None)
    ap.add_argument('--seed', type=int, default=67200)
    ap.add_argument('--count', type=int, default=12, help='paired seeds per condition')
    ap.add_argument('--jobs', type=int, default=1)
    ap.add_argument('--quick', action='store_true', help='only standard floor at 0.12 m/s')
    args = ap.parse_args()
    if args.seed<0 or args.count<1 or args.jobs<1:
        ap.error('require a nonnegative seed, positive count and positive jobs')
    cases = [CASES[1]] if args.quick else CASES
    dest = args.output or ROOT/'runs'/('straight-'+datetime.now().strftime('%Y%m%d-%H%M%S'))
    dest.mkdir(parents=True, exist_ok=False)
    specs = [(case, seed, mode, seed==args.seed and case==CASES[1]) for case in cases
             for seed in range(args.seed, args.seed+args.count) for mode in MODES]
    rows, traces = [], {}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool, (dest/'episodes.csv').open('w',newline='') as out:
        writer = None
        for row, trace in pool.map(evaluate, specs):
            if writer is None:
                writer = csv.DictWriter(out, fieldnames=row); writer.writeheader()
            writer.writerow(row); out.flush(); rows.append(row)
            if trace:
                traces[row['mode']] = trace
    (dest/'traces.json').write_text(json.dumps(traces))
    policy = POLICIES/'policies'/MODELS['walk']['file']
    result = dict(seed=args.seed,count=args.count,policy=policy.name,
                  policy_sha256=hashlib.sha256(policy.read_bytes()).hexdigest(), mujoco=mujoco.__version__,
                  hash_files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in (ROBOT/'robot.xml',ROOT/'ducklab/straight.py',ROOT/'ducklab/foot_odometry.py')},
                  reference='start pose after standing 1 s', gyro_test_deg_s=.12,
                  corridor_half_width_cm=5, cases=cases, groups=summarize(rows))
    (dest/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    print(f'{len(rows)} episodes, {dest}',flush=True)

if __name__=='__main__':
    main()
