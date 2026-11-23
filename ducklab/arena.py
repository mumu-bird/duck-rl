"""Finite episodes with a separate world-state referee and deployable policy inputs."""
import math
from contextlib import ExitStack

import mujoco
import numpy as np

from .assets import BALL_RADIUS, GOAL_HALF_WIDTH, GOAL_X, load_scene
from .bench import Bench
from .feedback_push import FeedbackPush
from .policy import Controller, HOME_ANGLES
from .video import Video

LABELS = {
    'stand':'原地站稳', 'walk_forward':'向前走', 'walk_backward':'倒车入位',
    'sidestep_left':'向左横移', 'sidestep_right':'向右横移', 'turn_left':'向左转身',
    'turn_right':'向右转身', 'sit_stand':'坐下再站起', 'ground_pick_rollout':'俯身点地再起身',
    'kick_left':'左脚原地点球', 'kick_right':'右脚原地点球', 'penalty_approach':'助跑点球',
    'penalty_baseline':'助跑点球 · 原地对照',
    'push_recovery':'轻推后站稳', 'rgb_locate':'头部相机找色块', 'table_fixture':'接触辅助 · 桌面叼放',
    'table_contact':'仅靠接触 · 桌面叼放', 'table_push':'用嘴把色块推到绿区',
    'table_push_feedback':'分段推送 · 抬头复查',
    'table_push_initial_image':'分段推送 · 只看首帧',
}
BENCH = {'rgb_locate':'locate', 'table_fixture':'fixture','table_contact':'contact_only','table_push':'push',
         'table_push_feedback':'push_feedback', 'table_push_initial_image':'push_feedback'}
WALK = {'walk_forward':(.12,0,0), 'walk_backward':(-.11,0,0), 'sidestep_left':(0,.12,0),
        'sidestep_right':(0,-.12,0),'turn_left':(0,0,.65),'turn_right':(0,0,-.65)}
KICKS = {'kick_left','kick_right','penalty_approach','penalty_baseline'}
STAGES = {'scan':'取图定位','identified':'目标已定位','approach':'接近目标','touch':'寻找接触',
          'lift':'抬头','carry':'横移','lower':'下降','release':'撤回并复查','push_setup':'绕到色块侧面',
          'push_lower':'降低嘴尖','push_slide':'横向推送','inspect':'抬头拍照','align':'对准色块',
          'descend':'降低嘴尖','stroke':'推一小步','lift_clear':'抬起嘴尖'}


def crossing(a,b):
    """First forward crossing of the whole-ball plane, inside the practice posts."""
    line = GOAL_X + BALL_RADIUS
    if not (a[0] < line <= b[0]):
        return None
    impact = a + (b-a)*((line-a[0])/(b[0]-a[0]))
    return bool(abs(impact[1])+BALL_RADIUS <= GOAL_HALF_WIDTH-.006 and impact[2]+BALL_RADIUS <= .116)


class Arena:
    def __init__(self):
        self.stack = ExitStack()
        self.models, self.renderers = {}, {}

    def close(self):
        self.stack.close()

    def model(self, bench):
        if bench not in self.models:
            self.models[bench] = load_scene(bench)
        return self.models[bench]

    def renderer(self, bench, head):
        key = (bench, head)
        if key not in self.renderers:
            width,height = (192,144) if head else (536,328)
            renderer = mujoco.Renderer(self.model(bench), width=width, height=height)
            self.renderers[key] = self.stack.enter_context(renderer)
        return self.renderers[key]

    def run(self, task, seed, video_path=None, approach=None):
        if task not in LABELS:
            raise ValueError(f'unknown task: {task}')
        rng = np.random.default_rng(seed)
        bench = task in BENCH
        model = self.model(bench)
        data = mujoco.MjData(model)
        controller = Controller(model,data)
        initial_jitter = rng.uniform(-.001,.001,len(controller.q))
        data.qpos[controller.q] += initial_jitter
        mujoco.mj_forward(model,data)
        head = self.renderer(True,True) if bench else None
        video = Video(video_path,self.renderer(bench,False),head,LABELS[task],seed) if video_path else None
        ref = Referee(controller)
        setup = {'joint_jitter_max_rad':.001}
        try:
            if bench:
                color = 'red' if seed%2 == 0 else 'blue'
                feedback = BENCH[task] == 'push_feedback'
                if feedback:
                    worker = FeedbackPush(controller,rng,color,head,reobserve=task=='table_push_feedback')
                else:
                    worker = Bench(controller,rng,color,head,BENCH[task])
                setup.update(color=color, target_xy=worker.initial[:2].tolist(),
                             pad_xy=data.geom('pad').xpos[:2].tolist(), other_xy=worker.other_start[:2].tolist(),
                             joint_jitter_rad=initial_jitter.tolist(), trunk_supported=True)
                ticks = worker.max_ticks if feedback else {'locate':60,'push':292}.get(worker.mode,242)
                for tick in range(ticks):
                    worker.step()
                    ref.sample()
                    if video and tick%5 == 4:
                        caption = STAGES[worker.stage]
                        if feedback and not worker.reobserve and worker.stage == 'inspect' and worker.planned_pick is not None:
                            caption = '按上次推程估计位置'
                        video.capture(data,[f'目标　{"红块" if color=="red" else "蓝块"} → 绿区',caption,
                                            '躯干有支撑', '接触辅助已启用' if worker.mode=='fixture' else '夹持辅助已关闭'])
                    if feedback and worker.stage == 'release' and worker.age >= 40:
                        break
                metrics = worker.result()
                task_pass = metrics.pop('success')
            else:
                task_pass,metrics = pitch_episode(task,controller,rng,ref,video,approach,setup)
        except FloatingPointError:
            ref.nonfinite = True
            task_pass,metrics = False, {'phase':'nonfinite'}
        except BaseException:
            if video:
                video.finish()
            raise
        fell = ref.terminal_fall(task)
        safe = not (fell or ref.nonfinite or ref.violations)
        result = dict(task=task, episode=int(seed), seed=int(seed), success=int(bool(task_pass and safe)),
                      task_criterion=int(bool(task_pass)), fell=int(fell), nonfinite=int(ref.nonfinite),
                      joint_limit_violations=ref.violations, max_exceed_deg=round(math.degrees(ref.max_exceed),3),
                      sim_s=round(float(data.time),3), min_root_z=round(ref.min_z,4), setup=setup, metrics=metrics)
        if video:
            video.finish('本回合通过' if result['success'] else '本回合未通过')
        return result


class Referee:
    def __init__(self, controller):
        self.c, self.d = controller,controller.d
        self.limits = controller.m.jnt_range[[controller.m.joint(k).id for k in HOME_ANGLES]]
        self.tip = controller.m.site('mouth_tip').id
        self.base = controller.m.body('trunk_base').id
        self.nonfinite = False
        self.min_z, self.min_tip = 1.,1.
        self.violations, self.max_exceed = 0,0.
        self.end_posture = []

    def sample(self):
        d = self.d
        if not (np.isfinite(d.qpos).all() and np.isfinite(d.qvel).all()) or any(
                d.warning[w].number for w in (mujoco.mjtWarning.mjWARN_BADQPOS,
                                                mujoco.mjtWarning.mjWARN_BADQVEL,mujoco.mjtWarning.mjWARN_BADQACC)):
            raise FloatingPointError('invalid state')
        joints = d.qpos[self.c.q]
        excess = np.maximum(self.limits[:,0]-joints,joints-self.limits[:,1])
        self.violations += int(np.count_nonzero(excess>1e-4))
        self.max_exceed = max(self.max_exceed,float(excess.max()))
        height = float(d.xpos[self.base,2])
        upright = float(d.xmat[self.base,8])
        self.end_posture.append((height,upright))
        self.min_z = min(self.min_z,height)
        self.min_tip = min(self.min_tip,float(d.site_xpos[self.tip,2]))

    def terminal_fall(self,task):
        # Scored after the stand/retreat window (controlled sitting is allowed).
        return bool(self.end_posture and all(z<.075 or up<.45 for z,up in self.end_posture[-10:]))


def pitch_episode(task,c,rng,ref,video,approach,setup):
    m,d = c.m,c.d
    is_kick = task in KICKS
    approach_ticks, speed, settle, foot = (0,0.,0,'kickL' if task=='kick_left' else 'kickR')
    if task=='penalty_approach':
        if approach is None:
            raise ValueError('助跑点球需要先校准参数')
        approach_ticks,speed,settle,foot = (approach[k] for k in ('walk_ticks','speed','settle_ticks','slot'))
    if is_kick:
        if task in ('penalty_baseline','penalty_approach'):
            kick_x, kick_y = .18, -.04
        else:
            kick_y = (.04 if foot == 'kickL' else -.04) + float(rng.uniform(-.003,.003))
            kick_x = .12 + float(rng.uniform(-.002,.002))
        d.joint('ball_free').qpos[:] = [kick_x,kick_y,BALL_RADIUS,1,0,0,0]
        mujoco.mj_forward(m,d)
        setup.update(ball_xyz=[kick_x,kick_y,BALL_RADIUS], walk_ticks=approach_ticks,
                     speed=speed,settle_ticks=settle,slot=foot)
    if task=='push_recovery':
        angle,force = rng.uniform(-math.pi,math.pi),rng.uniform(.75,1.8)
        push = force*np.array([math.cos(angle),math.sin(angle)])
        setup.update(push_newton=push.tolist(), push_seconds=.12)
    ticks = 315 if task in ('ground_pick_rollout','stand') else 240
    if is_kick:
        ticks = 35+approach_ticks+settle+170
    contact = False
    shot = None
    crossing_y = None
    prev = d.xpos[m.body('ball').id].copy()
    ball_id,ball_geom = m.body('ball').id,m.geom('ball').id
    foot_geoms = {m.geom(name).id for name in (('left_foot_collision','left_foot_upper_collision') if foot=='kickL'
                                                else ('right_foot_collision','right_foot_upper_collision'))}
    max_ball_x = float(d.body('ball').xpos[0])

    def shot_referee():
        nonlocal contact,shot,crossing_y,prev,max_ball_x
        point = d.xpos[ball_id].copy()
        if slot == foot:
            for event in d.contact:
                pair = {event.geom1,event.geom2}
                if ball_geom in pair and pair & foot_geoms:
                    contact = True
        if shot is None:
            at_line = crossing(prev,point)
            if at_line is not None:
                shot = bool(contact and at_line)
                crossing_y = float((prev+(point-prev)*((GOAL_X+BALL_RADIUS-prev[0])/(point[0]-prev[0])))[1])
        max_ball_x = max(max_ball_x,float(point[0]))
        prev = point

    for tick in range(ticks):
        active = tick-35
        slot = 'stand'
        cmd = np.zeros(13,np.float32)
        if task in WALK and 0<=active<120:
            slot,cmd[:3] = 'walk',WALK[task]
        elif task=='sit_stand' and 0<=active<140:
            slot,cmd[0] = 'sitstand',int(active<72)
        elif task=='ground_pick_rollout' and 0<=active<200:
            slot = 'groundpick'
            phase = active*np.pi/100
            cmd[:2] = [np.cos(phase),np.sin(phase)]
        elif is_kick and 0<=active<approach_ticks:
            slot,cmd[0] = 'walk',speed
        elif is_kick and 0<=active-approach_ticks-settle<100 and not c.finished:
            slot = foot
        if task=='push_recovery':
            d.xfrc_applied[ref.base,:2] = push if 65<=tick<71 else 0.
        last = c.last.copy() if slot == foot else None
        targets = c.act(slot,cmd)
        if is_kick and slot==foot and c.finished:
            # The published kicker returns control to stand on the recurrent done flag.
            slot = 'stand'
            c.last[:] = last
            targets = c.act(slot,np.zeros(13,np.float32))
        c.step(targets,slot,shot_referee if is_kick else None)
        ref.sample()
        if video and tick%5==4:
            lines = [f'策略　{slot}',f'机身高　{d.qpos[2]*100:.1f} cm']
            if is_kick:
                lines += [f'球的前向位置　{prev[0]*100:.1f} cm',f'指定脚触球　{"是" if contact else "否"}',
                          '已进门' if shot else '等待有效进球']
            video.capture(d,lines,kick=is_kick)
    dx,dy = d.qpos[:2]
    z = float(d.qpos[2])
    yaw = math.atan2(d.xmat[ref.base,3],d.xmat[ref.base,0])
    metrics = dict(dx_m=round(float(dx),4),dy_m=round(float(dy),4),yaw_rad=round(yaw,4),
                   end_z_m=round(z,4), min_bill_m=round(ref.min_tip,4))
    stable = all(z>.085 and up>.85 for z,up in ref.end_posture[-15:])
    if is_kick:
        metrics.update(foot_contact=contact, goal=bool(shot), crossing_y_m=crossing_y,
                       max_ball_x_m=round(max_ball_x,4), kick_completed=bool(c.finished))
        passed = bool(shot) and bool(c.finished)
    elif task in WALK:
        vx,vy,w = WALK[task]
        passed = (dx*np.sign(vx)>.08 if vx else dy*np.sign(vy)>.08 if vy else yaw*np.sign(w)>.35)
    elif task=='sit_stand':
        passed = ref.min_z<.08 and z>.095
    elif task=='ground_pick_rollout':
        passed = ref.min_tip<.012 and d.site_xpos[ref.tip,2]>.18
    else:
        passed = np.hypot(dx,dy)<(.08 if task=='push_recovery' else .02)
    return bool(passed and stable),metrics
