#!/usr/bin/env python3
"""Train deployable WonderDuck policies against the on-robot contract.

Tasks
-----
duck_lift    pick up the LINLEE duck: walk up, bill over the duck, beak close
             (rule controller, bus servo 34), lift and hold 1 s.
toss_catch   a ball/cube tossed from 0.9..1.6 m toward a point just ahead: walk
             forward to meet it and catch it in the beak before it lands, hold 1 s.

Both tasks are expressed purely through the deployed contract (obs[61] ->
actions[14], 50 Hz over 200 Hz physics, fitted servo law with the official
delay). The policy never sees the object; the command block carries the tracker
output, exactly as on the robot. The exported artefact is a single ONNX that
loads into any robotd policy slot.

Usage
-----
python3 tools/train_deploy.py --task toss_catch --episodes 4000 --workers 8
python3 tools/train_deploy.py --task duck_lift --object linlee_duck --episodes 4000
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

torch.set_num_threads(1)  # rollouts already parallelise across workers

from ducklab.deploy import contract as C
from ducklab.deploy.env import DeployEnv
from ducklab.deploy import scene
from ducklab.deploy.ppo import DeployPPO

OFFICIAL_WALK = (
    "vendor/official-handoff-20261113/wonderduck-simulator-handoff-20261113-144034"
    "/resources/policies/velocity_teacher_m6_013_004_p15_187251.onnx"
)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".partial")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


class Expert:
    """The official on-robot policies, used for the BC warm start.

    Mirrors the runtime's selection (control.rs): the stand net owns small
    command magnitudes, the walk net owns movement. Behaviour-cloning both
    gives the student the stand/walk switch the deployed contract expects.
    """

    def __init__(self, walk_path, stand_path, pick_path=None):
        import onnxruntime as ort
        self.walk = ort.InferenceSession(str(walk_path))
        self.stand = ort.InferenceSession(str(stand_path))
        self.walk_input = self.walk.get_inputs()[0].name
        self.stand_input = self.stand.get_inputs()[0].name
        self.pick = (ort.InferenceSession(str(pick_path)) if pick_path else None)
        self.pick_input = self.pick.get_inputs()[0].name if self.pick else None
        self.pick_phase = None

    def act(self, obs, ctx=None):
        """Runtime-style selection: walk when moving, ground-pick when arrived
        and over the object (phase-clocked, as the official runtime drives it),
        stand otherwise. The pick policy is what teaches the head-down motion
        the student would otherwise have to discover from scratch."""
        obs1 = np.asarray(obs, dtype=np.float32)[None, :]
        twist = obs1[0, 48:51]
        if (self.pick is not None and ctx is not None
                and np.abs(twist).max() < 0.05 and ctx["tip_dist"] < 0.16):
            if self.pick_phase is None:
                self.pick_phase = 0.0
            o = obs1.copy()
            o[0, 48] = np.cos(2 * np.pi * self.pick_phase)
            o[0, 49] = np.sin(2 * np.pi * self.pick_phase)
            o[0, 50] = 0.0
            o[0, 51:61] = 0.0
            self.pick_phase += 0.02 / 4.0
            if self.pick_phase >= 0.7:
                self.pick_phase = None
            out = self.pick.run(None, {self.pick_input: o})[0]
            return np.asarray(out, dtype=np.float64).reshape(-1)
        self.pick_phase = None
        if np.abs(twist).max() < 0.05:
            out = self.stand.run(None, {self.stand_input: obs1})[0]
        else:
            out = self.walk.run(None, {self.walk_input: obs1})[0]
        return np.asarray(out, dtype=np.float64).reshape(-1)


def run_rollout(env, policy, expert=None, greedy=False, max_ticks=None):
    """One episode; returns transition lists plus a summary."""
    obs = env.reset()
    obs_l, act_l, lp_l, v_l, r_l = [], [], [], [], []
    ticks = 0
    limit = max_ticks or env.max_ticks
    while ticks < limit:
        if expert is not None:
            action = expert.act(obs, env.task_ctx())
            log_prob, value = 0.0, 0.0
        else:
            action, log_prob, value = policy.act_batch(obs[None, :], deterministic=greedy)
            action = action[0]
            log_prob = float(log_prob[0]) if log_prob is not None else 0.0
            value = float(value[0])
        next_obs, reward, done, info = env.step(action)
        obs_l.append(obs)
        act_l.append(action)
        lp_l.append(log_prob)
        v_l.append(value)
        r_l.append(reward)
        obs = next_obs
        ticks += 1
        if done:
            break
    info = dict(info)
    info["min_horiz"] = getattr(env, "_min_horiz", 9.0)
    info["min_tip"] = getattr(env, "_min_tip", 9.0)
    return obs_l, act_l, lp_l, v_l, r_l, info, ticks


def evaluate(env, policy, episodes=6, greedy=True):
    results = []
    for _ in range(episodes):
        _, _, _, _, _, info, ticks = run_rollout(env, policy, greedy=greedy)
        results.append(dict(
            success=bool(info["success"]), ticks=ticks,
            fall=bool(info["fall"]), obj_z=info["obj_z"],
            min_horiz=round(info.get("min_horiz", 9.0), 3),
            min_tip=round(info.get("min_tip", 9.0), 3),
            hold_ticks=info.get("hold_ticks", 0),
        ))
    successes = sum(r["success"] for r in results)
    return dict(episodes=len(results), successes=successes,
                success_rate=successes / max(len(results), 1), rows=results)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--task", choices=["duck_lift", "toss_catch"], required=True)
    ap.add_argument("--object", default=None,
                    help="ball | cube | linlee_duck (default per task)")
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--target-kl", type=float, default=0.015)
    ap.add_argument("--bc-episodes", type=int, default=60,
                    help="expert rollouts for the behaviour-cloning warm start")
    ap.add_argument("--bc-epochs", type=int, default=10)
    ap.add_argument("--expert", default=OFFICIAL_WALK)
    ap.add_argument("--dagger-episodes", type=int, default=48)
    ap.add_argument("--dagger-rounds", type=int, default=4)
    ap.add_argument("--anchor-coef", type=float, default=0.5,
                    help="imitation regulariser toward the expert during PPO")
    ap.add_argument("--anchor-decay", type=int, default=150,
                    help="updates over which the anchor decays to zero")
    ap.add_argument("--no-bc", action="store_true")
    ap.add_argument("--update-interval", type=int, default=16,
                    help="episodes per PPO update")
    ap.add_argument("--eval-every", type=int, default=10, help="updates between evals")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--video-every", type=int, default=10,
                    help="updates between training-progress videos (0 disables)")
    ap.add_argument("--freeze-body", action="store_true",
                    help="phase 1: clamp leg joints to home, learn head-down only")
    ap.add_argument("--curriculum", default=None,
                    help="'far,near' metres: anneal object distance over training")
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out or f"runs/deploy/{args.task}-{stamp}")
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config.json", vars(args))

    expert = None
    if not args.no_bc and args.bc_episodes > 0:
        stand = Path(args.expert).with_name("standing_m6_013_002_p15_19611.onnx")
        pick = Path(args.expert).with_name("ground_pick_m6_013_004_p15_13750.onnx")
        expert = Expert(args.expert, stand, pick if pick.is_file() else None)

    def make_anchor(expert):
        def anchor(obs_batch):
            return np.stack([expert.act(o) for o in obs_batch])
        return anchor

    policy = DeployPPO(
        device=args.device, lr=args.lr, target_kl=args.target_kl,
        anchor=make_anchor(expert) if expert is not None else None,
        anchor_coef=args.anchor_coef,
    )
    if args.resume:
        policy.load(args.resume)
    # one immutable model shared by all workers (each keeps its own MjData)
    model, _, layout = scene.build(object_kind=args.object or
                                   DeployEnv.TASKS[args.task]["object"])
    shared = (model, layout)
    curriculum = None
    if args.curriculum:
        far, near = (float(v) for v in args.curriculum.split(","))
        curriculum = (far, near)
    envs = [DeployEnv(args.task, seed=args.seed * 1000 + i, object_kind=args.object,
                      shared=shared, curriculum=curriculum,
                      freeze_body=args.freeze_body)
            for i in range(args.workers)]

    # ---------------------------------------------------------- BC warm start
    if not args.no_bc and args.bc_episodes > 0:
        print(f"[bc] 官方行走/站立策略预热：{args.bc_episodes} 回合 x {args.workers} 并行",
              flush=True)
        obs_all, act_all = [], []
        # One task per env per wave: submitting more tasks than workers would let
        # two threads share one MjModel/MjData pair, which corrupts the solver.
        waves = (args.bc_episodes + args.workers - 1) // args.workers
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for wave in range(waves):
                lo = wave * args.workers
                hi = min(lo + args.workers, args.bc_episodes)
                futures = [
                    pool.submit(run_rollout, envs[i - lo], None, expert=expert)
                    for i in range(lo, hi)
                ]
                for f in futures:
                    obs_l, act_l, *_ = f.result()
                    obs_all.extend(obs_l)
                    act_all.extend(act_l)
        mse = policy.behaviour_clone(obs_all, act_all, epochs=args.bc_epochs)
        print(f"[bc] 专家轮：{len(obs_all)} 帧，MSE {mse:.5f}", flush=True)

        # DAgger rounds: roll the student out and relabel every visited state
        # with the expert. Closes the compounding-error gap that makes a pure
        # imitation student fall within a second even at low MSE. Measured on
        # the toss_catch task: 0.8 s -> 2.4 s -> 5.0 s (episode limit) over
        # three rounds.
        for rnd in range(args.dagger_rounds):
            dagger_obs, dagger_act = [], []
            waves = (args.dagger_episodes + args.workers - 1) // args.workers
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                for wave in range(waves):
                    lo = wave * args.workers
                    hi = min(lo + args.workers, args.dagger_episodes)
                    futures = [
                        pool.submit(run_rollout, envs[i - lo], policy, greedy=False)
                        for i in range(lo, hi)
                    ]
                    for f in futures:
                        obs_l, _, _, _, _, _, _ = f.result()
                        dagger_obs.extend(obs_l)
                        dagger_act.extend(expert.act(o) for o in obs_l)
            obs_all.extend(dagger_obs)
            act_all.extend(dagger_act)
            # BC on the ACCUMULATED dataset: training on the new frames alone
            # forgets the expert behaviour within a round or two.
            mse2 = policy.behaviour_clone(
                np.array(obs_all), np.array(act_all), epochs=args.bc_epochs
            )
            print(f"[bc] DAgger 轮 {rnd}: {len(dagger_obs)} 帧，MSE {mse2:.5f}",
                  flush=True)
            save_json(out / "bc.json", dict(
                expert_frames=len(obs_all) - len(dagger_obs) * (rnd + 1),
                expert_mse=mse, dagger_rounds=rnd + 1,
                dagger_frames=len(dagger_obs), dagger_mse=mse2,
            ))

    # ---------------------------------------------------------- PPO
    print(f"[ppo] 开始训练：{args.episodes} 回合，{args.workers} 并行 worker，"
          f"设备 {policy.device}", flush=True)
    started = time.time()
    episodes_done = 0
    best_rate = -1.0
    history = []
    eval_env = DeployEnv(args.task, seed=args.seed * 7919 + 5,
                         object_kind=args.object, shared=shared,
                         curriculum=curriculum, freeze_body=args.freeze_body)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        # every worker runs one episode at a time; the main thread batches actions
        pending = {i: pool.submit(run_rollout, envs[i], policy) for i in range(args.workers)}
        while episodes_done < args.episodes:
            ready = []
            # collect whichever worker finishes first without busy-waiting long
            import concurrent.futures as cf
            done_set, _ = cf.wait(list(pending.values()), return_when=cf.FIRST_COMPLETED)
            for env_index, future in list(pending.items()):
                if future in done_set:
                    obs_l, act_l, lp_l, v_l, r_l, info, ticks = future.result()
                    policy.store_episode(obs_l, act_l, lp_l, v_l, r_l,
                                         next_value=0.0, done=True)
                    episodes_done += 1
                    if episodes_done % args.update_interval == 0:
                        # Anchor decay: the expert prior contains no head-down
                        # behaviour, so a constant anchor actively prevents
                        # learning the very skill the task needs. Keep it early
                        # (protect the standing prior), then hand over to the
                        # task reward.
                        if curriculum is not None:
                            progress = min(1.0, episodes_done / args.episodes)
                            for env in (*envs, eval_env):
                                env.set_curriculum_progress(progress)
                        if args.anchor_decay > 0:
                            policy.anchor_coef = args.anchor_coef * max(
                                0.0, 1.0 - policy.update_count / args.anchor_decay)
                        stats = policy.update()
                        record = dict(update=policy.update_count, episodes=episodes_done,
                                      elapsed_s=round(time.time() - started, 1), **stats)
                        history.append(record)
                        print(f"[ppo] {record}", flush=True)
                        save_json(out / "training.json", history)
                        torch_save = policy.state_dict()
                        import torch
                        torch.save(dict(model=torch_save, update=policy.update_count),
                                   out / "policy-latest.pt")
                        if policy.update_count % args.eval_every == 0:
                            ev = evaluate(eval_env, policy)
                            record["eval"] = ev
                            print(f"[eval] {ev['successes']}/{ev['episodes']} "
                                  f"存活 {sum(r['ticks'] for r in ev['rows']) // max(ev['episodes'], 1)}",
                                  flush=True)
                            if args.video_every > 0 and \
                                    policy.update_count % args.video_every == 0:
                                try:
                                    from ducklab.deploy.video import record_episode
                                    metrics = {
                                        "update": policy.update_count,
                                        "episodes": episodes_done,
                                        "success_rate": round(ev["success_rate"], 2),
                                        "mean_ticks": round(
                                            sum(r["ticks"] for r in ev["rows"])
                                            / max(ev["episodes"], 1)),
                                        "min_tip_dist": round(
                                            min(r["min_tip"] for r in ev["rows"]), 3),
                                    }
                                    mp4, png = record_episode(
                                        eval_env,
                                        lambda o: policy.act_batch(
                                            o[None, :], deterministic=True)[0][0],
                                        out / "videos" /
                                        f"stage-{policy.update_count:04d}.mp4",
                                        label=f"update {policy.update_count} ({args.task})",
                                        metrics=metrics,
                                    )
                                    print(f"[video] {mp4.name}", flush=True)
                                except Exception as exc:
                                    print(f"[video] 跳过: {exc}", flush=True)
                            save_json(out / "evaluations.json",
                                      [h for h in history if "eval" in h])
                            if ev["success_rate"] > best_rate:
                                best_rate = ev["success_rate"]
                                torch.save(dict(model=policy.state_dict(),
                                                update=policy.update_count),
                                           out / "policy-best.pt")
                                policy.export_onnx(out / "policy-best.onnx")
                    if episodes_done < args.episodes:
                        pending[env_index] = pool.submit(run_rollout, envs[env_index], policy)

    summary = dict(task=args.task, episodes=episodes_done,
                   best_eval_success_rate=best_rate,
                   elapsed_s=round(time.time() - started, 1),
                   device=str(policy.device))
    save_json(out / "summary.json", summary)
    policy.save(out / "policy-final.pt")
    policy.export_onnx(out / "policy-final.onnx")
    print(f"[done] {summary}", flush=True)
    print(f"输出目录: {out}", flush=True)


if __name__ == "__main__":
    main()
