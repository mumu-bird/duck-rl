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

    def __init__(self, walk_path, stand_path):
        import onnxruntime as ort
        self.walk = ort.InferenceSession(str(walk_path))
        self.stand = ort.InferenceSession(str(stand_path))
        self.walk_input = self.walk.get_inputs()[0].name
        self.stand_input = self.stand.get_inputs()[0].name

    def act(self, obs):
        obs = np.asarray(obs, dtype=np.float32)[None, :]
        twist = obs[0, 48:51]
        if np.abs(twist).max() < 0.05:
            out = self.stand.run(None, {self.stand_input: obs})[0]
        else:
            out = self.walk.run(None, {self.walk_input: obs})[0]
        return np.asarray(out, dtype=np.float64).reshape(-1)


def run_rollout(env, policy, expert=None, greedy=False, max_ticks=None):
    """One episode; returns transition lists plus a summary."""
    obs = env.reset()
    obs_l, act_l, lp_l, v_l, r_l = [], [], [], [], []
    ticks = 0
    limit = max_ticks or env.max_ticks
    while ticks < limit:
        if expert is not None:
            action = expert.act(obs)
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
    return obs_l, act_l, lp_l, v_l, r_l, info, ticks


def evaluate(env, policy, episodes=6, greedy=True):
    results = []
    for _ in range(episodes):
        _, _, _, _, _, info, ticks = run_rollout(env, policy, greedy=greedy)
        results.append(dict(success=bool(info["success"]), ticks=ticks,
                            fall=bool(info["fall"]), obj_z=info["obj_z"]))
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
    ap.add_argument("--bc-episodes", type=int, default=60,
                    help="expert rollouts for the behaviour-cloning warm start")
    ap.add_argument("--bc-epochs", type=int, default=10)
    ap.add_argument("--expert", default=OFFICIAL_WALK)
    ap.add_argument("--dagger-episodes", type=int, default=48)
    ap.add_argument("--dagger-rounds", type=int, default=4)
    ap.add_argument("--no-bc", action="store_true")
    ap.add_argument("--update-interval", type=int, default=16,
                    help="episodes per PPO update")
    ap.add_argument("--eval-every", type=int, default=10, help="updates between evals")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()

    stamp = time.strftime("%Y%m%d-%H%M%S")
    out = Path(args.out or f"runs/deploy/{args.task}-{stamp}")
    out.mkdir(parents=True, exist_ok=True)
    save_json(out / "config.json", vars(args))

    policy = DeployPPO(device=args.device)
    if args.resume:
        policy.load(args.resume)
    # one immutable model shared by all workers (each keeps its own MjData)
    model, _, layout = scene.build(object_kind=args.object or
                                   DeployEnv.TASKS[args.task]["object"])
    shared = (model, layout)
    envs = [DeployEnv(args.task, seed=args.seed * 1000 + i, object_kind=args.object,
                      shared=shared)
            for i in range(args.workers)]

    # ---------------------------------------------------------- BC warm start
    if not args.no_bc and args.bc_episodes > 0:
        stand = Path(args.expert).with_name("standing_m6_013_002_p15_19611.onnx")
        expert = Expert(args.expert, stand)
        print(f"[bc] 官方行走策略预热：{args.bc_episodes} 回合 x {args.workers} 并行",
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
            mse2 = policy.behaviour_clone(
                np.array(dagger_obs), np.array(dagger_act), epochs=args.bc_epochs
            )
            obs_all.extend(dagger_obs)
            act_all.extend(dagger_act)
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
                         object_kind=args.object, shared=shared)

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
                            print(f"[eval] {ev['successes']}/{ev['episodes']}", flush=True)
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
