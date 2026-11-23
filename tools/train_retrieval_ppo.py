#!/usr/bin/env python3
"""Train a bounded phase-level PPO policy for simulated pickup and delivery."""

import argparse
from contextlib import ExitStack
import json
import multiprocessing as mp
import os
from pathlib import Path
import queue
import random
import sys
import subprocess
import time
import traceback

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
import torch

from ducklab.assets import ROOT
from ducklab.retrieval.episode import Episode
from ducklab.retrieval.grasp import load_profiles
from ducklab.retrieval.ppo import PPOPolicy
from ducklab.retrieval.scene import BIN, HEIGHT, WIDTH, OBJECTS, make_scene
from ducklab.video import Video


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def make_episode_scene(item, bin_xy, mass_grams=4.0, friction=None):
    """Build the measured/candidate payload and randomize every parcel contact.

    Explicit MuJoCo contact pairs override geom friction. Update both so the
    sampled coefficient reaches the bill, not just the floor contact.
    """
    if not np.isfinite(mass_grams) or mass_grams <= 0:
        raise ValueError("mass_grams must be finite and positive")
    if friction is not None and (not np.isfinite(friction) or friction <= 0):
        raise ValueError("friction must be finite and positive")
    model, half = make_scene({**item, "mass_kg": mass_grams / 1000.0}, bin_xy,
                             moving_jaw=True)
    if friction is not None:
        parcel_geoms = model.geom_bodyid == model.body("parcel").id
        model.geom_friction[parcel_geoms, 0] = friction
        pairs = parcel_geoms[model.pair_geom1] | parcel_geoms[model.pair_geom2]
        model.pair_friction[pairs, :2] = friction
    return model, half


def run_episode(
    item,
    seed,
    bin_xy,
    detector,
    profiles,
    policy,
    video_path=None,
    label="PPO",
    friction=None,
    mass_grams=4.0,
):
    model, half = make_episode_scene(item, bin_xy, mass_grams, friction)
    with ExitStack() as stack:
        camera = stack.enter_context(mujoco.Renderer(model, HEIGHT, WIDTH))
        video = None
        if video_path is not None:
            wide = stack.enter_context(mujoco.Renderer(model, 320, 536))
            video = Video(
                video_path,
                wide,
                camera,
                f"拾取 PPO · {item['id']} · {label}",
                seed,
                arena_view=(
                    [bin_xy[0] / 2, bin_xy[1] / 2, 0.078],
                    max(0.90, 1.9 * max(map(abs, bin_xy))),
                ),
            )
        try:
            result = Episode(
                model,
                half,
                camera,
                detector,
                "jaw",
                seed,
                bin_xy,
                video=video,
                profiles=profiles,
                policy=policy,
            ).run()
        except BaseException:
            if video is not None:
                video.finish()
            raise
        if video is not None:
            video.finish("完整通过" if result["success"] else "本轮未通过")
        result.update(mass_grams=float(mass_grams), contact_friction=friction)
        return result


class _RemoteServices:
    def __init__(self, worker_id, episode, requests, responses):
        self.worker_id = worker_id
        self.episode = episode
        self.requests = requests
        self.responses = responses
        self.next_id = 0

    def request(self, kind, payload):
        request_id = self.next_id
        self.next_id += 1
        self.requests.put(
            {
                "type": "request",
                "kind": kind,
                "worker": self.worker_id,
                "episode": self.episode,
                "request_id": request_id,
                "payload": payload,
            }
        )
        response_id, result = self.responses.get()
        if response_id != request_id:
            raise RuntimeError(
                f"worker {self.worker_id} expected response {request_id}, got {response_id}"
            )
        return result


class _RemoteDetector:
    def __init__(self, services):
        self.services = services

    def locate(self, before, after, roi=None):
        return self.services.request("vision", (before, after, roi))


class _RemotePolicy:
    def __init__(self, services):
        self.services = services
        self.pending = {}
        self.path = []
        self.terminal_reward = 0.0

    def act(self, stage, observation):
        action, record = self.services.request("policy", (stage, observation))
        self.pending[stage] = dict(record)
        return action

    def resolve(self, stage, reward, done=False):
        if stage not in self.pending:
            raise RuntimeError(f"no pending PPO decision for {stage}")
        step = self.pending.pop(stage)
        step.update(reward=float(reward), done=bool(done))
        self.path.append(step)

    def finish_episode(self, terminal_reward=0.0):
        if self.pending:
            raise RuntimeError("cannot finish an episode with unresolved PPO decisions")
        self.terminal_reward = float(terminal_reward)


def _retrieval_rollout_worker(worker_id, task_queue, response_queue, event_queue):
    torch.set_num_threads(1)
    profiles = load_profiles(ROOT / "configs/retrieval-grasps-v6.json")
    while True:
        jobs = task_queue.get()
        if jobs is None:
            return
        for job in jobs:
            services = _RemoteServices(
                worker_id, job["episode"], event_queue, response_queue
            )
            policy = _RemotePolicy(services)
            try:
                row = run_episode(
                    job["item"],
                    job["seed"],
                    job["bin_xy"],
                    _RemoteDetector(services),
                    profiles,
                    policy,
                    video_path=(
                        Path(job["video_path"])
                        if job["video_path"] is not None
                        else None
                    ),
                    label="training",
                    friction=job["friction"],
                    mass_grams=job.get("mass_grams", 4.0),
                )
                event_queue.put(
                    {
                        "type": "episode",
                        "worker": worker_id,
                        "episode": job["episode"],
                        "row": row,
                        "path": policy.path,
                        "terminal_reward": policy.terminal_reward,
                    }
                )
            except BaseException:
                event_queue.put(
                    {
                        "type": "error",
                        "worker": worker_id,
                        "episode": job["episode"],
                        "traceback": traceback.format_exc(),
                    }
                )
                return


class ParallelRolloutPool:
    """CPU MuJoCo workers with centralized, batched CUDA vision and PPO."""

    def __init__(self, workers):
        self.workers = workers
        self.context = mp.get_context("spawn")
        self.events = self.context.Queue()
        self.task_queues = [self.context.Queue() for _ in range(workers)]
        self.response_queues = [self.context.Queue() for _ in range(workers)]
        self.processes = [
            self.context.Process(
                target=_retrieval_rollout_worker,
                args=(
                    worker_id,
                    self.task_queues[worker_id],
                    self.response_queues[worker_id],
                    self.events,
                ),
                name=f"retrieval-env-{worker_id:02d}",
            )
            for worker_id in range(workers)
        ]
        self.closed = False
        self.stats = {
            "workers": workers,
            "episodes": 0,
            "vision_batches": 0,
            "vision_samples": 0,
            "vision_max_batch": 0,
            "policy_batches": 0,
            "policy_samples": 0,
            "policy_max_batch": 0,
        }
        for process in self.processes:
            process.start()

    def _dispatch(self, kind, requests, policy, detector):
        requests.sort(
            key=lambda event: (
                event["episode"],
                event["worker"],
                event["request_id"],
            )
        )
        if kind == "vision":
            results = detector.locate_batch([event["payload"] for event in requests])
            self.stats["vision_batches"] += 1
            self.stats["vision_samples"] += len(requests)
            self.stats["vision_max_batch"] = max(
                self.stats["vision_max_batch"], len(requests)
            )
        else:
            stages, observations = zip(
                *(event["payload"] for event in requests), strict=True
            )
            actions, records = policy.act_batch(stages, observations)
            results = list(zip(actions, records, strict=True))
            self.stats["policy_batches"] += 1
            self.stats["policy_samples"] += len(requests)
            self.stats["policy_max_batch"] = max(
                self.stats["policy_max_batch"], len(requests)
            )
        for event, result in zip(requests, results, strict=True):
            self.response_queues[event["worker"]].put(
                (event["request_id"], result)
            )

    def run(self, jobs, policy, detector, batch_wait=0.05):
        if not jobs:
            return []
        groups = [jobs[index :: self.workers] for index in range(self.workers)]
        active_workers = sum(bool(group) for group in groups)
        for worker_id, group in enumerate(groups):
            if group:
                self.task_queues[worker_id].put(group)

        pending = {"vision": [], "policy": []}
        deadlines = {}
        results = {}
        while len(results) < len(jobs):
            crashed = [
                process
                for process in self.processes
                if process.exitcode is not None
            ]
            if crashed:
                detail = ", ".join(
                    f"{process.name} exit={process.exitcode}" for process in crashed
                )
                raise RuntimeError(
                    f"rollout worker exited before completing its batch: {detail}"
                )
            now = time.monotonic()
            for kind in ("vision", "policy"):
                if pending[kind] and (
                    len(pending[kind]) >= active_workers
                    or now >= deadlines[kind]
                ):
                    self._dispatch(kind, pending[kind], policy, detector)
                    pending[kind] = []
                    deadlines.pop(kind, None)

            now = time.monotonic()
            active_deadlines = [
                deadline for kind, deadline in deadlines.items() if pending[kind]
            ]
            timeout = (
                max(0.001, min(active_deadlines) - now)
                if active_deadlines
                else 0.5
            )
            try:
                event = self.events.get(timeout=timeout)
            except queue.Empty:
                continue

            if event["type"] == "request":
                kind = event["kind"]
                pending[kind].append(event)
                deadlines.setdefault(kind, time.monotonic() + batch_wait)
            elif event["type"] == "episode":
                policy.finish_external_episode(
                    event["path"], event["terminal_reward"]
                )
                results[event["episode"]] = event["row"]
                self.stats["episodes"] += 1
            elif event["type"] == "error":
                raise RuntimeError(
                    f"parallel rollout failed in worker {event['worker']} "
                    f"episode {event['episode']}:\n{event['traceback']}"
                )
            else:
                raise RuntimeError(f"unknown rollout worker event: {event['type']}")

        return [(job, results[job["episode"]]) for job in jobs]

    def close(self):
        if self.closed:
            return
        self.closed = True
        for task_queue in self.task_queues:
            task_queue.put(None)
        for process in self.processes:
            process.join(timeout=3)
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
        for channel in [self.events, *self.task_queues, *self.response_queues]:
            channel.close()


def summarize(rows):
    if not rows:
        return {"episodes": 0, "successes": 0, "success_rate": None}
    return {
        "episodes": len(rows),
        "successes": sum(bool(row["success"]) for row in rows),
        "success_rate": float(np.mean([row["success"] for row in rows])),
        "lift_rate": float(np.mean([row["lifted"] for row in rows])),
        "held_at_release_rate": float(
            np.mean([row["held_at_release"] for row in rows])
        ),
        "in_tray_rate": float(np.mean([row["in_tray_at_end"] for row in rows])),
        "home_rate": float(np.mean([row["pose_restored"] for row in rows])),
        "fall_rate": float(np.mean([row["stop"] == "fall" for row in rows])),
    }


def evaluate(
    out,
    split,
    trials,
    seed,
    detector,
    profiles,
    trained_policy,
    record_videos,
    tag="final",
    masses_grams=(4.0,),
):
    items = [item for item in OBJECTS if item["split"] == split]
    if not items:
        raise ValueError(f"no objects in evaluation split {split}")
    result_rows = []
    trained_policy.train = False
    for item_index, item in enumerate(items):
        for trial in range(trials * len(masses_grams)):
            mass_grams = masses_grams[trial % len(masses_grams)]
            episode_seed = seed + item_index * 101 + trial // len(masses_grams)
            bin_xy = BIN
            for name, policy in (("baseline", None), ("ppo", trained_policy)):
                video_path = None
                if record_videos:
                    video_path = (
                        out
                        / "videos"
                        / "evaluation"
                        / tag
                        / split
                        / name
                        / f"{item['id']}-{episode_seed}-{mass_grams:g}g.mp4"
                    )
                row = run_episode(
                    item,
                    episode_seed,
                    bin_xy,
                    detector,
                    profiles,
                    policy,
                    video_path=video_path,
                    label=name,
                    mass_grams=mass_grams,
                )
                row.update(object=item["id"], split=split, controller=name)
                result_rows.append(row)
                print(
                    f"eval split={split} object={item['id']} controller={name} "
                    f"seed={episode_seed} success={row['success']} stop={row['stop']}",
                    flush=True,
                )
    trained_policy.train = True
    grouped = {
        name: summarize([row for row in result_rows if row["controller"] == name])
        for name in ("baseline", "ppo")
    }
    return {"split": split, "summary": grouped, "rows": result_rows}


def training_curriculum(
    progress, rng, schedule="sequential", include_challenge=False
):
    base = [item for item in OBJECTS if item["split"] == "train"]
    flat = [item for item in OBJECTS if item["split"] == "flat_train"]
    challenge = [item for item in OBJECTS if item["split"] == "challenge"]

    if schedule == "replay":
        if progress < 0.20:
            phase = "easy"
        elif progress < 0.50:
            phase = rng.choice(("easy", "mixed"), p=(0.65, 0.35))
        else:
            phase = rng.choice(
                ("easy", "mixed", "randomized"), p=(0.45, 0.35, 0.20)
            )
    elif progress < 1 / 3:
        phase = "easy"
    elif progress < 2 / 3:
        phase = "mixed"
    else:
        phase = "randomized"

    if phase == "easy":
        pool, xy_span, friction = base, (0.006, 0.008), (0.90, 1.02)
    elif phase == "mixed":
        pool = base + flat + (challenge if include_challenge else [])
        xy_span, friction = (0.012, 0.016), (0.80, 1.02)
    else:
        pool = base + flat + (challenge if include_challenge else [])
        xy_span, friction = (0.018, 0.025), (0.68, 1.02)
    item = pool[int(rng.integers(len(pool)))]
    bin_xy = (
        float(BIN[0] + rng.uniform(-xy_span[0], xy_span[0])),
        float(BIN[1] + rng.uniform(-xy_span[1], xy_span[1])),
    )
    return item, bin_xy, float(rng.uniform(*friction)), phase


def validation_key(result):
    summary = result["summary"]["ppo"]
    return (
        summary["successes"],
        summary["in_tray_rate"],
        summary["held_at_release_rate"],
        summary["lift_rate"],
        summary["home_rate"],
        -summary["fall_rate"],
    )


def cuda_telemetry(device):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        return {"cuda_device_utilization_pct": None, "cuda_memory_used_mib": None}
    index = torch.device(device).index
    if index is None:
        index = torch.cuda.current_device()
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--id={index}",
                "--query-gpu=utilization.gpu,memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        utilization, memory = map(int, completed.stdout.strip().split(",", 1))
        return {
            "cuda_device_utilization_pct": utilization,
            "cuda_memory_used_mib": memory,
        }
    except (OSError, subprocess.SubprocessError, ValueError):
        try:
            utilization = int(torch.cuda.utilization(index))
        except (AttributeError, ImportError, RuntimeError, TypeError):
            utilization = None
        return {
            "cuda_device_utilization_pct": utilization,
            "cuda_memory_used_mib": None,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=96)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, default=ROOT / "runs/fetch/ppo-retrieval-v1")
    parser.add_argument("--batch-decisions", type=int, default=64)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-wait-ms", type=float, default=50.0)
    parser.add_argument("--record-every", type=int, default=24)
    parser.add_argument("--eval-trials", type=int, default=1)
    parser.add_argument("--final-trials", type=int, default=2)
    parser.add_argument("--eval-every", type=int, default=48)
    parser.add_argument("--learning-rate", type=float, default=8e-5)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--target-kl", type=float, default=0.012)
    parser.add_argument(
        "--curriculum", choices=("sequential", "replay"), default="sequential"
    )
    parser.add_argument("--include-challenge", action="store_true")
    parser.add_argument(
        "--selection-split", choices=("val", "blind"), default="val"
    )
    parser.add_argument(
        "--final-splits",
        nargs="+",
        choices=("val", "test", "blind", "blind2", "blind3"),
        default=("val", "test"),
    )
    parser.add_argument("--mass-grams", nargs=2, type=float, default=(4.0, 4.0),
                        metavar=("MIN", "MAX"), help="training payload mass range; uncalibrated unless measured")
    parser.add_argument("--eval-masses-grams", nargs="+", type=float, default=[4.0])
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--weights",
        type=Path,
        default=ROOT / "runs/fetch/round3-vision/change.pt",
    )
    args = parser.parse_args()
    if (not all(np.isfinite(m) and m > 0 for m in (*args.mass_grams, *args.eval_masses_grams))
            or args.mass_grams[0] > args.mass_grams[1]
            or len(set(args.eval_masses_grams)) != len(args.eval_masses_grams)):
        parser.error("mass values must be finite, positive, ordered, and evaluation masses unique")
    if (
        args.episodes < 1
        or args.batch_decisions < 2
        or args.workers < 1
        or args.eval_trials < 1
        or args.final_trials < 1
        or args.eval_every < 0
    ):
        parser.error("episode/trial/worker counts must be positive, eval-every non-negative")
    if (
        args.record_every < 0
        or args.learning_rate <= 0
        or args.target_kl <= 0
        or args.batch_wait_ms < 0
    ):
        parser.error("record-every/batch-wait must be non-negative and PPO scales positive")
    if len(args.final_splits) != len(set(args.final_splits)):
        parser.error("final splits must be unique")
    if not any(item["split"] == args.selection_split for item in OBJECTS):
        parser.error(f"no objects in selection split {args.selection_split}")
    for split in args.final_splits:
        if not any(item["split"] == split for item in OBJECTS):
            parser.error(f"no objects in final split {split}")
    if not args.weights.is_file():
        parser.error(f"vision weights not found: {args.weights}")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        parser.error("CUDA requested but not available")

    args.out.mkdir(parents=True, exist_ok=True)
    episodes_path = args.out / "training-episodes.json"
    if episodes_path.exists() and not args.resume:
        parser.error("output already contains training episodes; choose a new --out or --resume")
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_num_threads(2)
    torch.backends.cudnn.benchmark = True

    from ducklab.retrieval.vision import Detector

    detector = Detector(args.weights, args.device)
    profiles = load_profiles(ROOT / "configs/retrieval-grasps-v6.json")
    policy = PPOPolicy(
        device=args.device,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        target_kl=args.target_kl,
        train=True,
    )
    if args.resume:
        policy.load(args.resume)
        for group in policy.optimizer.param_groups:
            group["lr"] = args.learning_rate
    if not any(item["split"] == "train" for item in OBJECTS):
        raise ValueError("no training objects found")

    episode_rows = []
    update_rows = []
    validation_rows = []
    if episodes_path.exists() and args.resume:
        episode_rows = json.loads(episodes_path.read_text())
        update_path = args.out / "updates.json"
        if update_path.exists():
            update_rows = json.loads(update_path.read_text())
        validation_path = args.out / "validation-history.json"
        if validation_path.exists():
            validation_rows = json.loads(validation_path.read_text())
    best_key = max(
        (tuple(row["selection_key"]) for row in validation_rows), default=None
    )
    best_tag = next(
        (row["tag"] for row in reversed(validation_rows) if tuple(row["selection_key"]) == best_key),
        None,
    )
    total_episodes = len(episode_rows) + args.episodes
    started = time.monotonic()
    if args.resume and not validation_rows and not args.skip_eval:
        initial = evaluate(
            args.out,
            args.selection_split,
            args.eval_trials,
            args.seed + 50000,
            detector,
            profiles,
            policy,
            record_videos=True,
            tag="initial",
            masses_grams=args.eval_masses_grams,
        )
        key = validation_key(initial)
        initial.update(selection_key=list(key), tag="initial")
        validation_rows.append(initial)
        save_json(
            args.out / f"evaluation-{args.selection_split}-initial.json", initial
        )
        save_json(args.out / "validation-history.json", validation_rows)
        best_key, best_tag = key, "initial"
        policy.save(
            args.out / "policy-best-val.pt",
            extra={"source_checkpoint": str(args.resume), "validation_key": list(key)},
        )
        print(
            f"validation tag=initial ppo={initial['summary']['ppo']} "
            f"baseline={initial['summary']['baseline']} selected=initial",
            flush=True,
        )
    rollout_metrics_path = args.out / "parallel-rollout-metrics.json"
    parallel_metrics = []
    if rollout_metrics_path.is_file() and args.resume:
        parallel_metrics = json.loads(rollout_metrics_path.read_text())
    rollout_pool = ParallelRolloutPool(args.workers) if args.workers > 1 else None
    try:
        processed = 0
        while processed < args.episodes:
            group_count = min(
                args.workers if rollout_pool is not None else 1,
                args.episodes - processed,
            )
            if args.eval_every:
                until_evaluation = args.eval_every - (
                    len(episode_rows) % args.eval_every
                )
                group_count = min(group_count, until_evaluation)

            jobs = []
            for offset in range(group_count):
                index = processed + offset
                episode_number = len(episode_rows) + offset + 1
                episode_seed = args.seed + 1000 + episode_number - 1
                item, bin_xy, friction, curriculum = training_curriculum(
                    (index + 1) / args.episodes,
                    rng,
                    schedule=args.curriculum,
                    include_challenge=args.include_challenge,
                )
                record = (
                    args.record_every > 0
                    and episode_number % args.record_every == 0
                )
                video_path = (
                    args.out
                    / "videos"
                    / "training"
                    / f"{episode_number:05d}-{item['id']}-{episode_seed}.mp4"
                    if record
                    else None
                )
                jobs.append(
                    {
                        "episode": episode_number,
                        "seed": episode_seed,
                        "item": item,
                        "bin_xy": bin_xy,
                        "friction": friction,
                        "mass_grams": float(np.random.default_rng(episode_seed ^ 0x4D415353).uniform(*args.mass_grams)),
                        "curriculum": curriculum,
                        "video_path": video_path,
                    }
                )

            batch_started = time.monotonic()
            if rollout_pool is None:
                results = []
                for job in jobs:
                    row = run_episode(
                        job["item"],
                        job["seed"],
                        job["bin_xy"],
                        detector,
                        profiles,
                        policy,
                        video_path=job["video_path"],
                        label="training",
                        friction=job["friction"],
                        mass_grams=job.get("mass_grams", 4.0),
                    )
                    results.append((job, row))
            else:
                results = rollout_pool.run(
                    jobs,
                    policy,
                    detector,
                    batch_wait=args.batch_wait_ms / 1000.0,
                )

            for job, row in results:
                row.update(
                    episode=job["episode"],
                    object=job["item"]["id"],
                    bin_m=list(job["bin_xy"]),
                    friction=job["friction"],
                    mass_grams=job["mass_grams"],
                    curriculum=job["curriculum"],
                )
                episode_rows.append(row)
                print(
                    f"train episode={job['episode']}/{total_episodes} "
                    f"object={job['item']['id']} curriculum={job['curriculum']} "
                    f"success={row['success']} lifted={row['lifted']} "
                    f"tray={row['in_tray_at_end']} home={row['pose_restored']}",
                    flush=True,
                )

            episode_number = jobs[-1]["episode"]
            should_evaluate = bool(
                args.eval_every and episode_number % args.eval_every == 0
            )
            update = None
            if len(policy.buffer) >= args.batch_decisions or should_evaluate:
                update = policy.update()
                if update is not None:
                    update["episode"] = episode_number
                    update["rollout_workers"] = (
                        1 if rollout_pool is None else rollout_pool.workers
                    )
                    update_rows.append(update)
                    policy.save(
                        args.out / "policy-latest.pt",
                        extra={"episode": episode_number},
                    )

            save_json(episodes_path, episode_rows)
            save_json(args.out / "updates.json", update_rows)
            elapsed = time.monotonic() - batch_started
            batch_metric = {
                "episode_start": jobs[0]["episode"],
                "episode_end": episode_number,
                "episodes": len(jobs),
                "workers": 1 if rollout_pool is None else rollout_pool.workers,
                "rollout_seconds": round(elapsed, 3),
                "episodes_per_second": round(len(jobs) / max(elapsed, 1e-6), 4),
                "decisions_buffered": len(policy.buffer),
            }
            batch_metric.update(cuda_telemetry(args.device))
            parallel_metrics.append(batch_metric)
            save_json(rollout_metrics_path, parallel_metrics)
            print(
                f"rollout episodes={batch_metric['episode_start']}–{episode_number} "
                f"workers={batch_metric['workers']} "
                f"eps_per_sec={batch_metric['episodes_per_second']:.3f} "
                f"cuda_util={batch_metric.get('cuda_device_utilization_pct')}",
                flush=True,
            )
            processed += len(jobs)

            if should_evaluate:
                tag = f"checkpoint-{episode_number:05d}"
                result = evaluate(
                    args.out,
                    args.selection_split,
                    args.eval_trials,
                    args.seed + 50000,
                    detector,
                    profiles,
                    policy,
                    record_videos=True,
                    tag=tag,
                    masses_grams=args.eval_masses_grams,
                )
                key = validation_key(result)
                result["selection_key"] = list(key)
                result["tag"] = tag
                validation_rows.append(result)
                save_json(
                    args.out / f"evaluation-{args.selection_split}-{tag}.json",
                    result,
                )
                save_json(args.out / "validation-history.json", validation_rows)
                if best_key is None or key > best_key:
                    best_key, best_tag = key, tag
                    policy.save(
                        args.out / "policy-best-val.pt",
                        extra={"episode": episode_number, "validation_key": list(key)},
                    )
                print(
                    f"validation tag={tag} ppo={result['summary']['ppo']} "
                    f"baseline={result['summary']['baseline']} selected={best_tag}",
                    flush=True,
                )
    finally:
        if rollout_pool is not None:
            rollout_pool.close()

    if policy.buffer:
        update = policy.update()
        if update is not None:
            update_rows.append(update)
    policy.save(args.out / "policy-last.pt", extra={"episodes": args.episodes})
    save_json(args.out / "updates.json", update_rows)

    evaluations = []
    if not args.skip_eval:
        if args.out.joinpath("policy-best-val.pt").is_file():
            policy.load(args.out / "policy-best-val.pt")
        policy.save(
            args.out / "policy-final.pt",
            extra={"episodes": args.episodes, "selected_validation_tag": best_tag},
        )
        split_offsets = {
            "val": 70000,
            "test": 60000,
            "blind": 80000,
            "blind2": 90000,
            "blind3": 100000,
        }
        for split in args.final_splits:
            result = evaluate(
                args.out,
                split,
                args.final_trials,
                args.seed + split_offsets[split],
                detector,
                profiles,
                policy,
                record_videos=True,
                tag="final-selected",
                masses_grams=args.eval_masses_grams,
            )
            evaluations.append(result)
            save_json(args.out / f"evaluation-{split}.json", result)

    summary = {
        "algorithm": "PPO clipped policy gradient",
        "policy_scope": "phase-level grasp and delivery parameters; official low-level ONNX policies frozen",
        "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "episodes_this_run": args.episodes,
        "total_training_episodes": len(episode_rows),
        "seed": args.seed,
        "curriculum": args.curriculum,
        "training_mass_grams": args.mass_grams,
        "evaluation_mass_grams": args.eval_masses_grams,
        "friction_randomization": "parcel geom and explicit contact pairs",
        "physical_calibration": "not_measured_on_hardware",
        "include_challenge": args.include_challenge,
        "selection_split": args.selection_split,
        "final_splits": args.final_splits,
        "learning_rate": args.learning_rate,
        "rollout_workers": args.workers,
        "batch_wait_ms": args.batch_wait_ms,
        "initialization_checkpoint": str(args.resume) if args.resume else None,
        "training": summarize(episode_rows),
        "parallel_rollout": (
            rollout_pool.stats
            if rollout_pool is not None
            else {
                "workers": 1,
                "episodes": args.episodes,
                "vision_batches": 0,
                "policy_batches": policy.action_batches,
                "policy_samples": policy.action_decisions,
            }
        ),
        "rollout_batches": parallel_metrics,
        "cuda_device_utilization_mean_pct": (
            round(
                float(
                    np.mean(
                        [
                            row["cuda_device_utilization_pct"]
                            for row in parallel_metrics
                            if row.get("cuda_device_utilization_pct") is not None
                        ]
                    )
                ),
                1,
            )
            if any(
                row.get("cuda_device_utilization_pct") is not None
                for row in parallel_metrics
            )
            else None
        ),
        "updates": update_rows,
        "selected_validation_tag": best_tag,
        "validation_history": [
            {"tag": row["tag"], "summary": row["summary"], "selection_key": row["selection_key"]}
            for row in validation_rows
        ],
        "evaluations": [
            {"split": value["split"], "summary": value["summary"]}
            for value in evaluations
        ],
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "checkpoint": str(args.out / "policy-final.pt"),
        "peak_gpu_mib": (
            round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if args.device.startswith("cuda")
            else None
        ),
    }
    save_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
