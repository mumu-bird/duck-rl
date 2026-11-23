#!/usr/bin/env python3
"""Train a CUDA PPO policy for picking up procedurally randomized LINLEE-like ducks.

The official low-level walking, standing and head-pick ONNX policies stay frozen.
MuJoCo rollouts run on CPU worker threads, while the PPO actor/critic inference
and clipped policy-gradient updates run in batched form on the requested device.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
import os
from pathlib import Path
from queue import Empty, Queue
import random
import sys
import threading
import time

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
import torch

from ducklab.assets import ROOT
from ducklab.retrieval.duck import load_duck_variants, sample_duck
from ducklab.retrieval.episode import Episode
from ducklab.retrieval.grasp import load_profiles
from ducklab.retrieval.ppo import PPOPolicy
from ducklab.retrieval.scene import BIN, HEIGHT, WIDTH, make_scene
from ducklab.retrieval.vision import Detector
from ducklab.video import Video


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".partial")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def summarize(rows):
    if not rows:
        return {
            "episodes": 0,
            "successes": 0,
            "success_rate": None,
            "lift_rate": None,
            "held_at_release_rate": None,
            "in_tray_rate": None,
            "home_rate": None,
            "fall_rate": None,
        }
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


def physical_parameters(item):
    """Keep the sampled domain-randomization values beside every trajectory."""
    return {
        "variant_id": item.get("id"),
        "half_m": [float(value) for value in item.get("half_m", ())],
        "mass_kg": float(item.get("mass_kg", 0.0)),
        "mass_grams": float(item.get("mass_kg", 0.0)) * 1000.0,
        "sample_scale": float(item.get("sample_scale", 1.0)),
        "friction": item.get("friction"),
        "contact_friction": item.get("contact_friction"),
        "solref": item.get("solref"),
        "solimp": item.get("solimp"),
        "body_rgba": item.get("body_rgba"),
        "head_rgba": item.get("head_rgba"),
        "bill_rgba": item.get("bill_rgba"),
    }


class LockedDetector:
    """Serialize the shared TorchScript detector while physics stays parallel."""

    def __init__(self, detector):
        self.detector = detector
        self.lock = threading.Lock()

    def locate(self, *args, **kwargs):
        with self.lock:
            return self.detector.locate(*args, **kwargs)


class BatchedActionServer:
    """Collect synchronous worker requests and run one CUDA policy batch."""

    def __init__(self, policy, max_batch=32, wait_ms=8):
        self.policy = policy
        self.max_batch = max(1, int(max_batch))
        self.wait_seconds = max(0.0, float(wait_ms) / 1000.0)
        self.requests = Queue()
        self.completed = []
        self.completed_lock = threading.Lock()
        self.thread = None
        self.batch_count = 0
        self.decision_count = 0
        self.batch_sizes = []

    def start(self):
        if self.thread is not None:
            raise RuntimeError("action server already started")
        self.thread = threading.Thread(
            target=self._serve, name="ppo-cuda-action-server", daemon=True
        )
        self.thread.start()

    def request(self, stage, observation):
        response = Queue(maxsize=1)
        self.requests.put((stage, np.asarray(observation, dtype=np.float32), response))
        result = response.get()
        if isinstance(result, BaseException):
            raise result
        return result

    def submit_episode(self, path, terminal_reward):
        if not path:
            return
        with self.completed_lock:
            self.completed.append((path, float(terminal_reward)))

    def drain_episodes(self):
        with self.completed_lock:
            paths = self.completed
            self.completed = []
        return paths

    def _serve(self):
        while True:
            request = self.requests.get()
            if request is None:
                return
            batch = [request]
            deadline = time.monotonic() + self.wait_seconds
            while len(batch) < self.max_batch:
                timeout = deadline - time.monotonic()
                if timeout <= 0:
                    break
                try:
                    request = self.requests.get(timeout=timeout)
                except Empty:
                    break
                if request is None:
                    raise RuntimeError("action server stopped with pending requests")
                batch.append(request)
            stages = [request[0] for request in batch]
            observations = [request[1] for request in batch]
            try:
                actions, records = self.policy.act_batch(stages, observations)
            except BaseException as error:
                for _, _, response in batch:
                    response.put(error)
                continue
            self.batch_count += 1
            self.decision_count += len(batch)
            self.batch_sizes.append(len(batch))
            for request, action, record in zip(batch, actions, records, strict=True):
                request[2].put((action, record))

    def stop(self):
        if self.thread is None:
            return
        self.requests.put(None)
        self.thread.join(timeout=30)
        if self.thread.is_alive():
            raise RuntimeError("timed out stopping PPO action server")
        self.thread = None

    def metrics(self):
        return {
            "batches": self.batch_count,
            "decisions": self.decision_count,
            "mean_batch_size": (
                float(np.mean(self.batch_sizes)) if self.batch_sizes else 0.0
            ),
            "max_batch_size": max(self.batch_sizes, default=0),
        }


class ExternalRolloutPolicy:
    """Episode-facing adapter whose action/value calls are served in batches."""

    train = True

    def __init__(self, server):
        self.server = server
        self.pending = {}
        self.path = []

    def act(self, stage, observation):
        if stage in self.pending:
            raise RuntimeError(f"unresolved rollout decision for {stage}")
        action, record = self.server.request(stage, observation)
        self.pending[stage] = record
        return action

    def resolve(self, stage, reward, done=False):
        if stage not in self.pending:
            raise RuntimeError(f"no pending rollout decision for {stage}")
        step = dict(self.pending.pop(stage))
        step.update(reward=float(reward), done=bool(done))
        self.path.append(step)

    def finish_episode(self, terminal_reward=0.0):
        if self.pending:
            raise RuntimeError("cannot finish rollout with unresolved decisions")
        self.server.submit_episode(list(self.path), terminal_reward)
        self.path.clear()


def run_episode(
    item,
    seed,
    bin_xy,
    detector,
    profiles,
    policy,
    video_path=None,
    label="PPO",
):
    # Do not overwrite mass or contact parameters here: they are the sampled
    # domain-randomized instance the policy is supposed to learn to handle.
    model, half = make_scene(item, bin_xy, moving_jaw=True)
    with ExitStack() as stack:
        camera = stack.enter_context(mujoco.Renderer(model, HEIGHT, WIDTH))
        video = None
        if video_path is not None:
            wide = stack.enter_context(mujoco.Renderer(model, 320, 536))
            video = Video(
                video_path,
                wide,
                camera,
                f"叼取 PPO · {item['id']} · {label}",
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
                video.finish("异常终止")
            raise
        if video is not None:
            video.finish("完整通过" if result["success"] else "本轮未通过")
            result["artifacts"] = {
                "video": str(Path(video_path).resolve()),
                "frames": [str(Path(path).resolve()) for path in video.snapshots],
            }
        else:
            result["artifacts"] = {"video": None, "frames": []}
        result["duck_physics"] = physical_parameters(item)
        result["bin_xy"] = [float(value) for value in bin_xy]
        return result


def _difficulty_template(template, phase):
    item = dict(template)
    if phase == "easy":
        item["mass_grams"] = (19.0, 22.0)
        item["scale"] = (0.98, 1.02)
        item["friction"] = (0.78, 0.88)
    elif phase == "mixed":
        item["mass_grams"] = (15.0, 30.0)
        item["scale"] = (0.94, 1.08)
        item["friction"] = (0.64, 0.90)
    return item


def duck_curriculum(progress, rng, variants, schedule="replay"):
    if schedule == "replay":
        if progress < 0.20:
            phase = "easy"
        elif progress < 0.50:
            phase = str(rng.choice(("easy", "mixed"), p=(0.65, 0.35)))
        else:
            phase = str(
                rng.choice(
                    ("easy", "mixed", "randomized"), p=(0.45, 0.35, 0.20)
                )
            )
    else:
        if progress < 1 / 3:
            phase = "easy"
        elif progress < 2 / 3:
            phase = "mixed"
        else:
            phase = "randomized"

    if phase == "easy":
        pool = [template for template in variants if template["id"] == "linlee_round_yellow"]
        if not pool:
            pool = variants[:1]
        span = (0.004, 0.006)
    elif phase == "mixed":
        pool = variants
        span = (0.010, 0.014)
    else:
        pool = variants
        span = (0.018, 0.025)
    template = pool[int(rng.integers(len(pool)))]
    item = sample_duck(_difficulty_template(template, phase), rng)
    bin_xy = (
        float(BIN[0] + rng.uniform(-span[0], span[0])),
        float(BIN[1] + rng.uniform(-span[1], span[1])),
    )
    return item, bin_xy, phase


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


def evaluate(out, variants, trials, seed, detector, profiles, policy, tag):
    rng = np.random.default_rng(seed)
    rows = []
    policy.train = False
    try:
        for variant_index, template in enumerate(variants):
            for trial in range(trials):
                episode_seed = seed + variant_index * 1009 + trial
                item = sample_duck(template, rng)
                for controller, controller_policy in (("baseline", None), ("ppo", policy)):
                    video_path = (
                        out
                        / "videos"
                        / "evaluation"
                        / tag
                        / controller
                        / f"{item['id']}-{episode_seed}.mp4"
                    )
                    row = run_episode(
                        item,
                        episode_seed,
                        BIN,
                        detector,
                        profiles,
                        controller_policy,
                        video_path=video_path,
                        label=controller,
                    )
                    row.update(
                        variant=item["id"],
                        trial=trial,
                        controller=controller,
                        split="linlee-validation",
                    )
                    rows.append(row)
                    print(
                        f"eval tag={tag} variant={item['id']} controller={controller} "
                        f"seed={episode_seed} success={row['success']} stop={row['stop']}",
                        flush=True,
                    )
    finally:
        policy.train = True
    grouped = {
        controller: summarize(
            [row for row in rows if row["controller"] == controller]
        )
        for controller in ("baseline", "ppo")
    }
    return {"tag": tag, "summary": grouped, "rows": rows}


def gpu_info(device):
    info = {
        "requested_device": str(device),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
    }
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        info.update(
            {
                "device_index": index,
                "device_name": properties.name,
                "total_memory_mib": round(properties.total_memory / 2**20, 1),
                "capability": f"{properties.major}.{properties.minor}",
            }
        )
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--out", type=Path, default=ROOT / "runs/fetch/linlee-duck-ppo-v1"
    )
    parser.add_argument("--rollout-workers", type=int, default=4)
    parser.add_argument("--batch-decisions", type=int, default=32)
    parser.add_argument("--action-batch-size", type=int, default=32)
    parser.add_argument("--action-wait-ms", type=float, default=8.0)
    parser.add_argument("--record-every", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--eval-trials", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--entropy-coef", type=float, default=0.003)
    parser.add_argument("--target-kl", type=float, default=0.012)
    parser.add_argument(
        "--curriculum", choices=("sequential", "replay"), default="replay"
    )
    parser.add_argument("--weights", type=Path, default=ROOT / "runs/fetch/round3-vision/change.pt")
    parser.add_argument(
        "--profiles", type=Path, default=ROOT / "configs/linlee-duck-profiles.json"
    )
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--skip-eval", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.rollout_workers < 1 or args.batch_decisions < 8:
        parser.error("episodes/workers must be positive and batch-decisions must be >= 8")
    if args.action_batch_size < 1 or args.action_wait_ms < 0:
        parser.error("action batch settings must be non-negative and positive")
    if args.record_every < 0 or args.eval_every < 0 or args.eval_trials < 1:
        parser.error("record/evaluation settings are invalid")
    if args.learning_rate <= 0 or args.target_kl <= 0:
        parser.error("PPO scales must be positive")
    if not args.weights.is_file():
        parser.error(f"vision weights not found: {args.weights}")
    if not args.profiles.is_file():
        parser.error(f"profile config not found: {args.profiles}")
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
        torch.backends.cudnn.benchmark = True
        torch.cuda.reset_peak_memory_stats()
    torch.set_num_threads(max(1, min(2, args.rollout_workers)))

    variants = load_duck_variants()
    detector = LockedDetector(Detector(args.weights, args.device))
    profiles = load_profiles(args.profiles)
    policy = PPOPolicy(
        device=args.device,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        target_kl=args.target_kl,
        minibatch_size=max(32, min(args.batch_decisions, 256)),
        train=True,
    )
    if args.resume:
        policy.load(args.resume)
        for group in policy.optimizer.param_groups:
            group["lr"] = args.learning_rate

    episode_rows = []
    update_rows = []
    validation_rows = []
    if args.resume:
        if episodes_path.exists():
            episode_rows = json.loads(episodes_path.read_text())
        if (args.out / "updates.json").exists():
            update_rows = json.loads((args.out / "updates.json").read_text())
        if (args.out / "validation-history.json").exists():
            validation_rows = json.loads(
                (args.out / "validation-history.json").read_text()
            )
    # A selected validation checkpoint can intentionally lag behind the latest
    # rollout checkpoint. Keep the update IDs monotonic when resuming from it.
    if update_rows:
        policy.update_count = max(
            policy.update_count,
            max(int(row.get("update", 0)) for row in update_rows),
        )
    best_key = max(
        (tuple(row["selection_key"]) for row in validation_rows), default=None
    )
    best_tag = next(
        (
            row["tag"]
            for row in reversed(validation_rows)
            if tuple(row["selection_key"]) == best_key
        ),
        None,
    )

    server = BatchedActionServer(
        policy, max_batch=args.action_batch_size, wait_ms=args.action_wait_ms
    )
    server.start()
    started = time.monotonic()
    rollout_batches = []
    initial_episode_count = len(episode_rows)
    try:
        for batch_offset in range(0, args.episodes, args.rollout_workers):
            batch_size = min(args.rollout_workers, args.episodes - batch_offset)
            cases = []
            for worker_index in range(batch_size):
                episode_number = initial_episode_count + batch_offset + worker_index + 1
                progress = (batch_offset + worker_index + 1) / args.episodes
                item, bin_xy, phase = duck_curriculum(
                    min(1.0, progress), rng, variants, schedule=args.curriculum
                )
                episode_seed = args.seed + 1000 + episode_number * 31
                record = args.record_every > 0 and episode_number % args.record_every == 0
                video_path = (
                    args.out
                    / "videos"
                    / "training"
                    / f"{episode_number:05d}-w{worker_index:02d}-{item['id']}-{episode_seed}.mp4"
                    if record
                    else None
                )
                cases.append((item, episode_seed, bin_xy, phase, video_path))

            batch_started = time.monotonic()
            with ThreadPoolExecutor(
                max_workers=args.rollout_workers,
                thread_name_prefix="mujoco-rollout",
            ) as executor:
                futures = [
                    executor.submit(
                        run_episode,
                        item,
                        episode_seed,
                        bin_xy,
                        detector,
                        profiles,
                        ExternalRolloutPolicy(server),
                        video_path,
                        "training",
                    )
                    for item, episode_seed, bin_xy, _, video_path in cases
                ]
                results = [future.result() for future in futures]
            for path, terminal_reward in server.drain_episodes():
                policy.finish_external_episode(path, terminal_reward)
            rollout_seconds = time.monotonic() - batch_started
            rollout_rows = []
            for (item, episode_seed, bin_xy, phase, _), row in zip(
                cases, results, strict=True
            ):
                episode_number = initial_episode_count + batch_offset + len(rollout_rows) + 1
                row.update(
                    episode=episode_number,
                    worker_seed=episode_seed,
                    curriculum=phase,
                    variant=item["id"],
                )
                episode_rows.append(row)
                rollout_rows.append(row)
                print(
                    f"train episode={episode_number} variant={item['id']} "
                    f"phase={phase} success={row['success']} lifted={row['lifted']} "
                    f"tray={row['in_tray_at_end']} mass={row['duck_physics']['mass_grams']:.1f}g",
                    flush=True,
                )
            last_episode = initial_episode_count + batch_offset + batch_size
            rollout_batches.append(
                {
                    "episode": last_episode,
                    "episodes": len(rollout_rows),
                    "seconds": float(rollout_seconds),
                    "physics_steps": int(
                        sum(row.get("physics_steps", 0) for row in rollout_rows)
                    ),
                    "episodes_per_second": len(rollout_rows) / max(rollout_seconds, 1e-9),
                    "physics_steps_per_second": sum(
                        row.get("physics_steps", 0) for row in rollout_rows
                    )
                    / max(rollout_seconds, 1e-9),
                }
            )

            if len(policy.buffer) >= args.batch_decisions:
                update = policy.update()
                if update is not None:
                    update["after_episode"] = last_episode
                    update_rows.append(update)
                    policy.save(
                        args.out / "policy-latest.pt",
                        extra={"episode": last_episode, "domain": "linlee-duck"},
                    )
                    print(
                        f"ppo update={update['update']} decisions={update['decisions']} "
                        f"gpu_update_s={update['update_seconds']:.4f} "
                        f"gpu_mem={update['gpu_memory_allocated_mib']:.1f}MiB",
                        flush=True,
                    )
            save_json(episodes_path, episode_rows)
            save_json(args.out / "updates.json", update_rows)
            save_json(args.out / "rollout-batches.json", rollout_batches)

            should_evaluate = (
                not args.skip_eval
                and args.eval_every > 0
                and last_episode % args.eval_every == 0
            )
            if should_evaluate:
                result = evaluate(
                    args.out,
                    variants,
                    args.eval_trials,
                    args.seed + 50000 + last_episode,
                    detector,
                    profiles,
                    policy,
                    tag=f"checkpoint-{last_episode:05d}",
                )
                key = validation_key(result)
                result["selection_key"] = list(key)
                validation_rows.append(result)
                best_tag = result["tag"] if best_key is None or key > best_key else best_tag
                best_key = key if best_key is None or key > best_key else best_key
                save_json(
                    args.out / f"evaluation-{result['tag']}.json", result
                )
                save_json(args.out / "validation-history.json", validation_rows)
                if best_tag == result["tag"]:
                    policy.save(
                        args.out / "policy-best-val.pt",
                        extra={
                            "episode": last_episode,
                            "validation_key": list(key),
                        },
                    )
    finally:
        server.stop()

    if policy.buffer:
        update = policy.update()
        if update is not None:
            update["after_episode"] = len(episode_rows)
            update_rows.append(update)
    policy.save(
        args.out / "policy-last.pt",
        extra={"episodes": len(episode_rows), "domain": "linlee-duck"},
    )

    final_evaluation = None
    if not args.skip_eval:
        if (args.out / "policy-best-val.pt").is_file():
            policy.load(args.out / "policy-best-val.pt")
        final_evaluation = evaluate(
            args.out,
            variants,
            args.eval_trials,
            args.seed + 90000,
            detector,
            profiles,
            policy,
            tag="final-selected",
        )
        save_json(args.out / "evaluation-final.json", final_evaluation)
        policy.save(
            args.out / "policy-final.pt",
            extra={
                "episodes": len(episode_rows),
                "selected_validation_tag": best_tag,
                "domain": "linlee-duck",
            },
        )
    else:
        policy.save(
            args.out / "policy-final.pt",
            extra={"episodes": len(episode_rows), "domain": "linlee-duck"},
        )

    summary = {
        "algorithm": "PPO-Clip with GAE and policy-gradient updates",
        "task": "single LINLEE-like duck pickup, carry, release and return",
        "policy_scope": "phase-level grasp and delivery parameters; official low-level ONNX policies frozen",
        "physics_device": "CPU (MuJoCo 3.3.7)",
        "policy_device": str(policy.device),
        "gpu": gpu_info(args.device),
        "episodes_this_run": args.episodes,
        "total_training_episodes": len(episode_rows),
        "seed": args.seed,
        "curriculum": args.curriculum,
        "rollout_workers": args.rollout_workers,
        "batch_decisions": args.batch_decisions,
        "action_server": server.metrics(),
        "rollout_batches": rollout_batches,
        "updates": update_rows,
        "training": summarize(episode_rows),
        "selected_validation_tag": best_tag,
        "validation_history": [
            {
                "tag": row["tag"],
                "summary": row["summary"],
                "selection_key": row["selection_key"],
            }
            for row in validation_rows
        ],
        "final_evaluation": None
        if final_evaluation is None
        else final_evaluation["summary"],
        "elapsed_seconds": round(time.monotonic() - started, 2),
        "checkpoint": str((args.out / "policy-final.pt").resolve()),
        "peak_gpu_memory_allocated_mib": (
            round(torch.cuda.max_memory_allocated() / 2**20, 1)
            if policy.device.type == "cuda"
            else None
        ),
        "artifacts": {
            "episodes": str(episodes_path.resolve()),
            "updates": str((args.out / "updates.json").resolve()),
            "rollouts": str((args.out / "rollout-batches.json").resolve()),
            "videos": str((args.out / "videos").resolve()),
        },
    }
    save_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
