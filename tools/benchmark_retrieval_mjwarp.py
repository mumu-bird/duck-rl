#!/usr/bin/env python3
"""Benchmark batched GPU physics on the retrieval MJCF without changing training."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import mujoco
import mujoco_warp as mjw
import numpy as np
import warp as wp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def integrator_name(value):
    for name, member in mujoco.mjtIntegrator.__members__.items():
        if int(member) == int(value):
            return name
    return str(int(value))


class GpuSampler:
    def __init__(self):
        self.samples = []
        self.stop = threading.Event()
        self.thread = None

    def _sample(self):
        command = [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,power.draw",
            "--format=csv,noheader,nounits",
        ]
        while not self.stop.is_set():
            try:
                result = subprocess.run(
                    command, capture_output=True, text=True, timeout=2, check=True
                )
                values = [part.strip() for part in result.stdout.splitlines()[0].split(",")]
                parsed = [float(value) if value not in ("N/A", "[Not Supported]") else None
                          for value in values]
                self.samples.append(
                    {"gpu_utilization_pct": parsed[0], "memory_used_mib": parsed[1],
                     "power_w": parsed[2]}
                )
            except (IndexError, OSError, subprocess.SubprocessError, ValueError):
                pass
            self.stop.wait(0.5)

    def __enter__(self):
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join(timeout=3)


def initial_states(model, item, count, seed):
    from ducklab.policy import Controller
    from ducklab.retrieval.episode import seed_scene, toss

    qpos, qvel = [], []
    for index in range(count):
        data = mujoco.MjData(model)
        Controller(model, data)
        rng = np.random.default_rng(seed + index)
        seed_scene(model, data, rng)
        toss(data, rng)
        mujoco.mj_forward(model, data)
        qpos.append(data.qpos.copy())
        qvel.append(data.qvel.copy())
    return np.asarray(qpos), np.asarray(qvel)


def cpu_data(model, qpos, qvel):
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    data.ctrl[:] = 0
    data.qacc_warmstart[:] = 0
    if hasattr(data, "eq_active"):
        data.eq_active[:] = model.eq_active0
    mujoco.mj_forward(model, data)
    return data


def run_cpu(model, qpos, qvel, steps):
    worlds = len(qpos)
    states = [cpu_data(model, qpos[index], qvel[index]) for index in range(worlds)]
    for _ in range(2):
        for state in states:
            mujoco.mj_step(model, state)
    states = [cpu_data(model, qpos[index], qvel[index]) for index in range(worlds)]
    start = time.perf_counter()
    for _ in range(steps):
        for state in states:
            mujoco.mj_step(model, state)
    elapsed = time.perf_counter() - start
    return {
        "seconds": elapsed,
        "env_steps_per_second": worlds * steps / elapsed,
        "qpos": np.asarray([state.qpos.copy() for state in states]),
        "contact_counts": np.asarray([state.ncon for state in states]),
    }


def gpu_array(values, device):
    return wp.array(np.asarray(values, dtype=np.float32), dtype=wp.float32, device=device)


def set_gpu_state(data, qpos, qvel, device):
    wp.copy(data.qpos, gpu_array(qpos, device))
    wp.copy(data.qvel, gpu_array(qvel, device))
    wp.copy(data.ctrl, gpu_array(np.zeros((len(qpos), data.ctrl.shape[1])), device))
    wp.copy(data.time, gpu_array(np.zeros(len(qpos)), device))
    wp.copy(
        data.qacc_warmstart,
        gpu_array(np.zeros((len(qpos), data.qacc_warmstart.shape[1])), device),
    )
    if data.act.shape[1]:
        wp.copy(data.act, gpu_array(np.zeros((len(qpos), data.act.shape[1])), device))


def gpu_contact_counts(data, worlds):
    active = int(data.nacon.numpy()[0])
    world_ids = data.contact.worldid.numpy()[:active]
    return np.bincount(world_ids, minlength=worlds)


def assert_no_gpu_overflow(data, history, context):
    if not hasattr(data, "overflow"):
        return False
    current = np.asarray(data.overflow.numpy(), dtype=np.uint32).reshape(-1)
    if current.shape != history.shape:
        raise RuntimeError(
            f"unexpected overflow shape at {context}: {current.shape} != {history.shape}"
        )
    history |= current
    if not np.any(current):
        return
    overflow_type = getattr(mjw, "OverflowType", None)
    names = [
        member.name
        for member in overflow_type or ()
        if member.value
        and not (member.value & (member.value - 1))
        and np.any(current & member.value)
    ]
    worlds = np.flatnonzero(current).tolist()
    detail = f" ({', '.join(names)})" if names else ""
    raise RuntimeError(
        f"MJWarp overflow at {context}: worlds={worlds}{detail}; "
        "increase nconmax/njmax or reduce the batch before using these results"
    )


def run_gpu(model, warp_model, qpos, qvel, steps, graph_steps, nconmax, njmax):
    worlds = len(qpos)
    device = wp.get_device("cuda:0")
    data = mjw.make_data(
        model, nworld=worlds, nconmax=nconmax, njmax=njmax
    )
    overflow_check_supported = hasattr(data, "overflow")
    overflow_history = np.zeros(worlds, dtype=np.uint32)
    set_gpu_state(data, qpos, qvel, device)
    mjw.forward(warp_model, data)
    wp.synchronize_device(device)
    assert_no_gpu_overflow(data, overflow_history, "initial forward")
    for _ in range(3):
        mjw.step(warp_model, data)
    wp.synchronize_device(device)
    assert_no_gpu_overflow(data, overflow_history, "warmup")

    set_gpu_state(data, qpos, qvel, device)
    mjw.forward(warp_model, data)
    wp.synchronize_device(device)
    assert_no_gpu_overflow(data, overflow_history, "graph capture setup")
    with wp.ScopedCapture() as capture:
        for _ in range(graph_steps):
            mjw.step(warp_model, data)
    graph = capture.graph
    wp.synchronize_device(device)

    set_gpu_state(data, qpos, qvel, device)
    mjw.forward(warp_model, data)
    wp.synchronize_device(device)
    assert_no_gpu_overflow(data, overflow_history, "timed run setup")
    sampler = GpuSampler()
    with sampler:
        start = time.perf_counter()
        for _ in range(steps // graph_steps):
            wp.capture_launch(graph)
        wp.synchronize_device(device)
        assert_no_gpu_overflow(data, overflow_history, "timed run")
        elapsed = time.perf_counter() - start

    final_qpos = data.qpos.numpy().copy()
    if not np.isfinite(final_qpos).all():
        raise FloatingPointError(
            f"MJWarp produced non-finite qpos at {worlds} worlds; "
            "increase contact/constraint capacity or reject this batch size"
        )
    return {
        "seconds": elapsed,
        "env_steps_per_second": worlds * steps / elapsed,
        "qpos": final_qpos,
        "contact_counts": gpu_contact_counts(data, worlds),
        "gpu_samples": sampler.samples,
        "overflow_check_supported": overflow_check_supported,
        "overflow_flags": overflow_history.tolist(),
        "max_nefc": (
            int(np.max(data.nefc.numpy())) if hasattr(data, "nefc") else None
        ),
        "max_solver_iterations": (
            int(np.max(data.solver_niter.numpy()))
            if hasattr(data, "solver_niter")
            else None
        ),
    }


def has_parcel_contact(model, data):
    active = int(data.nacon.numpy()[0])
    if active == 0:
        return False
    world_ids = data.contact.worldid.numpy()[:active]
    geom_ids = data.contact.geom.numpy()[:active]
    parcel_geoms = np.flatnonzero(model.geom_bodyid == model.body("parcel").id)
    return bool(
        np.any(
            (world_ids == 0)
            & (np.isin(geom_ids[:, 0], parcel_geoms)
               | np.isin(geom_ids[:, 1], parcel_geoms))
        )
    )


def record_gpu_video(
    model, warp_model, qpos, qvel, steps, frame_stride, path, object_id,
    nconmax, njmax,
):
    from ducklab.retrieval.scene import BIN, HEIGHT, WIDTH
    from ducklab.video import Video

    device = wp.get_device("cuda:0")
    data = mjw.make_data(model, nworld=1, nconmax=nconmax, njmax=njmax)
    set_gpu_state(data, qpos[:1], qvel[:1], device)
    mjw.forward(warp_model, data)
    with wp.ScopedCapture() as capture:
        for _ in range(frame_stride):
            mjw.step(warp_model, data)
    graph = capture.graph
    wp.synchronize_device(device)
    set_gpu_state(data, qpos[:1], qvel[:1], device)
    mjw.forward(warp_model, data)

    frames = []
    parcel_contact = False
    for frame_index in range(steps // frame_stride):
        wp.capture_launch(graph)
        wp.synchronize_device(device)
        parcel_contact |= has_parcel_contact(model, data)
        frames.append(
            (data.qpos.numpy()[0].copy(), float(data.time.numpy()[0]), parcel_contact)
        )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    snapshots = []
    with mujoco.Renderer(model, 320, 536) as wide, mujoco.Renderer(
        model, HEIGHT, WIDTH
    ) as head:
        video = Video(
            path,
            wide,
            head,
            f"MJWarp GPU 物理试验 · {object_id}",
            0,
            arena_view=([BIN[0] / 2, BIN[1] / 2, 0.078], 0.90),
        )
        render_data = mujoco.MjData(model)
        contact_snapshot_saved = False
        for index, (state, sim_time, touched) in enumerate(frames):
            render_data.qpos[:] = state
            render_data.time = sim_time
            mujoco.mj_forward(model, render_data)
            labels = [
                "MJWarp GPU",
                f"物品 {object_id}",
                f"物体接触 {touched}",
            ]
            video.capture(render_data, labels)
            if index == 0:
                snapshots.append(str(video.snapshot(render_data, "initial", labels)))
            if touched and not contact_snapshot_saved:
                snapshots.append(str(video.snapshot(render_data, "parcel-contact", labels)))
                contact_snapshot_saved = True
        video.finish("GPU 轨迹已保存")
        snapshots.extend(video.snapshots)
    return {
        "video": str(path),
        "snapshots": sorted(set(snapshots)),
        "frames": len(frames),
        "parcel_contact_seen": parcel_contact,
    }


def main():
    from ducklab.retrieval.scene import BIN, OBJECTS, make_scene

    default_out = ROOT / "runs/fetch/retrieval-mjwarp-probe-20261123"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--object-id", default="bath_bird")
    parser.add_argument("--worlds", type=int, nargs="+", default=(1, 8, 16, 32, 64))
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--graph-steps", type=int, default=8)
    parser.add_argument("--frame-stride", type=int, default=20)
    parser.add_argument("--nconmax", type=int, default=128)
    parser.add_argument("--njmax", type=int, default=128)
    parser.add_argument(
        "--physics-profile",
        choices=("legacy-no-slip-off", "implicit-noslip"),
        default="legacy-no-slip-off",
        help="Compare the legacy MJWarp workaround or a near-production implicit/no-slip profile.",
    )
    parser.add_argument("--seed", type=int, default=20261123)
    parser.add_argument("--out", type=Path, default=default_out)
    args = parser.parse_args()
    if not args.worlds or min(args.worlds) < 1 or len(set(args.worlds)) != len(args.worlds):
        parser.error("world counts must be positive and unique")
    if args.steps < 1 or args.graph_steps < 1 or args.steps % args.graph_steps:
        parser.error("steps must be positive and divisible by graph-steps")
    if args.frame_stride < 1 or args.steps % args.frame_stride:
        parser.error("steps must be divisible by frame-stride")
    if args.nconmax < 1 or args.njmax < 1:
        parser.error("contact and constraint capacities must be positive")
    if args.out.joinpath("summary.json").exists():
        parser.error(f"output already has a summary; choose a new --out: {args.out}")

    items = {item["id"]: item for item in OBJECTS}
    if args.object_id not in items:
        parser.error(f"unknown object id: {args.object_id}")
    item = items[args.object_id]
    if item["split"] not in ("train", "flat_train", "challenge"):
        parser.error("GPU probe objects must come from train/flat_train/challenge splits")

    production_model, _ = make_scene(item, BIN, moving_jaw=True)
    model, _ = make_scene(item, BIN, moving_jaw=True)
    production_physics = {
        "integrator": int(production_model.opt.integrator),
        "integrator_name": integrator_name(production_model.opt.integrator),
        "noslip_iterations": int(production_model.opt.noslip_iterations),
        "solver_tolerance": float(production_model.opt.tolerance),
    }
    if args.physics_profile == "implicit-noslip":
        version = tuple(int(part) for part in mjw.__version__.split(".")[:3])
        if version < (4, 0, 1):
            parser.error("implicit-noslip profile requires mujoco-warp >= 4.0.1")
        model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICIT
        model.opt.noslip_iterations = production_model.opt.noslip_iterations
    else:
        model.opt.noslip_iterations = 0
    model.opt.tolerance = max(production_model.opt.tolerance, 1e-6)
    candidate_physics = {
        "integrator": int(model.opt.integrator),
        "integrator_name": integrator_name(model.opt.integrator),
        "noslip_iterations": int(model.opt.noslip_iterations),
        "solver_tolerance": float(model.opt.tolerance),
    }
    if model.nv > 60 and model.opt.jacobian == mujoco.mjtJacobian.mjJAC_DENSE:
        parser.error(f"MJWarp dense solver limit exceeded: nv={model.nv}")

    wp.init()
    wp.set_device("cuda:0")
    device = wp.get_device("cuda:0")
    warp_model = mjw.put_model(model)

    rows = []
    max_worlds = max(args.worlds)
    all_qpos, all_qvel = initial_states(
        production_model, item, max_worlds, args.seed
    )
    for world_count in sorted(args.worlds):
        qpos, qvel = all_qpos[:world_count], all_qvel[:world_count]
        cpu_production_config = run_cpu(
            production_model, qpos, qvel, args.steps
        )
        cpu = run_cpu(model, qpos, qvel, args.steps)
        gpu = run_gpu(
            model,
            warp_model,
            qpos,
            qvel,
            args.steps,
            args.graph_steps,
            args.nconmax,
            args.njmax,
        )
        qpos_error = np.abs(cpu["qpos"] - gpu["qpos"])
        cpu_physics_error = np.abs(
            cpu_production_config["qpos"] - cpu["qpos"]
        )
        gpu_production_error = np.abs(
            cpu_production_config["qpos"] - gpu["qpos"]
        )
        contact_delta = np.abs(cpu["contact_counts"] - gpu["contact_counts"])
        production_contact_delta = np.abs(
            cpu_production_config["contact_counts"] - cpu["contact_counts"]
        )
        rows.append(
            {
                "worlds": world_count,
                "steps_per_world": args.steps,
                "production_config_cpu_seconds": round(
                    cpu_production_config["seconds"], 4
                ),
                "cpu_seconds": round(cpu["seconds"], 4),
                "cpu_env_steps_per_second": round(cpu["env_steps_per_second"], 1),
                "gpu_seconds": round(gpu["seconds"], 4),
                "gpu_env_steps_per_second": round(gpu["env_steps_per_second"], 1),
                "speedup_vs_sequential_cpu": round(
                    gpu["env_steps_per_second"] / cpu["env_steps_per_second"], 3
                ),
                "qpos_error_max": float(qpos_error.max()),
                "qpos_error_rms": float(np.sqrt(np.mean(qpos_error**2))),
                "candidate_cpu_vs_production_qpos_error_max": float(
                    cpu_physics_error.max()
                ),
                "candidate_gpu_vs_production_qpos_error_max": float(
                    gpu_production_error.max()
                ),
                "contact_count_production_config_cpu_mean": float(
                    cpu_production_config["contact_counts"].mean()
                ),
                "contact_count_cpu_mean": float(cpu["contact_counts"].mean()),
                "contact_count_gpu_mean": float(gpu["contact_counts"].mean()),
                "contact_count_absdiff_mean": float(contact_delta.mean()),
                "max_nefc": gpu["max_nefc"],
                "max_solver_iterations": gpu["max_solver_iterations"],
                "overflow_check_supported": gpu["overflow_check_supported"],
                "candidate_cpu_vs_production_contact_absdiff_mean": float(
                    production_contact_delta.mean()
                ),
                "gpu_samples": gpu["gpu_samples"],
            }
        )
        print(json.dumps(rows[-1], indent=2), flush=True)
        save_json(
            args.out / "progress.json",
            {
                "physics_profile": args.physics_profile,
                "completed_worlds": [row["worlds"] for row in rows],
                "results": rows,
            },
        )

    media = record_gpu_video(
        model,
        warp_model,
        all_qpos,
        all_qvel,
        args.steps,
        args.frame_stride,
        args.out / "videos" / f"mjwarp-{args.object_id}-{max_worlds}world-batch.mp4",
        args.object_id,
        args.nconmax,
        args.njmax,
    )
    summary = {
        "purpose": "MJWarp physics-only throughput and parity probe; not a PPO training run",
        "physics_profile": args.physics_profile,
        "object_id": args.object_id,
        "object_split": item["split"],
        "seed": args.seed,
        "mujoco_version": mujoco.__version__,
        "mujoco_warp_version": mjw.__version__,
        "warp_version": wp.__version__,
        "gpu": str(device),
        "model": {"nq": model.nq, "nv": model.nv, "nu": model.nu,
                  "nbody": model.nbody, "ngeom": model.ngeom},
        "physics_deviation": {
            "production": production_physics,
            "candidate": candidate_physics,
            "differences": [
                key
                for key in production_physics
                if production_physics[key] != candidate_physics[key]
            ],
        },
        "steps_per_world": args.steps,
        "nconmax_per_world": args.nconmax,
        "njmax_per_world": args.njmax,
        "graph_steps": args.graph_steps,
        "results": rows,
        "media": media,
        "note": "The production-parameter CPU, candidate-parameter CPU, and candidate GPU runs use the same MuJoCo version, identical initial states, and zero motor control; CPU throughput is sequential, not the existing multiprocessing baseline. This is not a controlled grasp task or PPO training run.",
    }
    save_json(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
