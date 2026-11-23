#!/usr/bin/env python3
"""Parallel on-policy PPO for the two-decision Wonder Duck self-catch drill."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.distributions import Normal

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ducklab.self_catch import ACT, OBS, SelfCatchEnv  # noqa: E402


class Policy(nn.Module):
    def __init__(self):
        super().__init__()

        def net(out):
            return nn.Sequential(nn.Linear(OBS, 32), nn.Tanh(), nn.Linear(32, out))

        self.actors = nn.ModuleList([net(ACT), net(ACT)])
        self.critics = nn.ModuleList([net(1), net(1)])
        self.logstd = nn.Parameter(torch.full((2, ACT), -0.16))
        for actor in self.actors:
            nn.init.orthogonal_(actor[-1].weight, gain=0.01)
            nn.init.zeros_(actor[-1].bias)

    def dist(self, x, phase):
        return Normal(self.actors[phase](x), self.logstd[phase].clamp(-3, 0.4).exp())


def init_worker(clearance, fine_jaw):
    global env, local_policy, clearance_curriculum
    torch.set_num_threads(1)
    env, local_policy = SelfCatchEnv(fine_jaw=fine_jaw), Policy().eval()
    clearance_curriculum = clearance


def margin(info):
    return max(
        0.0,
        min(
            info["air_s"] / 0.08,
            info["rise_m"] / 0.012,
            info["forward_m"] / 0.015,
            info["chase_m"] / 0.005,
            1.7,
        ),
    )


def seed_from_rollout(policy, log):
    # Seed the second curriculum stage from a stochastic PPO trial, never test or validation.
    with log.open() as f:
        r = max(
            (r for line in f if (r := json.loads(line))["held"] and r["recaught"]),
            key=margin,
        )
    for phase, act in enumerate(r["actions"]):
        bias = torch.atanh(torch.tensor(act).clamp(-0.985, 0.985))
        with torch.no_grad():
            policy.actors[phase][-1].bias.copy_(bias)
    with torch.no_grad():
        policy.logstd.fill_(-1.25)
    return dict(source=str(log), seed=r["seed"], actions=r["actions"], margin=margin(r))


@torch.inference_mode()
def collect(job):
    state, seeds, training, capture = job
    local_policy.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
    rows = []
    for idx, seed in enumerate(seeds):
        rng = torch.Generator().manual_seed(3_500_000 + seed)
        rec = idx == 0 and capture
        obs = env.reset(seed, record=rec)
        oo, aa, lp, vv = [], [], [], []
        for phase in range(2):
            x = torch.from_numpy(obs)
            dist = local_policy.dist(x, phase)
            raw = (
                dist.loc + dist.scale * torch.randn(ACT, generator=rng)
                if training
                else dist.loc
            )
            oo.append(obs)
            aa.append(raw.numpy())
            lp.append(float(dist.log_prob(raw).sum()))
            vv.append(float(local_policy.critics[phase](x)[0]))
            obs, ret, done, info = env.step(torch.tanh(raw).numpy())
            assert done == (phase == 1)
        if clearance_curriculum and info["recaught"] and info["held"]:
            ret += 8 * margin(info)
        tape = (np.array(env.frames), np.array(env.trace)) if rec else None
        rows.append((oo, aa, lp, vv, ret, dict(seed=seed, **info), tape))
    return os.getpid(), rows


def rollout(pool, policy, seeds, workers, training=False, capture=False):
    state = {k: v.detach().cpu().numpy() for k, v in policy.state_dict().items()}
    jobs = [
        (state, seeds[n::workers], training, capture)
        for n in range(min(workers, len(seeds)))
    ]
    results = list(pool.map(collect, jobs))
    return sorted(
        [r for _, rs in results for r in rs], key=lambda r: r[5]["seed"]
    ), sorted({p for p, _ in results})


def stats(rows):
    return dict(
        n=len(rows),
        **{
            k: sum(r[5][k] for r in rows)
            for k in ("armed", "released", "recaught", "held", "success", "ground")
        },
        mean_return=float(np.mean([r[4] for r in rows])),
    )


def store_tapes(out, rows, tag):
    for r in rows:
        if r[6] is not None:
            qpos, trace = r[6]
            np.savez_compressed(
                out / f"{tag}-{r[5]['seed']}.npz",
                qpos=qpos,
                trace=trace,
                raw_actions=r[1],
                meta=json.dumps(r[5]),
            )


def update(policy, opt, rows):
    obs, raw, oldlog, v_old, terminal = [
        torch.tensor(np.asarray([r[k] for r in rows]), dtype=torch.float32)
        for k in range(5)
    ]
    # GAE for a two-step MDP: first reward zero; actual contact score is terminal.
    a1 = terminal - v_old[:, 1]
    a0 = 0.99 * v_old[:, 1] - v_old[:, 0] + 0.99 * 0.95 * a1
    adv = torch.stack([a0, a1], 1)
    target = adv + v_old
    adv = (adv - adv.mean(0)) / (adv.std(0) + 1e-7)
    for _ in range(8):
        for idx in torch.randperm(len(rows)).split(32):
            loss = torch.tensor(0.0)
            for p in range(2):
                d = policy.dist(obs[idx, p], p)
                # The buffer stores latent Gaussian actions; the fixed tanh Jacobian
                # cancels in the new/old likelihood ratio for the same raw action.
                ratio = (d.log_prob(raw[idx, p]).sum(-1) - oldlog[idx, p]).exp()
                clip = ratio.clamp(0.8, 1.2)
                surrogate = torch.minimum(
                    ratio * adv[idx, p], clip * adv[idx, p]
                ).mean()
                val = policy.critics[p](obs[idx, p]).squeeze(-1)
                loss = loss + 0.5 * (
                    -surrogate
                    + 0.25 * (val - target[idx, p]).square().mean()
                    - 0.0005 * d.entropy().mean()
                )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        with torch.no_grad():
            kl = (
                torch.stack(
                    [
                        (
                            oldlog[:, p]
                            - policy.dist(obs[:, p], p).log_prob(raw[:, p]).sum(-1)
                        ).mean()
                        for p in range(2)
                    ]
                )
                .mean()
                .item()
            )
        if kl > 0.065:
            break
    return kl


def protect():
    paths = sorted((ROOT / "configs").glob("*")) + sorted(
        (ROOT / "vendor/wonderduck/policies").glob("*.onnx")
    )
    paths += [
        ROOT / "tools/fit_retrieval_grasp.py",
        ROOT / "tools/run_retrieval.py",
        ROOT / "ducklab/retrieval/episode.py",
        ROOT / "docs/tasks/toss-retrieval.md",
        ROOT / "runs/fetch/round3-vision/change.pt",
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
        if p.is_file()
    }


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def assess(run, pool, workers, count, baseline, test_seed):
    out = run / "assessment"
    out.mkdir(exist_ok=False)
    tapes = out / "tapes"
    tapes.mkdir()
    policy = Policy()
    report = {}
    for label, file in [
        ("before", baseline or run / "initial.pt"),
        ("after", run / "selected.pt"),
    ]:
        policy.load_state_dict(torch.load(file, map_location="cpu", weights_only=True))
        rows, _ = rollout(
            pool,
            policy,
            list(range(test_seed, test_seed + count)),
            workers,
            capture=True,
        )
        store_tapes(tapes, rows, label)
        report[label] = dict(
            checkpoint=str(file),
            sha256=hashlib.sha256(file.read_bytes()).hexdigest(),
            stats=stats(rows),
            rows=[r[5] for r in rows],
        )
    write_json(out / "paired.json", report)
    print(
        "Paired independent test:",
        {k: v["stats"] for k, v in report.items()},
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--per-worker", type=int, default=16)
    p.add_argument("--updates", type=int, default=70)
    p.add_argument("--seed", type=int, default=4171)
    p.add_argument("--first-seed", type=int, default=120000)
    p.add_argument("--validation-seed", type=int, default=810000)
    p.add_argument("--test-seed", type=int, default=930000)
    p.add_argument(
        "--clearance",
        action="store_true",
        help="emphasize the weakest flight criterion on retained catches",
    )
    p.add_argument(
        "--fine-jaw",
        action="store_true",
        help="5 ms jaw schedule and full legal neck preparation range",
    )
    p.add_argument(
        "--bootstrap-log",
        type=Path,
        help="optional training-only PPO episode log to initialize motor means",
    )
    p.add_argument(
        "--baseline", type=Path, help="untrained policy for paired assessment"
    )
    p.add_argument(
        "--assess",
        type=int,
        metavar="N",
        help="evaluate existing run on untouched paired seeds",
    )
    a = p.parse_args()
    if (
        min(a.workers, a.per_worker, a.updates) < 1
        or a.workers * a.per_worker < 2
        or a.assess is not None
        and a.assess < 1
    ):
        p.error(
            "positive counts and at least two training trajectories per update required"
        )
    torch.set_num_threads(1)
    torch.manual_seed(a.seed)
    if not a.assess:
        a.out.mkdir(parents=True, exist_ok=False)
        (a.out / "tapes").mkdir()
        (a.out / "checkpoints").mkdir()
    elif (a.out / "run-args.json").exists():
        a.fine_jaw = json.loads((a.out / "run-args.json").read_text()).get(
            "fine_jaw", False
        )
    with ProcessPoolExecutor(
        a.workers,
        mp_context=mp.get_context("spawn"),
        initializer=init_worker,
        initargs=(a.clearance, a.fine_jaw),
    ) as pool:
        if a.assess:
            return assess(a.out, pool, a.workers, a.assess, a.baseline, a.test_seed)
        before = protect()
        write_json(a.out / "protected-before.json", before)
        policy = Policy()
        opt = torch.optim.Adam(policy.parameters(), lr=0.0016, eps=1e-5)
        write_json(
            a.out / "run-args.json",
            {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        )
        if a.bootstrap_log:
            write_json(
                a.out / "bootstrap.json", seed_from_rollout(policy, a.bootstrap_log)
            )
        torch.save(policy.state_dict(), a.out / "initial.pt")
        batch = a.workers * a.per_worker
        start = time.monotonic()
        history = []
        best = None
        chosen = None
        pids = set()
        for u in range(1, a.updates + 1):
            seed = a.first_seed + (u - 1) * batch
            rows, pp = rollout(
                pool,
                policy,
                list(range(seed, seed + batch)),
                a.workers,
                training=True,
                capture=u in {1, max(1, a.updates // 2), a.updates},
            )
            pids.update(pp)
            store_tapes(a.out / "tapes", rows, f"train-{u:03}")
            with (a.out / "episodes.jsonl").open("a") as f:
                for r in rows:
                    f.write(
                        json.dumps(
                            dict(
                                update=u,
                                return_=r[4],
                                actions=np.tanh(r[1]).tolist(),
                                **r[5],
                            )
                        )
                        + "\n"
                    )
            kl = update(policy, opt, rows)
            h = dict(
                update=u,
                train=stats(rows),
                kl=kl,
                seconds=round(time.monotonic() - start, 2),
            )
            if u == 1 or u % 3 == 0 or u == a.updates:
                val, _ = rollout(
                    pool,
                    policy,
                    list(range(a.validation_seed, a.validation_seed + 24)),
                    a.workers,
                )
                s = stats(val)
                h["validation"] = s
                key = (s["success"], s["mean_return"])
                if best is None or key > best:
                    best, chosen = key, u
                    torch.save(policy.state_dict(), a.out / "selected.pt")
                    torch.save(
                        policy.state_dict(), a.out / "checkpoints" / f"ppo-{u:03}.pt"
                    )
            history.append(h)
            print(json.dumps(h), flush=True)
            with (a.out / "progress.jsonl").open("a") as f:
                f.write(json.dumps(h) + "\n")
        torch.save(policy.state_dict(), a.out / "last.pt")
        after = protect()
        write_json(a.out / "protected-after.json", after)
        if after != before:
            raise RuntimeError("a protected older task file changed during training")
        write_json(
            a.out / "training-report.json",
            dict(
                algorithm="on-policy PPO-Clip with GAE",
                seed=a.seed,
                parallel_workers=a.workers,
                worker_pids=sorted(pids),
                torch=torch.__version__,
                device="cpu",
                episodes=a.updates * batch,
                high_level_steps=a.updates * batch * 2,
                physics_steps_per_episode=(45 + 64) * 4,
                selected_update=chosen,
                training_seconds=round(time.monotonic() - start, 2),
                unchanged_protected_files=len(before),
                history=history,
            ),
        )


if __name__ == "__main__":
    main()
