"""Fine-tune PPO against the package-matched gait and stock 50 Hz command path only."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing as mp
from pathlib import Path

if __package__:
    from .contract import FROZEN, GATES, TransferEnv, load_actor, torch
else:
    from contract import FROZEN, GATES, TransferEnv, load_actor, torch
from tools.train_self_catch import stats, update


def init_worker():
    global env, local_policy
    torch.set_num_threads(1)
    env = TransferEnv('handoff_intent_20ms')
    local_policy = load_actor()


@torch.inference_mode()
def collect(job):
    state, seeds, stochastic = job
    local_policy.load_state_dict({k: torch.from_numpy(v) for k, v in state.items()})
    rows = []
    for seed in seeds:
        rng = torch.Generator().manual_seed(3_500_000 + seed) if stochastic else None
        obs = env.reset(seed)
        oo, aa, lp, vv = [], [], [], []
        for phase in range(2):
            x = torch.from_numpy(obs)
            dist = local_policy.dist(x, phase)
            raw = (dist.loc + dist.scale * torch.randn(dist.loc.shape, generator=rng)
                   if stochastic else dist.loc)
            oo.append(obs)
            aa.append(raw.numpy())
            lp.append(float(dist.log_prob(raw).sum()))
            vv.append(float(local_policy.critics[phase](x)[0]))
            obs, ret, done, info = env.step(torch.tanh(raw).numpy())
            assert done == (phase == 1)
        rows.append((oo, aa, lp, vv, float(ret), dict(seed=seed, **info)))
    return rows


def rollout(pool, policy, seeds, workers, *, stochastic):
    state = {k: v.detach().cpu().numpy() for k, v in policy.state_dict().items()}
    jobs = [(state, seeds[i::workers], stochastic) for i in range(min(len(seeds), workers))]
    return sorted((r for batch in pool.map(collect, jobs) for r in batch), key=lambda r: r[5]['seed'])


def seed_ranges(args):
    train = range(args.seed, args.seed + args.episodes_per_update * args.updates)
    validation = range(args.validation_seed, args.validation_seed + 16)
    heldout = range(args.test_seed, args.test_seed + 100)
    ranges = sorted((train, validation, heldout), key=lambda seed_range: seed_range.start)
    if ranges[0].start < 0 or any(a.stop > b.start for a, b in zip(ranges, ranges[1:])):
        raise ValueError('seeds must be nonnegative; train, validation and held-out ranges must be disjoint')
    return train, validation, heldout


def evaluate(pool, policy, seeds, workers):
    rows = rollout(pool, policy, seeds, workers, stochastic=False)
    return dict(**stats(rows), action_mode='tanh(mean)', rows=[r[5] for r in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--episodes-per-update', type=int, default=96)
    parser.add_argument('--updates', type=int, default=40)
    parser.add_argument('--seed', type=int, default=7100000)
    parser.add_argument('--validation-seed', type=int, default=7500000)
    parser.add_argument('--test-seed', type=int, default=7600000)
    args = parser.parse_args()
    if args.workers < 1 or args.episodes_per_update < max(2, args.workers) or args.updates < 1:
        parser.error('need positive workers/updates and at least max(2,workers) training episodes')
    try:
        _, val_seeds, test_seeds = seed_ranges(args)
    except ValueError as error:
        parser.error(str(error))
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    policy = load_actor()
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-4)
    best_state = {k: v.detach().clone() for k,v in policy.state_dict().items()}
    selected_update = 0
    history = []
    seed_base = args.seed
    with ProcessPoolExecutor(args.workers, mp_context=mp.get_context('spawn'), initializer=init_worker) as pool:
        baseline = evaluate(pool, policy, val_seeds, args.workers)
        best = (baseline['success'], baseline['mean_return'])
        for update_no in range(args.updates):
            rows = rollout(pool, policy, range(seed_base, seed_base + args.episodes_per_update), args.workers, stochastic=True)
            kl = update(policy, optimizer, rows)
            validation = evaluate(pool, policy, val_seeds, args.workers)
            rank = (validation['success'], validation['mean_return'])
            if rank > best:
                best, best_state = rank, {k:v.detach().clone() for k,v in policy.state_dict().items()}
                selected_update = update_no + 1
            history.append(dict(update=update_no+1, train_success=sum(r[5]['success'] for r in rows),
                                train_n=len(rows), kl=kl, validation={k:v for k,v in validation.items() if k!='rows'}))
            print(history[-1], flush=True)
            seed_base += args.episodes_per_update
        policy.load_state_dict(best_state)
        torch.save(best_state, args.out/'selected.pt')
        held = evaluate(pool, policy, test_seeds, args.workers)
    report = dict(hardware=False, source_checkpoint=str(FROZEN),
                  source_sha256=hashlib.sha256(FROZEN.read_bytes()).hexdigest(),
                  selected_sha256=hashlib.sha256((args.out/'selected.pt').read_bytes()).hexdigest(),
                  gaits=GATES, mode='handoff_intent_20ms', selected_update=selected_update,
                  args={k:v for k,v in vars(args).items() if k!='out'},
                  action_modes={'train':'seeded Gaussian exploration', 'validation':'tanh(mean)', 'heldout':'tanh(mean)'},
                  selection='initial actor plus 40 updates by default: strict successes then return on 16 fixed validation seeds; freeze before 100 disjoint held-out seeds',
                  baseline_validation={k:v for k,v in baseline.items() if k!='rows'},
                  history=history, best_validation=best, heldout=held)
    (args.out/'report.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print('HELD OUT', held['success'], '/', held['n'], flush=True)


if __name__ == '__main__':
    main()
