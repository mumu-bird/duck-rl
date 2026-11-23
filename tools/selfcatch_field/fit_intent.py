"""Fit a bounded bias on top of the original PPO for the stock command path (simulation only)."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
from pathlib import Path

import numpy as np
if __package__:
    from .contract import TransferEnv, load_actor, torch
else:
    from contract import TransferEnv, load_actor, torch


def init():
    global env, actor
    torch.set_num_threads(1)
    env, actor = TransferEnv('handoff_intent_20ms'), load_actor()


def trial(job):
    bias, seed = job
    obs = env.reset(seed)
    with torch.inference_mode():
        for phase in (0, 1):
            mean = actor.actors[phase](torch.from_numpy(obs)).numpy()
            obs, _, _, r = env.step(np.tanh(mean + bias[phase]))
    tr = np.asarray(env.trace)
    air = tr[(tr[:, 0] >= 0) & (tr[:, 8] == 0) & (tr[:, 11] == 0)]
    h, near = 0., 0.
    if len(air) > 2:
        h = float(np.clip((air[:, 3].max() - air[0, 3]) / .02, 0., 1.6))
        later = air[8:25]
        if len(later):
            near = float(np.exp(-np.min(np.linalg.norm(later[:, 1:4]-later[:, 5:8], axis=1)) / .025))
    score = (2*r['armed'] + r['upright'] + 2*h + near + r['released'] + 3*r['recaught']
             + 3*(r['recaught'] and r['held']) + 20*r['success']
             + 2*min(1., r['rise_m']/.012, r['air_s']/.08, r['forward_m']/.015, r['chase_m']/.005))
    return float(score), r


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    p.add_argument('--pop', type=int, default=64)
    p.add_argument('--generations', type=int, default=30)
    a = p.parse_args()
    if a.workers < 1 or a.pop < 4 or a.generations < 1 or a.pop * a.generations > 300_000:
        p.error('need positive workers/generations and population >=4; training must not overlap validation')
    a.out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(61371)
    mean, sigma = np.zeros((2, 6)), np.full((2, 6), 1.65)
    bestval, best = (-1., -100.), mean.copy()
    history = []
    with ProcessPoolExecutor(a.workers, mp_context=mp.get_context('spawn'), initializer=init) as pool:
        for gen in range(a.generations):
            population = np.clip(rng.normal(mean, sigma, (a.pop, 2, 6)), -4, 4)
            population[0] = best
            rs = list(pool.map(trial, [(x, 6100000 + gen * a.pop + i) for i, x in enumerate(population)], chunksize=3))
            elite = population[np.argsort([r[0] for r in rs])[-max(4, a.pop//7):]]
            mean = .25*mean + .75*elite.mean(axis=0)
            sigma = np.maximum(.15, .25*sigma+.75*elite.std(axis=0))
            candidate = elite[-1]
            val = list(pool.map(trial, [(candidate, 6400000+i) for i in range(6)]))
            rank = (sum(r['success'] for _, r in val), float(np.mean([s for s, _ in val])))
            if rank > bestval:
                bestval, best = rank, candidate.copy()
            history.append(dict(generation=gen, train_success=sum(r['success'] for _,r in rs), validation=rank))
            print(gen, history[-1], flush=True)
        test = [dict(seed=6500000+i, **r) for i, (_,r) in enumerate(pool.map(trial, [(best,6500000+i) for i in range(100)], chunksize=3))]
    print('HELD OUT', sum(r['success'] for r in test), '/', len(test), flush=True)
    (a.out/'fit.json').write_text(json.dumps(dict(hardware=False, method='CEM residual on frozen PPO, factory-weight proxy',
        bias=best.tolist(), history=history, success=sum(r['success'] for r in test), n=len(test), test=test), indent=2)+'\n')


if __name__ == '__main__':
    main()
