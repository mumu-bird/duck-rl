#!/usr/bin/env python3
"""Search approach timing on development seeds, freeze it and score paired holdouts."""
import argparse
import csv
import itertools
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MUJOCO_GL','egl')
from ducklab.arena import Arena


def write_rows(path, rows):
    with path.open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def calibrate(arena, directory):
    trials = []
    for speed,ticks,settle in itertools.product((.08,.11,.14), (20,28,36,44,52,60), (0,12,25)):
        spec = dict(slot='kickR',speed=speed,walk_ticks=ticks,settle_ticks=settle)
        runs = [arena.run('penalty_approach',seed,approach=spec) for seed in range(401,405)]
        trials.append({**spec,'passes':sum(r['success'] for r in runs),
                       'goals':sum(r['metrics']['goal'] for r in runs),
                       'falls':sum(r['fell'] for r in runs),'max_exceed_deg':max(r['max_exceed_deg'] for r in runs),
                       'development_seeds':'401 402 403 404'})
    winner = max(trials,key=lambda r:(r['passes'],r['goals'],-r['falls'],-r['max_exceed_deg'],-r['walk_ticks']))
    write_rows(directory/'search.csv',trials)
    frozen = {k:winner[k] for k in spec}
    (directory/'controller.json').write_text(json.dumps(frozen,indent=2)+'\n')
    return frozen


def evaluate(arena,directory,frozen):
    for method,task in [('baseline','penalty_baseline'),('candidate','penalty_approach'),('push','push_recovery')]:
        rows = []
        for seed in range(18400,18500):
            result = arena.run(task,seed,approach=frozen)
            rows.append({k:result[k] for k in ('task','episode','seed','success','fell','joint_limit_violations','nonfinite')})
            rows[-1]['task'] = 'push_recovery' if method=='push' else 'penalty_kick'
            rows[-1]['details'] = json.dumps(result,ensure_ascii=False,separators=(',',':'))
        write_rows(directory/(method+'.csv'),rows)
        print(f"{method}: {sum(r['success'] for r in rows)}/{len(rows)}",flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    arena = Arena()
    try:
        config = calibrate(arena,args.output)
        print('Fixed approach:',config,flush=True)
        evaluate(arena,args.output,config)
    finally:
        arena.close()
