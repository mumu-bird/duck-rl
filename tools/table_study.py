#!/usr/bin/env python3
"""Paired tabletop ablation with identical poses, color, camera and motor parameters."""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MUJOCO_GL','egl')
from ducklab.arena import Arena
from ducklab.suite import EXPLORATORY, FIELDS, aggregate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--start',type=int,default=20500)
    parser.add_argument('--seeds',type=int,default=100)
    args = parser.parse_args()
    if args.seeds<1:
        parser.error('--seeds must be positive')
    directory = args.output
    directory.mkdir(parents=True,exist_ok=False)
    arena = Arena()
    try:
        with (directory/'episodes.csv').open('w',newline='') as stream:
            writer = csv.DictWriter(stream,fieldnames=FIELDS)
            writer.writeheader()
            for seed in range(args.start,args.start+args.seeds):
                for task in EXPLORATORY:
                    video = directory/'videos'/f'{task}-{seed}.mp4' if seed in (args.start,args.start+1) else None
                    result = arena.run(task,seed,video_path=video)
                    row = {key:result[key] for key in FIELDS if key in result}
                    row.update(setup=json.dumps(result['setup']),metrics=json.dumps(result['metrics']),
                               video=str(video.relative_to(directory)) if video else '')
                    writer.writerow(row)
                stream.flush()
        print(json.dumps(aggregate(directory)['tasks'],indent=2))
    finally:
        arena.close()


if __name__=='__main__':
    main()
