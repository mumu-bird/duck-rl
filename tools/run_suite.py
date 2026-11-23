#!/usr/bin/env python3
"""Run a finite or wall-clock-bounded local policy evaluation batch."""
import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('MUJOCO_GL','egl')
from ducklab.suite import SUPPORTED_TASKS, run_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/batch'))
    parser.add_argument('--episodes', type=int, default=0,
                        help='new episode count (0 means run until --duration-hours expires)')
    parser.add_argument('--duration-hours', type=float, default=0)
    parser.add_argument('--until', type=datetime.fromisoformat,
                        help='absolute timezone-aware ISO timestamp for the final episode')
    parser.add_argument('--seed', type=int, default=71000)
    parser.add_argument('--video-every', type=int, default=0,
                        help='record each Nth rollout (0 disables video)')
    parser.add_argument('--video-cap', type=int, default=64)
    parser.add_argument('--pace-seconds', type=float, default=0)
    parser.add_argument('--tasks', nargs='+', choices=SUPPORTED_TASKS)
    parser.add_argument('--exploratory', action='store_true',
                        help='include RGB and desktop interaction prototypes')
    parser.add_argument("--audio-endpoint", help="explicit real-device audio; default logs simulation events")
    parser.add_argument("--audio-token-file", type=Path)
    args = parser.parse_args()
    if min(args.episodes,args.duration_hours,args.video_every,args.video_cap,args.pace_seconds) < 0:
        parser.error('limits and pacing cannot be negative')
    if args.until is not None and (args.until.tzinfo is None or args.until.utcoffset() is None):
        parser.error('--until needs an explicit UTC offset')
    if not args.episodes and not args.duration_hours and args.until is None:
        parser.error('specify --episodes or --duration-hours')
    episodes = args.episodes if args.episodes else None
    summary = run_suite(args.output, episodes=episodes, seed=args.seed,
                        duration_hours=args.duration_hours, video_every=args.video_every,
                        task_filter=args.tasks, exploratory=args.exploratory, pace=args.pace_seconds,
                        video_cap=args.video_cap, deadline=args.until.timestamp() if args.until else None,
                        audio_endpoint=args.audio_endpoint, audio_token_file=args.audio_token_file)
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__ == '__main__':
    main()
