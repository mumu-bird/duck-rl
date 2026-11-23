#!/usr/bin/env python3
"""Run tonight's simulation queue until 07:00 Asia/Shanghai, with durable progress."""
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MUJOCO_GL','egl')
os.environ['OPENBLAS_NUM_THREADS'] = '1'
from ducklab.suite import TASKS, BatchAlreadyRunning, run_suite


def end_of_night(now):
    if 7 <= now.hour < 22:
        return None
    end = now.replace(hour=7,minute=0,second=0,microsecond=0)
    return end+timedelta(days=1) if now.hour>=22 else end


if __name__=='__main__':
    end = end_of_night(datetime.now(ZoneInfo('Asia/Shanghai')))
    if end is None:
        print('Outside the 22:00–07:00 simulation window.')
    else:
        day = end.strftime('%Y-%m-%d')
        try:
            run_suite(ROOT/'runs/nightly'/day,seed=end.date().toordinal()*100000,
                      task_filter=TASKS,video_every=503,pace=1,deadline=end.timestamp(),video_cap=64)
        except BatchAlreadyRunning as exc:
            print(exc)
