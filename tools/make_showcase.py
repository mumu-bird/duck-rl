#!/usr/bin/env python3
"""Assemble selected real episode replays with their sample sizes and outcomes."""
import csv
import json
import os
import sys
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('MUJOCO_GL','egl')
from ducklab.arena import Arena
from ducklab.video import font


def render_missing(arena, path, task, seed, expected):
    if not path.exists():
        spec = json.loads((ROOT/'runs/penalty-study/controller.json').read_text())
        result = arena.run(task,seed,path,spec)
        if result['success'] != expected:
            raise ValueError('Replay disagrees with source evaluation')


def main():
    runs = ROOT/'runs'
    directory = runs/'showcase'
    directory.mkdir(parents=True,exist_ok=True)
    selection = [
        ('table-paired','table_fixture',20500,'模拟夹持器开启','100 个摆位中成功 96 次。本回合看嘴端接触、抬升和释放。'),
        ('table-paired','table_contact',20500,'同一摆位，关闭夹持器','相同控制流程下成功 0/100。头部抬起后，红块留在桌上。'),
        ('table-paired','table_push',20521,'纯接触推送的成功样例','这次把方块推到了绿区。100 个摆位中仅有 11 次通过。'),
        ('penalty-study','penalty_baseline',18400,'固定球位的原地踢球对照','100 次均未进球。注意球和右脚起始位置的距离。'),
        ('penalty-study','penalty_approach',18400,'同一球位，增加助跑','冻结助跑参数后独立测试 100 次，94 次有效进球。'),
        ('penalty-study','penalty_approach',18405,'助跑点球的失败样例','保留这次未达标的回放，与上一球对照。'),
    ]
    arena = Arena()
    out = directory/'wonder-duck.mp4'
    temp = out.with_suffix('.partial.mp4')
    encoder = imageio_ffmpeg.write_frames(str(temp),(768,484),fps=10,codec='libx264',
                 pix_fmt_out='yuv420p',quality=8,macro_block_size=1,ffmpeg_log_level='error',
                 output_params=['-threads','2','-movflags','+faststart'])
    encoder.send(None)
    manifest,elapsed = [],0
    try:
        for folder,task,seed,title,subtitle in selection:
            source = runs/folder/'videos'/f'{task}-{seed}.mp4'
            if folder=='table-paired':
                table,task_name = 'episodes.csv',task
            else:
                table,task_name = ('baseline.csv' if task=='penalty_baseline' else 'candidate.csv'),'penalty_kick'
            with (runs/folder/table).open() as f:
                row = next(r for r in csv.DictReader(f) if r['task']==task_name and int(r['seed'])==seed)
            render_missing(arena,source,task,seed,int(row['success']))
            chapter = dict(file=str(source.relative_to(runs)),seed=seed,title=title,start_s=elapsed/10,
                           expected_success=int(row['success']))
            frames = imageio_ffmpeg.read_frames(str(source),pix_fmt='rgb24')
            try:
                meta = next(frames)
                if meta['size'] != (768,416) or meta['fps'] != 10:
                    raise ValueError(f'Unexpected replay format: {source}')
                for raw in frames:
                    canvas = Image.new('RGB',(768,484),'#17333c')
                    canvas.paste(Image.frombytes('RGB',(768,416),raw))
                    draw = ImageDraw.Draw(canvas)
                    draw.text((13,420),title,font=font(20),fill='#abefe2')
                    draw.text((13,451),subtitle,font=font(16),fill='#f1f4f1')
                    encoder.send(canvas.tobytes())
                    elapsed += 1
            finally:
                frames.close()
            chapter['end_s'] = elapsed/10
            manifest.append(chapter)
    finally:
        arena.close()
        encoder.close()
    temp.replace(out)
    (directory/'chapters.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(out,elapsed/10,'seconds')


if __name__=='__main__':
    main()
