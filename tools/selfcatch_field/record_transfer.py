"""Record fixed-seed MuJoCo transfer comparisons, including failures. No robot I/O."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MUJOCO_GL', 'egl')
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools'))
if __package__:
    from .contract import FROZEN, TransferEnv, load_actor, torch
else:
    from contract import FROZEN, TransferEnv, load_actor, torch
import mujoco
import numpy as np
from PIL import Image
from record_self_catch import (Camera, RH, RW, RED, SUB, TEAL, T, decode, encoder,
                               page, phase, read_tape, same_info, text)


def record(env, policy, seed, dest, expected=None):
    obs = env.reset(seed, record=True)
    with torch.inference_mode():
        actions = []
        for p in (0, 1):
            mean = policy.dist(torch.from_numpy(obs), p).loc
            actions.append(mean.numpy())
            obs, _, _, info = env.step(mean.tanh().numpy())
    if expected is not None:
        same_info(info, expected)
    np.savez_compressed(dest, qpos=env.frames, trace=env.trace, raw_actions=actions,
                        meta=json.dumps(dict(seed=seed, **info)))
    return read_tape(dest)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--qualification', type=Path, required=True)
    parser.add_argument('--finetune', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    qual = json.loads(args.qualification.read_text())
    fit = json.loads((args.finetune / 'report.json').read_text())
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    selected = args.finetune / 'selected.pt'
    if not (qual['hardware'] is False and fit['hardware'] is False and
            fit['source_sha256'] == qual['sha256'] == sha(FROZEN) and
            fit['selected_sha256'] == sha(selected) and fit['heldout']['action_mode'] == 'tanh(mean)'):
        raise ValueError('report/checkpoint mismatch or non-deterministic evaluation')
    args.out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    orig, tuned = load_actor(), load_actor()
    tuned.load_state_dict(torch.load(selected, map_location='cpu', weights_only=True))
    native, proxy = TransferEnv('trained_5ms'), TransferEnv('handoff_intent_20ms')
    first = min(qual['modes'][native.mode]['rows'], key=lambda x: x['seed'])
    peer = next(x for x in qual['modes'][proxy.mode]['rows'] if x['seed'] == first['seed'])
    held = min(fit['heldout']['rows'], key=lambda x: x['seed'])
    chapters = [
        ('同一个策略，换控制接口', first['seed'],
         [('原训练接口 · 鸭嘴 5 ms', native, orig, first), ('随包步态代理 · 鸭嘴 20 ms', proxy, orig, peer)]),
        ('同一套 50 Hz 接口，微调前后', held['seed'],
         [('原 PPO', proxy, orig, None), ('迁移微调后的 PPO', proxy, tuned, held)]),
    ]
    encoded, sources = 0, []
    path = args.out / 'transfer-comparison.mp4'
    with mujoco.Renderer(native.m, height=RH, width=RW) as renderer, encoder(path) as emit:
        cam = Camera(native, renderer)
        for chapter, (title, seed, panels) in enumerate(chapters, 1):
            tapes = []
            for j, (label, env, policy, expected) in enumerate(panels):
                dest = args.out / f'chapter-{chapter}-panel-{j+1}-{seed}.npz'
                tape = record(env, policy, seed, dest, expected)
                tapes.append(tape)
                sources.append(dict(file=dest.name, sha256=sha(dest), label=label, mode=env.mode,
                                    actor_sha256=sha(FROZEN if policy is orig else selected),
                                    matched_report=expected is not None, result=tape[2]))
            if not np.array_equal(tapes[0][1][:, T], tapes[1][1][:, T]):
                raise ValueError('different physics clocks')
            for i in range(0, len(tapes[0][0]), 2):
                image, draw = page('WONDERDUCK  /  MUJOCO  /  SIMULATION ONLY', title,
                                   '两侧均为物理仿真，不是真机录像。黄线是球的实际轨迹；碰嘴后掉地不算接住。')
                text(draw, 824, 40, f'实验 {chapter}/2    种子 {seed}', 21, TEAL)
                for j, (tape, panel) in enumerate(zip(tapes, panels)):
                    x = 12 + 596 * j
                    im, (mx, my) = cam.frame(tape, i)
                    image.paste(im, (x, 124))
                    text(draw, x, 83, panel[0], 23)
                    text(draw, x + 400, 89, phase(tape, i), 17, SUB)
                    crop = im.crop((round(mx) - 65, round(my) - 40, round(mx) + 65, round(my) + 40))
                    image.paste(crop.resize((260, 160), Image.Resampling.LANCZOS), (x, 455))
                    r = tape[2]
                    text(draw, x + 278, 460, '终局：' + ('夹稳' if r['success'] else '未达标'), 23,
                         TEAL if r['success'] else RED)
                    text(draw, x + 278, 511, '碰地：' + ('是' if r['ground'] else '否'), 20)
                    text(draw, x + 278, 551, f"鸭身前移：{r['chase_m']*1000:.1f} mm", 20)
                text(draw, 18, 642, f"鸭嘴局部放大     |     1/4 倍速     |     甩头后 {tapes[0][1][i,T]:+.3f} s", 19, SUB)
                emit(image)
                encoded += 1
            for _ in range(15):
                emit(image)
                encoded += 1
    manifest = decode(path, args.out)
    if manifest['frames'] != encoded:
        raise ValueError('video frame count mismatch')
    manifest.update(hardware=False, selection='first seed in each report, no outcome selection',
                    sources=sources, encoded_frames=encoded)
    (args.out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(path, manifest['frames'], manifest['seconds'])


if __name__ == '__main__':
    main()
