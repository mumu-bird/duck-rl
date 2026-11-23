#!/usr/bin/env python3
"""Encode actual PPO physics samples and matched held-out trials, never interpolated poses."""

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import subprocess
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import imageio_ffmpeg
import mujoco
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ducklab.bench import view_options  # noqa: E402
from ducklab.self_catch import T, X, Z, MX, MZ, BOTH, FLOOR, DT, SelfCatchEnv, judge  # noqa: E402
from ducklab.video import font  # noqa: E402
from retrieval_gallery import decode  # noqa: E402

W, H, FPS = 1200, 736, 25
PAPER, INK, SUB = "#f3f3ed", "#17292a", "#465d60"
TEAL, RED, GOLD = "#146b5d", "#983d37", "#ffc86f"
RW, RH = 584, 312


def text(draw, x, y, s, size=18, color=INK):
    draw.text((x, y), s, font=font(size), fill=color)


def page(kicker, title, note):
    image = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(image)
    d.rectangle((0, 0, W, 5), fill=TEAL)
    text(d, 22, 12, kicker, 15, TEAL)
    text(d, 20, 32, title, 33)
    text(d, 22, H - 36, note, 17, SUB)
    return image, d


@contextmanager
def encoder(path):
    if path.exists():
        raise FileExistsError(path)
    temp = path.with_suffix(".partial.mp4")
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-v",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{W}x{H}",
        "-r",
        str(FPS),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-crf",
        "18",
        "-preset",
        "veryfast",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temp),
    ]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as proc:
        with proc.stdin:
            yield lambda image: proc.stdin.write(image.tobytes())
        if proc.wait() != 0:
            raise RuntimeError("ffmpeg failed")
    temp.replace(path)


def same_info(info, row):
    for key, value in info.items():
        if isinstance(value, float):
            if not np.isclose(value, row[key], rtol=0, atol=1e-10):
                raise ValueError((key, value, row[key]))
        elif value != row[key]:
            raise ValueError((key, value, row[key]))


def read_tape(path):
    with np.load(path, allow_pickle=False) as f:
        q, tr, meta = f["qpos"], f["trace"], json.loads(str(f["meta"]))
    same_info(judge(tr), meta)
    if len(q) != len(tr) or not np.allclose(np.diff(tr[:, T]), DT):
        raise ValueError("mismatched state and referee clocks")
    return q, tr, meta


class Camera:
    def __init__(self, env, renderer):
        self.m, self.d, self.ball = env.m, mujoco.MjData(env.m), env.bid
        self.r = renderer
        self.opt = view_options()
        self.view = mujoco.MjvCamera()
        self.view.lookat[:] = (0.080, 0, 0.132)
        self.view.distance, self.view.elevation, self.view.azimuth = 0.44, -10, 99

    def frame(self, tape, i):
        q, tr, meta = tape
        self.d.qpos[:] = q[i]
        mujoco.mj_forward(self.m, self.d)
        # Refuse out-of-sync replay geometry, including a one-physics-step offset.
        if not np.allclose(self.d.xpos[self.ball], tr[i, X : Z + 1], atol=2e-8, rtol=0):
            raise ValueError(f"ball replay position differs at sample {i}")
        self.r.update_scene(self.d, camera=self.view, scene_option=self.opt)
        im = Image.fromarray(self.r.render())
        d = ImageDraw.Draw(im)
        cam, c2 = self.r.scene.camera
        pos = (cam.pos + c2.pos) / 2
        right = np.cross(cam.forward, cam.up)
        focal = RH * cam.frustum_near / (cam.frustum_top - cam.frustum_bottom)

        def pix(point):
            delta = point - pos
            z = delta @ cam.forward
            return (
                RW / 2 + focal * (delta @ right) / z,
                RH / 2 - focal * (delta @ cam.up) / z,
            )

        bx, by = pix(tr[i, X : Z + 1])
        mx, my = pix(tr[i, MX : MZ + 1])
        t = tr[i, T]
        if meta["release_s"] is not None and t >= meta["release_s"]:
            start = np.searchsorted(tr[:, T], meta["release_s"])
            stop = i + 1
            if meta["catch_s"] is not None:
                stop = min(stop, np.searchsorted(tr[:, T], meta["catch_s"]) + 1)
            prev = [pix(p) for p in tr[start:stop:2, X : Z + 1]]
            if len(prev) > 1:
                d.line(prev, fill="#ffdd8e", width=2)
        rr = max(6.0, 1.38 * 0.010 * focal / ((tr[i, X : Z + 1] - pos) @ cam.forward))
        for angle in (20, 110, 200, 290):
            d.arc(
                (bx - rr, by - rr, bx + rr, by + rr),
                angle,
                angle + 50,
                fill="#fff3b0",
                width=2,
            )
        return im, (mx, my)


def phase(tape, i):
    _, tr, r = tape
    t = tr[i, T]
    if tr[: i + 1, FLOOR].any():
        return "已落地"
    if r["catch_s"] is not None and t >= r["catch_s"]:
        return "已夹稳" if i >= 50 and tr[i - 49 : i + 1, BOTH].all() else "重新接触"
    if r["release_s"] is not None and t >= r["release_s"]:
        return "自由飞行"
    return "准备抛球" if t < 0 else "甩头抛球"


def training_video(run_dirs, dest, cam):
    path = dest / "training-six.mp4"
    sources = []
    count = 0
    with encoder(path) as emit:
        for stage, run, part in (
            (1, run_dirs[0], "first"),
            (len(run_dirs), run_dirs[-1], "last"),
        ):
            files = sorted((run / "tapes").glob("train-*.npz"))
            it = (files[0] if part == "first" else files[-1]).name.split("-")[1]
            group = sorted((run / "tapes").glob(f"train-{it}-*.npz"))[:6]
            if len(group) != 6:
                raise ValueError("expected recordings from all six training lanes")
            tapes = [read_tape(p) for p in group]
            sources.extend(str(p) for p in group)
            for i in range(0, len(tapes[0][0]), 2):
                image, d = page(
                    "WONDER DUCK  /  PPO  /  TRAINING",
                    "六只鸭子并行练习",
                    "同一轮的六个不同随机回合   ·   黄色细线是实测球路   ·   1/4 倍速",
                )
                text(
                    d,
                    790,
                    36,
                    f"阶段 {stage}    更新 {int(it):02d}    |    {tapes[0][1][i, T]:+.2f} s",
                    22,
                    TEAL,
                )
                for n, tape in enumerate(tapes):
                    col, row = n % 3, n // 3
                    x, y = 18 + col * 392, 96 + row * 285
                    im, _ = cam.frame(tape, i)
                    im = im.resize((380, 203), Image.Resampling.LANCZOS)
                    text(d, x, y, f"{n + 1:02d}   /   种子 {tape[2]['seed']}", 18, SUB)
                    image.paste(im, (x, y + 32))
                    text(d, x + 2, y + 240, phase(tape, i), 18)
                emit(image)
                count += 1
            for _ in range(12):
                emit(image)
                count += 1
    return dict(**decode(path, dest), sources=sources, encoded_frames=count)


def trial_tape(env, loc, report, label, seed):
    expected = next(r for r in report[label]["rows"] if r["seed"] == seed)
    path = loc / "tapes" / f"{label}-{seed}.npz"
    if not path.exists():
        import torch
        from train_self_catch import Policy

        policy = Policy().eval()
        ck = Path(report[label]["checkpoint"])
        ck = ck if ck.is_absolute() else ROOT / ck
        policy.load_state_dict(torch.load(ck, weights_only=True, map_location="cpu"))
        obs = env.reset(seed, record=True)
        actions = []
        with torch.inference_mode():
            for p in range(2):
                a = policy.actors[p](torch.from_numpy(obs))
                actions.append(a.numpy())
                obs, _, _, info = env.step(torch.tanh(a).numpy())
        same_info(info, expected)
        np.savez_compressed(
            path,
            qpos=env.frames,
            trace=env.trace,
            raw_actions=actions,
            meta=json.dumps(expected),
        )
    tape = read_tape(path)
    same_info(tape[2], expected)
    return tape


def paired_video(run, dest, cam, env):
    loc = run / "assessment"
    report = json.loads((loc / "paired.json").read_text())
    rows = report["after"]["rows"]
    reps = []
    for ok in (True, False):
        group = [r for r in rows if r["success"] == ok]
        if group:
            reps.append(group[0]["seed"])
    before, after = [report[k]["stats"] for k in ("before", "after")]
    path = dest / "paired-catches.mp4"
    count = 0
    with encoder(path) as emit:
        for seed in reps:
            pair = [trial_tape(env, loc, report, k, seed) for k in ("before", "after")]
            for stride, speed in ((8, "原速"), (2, "1/4 倍速")):
                for i in range(0, len(pair[0][0]), stride):
                    t = pair[0][1][i, T]
                    image, d = page(
                        "WONDER DUCK  /  PPO  /  HOLDOUT",
                        "同一起点，训练前后",
                        f"全组严格通过：训练前 {before['success']}/{before['n']}    训练后 {after['success']}/{after['n']}     ·     未换算为真机成功率",
                    )
                    text(d, 785, 36, f"种子 {seed}     |     {speed}", 22, TEAL)
                    for n, tape in enumerate(pair):
                        q, tr, r = tape
                        x = 12 + n * 596
                        im, (mx, my) = cam.frame(tape, i)
                        text(d, x + 3, 87, "训练前" if not n else "PPO 训练后", 23)
                        text(d, x + 330, 91, phase(tape, i), 19, SUB)
                        image.paste(im, (x, 127))
                        crop = im.crop(
                            (
                                round(mx) - 62,
                                round(my) - 35,
                                round(mx) + 62,
                                round(my) + 35,
                            )
                        )
                        image.paste(
                            crop.resize((248, 140), Image.Resampling.LANCZOS), (x, 457)
                        )
                        text(d, x + 4, 602, "鸭嘴局部  ×2", 16, SUB)
                        text(
                            d,
                            x + 266,
                            458,
                            "完整达标"
                            if r["success"]
                            else ("落地，未接住" if r["ground"] else "未达全部标准"),
                            23,
                            TEAL if r["success"] else RED,
                        )
                        text(
                            d,
                            x + 266,
                            499,
                            f"离嘴飞行   {r['air_s'] * 1000:.0f} ms",
                            18,
                        )
                        text(
                            d,
                            x + 266,
                            528,
                            f"离嘴后再升高   {1000 * r['rise_m']:.1f} mm",
                            18,
                        )
                        text(
                            d,
                            x + 266,
                            557,
                            f"鸭身前移   {1000 * r['chase_m']:.1f} mm",
                            18,
                        )
                    d.line((20, 655, W - 20, 655), fill="#b7c6bc", width=3)
                    xx = 20 + (W - 40) * i / (len(q) - 1)
                    d.ellipse((xx - 5, 650, xx + 5, 660), fill=TEAL)
                    text(
                        d,
                        20,
                        661,
                        f"甩头时刻为 0.00 s     当前 {t:+.3f} s     ·     黄线为实测球路",
                        16,
                        SUB,
                    )
                    emit(image)
                    count += 1
                    if stride == 2 and i == 104 and seed == reps[0]:
                        image.save(dest / "comparison-frame.png")
                for _ in range(8 if stride == 8 else 20):
                    emit(image)
                    count += 1
    return dict(**decode(path, dest), examples=reps, encoded_frames=count)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", type=Path, required=True, nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--mode", choices=["training", "paired", "both"], default="both")
    a = p.parse_args()
    a.out.mkdir(exist_ok=True, parents=True)
    config = a.runs[-1] / "run-args.json"
    fine_jaw = (
        json.loads(config.read_text()).get("fine_jaw", False)
        if config.exists()
        else False
    )
    env = SelfCatchEnv(fine_jaw=fine_jaw)
    manifest = {}
    with mujoco.Renderer(env.m, height=RH, width=RW) as renderer:
        cam = Camera(env, renderer)
        if a.mode != "paired":
            manifest["training"] = training_video(a.runs, a.out, cam)
        if a.mode != "training":
            manifest["comparison"] = paired_video(a.runs[-1], a.out, cam, env)
    for k, v in manifest.items():
        if v["encoded_frames"] != v["frames"]:
            raise ValueError("encoder and decoder disagree")
        (a.out / f"{k}-video.json").write_text(
            json.dumps(v, ensure_ascii=False, indent=2) + "\n"
        )
        print(k, v["path"], v["frames"], v["seconds"], flush=True)


if __name__ == "__main__":
    main()
