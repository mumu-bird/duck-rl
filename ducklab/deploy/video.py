"""Training-progress video capture for the deployable stack.

Every ``--video-every`` updates the trainer records one deterministic episode
and writes an MP4 (third-person overview with a head-camera inset and an
overlay of the stage metrics) plus a final-frame PNG, so the improvement over
training is watchable rather than inferred from numbers.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import mujoco
import numpy as np

import imageio_ffmpeg


def _text(draw, xy, text, font, fill=(255, 255, 255)):
    draw.text(xy, text, fill=fill, font=font)


def record_episode(env, act_fn, path, label="", metrics=None, fps=10,
                   width=640, height=360):
    """Run one deterministic episode while rendering, then encode an MP4."""
    from PIL import Image, ImageDraw, ImageFont

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".partial.mp4")

    renderer = mujoco.Renderer(env.model, height, width)
    inset_h, inset_w = height // 3, width // 3
    head_renderer = mujoco.Renderer(env.model, inset_h, inset_w)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", 14)
    except OSError:
        font = ImageFont.load_default()

    obs = env.reset()
    proc = subprocess.Popen([
        imageio_ffmpeg.get_ffmpeg_exe(), "-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s:v", f"{width}x{height}", "-r", str(fps),
        "-i", "pipe:0", "-an", "-c:v", "libx264", "-threads", "2",
        "-pix_fmt", "yuv420p", str(temp),
    ], stdin=subprocess.PIPE)

    tick = 0
    info = {}
    last_frame = None
    while True:
        action = act_fn(obs)
        obs, reward, done, info = env.step(action)
        tick += 1
        if tick % 2 == 0 or done:      # 50 Hz sim -> 25 fps capture, encoded at fps
            renderer.update_scene(env.data, camera="overview")
            frame = renderer.render()
            head_renderer.update_scene(env.data, camera="head_camera")
            inset = head_renderer.render()
            image = Image.fromarray(frame)
            image.paste(Image.fromarray(inset), (width - inset_w - 6, 6))
            draw = ImageDraw.Draw(image)
            overlay = [label]
            if metrics:
                overlay.extend(f"{k} {v}" for k, v in metrics.items())
            overlay.append(f"tick {tick}  {info.get('success') and 'SUCCESS' or info.get('fall') and 'FALL' or ''}")
            for i, line in enumerate(overlay):
                _text(draw, (8, 6 + i * 17), line, font)
            array = np.asarray(image)
            last_frame = array
            proc.stdin.write(array.tobytes())
        if done:
            break

    proc.stdin.close()
    proc.wait()
    temp.replace(path)
    png = path.with_suffix(".png")
    if last_frame is not None:
        from PIL import Image
        Image.fromarray(last_frame).save(png)
    return path, png
