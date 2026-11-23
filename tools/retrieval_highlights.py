#!/usr/bin/env python3
"""Cut three real paired replays into a compact, synchronised 2x viewing copy."""

import argparse
from itertools import zip_longest
import json
from pathlib import Path
import subprocess
import sys

import imageio_ffmpeg
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ducklab.video import font
from retrieval_gallery import decode

SCENES = [
    ("test", "040_large_marker", 97201, "马克笔"),
    ("test", "sony_duck", 97203, "鸭子玩偶"),
    ("blind", "038_padlock", 112308, "小挂锁"),
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    root = p.parse_args().root.resolve()
    path = root / "highlights.mp4"
    temp = path.with_suffix(".partial.mp4")
    sources = []
    cmd = [
        imageio_ffmpeg.get_ffmpeg_exe(),
        "-y",
        "-v",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "1536x472",
        "-r",
        "10",
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-threads",
        "2",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(temp),
    ]
    with subprocess.Popen(cmd, stdin=subprocess.PIPE) as writer:
        with writer.stdin:
            for panel, obj, seed, label in SCENES:
                readers = []
                outcomes = []
                clips = []
                for version in ("v1", "v6"):
                    folder = root / f"{version}-{panel}"
                    rows = json.loads((folder / "episodes.json").read_text())["rows"]
                    row = next(
                        r for r in rows if r["object"] == obj and r["seed"] == seed
                    )
                    src = folder / row["video"]
                    reader = imageio_ffmpeg.read_frames(str(src), pix_fmt="rgb24")
                    meta = next(reader)
                    if meta["fps"] != 10 or meta["size"] != (768, 416):
                        raise ValueError("input video format differs")
                    readers.append(reader)
                    outcomes.append("通过" if row["success"] else "未通过")
                    clips.append(src.relative_to(root).as_posix())
                sources.append(dict(panel=panel, object=obj, seed=seed, clips=clips))
                last = [None, None]
                try:
                    for idx, frames in enumerate(zip_longest(*readers)):
                        for i, frame in enumerate(frames):
                            if frame is not None:
                                last[i] = frame
                        if idx % 2:
                            continue
                        pic = Image.new("RGB", (1536, 472), "#162c2a")
                        draw = ImageDraw.Draw(pic)
                        for i, (version, result) in enumerate(
                            zip(("旧版", "第六轮"), outcomes)
                        ):
                            draw.text(
                                (14 + 768 * i, 9),
                                f"{label} · {version} · {result}",
                                font=font(26),
                                fill="#eff9f2",
                            )
                            pic.paste(
                                Image.frombytes("RGB", (768, 416), last[i]),
                                (768 * i, 56),
                            )
                        draw.text(
                            (1200, 15),
                            "同次投掷 · 2 倍速",
                            font=font(19),
                            fill="#b6e5de",
                        )
                        writer.stdin.write(pic.tobytes())
                finally:
                    for reader in readers:
                        reader.close()
        if writer.wait() != 0:
            raise RuntimeError("highlight encode failed")
    temp.replace(path)
    qa = decode(path, root)
    (root / "highlights.json").write_text(
        json.dumps(dict(qa=qa, sources=sources, speed=2), indent=2) + "\n"
    )
    print(qa)


if __name__ == "__main__":
    main()
