"""Render verified raw camera archives to PNG and a timestamp-paced MP4 on host."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.selfcatch_field.camera_probe import verify_archive
from tools.selfcatch_field.camera_board import png_preview


def render(path, out, requested=30, max_mib=128):
    report, _ = verify_archive(path, max_mib, requested)
    if not report["frames"]:
        raise ValueError("no camera frames to render")
    encoder = shutil.which("ffmpeg")
    if encoder is None:
        try:
            import imageio_ffmpeg
            encoder = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError as error:
            raise RuntimeError("ffmpeg unavailable; run with runs/fetch/train-env/bin/python") from error
    out.mkdir(parents=True, exist_ok=False)
    frames = report["frames"]
    concat = []
    with tarfile.open(path) as archive:
        for i, row in enumerate(frames):
            raw = archive.extractfile(row["file"]).read()
            image = png_preview(row["header"], raw, edge=1280)
            name = f"frame-{i:04d}.png"
            (out / name).write_bytes(image)
            concat.append(f"file '{name}'")
            if i + 1 < len(frames):
                delta = (frames[i+1]["header"]["captured_at_unix_us"] - row["header"]["captured_at_unix_us"]) / 1e6
                concat.append(f"duration {delta:.9f}")
    concat.append(f"file 'frame-{len(frames)-1:04d}.png'")
    (out / "frames.txt").write_text("\n".join(concat) + "\n")
    movie = out / "camera.mp4"
    subprocess.run([encoder, "-v", "error", "-n", "-f", "concat", "-safe", "1", "-i",
                    str((out / "frames.txt").resolve()), "-fps_mode", "vfr", "-vf",
                    "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    str(movie.resolve())], check=True, timeout=60)
    result = dict(source_archive=str(path.resolve()), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                  frames=len(frames), timestamp_kind=report["timestamp_kind"],
                  capture_code=report["code"], movie=str(movie.resolve()),
                  source_authenticity=report.get("source_authenticity", "NOT_VERIFIED"),
                  task_result="NOT_TESTED", motion_permitted=False)
    (out / "media.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive", type=Path)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--requested", type=int, default=30)
    args = p.parse_args()
    print(json.dumps(render(args.archive, args.out, args.requested), indent=2))


if __name__ == "__main__":
    main()
