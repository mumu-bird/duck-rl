#!/usr/bin/env python3
"""Build an offline paired replay page. Fully decode and fingerprint each linked video."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import hashlib
import json
from pathlib import Path
import subprocess

import imageio_ffmpeg

ROOT = Path(__file__).resolve().parents[1]
NAMES = dict(
    zip(
        (
            "040_large_marker",
            "058_golf_ball",
            "sony_duck",
            "014_lemon",
            "038_padlock",
            "062_dice",
            "009_gelatin_box",
            "033_spatula",
            "015_peach",
            "010_potted_meat_can",
            "051_large_clamp",
            "017_orange",
        ),
        (
            "马克笔",
            "小球",
            "鸭子玩偶",
            "小柠檬",
            "小挂锁",
            "骰子",
            "果冻粉纸盒",
            "小锅铲",
            "小桃子",
            "小罐头",
            "小夹子",
            "小橙子",
        ),
        strict=True,
    )
)
PANELS = {
    "test": "笔、球、玩偶",
    "blind": "第一批陌生物品",
    "blind2": "第二批陌生物品",
    "blind3": "第三批陌生物品",
}
PHASES = {
    "pick": "低头夹取",
    "approach": "接近",
    "carry": "携物行走",
    "stand_to_pick": "复查",
    "stand_to_drop": "到箱沿",
    "drop": "松嘴",
    "return": "归位",
    "brake": "归位制动",
    "restore": "恢复姿势",
}


def verdict(r):
    if r["success"]:
        return "整轮通过"
    stop = r["stop"]
    if stop == "fall":
        return PHASES.get(r["events"][-1][1], r["events"][-1][1]) + "时跌倒"
    if stop == "no_new_settled_object":
        return "等待目标停稳超时"
    if stop.endswith("_timeout"):
        return PHASES.get(stop[:-8], stop[:-8]) + "超时"
    if r["wrong_contact"]:
        return "夹到旧物"
    if not r["target_contact"]:
        return "夹空"
    if not r["lifted"]:
        return "接触后未抬起"
    if not r["held_at_release"]:
        return "曾抬起，投放时未持有"
    if not r["in_tray_at_end"]:
        return "物体未留在箱内"
    if not r["pose_restored"]:
        return "起点或姿势超差"
    return stop


def decode(path, root):
    proc = subprocess.run(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-threads",
            "1",
            "-i",
            str(path),
            "-map",
            "0:v:0",
            "-progress",
            "pipe:1",
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    tail = dict(line.split("=", 1) for line in proc.stdout.splitlines() if "=" in line)
    frames, duration = int(tail["frame"]), int(tail["out_time_us"]) / 1e6
    if tail["progress"] != "end" or frames < 1 or duration <= 0:
        raise ValueError(f"incomplete video: {path}")
    return dict(
        path=path.relative_to(root).as_posix(),
        frames=frames,
        seconds=duration,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--workers", type=int, default=6)
    a = p.parse_args()
    root = a.root.resolve()
    pairs = {}
    files = []
    scores = {}
    for path in sorted(root.glob("v*-*/episodes.json")):
        version, panel = path.parent.name.split("-", 1)
        if panel not in PANELS:
            continue
        run = json.loads(path.read_text())
        counts = Counter()
        for r in run["rows"]:
            if r["mode"] != "jaw" or r["fixture_activated"] or run["no_toss"]:
                raise ValueError("unexpected benchmark conditions")
            identity, seed = r["object"], r["seed"]
            key = (panel, identity, seed)
            signature = (run["mass_grams"], tuple(run["bin_m"]))
            pair = pairs.setdefault(
                key,
                dict(
                    panel=panel,
                    id=identity,
                    name=NAMES[identity],
                    seed=seed,
                    conditions=signature,
                    versions={},
                ),
            )
            if pair["conditions"] != signature or version in pair["versions"]:
                raise ValueError(f"nonpaired or duplicate: {key} {version}")
            media = (path.parent / r["video"]).resolve()
            if not media.is_file() or not media.is_relative_to(root):
                raise ValueError(f"missing replay {media}")
            files.append(media)
            pair["versions"][version] = dict(
                ok=r["success"],
                reason=verdict(r),
                url=media.relative_to(root).as_posix(),
                log=path.relative_to(root).as_posix(),
                attempts=r["attempts"],
                home=round(r["home_cm"], 1),
                yaw=round(r["yaw_deg"], 1),
            )
            counts.update(
                total=1,
                success=r["success"],
                falls=r["stop"] == "fall",
                lifted=r["lifted"],
            )
        scores[path.parent.name] = dict(counts)
    for panel in {k[0] for k in pairs}:
        available = [set(v["versions"]) for k, v in pairs.items() if k[0] == panel]
        if (
            not all(group == available[0] for group in available)
            or len(available[0]) < 2
        ):
            raise ValueError(f"missing paired trials: {panel}")
    if not files or len(set(files)) != len(files):
        raise ValueError("empty or duplicate replay files")
    extra = []
    for folder, label in [
        ("alternate-bin", "更换箱位"),
        ("payload-12g", "12 克负载"),
        ("no-toss", "没有投掷"),
    ]:
        log = root / folder / "episodes.json"
        if not log.exists():
            continue
        for r in json.loads(log.read_text())["rows"]:
            media = log.parent / f"{r['object']}-jaw-{r['seed']}.mp4"
            files.append(media)
            reason = (
                ("正确待命" if not r["detected"] else "误启动")
                if r["no_toss"]
                else verdict(r)
            )
            extra.append(
                dict(
                    group=label,
                    name=NAMES[r["object"]],
                    seed=r["seed"],
                    reason=reason,
                    url=media.relative_to(root).as_posix(),
                )
            )
    archive = root / "rounds" / "clips.json"
    if archive.exists():
        for clip in json.loads(archive.read_text()):
            r = clip["row"]
            media = archive.parent / clip["video"]
            files.append(media)
            extra.append(
                dict(
                    group=f"第 {clip['round']} 轮 · {clip['candidate']}",
                    name=NAMES.get(r["object"], r["object"]),
                    seed=r["seed"],
                    reason=verdict(r),
                    url=media.relative_to(root).as_posix(),
                )
            )
    with ThreadPoolExecutor(a.workers) as pool:
        records = list(pool.map(partial(decode, root=root), files))
    artifact = dict(
        videos=records,
        scores=scores,
        total_files=len(files),
        total_tosses=len(pairs),
        extra_files=len(extra),
    )
    (root / "video-qa.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2) + "\n"
    )
    data = sorted(
        pairs.values(),
        key=lambda r: (list(PANELS).index(r["panel"]), r["id"], r["seed"]),
    )
    templates = ROOT / "tools/templates"
    html = (templates / "retrieval-gallery.html").read_text().replace(
        "__SCRIPT__", (templates / "retrieval-gallery.js").read_text()
    )
    (root / "index.html").write_text(
        html.replace(
            "__DATA__", json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")
        ).replace(
            "__EXTRA__", json.dumps(extra, ensure_ascii=False).replace("<", "\\u003c")
        )
    )
    print(
        f"{len(records)} complete videos, {len(data)} paired tosses. {root / 'index.html'}"
    )


if __name__ == "__main__":
    main()
