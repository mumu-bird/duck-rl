#!/usr/bin/env python3
"""Reversible board installer; --root stages a complete layout without changing this host."""
from __future__ import annotations

import argparse
import stat
import grp
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

PACKAGE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGE))
MANAGED = "Managed by wonderduck-audio installer"


def checked(*args):
    subprocess.run(args, check=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check", "install", "rollback", "uninstall"))
    parser.add_argument("--root", type=Path, default=Path("/"), help="Staging root; no systemctl, sound or dependency installation")
    parser.add_argument("--robot-root", default="/home/wonderduck/wonderduck")
    parser.add_argument("--bind", default="127.0.0.1")
    parser.add_argument("--without-prompt-integration", action="store_true")
    parser.add_argument("--without-legacy-player", action="store_true", help="Keep existing robotd player config; manual shared-lock verification required")
    args = parser.parse_args(argv)
    root = args.root.resolve()
    staged = root != Path("/")
    if not Path(args.robot_root).is_absolute() or any(c in args.robot_root for c in '\n\r"%'):
        raise SystemExit("--robot-root 必须是绝对路径且不包含换行、引号或 %")
    def path(name): return root / name.lstrip("/")
    base = path("/opt/wonderduck-audio")
    current, previous = base / "current", base / "previous"
    unit = path("/etc/systemd/system/wonderduck-audio.service")
    dropin = path("/etc/systemd/system/wonderduck-prompt-sounds.service.d/60-duckaudio.conf")
    cli = path("/usr/local/bin/duckaudio")
    config = path("/etc/wonderduck-audio/audio.toml")
    state = path("/var/lib/wonderduck-audio")
    robot_config = path("/etc/robot/robotd.toml")
    legacy_meta = state / "legacy-player.json"
    legacy_backup = state / "robotd.toml.before-audio"
    if not staged and args.action != "check" and os.geteuid() != 0:
        raise SystemExit("板端安装需要 sudo；使用 --root /tmp/audio-stage 可生成可检查的安装目录")
    if args.action in ("check", "install") and not staged:
        errors = []
        if sys.version_info < (3, 10): errors.append("需要 Python >= 3.10")
        try:
            import duckaudio.config
        except ImportError: errors.append("Python 3.10 需要 python3-tomli")
        for program in ("aplay", "arecord", "amixer", "ffmpeg", "espeak-ng"):
            if not shutil.which(program): errors.append("缺少命令: " + program)
        try: grp.getgrnam("robot")
        except KeyError: errors.append("未找到原厂 robot 用户组")
        player = Path(args.robot_root) / "scripts/audio-output.py"
        if not player.is_file() or "--lock-fd" not in player.read_text():
            errors.append("未找到支持共享锁的原厂 audio-output.py")
        if not args.without_legacy_player and not robot_config.is_file():
            errors.append("未找到 /etc/robot/robotd.toml；需核对原厂配置位置")
        if not args.without_prompt_integration:
            from duckaudio.prompt_runner import SUPPORTED_SOURCE_SHA256
            source = Path(args.robot_root) / "scripts/prompt-sounds.py"
            if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != SUPPORTED_SOURCE_SHA256:
                errors.append("原厂提示音源码版本与已验证版本不同；可用 --without-prompt-integration 保留原厂提示音")
        if errors:
            raise SystemExit("\n".join(errors) + "\n依赖安装示例: sudo apt-get install alsa-utils ffmpeg espeak-ng python3-tomli")
    if args.action == "check":
        print(json.dumps({"check": "passed", "staged": staged}, ensure_ascii=False)); return

    def managed_remove(target):
        if target.exists():
            if MANAGED not in target.read_text():
                raise SystemExit("拒绝移除非本安装器管理的文件: " + str(target))
            target.unlink()

    def restore_legacy():
        if legacy_meta.exists() and legacy_backup.exists() and robot_config.exists():
            record = json.loads(legacy_meta.read_text())
            if hashlib.sha256(robot_config.read_bytes()).hexdigest() == record["installed_sha256"]:
                robot_config.write_bytes(legacy_backup.read_bytes())
                legacy_meta.unlink()
            else:
                print("原厂配置已被其他工具修改，保留当前文件；原始备份: " + str(legacy_backup))

    def link(target, location):
        location.parent.mkdir(parents=True, exist_ok=True)
        if location.exists() and not location.is_symlink():
            raise SystemExit("拒绝覆盖已有目录或文件: " + str(location))
        temporary = location.with_name(location.name + ".new")
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(target)
        os.replace(temporary, location)

    if args.action == "uninstall":
        # Keep all audio, settings, token, package versions and installation records.
        for target in (unit, dropin, cli):
            if target.exists() and MANAGED not in target.read_text():
                raise SystemExit("拒绝移除非本安装器管理的文件: " + str(target))
        if not staged: checked("systemctl", "disable", "--now", "wonderduck-audio.service")
        for target in (unit, dropin, cli): managed_remove(target)
        if not staged:
            checked("systemctl", "daemon-reload")
            checked("systemctl", "try-restart", "wonderduck-prompt-sounds.service")
        restore_legacy()
        print("已卸载服务入口；音频库、配置和安装版本保留。原厂播放器配置如被恢复，将在下次 robotd 启动时生效。")
        return
    if args.action == "rollback":
        if not previous.is_symlink(): raise SystemExit("没有可回退的上一版本")
        old, target = os.readlink(current), os.readlink(previous)
        link(target, current); link(old, previous)
        if not staged:
            checked("systemctl", "restart", "wonderduck-audio.service")
            checked("systemctl", "try-restart", "wonderduck-prompt-sounds.service")
        print("已切换上一版本，音频库与配置保留。")
        return

    digest = hashlib.sha256()
    for source in sorted((PACKAGE / "duckaudio").rglob("*")):
        if source.is_file() and "__pycache__" not in source.parts:
            digest.update(str(source.relative_to(PACKAGE)).encode()); digest.update(source.read_bytes())
    version = "1.0.0-" + digest.hexdigest()[:12]
    destination = base / "releases" / version
    if not destination.exists():
        destination.mkdir(parents=True)
        for folder in ("duckaudio", "assets/audio", "docs/audio", "examples/audio"):
            shutil.copytree(PACKAGE / folder, destination / folder, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    old = os.readlink(current) if current.is_symlink() else None
    planned = {
        unit: (PACKAGE / "deploy/audio/wonderduck-audio.service").read_text(),
        cli: '#!/bin/sh\n# ' + MANAGED + '\nexport PYTHONPATH=/opt/wonderduck-audio/current\nexec /usr/bin/python3 -m duckaudio "$@"\n',
    }
    if not args.without_prompt_integration:
        planned[dropin] = ("# " + MANAGED + "\n[Unit]\nWants=wonderduck-audio.service\nAfter=wonderduck-audio.service\n"
            "[Service]\nEnvironment=PYTHONPATH=/opt/wonderduck-audio/current\nExecStart=\n"
            'ExecStart=/usr/bin/python3 -m duckaudio.prompt_runner --source "' + args.robot_root + '/scripts/prompt-sounds.py"\n')
    for target in planned:
        if target.exists() and MANAGED not in target.read_text():
            raise SystemExit("拒绝覆盖非本安装器管理的文件: " + str(target))
    backups = {target: target.read_bytes() if target.exists() else None for target in planned}
    remove_prompt = args.without_prompt_integration and dropin.exists()
    if remove_prompt:
        if MANAGED not in dropin.read_text(): raise SystemExit("拒绝移除其他工具配置的提示音入口")
        backups[dropin] = dropin.read_bytes()
    legacy_original = robot_config.read_bytes() if robot_config.exists() else None
    legacy_changed = False
    legacy_updated = None
    if not args.without_legacy_player:
        if legacy_original is None: raise SystemExit("安装目录中缺少 /etc/robot/robotd.toml")
        from duckaudio.board_config import shared_player_config
        legacy_updated = shared_player_config(legacy_original.decode(), args.robot_root + "/scripts/audio-output.py").encode()
        legacy_changed = legacy_updated != legacy_original
    if not config.exists():
        config.parent.mkdir(parents=True, exist_ok=True)
        text = (PACKAGE / "deploy/audio/audio.toml").read_text().replace('"/home/wonderduck/wonderduck/scripts/audio-output.py"', json.dumps(args.robot_root + "/scripts/audio-output.py"))
        text = text.replace('host = "127.0.0.1"', "host = " + json.dumps(args.bind))
        config.write_text(text)
    state.mkdir(parents=True, exist_ok=True, mode=0o750)
    try:
        if remove_prompt: dropin.unlink()
        if legacy_changed:
            if legacy_backup.exists() and legacy_meta.exists():
                recorded = json.loads(legacy_meta.read_text())
                if hashlib.sha256(legacy_original).hexdigest() != recorded["installed_sha256"]:
                    shutil.copy2(legacy_backup, state / ("robotd.toml.backup-" + str(time.time_ns())))
                    legacy_backup.write_bytes(legacy_original)
            else:
                legacy_backup.write_bytes(legacy_original)
            temporary_config = robot_config.with_name(".robotd.toml.audio-new")
            temporary_config.write_bytes(legacy_updated)
            temporary_config.chmod(stat.S_IMODE(robot_config.stat().st_mode))
            os.chown(temporary_config, robot_config.stat().st_uid, robot_config.stat().st_gid)
            os.replace(temporary_config, robot_config)
        for target, text in planned.items():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        cli.chmod(0o755)
        link("releases/" + version, current)
        if not staged:
            checked("systemctl", "daemon-reload")
            checked("systemctl", "enable", "wonderduck-audio.service")
            checked("systemctl", "restart", "wonderduck-audio.service")
            from duckaudio.client import AudioClient
            client = AudioClient(timeout=.5)
            for attempt in range(50):
                try: client.health(); break
                except Exception:
                    if attempt == 49: raise
                    time.sleep(.2)
            from duckaudio.seed import seed
            seed(client, destination / "assets/audio")
            if not args.without_prompt_integration or remove_prompt:
                checked("systemctl", "try-restart", "wonderduck-prompt-sounds.service")
        if old and old != "releases/" + version: link(old, previous)
    except Exception:
        if legacy_changed: robot_config.write_bytes(legacy_original)
        if old: link(old, current)
        else: current.unlink(missing_ok=True)
        for target, contents in backups.items():
            if contents is None: target.unlink(missing_ok=True)
            else: target.write_bytes(contents)
        if not staged:
            subprocess.run(["systemctl", "stop", "wonderduck-audio.service"], check=False)
            subprocess.run(["systemctl", "daemon-reload"], check=False)
            if old: subprocess.run(["systemctl", "restart", "wonderduck-audio.service"], check=False)
        raise
    if legacy_changed:
        legacy_meta.write_text(json.dumps({"installed_sha256": hashlib.sha256(legacy_updated).hexdigest()}, indent=2))
    (state / "installation.json").write_text(json.dumps({"version": version, "previous": old, "staged": staged,
        "prompt_integration": not args.without_prompt_integration, "legacy_player_changed": legacy_changed,
        "robotd_restart_required": legacy_changed}, indent=2))
    print(json.dumps({"installed": version, "staged": staged, "root": str(root), "config": str(config),
                      "token_file": str(state / "access-token"), "robotd_restart_required": legacy_changed}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
