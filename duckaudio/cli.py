from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
import sys
import threading

from .client import AudioClient
from .config import Config
from .errors import AudioError


def parser():
    p = argparse.ArgumentParser(description="WonderDuck 公共音频服务与任务接口")
    p.add_argument("--endpoint", default="unix:/run/wonderduck-audio/audio.sock")
    p.add_argument("--token-file")
    commands = p.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--config")
    serve.add_argument("--data-dir")
    serve.add_argument("--backend", choices=("board", "alsa", "null"))
    serve.add_argument("--host")
    serve.add_argument("--port", type=int)
    serve.add_argument("--socket")
    serve.add_argument("--player")
    serve.add_argument("--tts-program")
    serve.add_argument("--ffmpeg")
    for name in ("list", "doctor", "events", "acceptance"):
        commands.add_parser(name)
    add = commands.add_parser("import")
    add.add_argument("file", type=Path)
    add.add_argument("--key")
    add.add_argument("--name")
    play = commands.add_parser("play")
    play.add_argument("clip")
    play.add_argument("--task-id", default="cli")
    play.add_argument("--repeat", type=int, default=1)
    play.add_argument("--priority", type=int, default=10)
    play.add_argument("--policy", choices=("enqueue", "replace_task", "interrupt", "reject"), default="enqueue")
    play.add_argument("--gain", type=float, default=1.0)
    play.add_argument("--ttl", type=float, default=120)
    play.add_argument("--request-id")
    play.add_argument("--wait", action="store_true")
    play.add_argument("--timeout", type=float, default=600)
    jobs = commands.add_parser("jobs")
    jobs.add_argument("--task-id")
    for action in ("status", "pause", "resume", "seek"):
        control = commands.add_parser(action)
        control.add_argument("job_id")
        if action == "seek": control.add_argument("position", type=float)
    rename = commands.add_parser("rename")
    rename.add_argument("clip")
    rename.add_argument("name")
    rename.add_argument("--key")
    delete = commands.add_parser("delete")
    delete.add_argument("clip")
    download = commands.add_parser("download")
    download.add_argument("clip")
    download.add_argument("file", type=Path)
    stop = commands.add_parser("stop")
    group = stop.add_mutually_exclusive_group(required=True)
    group.add_argument("--task-id")
    group.add_argument("--job-id")
    group.add_argument("--all", action="store_true")
    say = commands.add_parser("say")
    say.add_argument("text")
    say.add_argument("--voice")
    say.add_argument("--speed", type=int, default=160)
    say.add_argument("--key")
    say.add_argument("--save-only", action="store_true")
    say.add_argument("--wait", action="store_true")
    settings = commands.add_parser("settings")
    settings.add_argument("--volume", type=int)
    settings.add_argument("--mute", action=argparse.BooleanOptionalAction, default=None)
    settings.add_argument("--enabled", action=argparse.BooleanOptionalAction, default=None)
    record = commands.add_parser("record")
    record.add_argument("action", choices=("start", "stop", "status"))
    record.add_argument("--seconds", type=int, default=120)
    record.add_argument("--name", default="机身录音")
    record.add_argument("--playback", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "serve":
            from .server import Application
            changes = {k: getattr(args, k) for k in ("data_dir", "backend", "host", "port", "socket", "player", "tts_program", "ffmpeg")}
            if args.data_dir:
                changes["data_dir"] = str(Path(args.data_dir).resolve())
                if not args.socket:
                    changes["socket"] = str(Path(changes["data_dir"]) / "audio.sock")
            config = Config.load(args.config, **changes)
            if config.backend == "null" and args.data_dir:
                config.lock_file = str(Path(config.data_dir) / "output.lock")
            app = Application(config).start()
            stop = threading.Event()
            for sig in (signal.SIGTERM, signal.SIGINT):
                signal.signal(sig, lambda *_: stop.set())
            print(json.dumps({"listening": app.http.server_address, "socket": config.socket,
                              "backend": config.backend, "token_file": config.token_file}, ensure_ascii=False), flush=True)
            try:
                while not stop.wait(.5):
                    pass
            finally:
                app.close()
            return 0
        client = AudioClient(args.endpoint, token_file=args.token_file)
        if args.command == "doctor": result = client.health()
        elif args.command == "list": result = client.clips()
        elif args.command == "import": result = client.import_file(args.file, key=args.key, name=args.name)
        elif args.command == "play":
            job = client.play(args.clip, task_id=args.task_id, repeat=args.repeat, priority=args.priority, policy=args.policy, gain=args.gain, ttl=args.ttl, request_id=args.request_id)
            result = job.wait(args.timeout) if args.wait else job.initial
        elif args.command == "jobs":
            from urllib.parse import urlencode
            result = client.request("GET", "/api/jobs" + ("?" + urlencode({"task_id": args.task_id}) if args.task_id else ""))
        elif args.command in ("status", "pause", "resume", "seek"):
            path = "/api/jobs/" + args.job_id
            result = client.request("GET", path) if args.command == "status" else client.request("POST", path + "/" + args.command, {"position": args.position} if args.command == "seek" else {})
        elif args.command == "rename": result = client.rename(args.clip, args.name, key=args.key)
        elif args.command == "delete": result = client.delete(args.clip)
        elif args.command == "download":
            client.download(args.clip, args.file)
            result = {"file": str(args.file.resolve())}
        elif args.command == "stop":
            result = client.request("POST", "/api/stop-all", {}) if args.all else client.request("POST", "/api/stop", {k: v for k, v in {"task_id": args.task_id, "job_id": args.job_id}.items() if v})
        elif args.command == "say":
            result = client.say(args.text, voice=args.voice, speed=args.speed, key=args.key, play=not args.save_only)
            playback = result.pop("playback", None)
            if playback and args.wait: result["job"] = playback.wait(600)
        elif args.command == "settings":
            result = client.settings(**{k: v for k, v in {"volume": args.volume, "muted": args.mute, "enabled": args.enabled}.items() if v is not None})
        elif args.command == "record":
            if args.action == "status": result = client.request("GET", "/api/recording")
            elif args.action == "start": result = client.request("POST", "/api/recording/start", {"seconds": args.seconds, "name": args.name, "playback": args.playback})
            else: result = client.request("POST", "/api/recording/stop", {"playback": args.playback})
        elif args.command == "events": result = client.request("GET", "/api/events")
        elif args.command == "acceptance": result = client.request("GET", "/api/acceptance")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (AudioError, OSError, TimeoutError) as exc:
        print(json.dumps({"error": str(exc), "code": getattr(exc, "code", "command_failed")}, ensure_ascii=False), file=sys.stderr)
        return 1
