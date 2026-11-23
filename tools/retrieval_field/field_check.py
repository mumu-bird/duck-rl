"""Portable retrieval-specific read-only acceptance entry. Never sends motion."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import zipfile

if __package__:
    from . import shadow
    from tools.selfcatch_field import readonly_preflight, camera_probe
else:
    import shadow
    import readonly_preflight
    import camera_probe


def replay_bundle(read):
    manifest = json.loads(read("manifest.json"))
    if (manifest.get("schema") != shadow.SCHEMA or manifest.get("task") != "retrieval"
            or manifest.get("motion_permitted") is not False):
        raise ValueError("not a retrieval shadow bundle")
    required = {"actor.json", "replay-cases.json", "verification.json", "high-level-actor.onnx"}
    if not required <= set(manifest.get("files", {})):
        raise ValueError("incomplete bundle manifest")
    for name, digest in manifest["files"].items():
        if hashlib.sha256(read(name)).hexdigest() != digest:
            raise ValueError("bundle checksum mismatch: " + name)
    model = json.loads(read("actor.json"))
    if model.get("checkpoint_sha256") != manifest.get("checkpoint_sha256"):
        raise ValueError("checkpoint identity mismatch")
    report = shadow.verify_cases(model, json.loads(read("replay-cases.json")))
    report["checkpoint_sha256"] = model["checkpoint_sha256"]
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description="拾取任务只读检查：遥测、策略回放、可选相机采集；不发送运动。")
    parser.add_argument("--offline", action="store_true", help="仅在电脑上检查包完整性与策略回放")
    parser.add_argument("--camera", action="store_true", help="用已确认的 SSH 账号只读取30帧相机画面")
    parser.add_argument("--ssh-user", help="这台机器鸭已确认的 SSH 用户名")
    parser.add_argument("--bundle", type=Path, help="开发模式的导出目录；pyz 包无需指定")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    if (args.camera and not args.ssh_user) or (args.offline and args.camera):
        parser.error("相机采集需要 --ssh-user，不能与 --offline 同用")
    entry = Path(sys.argv[0]).resolve()
    out = args.out or entry.parent / "retrieval-reports"
    report = dict(task="retrieval", hardware_task_result="NOT_TESTED", retrieval_motion_ready=False,
                  motion_permitted=False, camera=dict(code="NOT_RUN"),
                  utc=datetime.now(timezone.utc).isoformat())
    try:
        if entry.suffix == ".pyz":
            with zipfile.ZipFile(entry) as archive:
                report["policy_replay"] = replay_bundle(archive.read)
        elif args.bundle:
            report["policy_replay"] = replay_bundle(lambda name: (args.bundle / name).read_bytes())
        else:
            parser.error("开发模式需要 --bundle")
        if report["policy_replay"]["code"] != "SHADOW_REPLAY_PASS":
            raise ValueError("policy replay failed")
        if args.offline:
            report["code"] = "OFFLINE_REPLAY_PASS"
            report["device"] = dict(code="NOT_RUN", identity_verified=False)
        else:
            report["device"] = readonly_preflight.run()
            report["code"] = report["device"]["code"]
        if args.camera and report["device"].get("identity_verified"):
            folder, result = camera_probe.capture(args.ssh_user, 30, 128, out / "camera")
            report["camera"] = dict(result, folder=str(folder))
            if result["code"] != "FRAME_CAPTURE_ONLY":
                report["code"] = "CHECK"
        report["unverified"] = ["real camera calibration and perception accuracy",
                                "encoder/IMU conventions and complete control adapter",
                                "bill force/contact feedback and measured object properties",
                                "bounded motion/stop validation on this robot",
                                "physical pickup, delivery and return"]
    except (OSError, ValueError, KeyError, TypeError, zipfile.BadZipFile) as error:
        report.update(code="CHECK", error=str(error))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"retrieval-{stamp}.json"
    path.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print("报告：", path)
    print("本结果不表示实体拾取通过，也不允许自动发送运动。")
    return 0 if report["code"] in ("OFFLINE_REPLAY_PASS", "PASS_READONLY") else (2 if report["code"] == "NO_LINK" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
