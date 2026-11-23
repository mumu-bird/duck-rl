"""Host launcher for camera acquisition. No motion service is contacted."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import zipfile

if __package__:
    from . import camera_board, readonly_preflight as telem
else:
    import camera_board
    import readonly_preflight as telem


def packed():
    entry = Path(sys.argv[0])
    return entry if entry.suffix == '.pyz' and entry.is_file() else None


def board_program():
    if packed():
        with zipfile.ZipFile(packed()) as archive:
            return archive.read('camera_board.py')
    return Path(camera_board.__file__).read_bytes()


def verify_archive(path, max_mib, requested):
    """Validate every member and its digest without extracting archive paths."""
    if path.stat().st_size > max_mib * 1024 * 1024 + 1024 * 1024:
        raise ValueError('camera archive exceeds configured raw image budget')
    with tarfile.open(path, 'r:') as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if len(names) != len(set(names)) or any(not member.isfile() for member in members):
            raise ValueError('invalid camera archive members')
        if 'report.json' not in names or archive.getmember('report.json').size > 256 * 1024:
            raise ValueError('missing or oversized camera report')
        report = json.load(archive.extractfile('report.json'))
        if (not isinstance(report, dict) or report.get('requested_frames') != requested
                or report.get('api_version') != camera_board.API_VERSION
                or report.get('motion_permitted') is not False or report.get('task_result') != 'NOT_TESTED'
                or report.get('perception_verified') is not False or report.get('max_raw_mib') != max_mib
                or report.get('code') not in ('FRAME_CAPTURE_ONLY', 'CAPTURE_PARTIAL')
                or not isinstance(report.get('frames'), list)):
            raise ValueError('camera report does not match requested capture')
        rows = report['frames']
        if (not 0 <= len(rows) <= requested or report.get('frames_received') != len(rows)
                or (report['code'] == 'FRAME_CAPTURE_ONLY' and len(rows) != requested)):
            raise ValueError('invalid camera frame count')
        expected = ['report.json'] + [f'frames/{i:04d}.uyvy' for i in range(len(rows))]
        if rows:
            expected.append('preview.png')
        if set(names) != set(expected):
            raise ValueError('unexpected camera archive paths')
        budget = max_mib * 1024 * 1024
        stamps, geometry = [], None
        for i, row in enumerate(rows):
            if not isinstance(row, dict) or row.get('file') != expected[i + 1]:
                raise ValueError('invalid camera frame path')
            header = row['header']
            camera_board.validate_header(header, budget)
            shape = tuple(header[k] for k in ('width', 'height', 'format', 'rotate'))
            if geometry is not None and shape != geometry:
                raise ValueError('camera geometry changed during capture')
            geometry = shape
            stamps.append(header['captured_at_unix_us'])
            member = archive.getmember(row['file'])
            if member.size != header['bytes']:
                raise ValueError('camera archive frame size mismatch')
            raw = archive.extractfile(member).read()
            if hashlib.sha256(raw).hexdigest() != row.get('sha256'):
                raise ValueError('camera archive frame checksum mismatch')
            budget -= len(raw)
        if any(b <= a for a, b in zip(stamps, stamps[1:])):
            raise ValueError('camera timestamps did not advance')
        if report.get('raw_bytes') != max_mib * 1024 * 1024 - budget:
            raise ValueError('camera archive total size mismatch')
        preview = None
        if rows:
            if archive.getmember('preview.png').size > 400_000:
                raise ValueError('oversized camera preview')
            preview = archive.extractfile('preview.png').read()
            if not preview.startswith(b'\x89PNG\r\n\x1a\n'):
                raise ValueError('invalid camera preview')
        return report, preview


def capture(user, count, max_mib, out_dir):
    if not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_.-]{0,31}', user):
        raise ValueError('请使用这台设备已确认的 SSH 用户名')
    if type(count) is not int or not 1 <= count <= 120 or type(max_mib) is not int or not 16 <= max_mib <= 256:
        raise ValueError('帧数须为 1..120，原始帧内存上限须为 16..256 MiB')
    preflight = telem.run()
    if not preflight['identity_verified'] or preflight['code'] == 'NO_LINK':
        raise RuntimeError('未确认热点直连的非演示设备；未运行 SSH。' + '；'.join(preflight['reasons']))
    if not telem.direct_route():
        raise RuntimeError('SSH 前复查时已不是热点直连；未采集图像')
    program = board_program()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    folder = Path(out_dir).resolve() / f'camera-{stamp}'
    folder.mkdir(parents=True, exist_ok=False)
    launch = {'board_program_sha256': hashlib.sha256(program).hexdigest(),
              'preflight': preflight, 'task_result': 'NOT_TESTED', 'motion_permitted': False,
              'code': 'CAPTURE_INCOMPLETE'}
    ssh = ['ssh', '-F', os.devnull, '-T', '-a', '-x', '-o', 'ClearAllForwardings=yes',
           '-o', 'StrictHostKeyChecking=ask', '-o', 'ConnectTimeout=5', '-o', 'ConnectionAttempts=1',
           '-o', 'NumberOfPasswordPrompts=1', '-l', user, telem.HOST,
           f'python3 -I -B -S - --count {count} --max-mib {max_mib}']
    try:
        path = folder / 'capture.tar'
        with path.open('xb') as output:
            proc = subprocess.run(ssh, input=program, stdout=output, timeout=60, check=False)
        report, preview = verify_archive(path, max_mib, count)
        launch.update(camera=report, ssh_exit_code=proc.returncode)
        expected_exit = 0 if report['code'] == 'FRAME_CAPTURE_ONLY' else 1
        if proc.returncode != expected_exit:
            raise ValueError('SSH 退出码与相机记录不符')
        if preview:
            (folder / 'preview.png').write_bytes(preview)
        launch['code'] = report['code']
    except (OSError, ValueError, KeyError, TypeError, tarfile.TarError, subprocess.SubprocessError) as error:
        launch['error'] = str(error)
    (folder / 'report.json').write_text(json.dumps(launch, ensure_ascii=False, allow_nan=False, indent=2) + '\n', encoding='utf-8')
    return folder, launch


def main(argv=None):
    parser = argparse.ArgumentParser(description='相机只读采集：板上先取帧，完成后传回电脑，不发动作。')
    parser.add_argument('--ssh-user', required=True, help='这台鸭子已确认的 SSH 账号；首次核对主机密钥')
    parser.add_argument('--count', type=int, default=30, help='目标帧数，默认30；取帧窗口最多5秒')
    parser.add_argument('--max-mib', type=int, default=128, help='板端原始图像内存上限，16..256 MiB')
    parser.add_argument('--out-dir', type=Path,
                        default=packed().resolve().parent / 'camera-records' if packed() else
                        Path(__file__).resolve().parents[2] / 'runs/self-catch-field/camera')
    args = parser.parse_args(argv)
    try:
        folder, report = capture(args.ssh_user, args.count, args.max_mib, args.out_dir)
    except (OSError, RuntimeError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(f'相机采集未开始：{error}', file=sys.stderr)
        return 2
    print(f"相机采集：{report['code']}；记录：{folder}")
    camera = report.get('camera', {})
    if camera:
        print(f"收到 {camera['frames_received']}/{args.count} 帧；{camera['stop_reason']}")
    if report.get('error'):
        print(report['error'])
    print('仅采集原始图像；曝光延迟、标定、看球策略和实体接球仍未验证。')
    return 0 if report['code'] == 'FRAME_CAPTURE_ONLY' else 1


if __name__ == '__main__':
    sys.exit(main())
