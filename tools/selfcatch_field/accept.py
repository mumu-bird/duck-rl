"""WonderDuck field checks: passive Studio telemetry and optional on-board replay only."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import zipfile

if __package__:
    from . import readonly_preflight as telem
else:
    import readonly_preflight as telem

ROOT = Path(__file__).resolve().parent.parent.parent
BUNDLE = ROOT / 'runs/self-catch-field/board-shadow-02/board-replay.py'
SCRIPT_SHA = 'a5d32fd912c4de25b51e2eea1a4fba0ffe107d17b215950767d9d4dd1e6e5ee1'
ACTOR_SHA = '0f8a47cc5dde3d3e56f2001d78dcb007f20147f89db2d55b9c2f37ae4af37a17'


def packed():
    path = Path(sys.argv[0])
    return path if path.suffix == '.pyz' and path.is_file() else None


def program_bytes():
    if packed():
        with zipfile.ZipFile(packed()) as archive:
            program = archive.read('board-replay.py')
    else:
        program = BUNDLE.read_bytes()
    if hashlib.sha256(program).hexdigest() != SCRIPT_SHA:
        raise ValueError('本地只读回放包与签发时指纹不同')
    return program


def replay(user):
    if not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_.-]{0,31}', user):
        return dict(code='NOT_RUN', reason='SSH 用户名格式不正确')
    try:
        if not telem.direct_route():
            return dict(code='NOT_RUN', reason='SSH 前复查时已不是热点直连，未执行板端回放')
        program = program_bytes()
        # Stream a pinned, inspectable script to ordinary Python's stdin. No install,
        # file copy, remote shell interpolation, robot socket or motion service.
        ssh = ['ssh', '-F', os.devnull, '-T', '-a', '-x', '-o', 'ClearAllForwardings=yes',
               '-o', 'StrictHostKeyChecking=ask', '-o', 'ConnectTimeout=5', '-o', 'ConnectionAttempts=1',
               '-o', 'NumberOfPasswordPrompts=1', '-l', user, telem.HOST, 'python3 -I -B -S -']
        proc = subprocess.run(ssh, input=program, stdout=subprocess.PIPE, timeout=90, check=False)
        lines = [line for line in proc.stdout.splitlines() if line.startswith(b'WONDERDUCK_SHADOW ')]
        if len(lines) == 1:
            result = json.loads(lines[0].partition(b' ')[2])
            if (isinstance(result, dict) and result.get('code') == 'BOARD_POLICY_PASS' and proc.returncode == 0
                    and result.get('rk3566') is True and result.get('checkpoint_sha256') == ACTOR_SHA
                    and result.get('inputs') == 200 and result.get('scalars') == 1200
                    and telem.number(result.get('max_abs_error')) and 0 <= result['max_abs_error'] <= 2e-5
                    and telem.number(result.get('forward_p95_ms')) and result['forward_p95_ms'] >= 0
                    and result.get('task_result') == 'NOT_TESTED' and result.get('motion_permitted') is False):
                return {k: result[k] for k in ('code', 'inputs', 'scalars', 'max_abs_error', 'forward_p95_ms', 'checkpoint_sha256', 'rk3566')}
            if isinstance(result, dict) and result.get('code') == 'HOST_PARITY_ONLY':
                return dict(code='NOT_VERIFIED', reason='回放可算，但目标未自报 RK3566，不能记为板端通过')
        return dict(code='NOT_VERIFIED', reason=f'SSH、板端 Python 或输出核对失败（退出码 {proc.returncode}）')
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError, KeyError, zipfile.BadZipFile) as error:
        return dict(code='NOT_VERIFIED', reason=f'未完成板端自测：{error}')


def main(argv=None):
    p = argparse.ArgumentParser(description='只读验收：不运行机器人动作，不改模型。')
    choice = p.add_mutually_exclusive_group()
    choice.add_argument('--ssh-user', help='可选：已获准使用的机器人 SSH 账号；仅运行板端数值回放')
    choice.add_argument('--telemetry-only', action='store_true', help='只读 Studio；不使用 SSH')
    choice.add_argument('--offline', action='store_true', help='仅在本机验证回放包，不访问 Studio 或 SSH')
    p.add_argument('--out-dir', type=Path, default=packed().resolve().parent / 'reports' if packed() else telem.DEFAULT_OUT)
    args = p.parse_args(argv)
    if args.ssh_user is not None and not re.fullmatch(r'[a-zA-Z_][a-zA-Z0-9_.-]{0,31}', args.ssh_user):
        p.error('请提供这台设备已确认的 SSH 用户名；没有账号请省略 --ssh-user')
    if args.offline:
        try:
            proc = subprocess.run([sys.executable, '-I', '-B', '-S', '-'], input=program_bytes(),
                                  stdout=subprocess.PIPE, timeout=20, check=False)
            if proc.returncode not in (0, 3) or not proc.stdout.startswith(b'WONDERDUCK_SHADOW '):
                raise ValueError('回放程序输出异常')
            result = json.loads(proc.stdout.partition(b' ')[2])
            if (not isinstance(result, dict) or result.get('checkpoint_sha256') != ACTOR_SHA
                    or result.get('scalars') != 1200 or result.get('inputs') != 200
                    or (proc.returncode, result.get('code')) not in ((3, 'HOST_PARITY_ONLY'), (0, 'BOARD_POLICY_PASS'))
                    or result.get('motion_permitted') is not False or result.get('task_result') != 'NOT_TESTED'
                    or not telem.number(result.get('max_abs_error')) or not 0 <= result['max_abs_error'] <= 2e-5):
                raise ValueError('与冻结策略不符')
        except (OSError, subprocess.TimeoutExpired, ValueError, KeyError, zipfile.BadZipFile) as error:
            print(f'本地包自检失败：{error}', file=sys.stderr)
            return 2
        print(f"LOCAL_PARITY_OK：本机 1,200 个输出与原模型一致，最大误差 {result['max_abs_error']:.3g}。未连接鸭子，未验证实体动作。")
        return 0
    report = telem.run()
    print(f"基础设备检查：{report['basic_code']}")
    print(f"完整只读诊断：{report['code']}", flush=True)
    if report['samples']:
        last = report['samples'][-1]
        voltage = f"{last['battery_v']} V" if last['battery_v'] is not None else '无读数'
        print(f"实测舵机 {last['measured_servos']}/15，IMU {'有效' if last['imu_valid'] else '无有效读数'}，电池 {voltage}")
    for reason in report['reasons']:
        print(' -', reason)
    user = args.ssh_user
    board = dict(code='NOT_RUN', reason='默认不连接 SSH')
    if user:
        board = (replay(user) if report['identity_verified'] and report['code'] != 'NO_LINK'
                 else dict(code='NOT_RUN', reason='没有核实直连设备的 Studio 身份，未执行板端回放'))
    print('板端计算：', board['code'], board.get('reason', '原策略的 1,200 个输出与参考值一致'))
    report.update(board_replay=board, task_result='NOT_TESTED', motion_permitted=False)
    now = datetime.now(timezone.utc)
    report['utc'] = now.isoformat()
    out = args.out_dir / f"field-{now.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with out.open('x', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write('\n')
    except OSError as error:
        print(f'报告未保存：{error}', file=sys.stderr)
        return 2
    print('自动接球：NOT_TESTED；不会发出动作。\n电脑上的报告：', out.resolve())
    if report['code'] == 'NO_LINK':
        return 2
    return 0 if report['code'] == 'PASS_READONLY' and (not user or board['code'] == 'BOARD_POLICY_PASS') else 1


if __name__ == '__main__':
    sys.exit(main())
