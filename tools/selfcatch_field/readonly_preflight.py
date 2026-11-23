"""Read-only WonderDuck Studio telemetry preflight. Sends GET requests only."""

import argparse
from datetime import datetime, timezone
import http.client
import ipaddress
import json
import math
from pathlib import Path
import re
import subprocess
import sys
import time

HOST = '192.168.149.1'
PORT = 8090
MAX_RESPONSE = 1_048_576
MAX_AGE_MS = 500
MIN_BATTERY_V = 7.0
ALLOWED_PATHS = ('/api/info', '/api/state')
JOINT_NAMES = (
    'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
    'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll', 'mouth',
    'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle',
)
IDS = (20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 10, 11, 12, 13, 14)
JOINTS = dict(zip(IDS, JOINT_NAMES))
DEFAULT_OUT = Path(__file__).resolve().parent.parent.parent / 'runs/self-catch-field/device'


def hotspot_route(routes):
    if not isinstance(routes, list) or len(routes) != 1 or not isinstance(routes[0], dict):
        return False
    row = routes[0]
    try:
        source = ipaddress.IPv4Address(row['prefsrc'])
    except (KeyError, ValueError):
        return False
    net = ipaddress.IPv4Network('192.168.149.0/24')
    return (row.get('dst') == HOST and row.get('dev') not in ('lo', None, '')
            and row.get('type', 'unicast') == 'unicast' and not row.get('gateway') and not row.get('via')
            and source in net and source not in (net.network_address, net.broadcast_address, ipaddress.IPv4Address(HOST)))


def direct_route():
    # Ask the local Linux routing table. No ping, probe, DNS or datagram.
    if not sys.platform.startswith('linux'):
        raise RuntimeError('请在本项目所在的 Linux 电脑上运行热点直连检查')
    proc = subprocess.run(['ip', '-j', '-4', 'route', 'get', HOST], capture_output=True, timeout=3)
    return proc.returncode == 0 and hotspot_route(json.loads(proc.stdout))


def get_json(path):
    if path not in ALLOWED_PATHS:
        raise ValueError('不是预检只读接口')
    connection = http.client.HTTPConnection(HOST, PORT, timeout=2.0)
    try:
        headers = {'Accept': 'application/json', 'Connection': 'close'}
        connection.request('GET', path, headers=headers)
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f'{path}: HTTP {response.status}')
        raw = response.read(MAX_RESPONSE + 1)
        if len(raw) > MAX_RESPONSE:
            raise RuntimeError(f'{path}: 响应过大')
        try:
            data = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RuntimeError(f'{path}: 未收到 JSON；核对热点和 Studio 版本') from error
        if not isinstance(data, dict):
            raise RuntimeError(f'{path}: 数据格式不匹配')
        return data
    finally:
        connection.close()


def number(value):
    return type(value) in (int, float) and abs(value) < 1e30 and math.isfinite(value)


def fresh(device, field='age_ms', limit=MAX_AGE_MS, elapsed_ms=0):
    age = device.get(field)
    return number(age) and age >= 0 and age + elapsed_ms < limit


def obj(value):
    return value if isinstance(value, dict) else {}


def stamp(value):
    return value if type(value) is int and 0 < value < 2**64 else None


def identity_ok(info):
    joints = info.get('joints')
    if info.get('studio') is not True or info.get('demo') is not False:
        return False
    if not isinstance(joints, list) or len(joints) != len(IDS):
        return False
    return all(type(obj(j).get('id')) is int for j in joints) and {
        j['id']: j.get('name') for j in joints} == JOINTS


def inspect(info, state, elapsed_ms=0):
    failures = []
    if not identity_ok(info) or state.get('demo') is True:
        failures.append('不是已核对的非演示 Studio / 15 关节映射')
    if (state.get('online') is not True or state.get('control_online') is False or
            any(state.get(k) for k in ('error', 'control_error', 'bus_error', 'bus_waiting', 'device_paused',
                                       'maintenance_active', 'maintenance_error'))):
        failures.append('设备离线、错误或维护占用')
    if not fresh(state, elapsed_ms=elapsed_ms):
        failures.append('控制状态缺失或过期')
    control_ok = not failures
    # Studio snapshot publishes the device rows here. The control sidebar can
    # contain angles converted back to counts; only accept explicit diagnostics.
    rows = state.get('servos')
    rows = rows if isinstance(rows, list) else []
    servos = {row['id']: row for row in rows if type(obj(row).get('id')) is int}
    if len(rows) != len(IDS) or set(servos) != set(IDS):
        failures.append('舵机反馈 ID 缺失、重复或不符')
    valid = []
    head = {}
    stamps = {'frame': stamp(state.get('t_ns'))}
    limit = 3000 if 'device_report_age_ms' in state else MAX_AGE_MS
    if 'device_report_age_ms' in state and not fresh(state, 'device_report_age_ms', limit, elapsed_ms):
        failures.append('逐舵机状态流未更新')
    for idx in IDS:
        row = servos.get(idx, {})
        raw = row.get('raw')
        real = (row.get('diagnostics_available') is True and row.get('encoder_reconstructed') is False
                and type(raw) is int and 0 <= raw < 4096)
        ok = (real and row.get('valid') is True and row.get('online') is True and not row.get('error')
              and type(row.get('packet_status')) is int and row['packet_status'] == 0
              and type(row.get('register_status')) is int and row['register_status'] == 0
              and row.get('faults') == [] and type(row.get('status')) is int and row['status'] == 0
              and stamp(row.get('t_ns')) is not None
              and fresh(row, limit=limit, elapsed_ms=elapsed_ms)
              and fresh(row, 'observation_age_ms', limit, elapsed_ms))
        if ok:
            valid.append(idx)
        name = JOINTS[idx]
        stamps[name] = stamp(row.get('t_ns')) if ok else None
        if idx in (30, 31, 34):
            head[name] = dict(id=idx, measured_raw=raw if ok else None, valid=ok)
    if len(valid) != len(IDS):
        failures.append('没有新鲜、无故障的实测编码器反馈，ID：' + ','.join(str(i) for i in IDS if i not in valid))
        if any(obj(r).get('encoder_reconstructed') is True for r in rows):
            failures.append('固件只发布了部分或全部重建角度；无法从匿名接口确认全部编码器，不等于舵机已坏')
    imu = obj(state.get('imu'))
    quat = imu.get('quaternion_wxyz')
    imu_ok = (imu.get('valid') is True and not imu.get('error') and fresh(imu, elapsed_ms=elapsed_ms)
              and isinstance(quat, list) and len(quat) == 4 and all(number(v) for v in quat)
              and abs(math.sqrt(sum(v*v for v in quat)) - 1) < .05)
    stamps['imu'] = stamp(imu.get('t_ns'))
    if not imu_ok:
        failures.append('IMU 姿态缺失、无效或过期')
    robot = obj(state.get('robot'))
    battery = obj(obj(robot.get('health')).get('battery')).get('volts')
    low = obj(robot.get('low_voltage'))
    threshold = obj(low.get('settings')).get('threshold_v')
    threshold = max(MIN_BATTERY_V, threshold) if number(threshold) else MIN_BATTERY_V
    battery_ok = (number(battery) and threshold <= battery <= 9 and low.get('triggered') is False
                  and not (number(low.get('voltage_v')) and low['voltage_v'] < threshold))
    if not battery_ok:
        failures.append('电压 / 低压保护数据缺失、异常或已告警；低压先充电')
    return dict(failures=failures, control_ok=control_ok, battery_ok=battery_ok,
                measured_servos=len(valid), imu_valid=imu_ok, head=head,
                imu_quaternion_wxyz=quat if imu_ok else None,
                measured_raw={JOINTS[i]: servos[i]['raw'] for i in valid},
                battery_v=battery if number(battery) else None, t_ns=stamps,
                request_ms=round(elapsed_ms, 1))


def run(fetch=get_json, sleep=time.sleep):
    report = dict(target=f'{HOST}:{PORT}', code='NO_LINK', identity_verified=False, motion_permitted=False,
                  task_result='NOT_TESTED', basic_code='NOT_RUN', versions={}, samples=[], reasons=[])
    try:
        if not direct_route():
            report['reasons'].append('目标没有热点直连路由；请连接 WD- 热点后重试。未向目标发送请求。')
            return report
    except (OSError, RuntimeError, subprocess.SubprocessError, ValueError) as error:
        report['reasons'].append(f'无法核对本机直连路由（需要 Linux 的 ip 工具）；未向目标发送请求：{error}')
        return report
    try:
        info = fetch('/api/info')
        report['versions'] = {k: v for k, v in obj(info.get('device_versions')).items()
                              if k in ('image_version', 'hardware_version', 'imu_firmware_version')
                              and isinstance(v, str) and re.fullmatch(r'[A-Za-z0-9._+ -]{1,64}', v)}
        if not identity_ok(info):
            report.update(code='CHECK', reasons=['Studio 为演示模式、版本不兼容或关节映射不符；未继续读取'])
            return report
        report['identity_verified'] = True
        report['basic_code'] = 'CHECK'
        for i in range(3):
            if i:
                sleep(.65)
            start = time.monotonic()
            state = fetch('/api/state')
            report['samples'].append(inspect(info, state, (time.monotonic()-start)*1000))
    except (OSError, http.client.HTTPException, RuntimeError, ValueError, TypeError, KeyError, RecursionError) as error:
        report['reasons'].append(f'未能完整读取 Studio：{error}')
    if report['reasons']:
        report['code'] = 'CHECK' if report['identity_verified'] else 'NO_LINK'
        return report
    if not report['samples']:
        return report
    reasons = [reason for sample in report['samples'] for reason in sample['failures']]
    stale = []
    for name in report['samples'][0]['t_ns']:
        times = [s['t_ns'][name] for s in report['samples']]
        if None in times or not all(a < b for a, b in zip(times, times[1:])):
            stale.append(name)
    if stale:
        reasons.append('三次实测时间戳缺失或没有连续前进：' + ', '.join(stale))
    if (all(s['control_ok'] and s['imu_valid'] and s['battery_ok'] for s in report['samples'])
            and 'imu' not in stale and 'frame' not in stale):
        report['basic_code'] = 'BASIC_READONLY_OK'
    report.update(code='CHECK' if reasons else 'PASS_READONLY', reasons=list(dict.fromkeys(reasons)))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description='WonderDuck 只读遥测检查，不控制机器人。')
    parser.add_argument('--out-dir', default=DEFAULT_OUT, type=Path, help='电脑上保存报告的目录')
    args = parser.parse_args(argv)
    result = run()
    now = datetime.now(timezone.utc)
    result['utc'] = now.isoformat()
    path = args.out_dir / f"preflight-{now.strftime('%Y%m%dT%H%M%S%fZ')}.json"
    try:
        args.out_dir.mkdir(parents=True, exist_ok=True)
        with path.open('x', encoding='utf-8') as out:
            json.dump(result, out, indent=2, ensure_ascii=False, allow_nan=False)
            out.write('\n')
    except OSError as error:
        print(f'报告未保存：{error}', file=sys.stderr)
        return 2
    print(result['code'], '｜ 仅检查遥测；未运行接球、未授权动作。')
    if result['samples']:
        last = result['samples'][-1]
        print(f"实测舵机 {last['measured_servos']}/15，IMU {'有效' if last['imu_valid'] else '无效'}，电池 {last['battery_v']} V")
    for reason in result['reasons']:
        print(' -', reason)
    print('电脑上的报告：', path.resolve())
    return {'PASS_READONLY': 0, 'CHECK': 1, 'NO_LINK': 2}[result['code']]


if __name__ == '__main__':
    sys.exit(main())
