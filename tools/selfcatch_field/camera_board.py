"""Bounded camera-only acquisition; streamed to ordinary board Python over SSH.

Only hello and media.frame are sent, to mediad's dedicated Unix socket. Images
are buffered before transfer so Wi-Fi cannot throttle the acquisition loop.
"""

import argparse
import hashlib
import io
import json
import socket
import struct
import sys
import tarfile
import time
import zlib

API_VERSION = 62  # source snapshot 86f31bfee90ba96cd4990320b7bdcb9dd7864612
MEDIA_SOCKET = '/run/mediad/media.sock'
MAX_FRAME = 16 * 1024 * 1024
MAX_LINE = 4096


class CapacityLimit(ValueError):
    pass


def exact(reader, size):
    result = bytearray()
    while len(result) < size:
        chunk = reader.read(size - len(result))
        if not chunk:
            raise EOFError('camera binary payload ended early')
        result.extend(chunk)
    return bytes(result)


def response(reader, ident):
    line = reader.readline(MAX_LINE + 1)
    if len(line) > MAX_LINE or not line.endswith(b'\n'):
        raise ValueError('camera JSON line is incomplete or too long')
    value = json.loads(line)
    if (not isinstance(value, dict) or value.get('jsonrpc') != '2.0'
            or type(value.get('id')) is not int or value['id'] != ident):
        raise ValueError('camera response ID or JSON-RPC version mismatch')
    if value.get('error') is not None:
        raise ValueError('camera service rejected request: ' + str(value['error'])[:300])
    if not isinstance(value.get('result'), dict):
        raise ValueError('camera response has no object result')
    return value['result']


def validate_header(h, budget):
    if not isinstance(h, dict):
        raise ValueError('frame header is not an object')
    for key in ('width', 'height', 'bytes', 'rotate', 'captured_at_unix_us'):
        if type(h.get(key)) is not int:
            raise ValueError('invalid integer field: ' + key)
    if (h['width'] <= 0 or h['height'] <= 0 or h['width'] % 2
            or h['bytes'] != h['width'] * h['height'] * 2
            or h['bytes'] > MAX_FRAME or h.get('format') != 'UYVY'
            or h['rotate'] not in (0, 90, 180, 270)
            or not 0 < h['captured_at_unix_us'] < 2**64):
        raise ValueError('unsupported frame geometry, format, rotation or timestamp')
    if h['bytes'] > budget:
        raise CapacityLimit('raw image memory budget reached')


class DeadlineReader:
    """A shared buffer for JSON and binary, with a total socket deadline."""

    def __init__(self, sock, deadline):
        self.sock, self.deadline, self.pending = sock, deadline, bytearray()

    def timeout(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('camera frame deadline exceeded')
        self.sock.settimeout(remaining)

    def read(self, size):
        if self.pending:
            result = bytes(self.pending[:size])
            del self.pending[:size]
            return result
        self.timeout()
        return self.sock.recv(min(size, 1024 * 1024))

    def readline(self, limit):
        while True:
            newline = self.pending.find(b'\n')
            if newline >= 0 or len(self.pending) >= limit:
                end = min(newline + 1, limit) if newline >= 0 else limit
                result = bytes(self.pending[:end])
                del self.pending[:end]
                return result
            self.timeout()
            chunk = self.sock.recv(min(4096, limit - len(self.pending)))
            if not chunk:
                result = bytes(self.pending)
                self.pending.clear()
                return result
            self.pending.extend(chunk)


def fetch(budget, path=MEDIA_SOCKET, timeout=3.0):
    requested_ns = time.monotonic_ns()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
        reader = DeadlineReader(sock, time.monotonic() + timeout)
        reader.timeout()
        sock.connect(path)
        reader.timeout()
        sock.sendall((json.dumps({'jsonrpc': '2.0', 'id': 1, 'method': 'hello',
                                 'params': {'api_version': API_VERSION}}) + '\n').encode())
        hello = response(reader, 1)
        if type(hello.get('api_version')) is not int or hello['api_version'] != API_VERSION:
            raise ValueError('camera API version differs from reviewed source; adaptation required')
        reader.timeout()
        sock.sendall(b'{"jsonrpc":"2.0","id":2,"method":"media.frame"}\n')
        header = response(reader, 2)
        validate_header(header, budget)
        data = exact(reader, header['bytes'])
    received_ns = time.monotonic_ns()
    timing = {'request_monotonic_ns': requested_ns, 'received_monotonic_ns': received_ns,
              'received_unix_us': time.time_ns() // 1000}
    return header, data, hello, timing


def png_preview(header, data, edge=320):
    """Small upright BT.601 preview. Full unmodified UYVY is preserved separately."""
    w, h, rotation = header['width'], header['height'], header['rotate']
    uw, uh = (h, w) if rotation in (90, 270) else (w, h)
    scale = min(1.0, edge / max(uw, uh))
    ow, oh = max(1, int(uw * scale)), max(1, int(uh * scale))
    rows = bytearray()
    for y in range(oh):
        rows.append(0)  # PNG filter: none
        uy = y * uh // oh
        for x in range(ow):
            ux = x * uw // ow
            sx, sy = ((ux, uy) if rotation == 0 else
                      (uy, h - 1 - ux) if rotation == 90 else
                      (w - 1 - ux, h - 1 - uy) if rotation == 180 else
                      (w - 1 - uy, ux))
            off = (sy * w + (sx // 2) * 2) * 2
            u, v, c = data[off] - 128, data[off + 2] - 128, max(0, data[off + 1 + 2 * (sx % 2)] - 16)
            for value in (298*c + 409*v, 298*c - 100*u - 208*v, 298*c + 516*u):
                rows.append(max(0, min(255, (value + 128) >> 8)))

    def chunk(kind, payload):
        body = kind + payload
        return struct.pack('>I', len(payload)) + body + struct.pack('>I', zlib.crc32(body) & 0xffffffff)

    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>2I5B', ow, oh, 8, 2, 0, 0, 0))
            + chunk(b'IDAT', zlib.compress(rows, 3)) + chunk(b'IEND', b''))


def collect(count=30, max_mib=128, reader=None):
    if type(count) is not int or not 1 <= count <= 120:
        raise ValueError('frame count must be 1..120')
    if type(max_mib) is not int or not 16 <= max_mib <= 256:
        raise ValueError('raw image budget must be 16..256 MiB')
    report = {'code': 'CAPTURE_PARTIAL', 'requested_frames': count, 'frames': [], 'api_version': API_VERSION,
              'task_result': 'NOT_TESTED', 'motion_permitted': False, 'perception_verified': False,
              'timestamp_kind': 'software appsink callback wall clock in reviewed source; NOT sensor exposure',
              'end_to_end_visual_latency_ms': None, 'source_authenticity': 'NOT_VERIFIED',
              'max_raw_mib': max_mib, 'stop_reason': None}
    frames, budget = [], max_mib * 1024 * 1024
    started = time.monotonic()
    identity, geometry = None, None
    try:
        for i in range(count):
            remaining_s = 5 - (time.monotonic() - started)
            if remaining_s <= 0:
                report['stop_reason'] = 'five second acquisition window exhausted'
                break
            header, data, hello, timing = (fetch(budget, timeout=min(3.0, remaining_s))
                                            if reader is None else reader(budget))
            validate_header(header, budget)
            if len(data) != header['bytes']:
                raise ValueError('camera payload size mismatch')
            shape = tuple(header[k] for k in ('width', 'height', 'format', 'rotate'))
            if geometry is not None and (geometry != shape or identity != hello):
                raise ValueError('camera service or geometry changed during capture')
            if frames and header['captured_at_unix_us'] <= report['frames'][-1]['header']['captured_at_unix_us']:
                raise ValueError('camera software timestamps are repeated or reversed')
            geometry, identity = shape, hello
            budget -= len(data)
            frames.append(data)
            report['frames'].append({'file': f'frames/{i:04d}.uyvy', 'header': header, **timing})
        if len(frames) == count:
            report['code'] = 'FRAME_CAPTURE_ONLY'
            report['stop_reason'] = 'requested frame count captured'
    except (OSError, ValueError, EOFError) as error:
        report['stop_reason'] = str(error)
    report['acquisition_s'] = time.monotonic() - started
    report['service'] = identity
    report['frames_received'] = len(frames)
    report['raw_bytes'] = sum(map(len, frames))
    if len(frames) > 1:
        stamps = [f['header']['captured_at_unix_us'] for f in report['frames']]
        report['software_timestamp_fps'] = 1e6 * (len(stamps) - 1) / (stamps[-1] - stamps[0])
    else:
        report['software_timestamp_fps'] = None
    # Hashing and preview generation happen after the measured acquisition window.
    for row, pixels in zip(report['frames'], frames):
        row['sha256'] = hashlib.sha256(pixels).hexdigest()
    return report, frames


def write_archive(stream, report, frames):
    with tarfile.open(fileobj=stream, mode='w|') as archive:
        def add(name, data):
            info = tarfile.TarInfo(name)
            info.size, info.mode = len(data), 0o600
            archive.addfile(info, io.BytesIO(data))
        add('report.json', (json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2) + '\n').encode())
        for row, pixels in zip(report['frames'], frames):
            add(row['file'], pixels)
        if frames:
            add('preview.png', png_preview(report['frames'][0]['header'], frames[0]))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Camera-only buffered collection; no motion commands.')
    parser.add_argument('--count', type=int, default=30)
    parser.add_argument('--max-mib', type=int, default=128)
    args = parser.parse_args(argv)
    report, frames = collect(args.count, args.max_mib)
    write_archive(sys.stdout.buffer, report, frames)
    return 0 if report['code'] == 'FRAME_CAPTURE_ONLY' else 1


if __name__ == '__main__':
    sys.exit(main())
