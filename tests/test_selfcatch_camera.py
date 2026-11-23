"""Camera wire, frame integrity, image orientation and no-link regressions."""

import io
import json
from pathlib import Path
import socket
import struct
import subprocess
import tarfile
import tempfile
import threading
import unittest
from unittest.mock import patch
import zlib

from tools.selfcatch_field import camera_board as board, camera_probe as host


def fixture(stamp=1_000_000, rotate=0):
    # Distinct coloured pixels make channel and rotation errors observable.
    raw = bytes([100, 32, 180, 80, 190, 144, 60, 208])
    header = dict(width=2, height=2, bytes=8, format='UYVY', rotate=rotate,
                  captured_at_unix_us=stamp)
    timing = dict(request_monotonic_ns=1, received_monotonic_ns=2, received_unix_us=stamp + 1)
    hello = dict(api_version=62, revision='test-fixture', daemon_version='0.0.0')
    return header, raw, hello, timing


def archive_bytes(count=2):
    items = iter(fixture(1_000_000 + i * 33333) for i in range(count))
    report, frames = board.collect(count, 16, reader=lambda _: next(items))
    out = io.BytesIO()
    board.write_archive(out, report, frames)
    return out.getvalue(), report


def png_pixels(data):
    pos, compressed, dimensions = 8, bytearray(), None
    while pos < len(data):
        length = struct.unpack('>I', data[pos:pos + 4])[0]
        kind, body = data[pos + 4:pos + 8], data[pos + 8:pos + 8 + length]
        if kind == b'IHDR':
            dimensions = struct.unpack('>II', body[:8])
        if kind == b'IDAT':
            compressed.extend(body)
        pos += length + 12
    width, height = dimensions
    raw = zlib.decompress(compressed)
    return [[tuple(raw[y * (width*3+1) + 1 + x*3:y * (width*3+1) + 4 + x*3])
             for x in range(width)] for y in range(height)]


class CameraTests(unittest.TestCase):
    def test_real_unix_protocol_reconnects_per_frame_and_never_sends_motion(self):
        with tempfile.TemporaryDirectory() as tmp:
            address = str(Path(tmp) / 'camera.sock')
            received, errors = [], []
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
                server.bind(address)
                server.listen(2)
                server.settimeout(3)

                def serve():
                    try:
                        for i in range(2):
                            conn, _ = server.accept()
                            with conn, conn.makefile('rb') as stream:
                                hello_req = json.loads(stream.readline())
                                received.append(hello_req)
                                h, pixels, hello, _ = fixture(1_000_000 + i*33333)
                                conn.sendall((json.dumps(dict(jsonrpc='2.0', id=1, result=hello)) + '\n').encode())
                                frame_req = json.loads(stream.readline())
                                received.append(frame_req)
                                line = (json.dumps(dict(jsonrpc='2.0', id=2, result=h)) + '\n').encode()
                                conn.sendall(line + pixels[:3])
                                conn.sendall(pixels[3:])
                    except Exception as error:
                        errors.append(error)

                thread = threading.Thread(target=serve)
                thread.start()
                report, frames = board.collect(2, 16, reader=lambda budget: board.fetch(budget, address))
                thread.join(4)
                self.assertFalse(thread.is_alive())
                self.assertFalse(errors)
            self.assertEqual(report['code'], 'FRAME_CAPTURE_ONLY')
            self.assertEqual([row['method'] for row in received], ['hello', 'media.frame'] * 2)
            self.assertEqual([row['params']['api_version'] for row in received if row['method'] == 'hello'], [62, 62])
            self.assertEqual(len(frames), 2)
            self.assertEqual(report['task_result'], 'NOT_TESTED')
            self.assertFalse(report['perception_verified'])
            self.assertIsNone(report['end_to_end_visual_latency_ms'])

    def test_wrong_ids_version_errors_and_truncated_payload_fail(self):
        for message in ({'jsonrpc': '2.0', 'id': True, 'result': {}},
                        {'jsonrpc': '2.0', 'id': 9, 'result': {}},
                        {'jsonrpc': '2.0', 'id': 1, 'error': {'message': 'offline'}},
                        {'jsonrpc': '2.0', 'id': 1, 'result': []}):
            with self.subTest(message=message), self.assertRaises(ValueError):
                board.response(io.BytesIO((json.dumps(message) + '\n').encode()), 1)
        with self.assertRaises(ValueError):
            board.response(io.BytesIO(b'x' * 4097), 1)
        with self.assertRaises(EOFError):
            board.exact(io.BytesIO(b'short'), 8)

    def test_unknown_api_stops_before_requesting_any_frame(self):
        with patch.object(board.socket, 'socket') as factory, \
                patch.object(board, 'response', return_value={'api_version': 61}):
            with self.assertRaisesRegex(ValueError, 'API version'):
                board.fetch(1024)
            connection = factory.return_value.__enter__.return_value
            self.assertEqual(connection.sendall.call_count, 1)
            self.assertEqual(json.loads(connection.sendall.call_args.args[0])['method'], 'hello')

    def test_deadline_stops_without_another_blocking_read(self):
        with patch.object(board.socket, 'socket') as factory, patch.object(board.time, 'monotonic', return_value=2.0):
            sock = factory.return_value
            reader = board.DeadlineReader(sock, 1.0)
            with self.assertRaises(TimeoutError):
                reader.read(1)
            sock.recv.assert_not_called()

    def test_malformed_geometry_and_budget_are_rejected_before_pixels(self):
        good = fixture()[0]
        for change in ({'width': True}, {'width': 3}, {'height': -1}, {'rotate': 45},
                       {'bytes': 9}, {'captured_at_unix_us': 0}, {'format': 'RGB'}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                board.validate_header(dict(good, **change), 1024)
        with self.assertRaises(board.CapacityLimit):
            board.validate_header(good, 7)

    def test_repeat_frame_and_changed_geometry_cannot_complete_capture(self):
        for bad in (fixture(), fixture(2_000_000, 90)):
            items = iter([fixture(), bad])
            report, frames = board.collect(2, 16, reader=lambda _: next(items))
            self.assertEqual(report['code'], 'CAPTURE_PARTIAL')
            self.assertEqual(len(frames), 1)
            self.assertFalse(report['motion_permitted'])

    def test_connection_failure_is_a_report_not_camera_success(self):
        def fail(_):
            raise FileNotFoundError('mediad not running')
        report, frames = board.collect(2, 16, reader=fail)
        self.assertEqual(report['code'], 'CAPTURE_PARTIAL')
        self.assertEqual(frames, [])
        self.assertIn('mediad not running', report['stop_reason'])

    def test_rotated_preview_preserves_pixel_channels(self):
        h, data, _, _ = fixture()
        baseline = png_pixels(board.png_preview(h, data))
        for degrees, expected in (
                (90, [[baseline[1][0], baseline[0][0]], [baseline[1][1], baseline[0][1]]]),
                (180, [[baseline[1][1], baseline[1][0]], [baseline[0][1], baseline[0][0]]]),
                (270, [[baseline[0][1], baseline[1][1]], [baseline[0][0], baseline[1][0]]])):
            self.assertEqual(png_pixels(board.png_preview(dict(h, rotate=degrees), data)), expected)

    def test_archive_roundtrip_checks_hashes_and_rejects_extra_paths(self):
        good, report = archive_bytes()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'capture.tar'
            path.write_bytes(good)
            actual, preview = host.verify_archive(path, 16, 2)
            self.assertEqual(actual, report)
            self.assertTrue(preview.startswith(b'\x89PNG'))
            broken = bytearray(good)
            with tarfile.open(fileobj=io.BytesIO(good)) as archive:
                offset = archive.getmember('frames/0000.uyvy').offset_data
            broken[offset] ^= 1
            path.write_bytes(broken)
            with self.assertRaisesRegex(ValueError, 'checksum'):
                host.verify_archive(path, 16, 2)
            path.write_bytes(good)
            with tarfile.open(path, 'a') as archive:
                entry = tarfile.TarInfo('../escape')
                entry.size = 1
                archive.addfile(entry, io.BytesIO(b'x'))
            with self.assertRaisesRegex(ValueError, 'paths'):
                host.verify_archive(path, 16, 2)

    def test_no_link_or_changed_route_never_starts_ssh(self):
        for preflight, route in ((dict(identity_verified=False, code='NO_LINK', reasons=['not connected']), True),
                                 (dict(identity_verified=True, code='CHECK', reasons=[]), False)):
            with patch.object(host.telem, 'run', return_value=preflight), \
                    patch.object(host.telem, 'direct_route', return_value=route), \
                    patch.object(host.subprocess, 'run') as ssh, tempfile.TemporaryDirectory() as tmp:
                with self.assertRaises(RuntimeError):
                    host.capture('test_account', 2, 16, tmp)
                ssh.assert_not_called()
                self.assertEqual(list(Path(tmp).iterdir()), [])

    def test_host_capture_and_ssh_failure_keep_truthful_reports(self):
        payload, _ = archive_bytes()
        def ssh_ok(argv, **kwargs):
            self.assertIn('python3 -I -B -S - --count 2 --max-mib 16', argv)
            self.assertEqual(kwargs['input'], host.board_program())
            kwargs['stdout'].write(payload)
            return subprocess.CompletedProcess(argv, 0)
        with patch.object(host.telem, 'run', return_value=dict(identity_verified=True, code='CHECK', reasons=[])), \
                patch.object(host.telem, 'direct_route', return_value=True), tempfile.TemporaryDirectory() as tmp:
            with patch.object(host.subprocess, 'run', side_effect=ssh_ok):
                folder, report = host.capture('test_account', 2, 16, tmp)
            self.assertEqual(report['code'], 'FRAME_CAPTURE_ONLY')
            self.assertTrue((folder / 'preview.png').is_file())
            self.assertFalse(report['motion_permitted'])
            with patch.object(host.subprocess, 'run', side_effect=subprocess.TimeoutExpired('ssh', 60)):
                other, failed = host.capture('test_account', 2, 16, tmp)
            self.assertNotEqual(folder, other)
            self.assertEqual(failed['code'], 'CAPTURE_INCOMPLETE')
            self.assertTrue((other / 'report.json').is_file())


if __name__ == '__main__':
    unittest.main()
