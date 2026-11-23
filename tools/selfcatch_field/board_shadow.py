"""Portable replay of the trained actor. Only input is the embedded test corpus.

This standalone program reads the SoC description, not live sensors. It has no
network, serial, GPIO, robot SDK, subprocess or file-writing interface.
"""

import base64
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time
import zlib

PAYLOAD_B85 = b""
PAYLOAD_SHA256 = ""
TOL = 2e-5


def policy(phase, obs):
    if len(obs) != 20 or not all(math.isfinite(x) for x in obs):
        raise ValueError('invalid observation')
    for layer, n_in, n_out in zip(phase, (20, 32), (32, 6)):
        weights, biases = layer
        if len(weights) != n_out or len(biases) != n_out or any(len(w) != n_in for w in weights):
            raise ValueError('unexpected actor shape')
        obs = [math.tanh(b + sum(w*v for w, v in zip(row, obs))) for row, b in zip(weights, biases)]
    return obs


def rk3566():
    for path in ('/proc/device-tree/compatible', '/sys/firmware/devicetree/base/compatible'):
        try:
            compatibles = Path(path).read_bytes().split(b'\0')
        except OSError:
            continue
        return b'rockchip,rk3566' in compatibles
    return False


def verify(corpus, on_board):
    phases, cases = corpus['phases'], corpus['cases']
    expected = [(seed, phase) for seed in range(990000, 990100) for phase in (0, 1)]
    if len(phases) != 2 or any(len(phase) != 2 for phase in phases) or [tuple(c[:2]) for c in cases] != expected:
        raise ValueError('unexpected phases or test corpus')
    times, error = [], 0.0
    for seed, phase, obs, reference in cases:
        if len(reference) != 6 or not all(math.isfinite(x) for x in reference):
            raise ValueError('invalid reference')
        start = time.monotonic()
        output = policy(phases[phase], obs)
        times.append((time.monotonic() - start)*1000)
        if not all(math.isfinite(x) for x in output):
            raise ValueError('non-finite action')
        error = max(error, max(abs(x-y) for x, y in zip(output, reference)))
    good = error <= TOL
    return dict(code=('BOARD_POLICY_PASS' if on_board else 'HOST_PARITY_ONLY') if good else 'PARITY_FAIL',
                rk3566=on_board, architecture=platform.machine(), input='fixed MuJoCo observations',
                checkpoint_sha256=corpus['source_sha256'], inputs=len(cases), scalars=6*len(cases),
                max_abs_error=error, tolerance=TOL, forward_p95_ms=sorted(times)[math.ceil(.95*len(times))-1],
                motion_permitted=False, task_result='NOT_TESTED')


def main():
    raw = zlib.decompress(base64.b85decode(PAYLOAD_B85))
    if hashlib.sha256(raw).hexdigest() != PAYLOAD_SHA256:
        raise ValueError('corpus checksum mismatch')
    result = verify(json.loads(raw), rk3566())
    print('WONDERDUCK_SHADOW ' + json.dumps(result, allow_nan=False))
    return {'BOARD_POLICY_PASS': 0, 'HOST_PARITY_ONLY': 3, 'PARITY_FAIL': 1}[result['code']]


if __name__ == '__main__':
    sys.exit(main())
