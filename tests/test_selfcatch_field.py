"""Small offline regressions for deterministic evaluation and safe preflight."""

import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch

from tools.selfcatch_field import accept, board_shadow, finetune_intent, readonly_preflight
from tools.train_self_catch import Policy


class DeterministicEvaluationTests(unittest.TestCase):
    def test_mean_actions_are_deterministic_and_exploration_is_seeded(self):
        finetune_intent.local_policy = Policy().eval()

        class FakeEnv:
            def reset(self, seed):
                self.steps = 0
                return np.full(20, seed / 10_000_000, dtype=np.float32)

            def step(self, action):
                self.steps += 1
                return np.zeros(20, dtype=np.float32), 0.0, self.steps == 2, {
                    'success': False, 'released': True, 'recaught': False, 'ground': True,
                    'held': False, 'armed': True, 'upright': True,
                }

        finetune_intent.env = FakeEnv()
        state = {key: value.detach().cpu().numpy() for key, value in finetune_intent.local_policy.state_dict().items()}
        first = finetune_intent.collect((state, [123], False))[0]
        second = finetune_intent.collect((state, [123], False))[0]
        self.assertTrue(np.array_equal(first[1], second[1]))
        for phase in (0, 1):
            observation = torch.from_numpy(first[0][phase])
            with torch.inference_mode():
                expected = finetune_intent.local_policy.dist(observation, phase).loc.numpy()
            np.testing.assert_array_equal(first[1][phase], expected)

        stochastic_a = finetune_intent.collect((state, [123], True))[0]
        stochastic_b = finetune_intent.collect((state, [123], True))[0]
        self.assertTrue(np.array_equal(stochastic_a[1], stochastic_b[1]))
        self.assertFalse(np.array_equal(first[1], stochastic_a[1]))

    def test_board_scalar_forward_matches_torch_actor(self):
        actor = Policy().eval()
        for phase in (0, 1):
            net = actor.actors[phase]
            weights = [[net[n].weight.detach().tolist(), net[n].bias.detach().tolist()] for n in (0, 2)]
            with torch.no_grad():
                net[-1].bias.fill_(1.3)
            weights[1][1] = net[-1].bias.detach().tolist()
            for obs in np.random.default_rng(41 + phase).normal(size=(32, 20)).astype(np.float32):
                with torch.inference_mode():
                    expected = net(torch.from_numpy(obs)).tanh().numpy()
                actual = board_shadow.policy(weights, obs.tolist())
                np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


class ReadOnlyPreflightTests(unittest.TestCase):
    def setUp(self):
        route = patch.object(readonly_preflight, 'direct_route', return_value=True)
        self.route = route.start()
        self.addCleanup(route.stop)
        self.info = {
            'joints': [{'id': idx, 'name': readonly_preflight.JOINTS[idx]}
                       for idx in readonly_preflight.IDS],
            'studio': True, 'demo': False, 'hz': 50, 'token': 'must-not-be-printed',
        }
        self.state = {
            'online': True, 'age_ms': 20, 't_ns': 1_000_000,
            'servos': [
                {'id': servo_id, 'raw': 2000, 'valid': True, 'online': True, 'age_ms': 30,
                 'observation_age_ms': 35, 'packet_status': 0, 'register_status': 0,
                 'diagnostics_available': True, 'encoder_reconstructed': False, 'faults': [],
                 'status': 0, 'torque': 1, 't_ns': 1_000_000 + i, 'error': None}
                for i, servo_id in enumerate(readonly_preflight.IDS)
            ],
            'imu': {'valid': True, 'age_ms': 20, 'quaternion_wxyz': [1.0, 0.0, 0.0, 0.0],
                    't_ns': 1_000_000},
            'robot': {'health': {'battery': {'volts': 7.4}},
                      'low_voltage': {'settings': {'threshold_v': 7.0}, 'triggered': False}},
        }

    def test_accepts_only_fresh_complete_telemetry(self):
        self.assertEqual([], readonly_preflight.inspect(self.info, self.state)['failures'])

    def test_rejects_stale_servo_imu_and_low_battery(self):
        self.state['servos'][9]['valid'] = False
        self.state['servos'][9]['age_ms'] = 3000
        self.state['imu']['valid'] = False
        self.state['robot']['health']['battery']['volts'] = 6.8
        result = readonly_preflight.inspect(self.info, self.state)
        self.assertGreaterEqual(len(result['failures']), 3)

    def test_run_requests_only_the_allowlisted_get_endpoints(self):
        paths, sample = [], 0

        def fetch(path):
            nonlocal sample
            paths.append(path)
            if path == '/api/info':
                return self.info
            sample += 1
            value = copy.deepcopy(self.state)
            value['t_ns'] += sample * 1_000_000_000
            value['imu']['t_ns'] += sample * 1_000_000_000
            for row in value['servos']:
                row['t_ns'] += sample * 1_000_000_000
            return value

        result = readonly_preflight.run(fetch, sleep=lambda _: None)
        self.assertEqual(result['code'], 'PASS_READONLY')
        self.assertEqual(result['basic_code'], 'BASIC_READONLY_OK')
        self.assertEqual(paths, ['/api/info'] + ['/api/state'] * 3)
        self.assertNotIn('must-not-be-printed', json.dumps(result))

    def test_foreign_route_stops_before_any_device_request(self):
        fetch = Mock(side_effect=AssertionError('must not query a non-hotspot route'))
        self.route.return_value = False
        result = readonly_preflight.run(fetch, sleep=lambda _: None)
        self.assertEqual(result['code'], 'NO_LINK')
        self.assertEqual(result['basic_code'], 'NOT_RUN')
        fetch.assert_not_called()

    def test_route_requires_local_hotspot_subnet_without_gateway(self):
        direct = [{'dst': '192.168.149.1', 'dev': 'wlan0', 'prefsrc': '192.168.149.20', 'flags': []}]
        self.assertTrue(readonly_preflight.hotspot_route(direct))
        office_route = [{'dst': '192.168.149.1', 'gateway': '10.11.104.1', 'dev': 'wlo1',
                         'prefsrc': '10.11.107.79', 'flags': []}]
        self.assertFalse(readonly_preflight.hotspot_route(office_route))
        self.assertFalse(readonly_preflight.hotspot_route([]))

    def test_basic_check_can_pass_when_public_endpoint_omits_servo_diagnostics(self):
        sample = copy.deepcopy(self.state)
        for row in sample['servos']:
            row.pop('diagnostics_available')
            row.pop('encoder_reconstructed')

        def fetch(path):
            if path == '/api/info':
                return self.info
            fetch.index += 1
            value = copy.deepcopy(sample)
            value['t_ns'] += fetch.index * 1_000_000_000
            value['imu']['t_ns'] += fetch.index * 1_000_000_000
            for row in value['servos']:
                row['t_ns'] += fetch.index * 1_000_000_000
            return value

        fetch.index = 0
        result = readonly_preflight.run(fetch, sleep=lambda _: None)
        self.assertEqual(result['basic_code'], 'BASIC_READONLY_OK')
        self.assertEqual(result['code'], 'CHECK')
        self.assertEqual(result['samples'][-1]['measured_servos'], 0)

    def test_stale_or_reconstructed_feedback_never_qualifies(self):
        def fetch(path):
            return self.info if path == '/api/info' else self.state

        result = readonly_preflight.run(fetch, sleep=lambda _: None)
        self.assertEqual(result['code'], 'CHECK')
        self.assertEqual(result['basic_code'], 'CHECK')
        self.assertTrue(any('时间戳' in s for s in result['reasons']))
        self.state['servos'][9]['encoder_reconstructed'] = True
        reading = readonly_preflight.inspect(self.info, self.state)
        self.assertIsNone(reading['head']['mouth']['measured_raw'])
        self.assertEqual(reading['measured_servos'], 14)

    def test_demo_stops_at_info_and_unreachable_stays_no_link(self):
        fake = Mock(return_value={**self.info, 'demo': True})
        self.assertEqual(readonly_preflight.run(fake)['code'], 'CHECK')
        fake.assert_called_once_with('/api/info')
        self.assertEqual(readonly_preflight.run(Mock(side_effect=TimeoutError('timeout')))['code'], 'NO_LINK')

    def test_get_json_uses_only_allowlisted_get_path(self):
        response = Mock(status=200)
        response.read.return_value = json.dumps({'token': 'secret'}).encode()
        connection = Mock()
        connection.getresponse.return_value = response
        with patch.object(readonly_preflight.http.client, 'HTTPConnection', return_value=connection):
            self.assertEqual(readonly_preflight.get_json('/api/info'), {'token': 'secret'})
        connection.request.assert_called_once()
        self.assertEqual(connection.request.call_args.args[:2], ('GET', '/api/info'))
        headers = connection.request.call_args.kwargs['headers']
        self.assertNotIn('X-Console-Token', headers)
        with self.assertRaises(ValueError):
            readonly_preflight.get_json('/api/command')


class FieldCommandTests(unittest.TestCase):
    def test_default_never_prompts_or_uses_ssh_and_writes_a_fresh_report(self):
        with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()), \
                patch('builtins.input', side_effect=AssertionError('no prompt')), \
                patch.object(accept, 'replay', side_effect=AssertionError('no ssh')), \
                patch.object(accept.telem, 'run') as run:
            for basic, code, exit_code in [('BASIC_READONLY_OK', 'PASS_READONLY', 0),
                                           ('BASIC_READONLY_OK', 'CHECK', 1), ('NOT_RUN', 'NO_LINK', 2)]:
                run.return_value = dict(basic_code=basic, code=code, samples=[], reasons=[],
                                        identity_verified=code != 'NO_LINK')
                self.assertEqual(accept.main(['--out-dir', tmp]), exit_code)
            reports = [json.loads(p.read_text()) for p in Path(tmp).glob('field-*.json')]
        self.assertEqual(len(reports), 3)
        for report in reports:
            self.assertEqual(report['board_replay']['code'], 'NOT_RUN')
            self.assertEqual(report['task_result'], 'NOT_TESTED')
            self.assertIs(report['motion_permitted'], False)

    def test_board_phase_rechecks_route_and_accepts_only_verified_fixed_outputs(self):
        result = dict(code='BOARD_POLICY_PASS', rk3566=True, inputs=200, scalars=1200, max_abs_error=5e-8,
                      motion_permitted=False, task_result='NOT_TESTED', forward_p95_ms=.1,
                      checkpoint_sha256=accept.ACTOR_SHA)
        with patch.object(accept.telem, 'direct_route', return_value=False) as route, \
                patch.object(accept, 'program_bytes', return_value=b'fixture'), \
                patch.object(accept.subprocess, 'run') as cmd:
            self.assertEqual(accept.replay('owner')['code'], 'NOT_RUN')
            cmd.assert_not_called()
            route.return_value = True
            for override, expected in [({}, 'BOARD_POLICY_PASS'), ({'rk3566': False}, 'NOT_VERIFIED'),
                                       ({'max_abs_error': .2}, 'NOT_VERIFIED'), ({'scalars': 1199}, 'NOT_VERIFIED'),
                                       ({'code': 'HOST_PARITY_ONLY'}, 'NOT_VERIFIED')]:
                line = b'WONDERDUCK_SHADOW ' + json.dumps({**result, **override}).encode()
                cmd.return_value = subprocess.CompletedProcess([], 0, line)
                self.assertEqual(accept.replay('owner')['code'], expected)
            self.assertEqual(cmd.call_args.args[0][-3:], ['owner', readonly_preflight.HOST, 'python3 -I -B -S -'])
            self.assertEqual(cmd.call_args.kwargs['input'], b'fixture')


if __name__ == '__main__':
    unittest.main()
