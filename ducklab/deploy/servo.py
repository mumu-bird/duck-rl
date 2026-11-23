"""Command conditioning for the official position actuators.

The official model ships ``<position kp=0.55 kv=0 forcerange=0.96>`` actuators
and the official simulator writes ``ctrl = DEFAULT_POSE + action`` straight into
them (game.js controlStep). The official policies stand and walk for minutes on
that stack, which is the proof that this is the dynamics they were trained
against — so this module deliberately does NOT model a motor: it reproduces the
two pieces of command conditioning that are part of the deployed contract and
are missing from the raw web loop:

* the 3–6 physics step command delay (fitted_control.DELAY_STEPS['official']);
* the firmware target slew (max_velocity rad/s, fitted_control SlewServoActuator).

Everything else (kp/kv/force range, joint damping/frictionloss/armature) stays
exactly as the official XML declares it.
"""

from __future__ import annotations

import numpy as np

from . import contract as C


class CommandConditioner:
    """Delay + slew for absolute joint targets (radians)."""

    def __init__(self, n, delay_steps=None, seed=None):
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.delay_steps = C.DELAY_STEPS_OFFICIAL if delay_steps is None else delay_steps
        self._delay = self._sample_delay()
        self._buffer = None
        self._smooth = None

    def _sample_delay(self):
        lo, hi = self.delay_steps
        return int(self.rng.integers(lo, hi + 1))

    def reset(self, q0):
        q0 = np.asarray(q0, dtype=np.float64).reshape(self.n)
        self._delay = self._sample_delay()
        self._buffer = np.repeat(q0[None, :], self._delay + 1, axis=0)
        self._smooth = q0.copy()

    def step(self, targets, dt):
        """One physics step: returns the conditioned target for ``data.ctrl``."""
        delayed = self._buffer[0]
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = targets
        max_step = C.MOTOR["max_velocity"] * dt
        self._smooth = self._smooth + np.clip(delayed - self._smooth, -max_step, max_step)
        return self._smooth
