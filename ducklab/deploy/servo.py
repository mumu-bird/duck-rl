"""Fitted behavioural servo model, ported from the official reference parameters.

The official training stack uses the BAM library (``bam.actuator`` +
``mjlab``), which is not redistributable and not vendored here. This module
re-implements the *same fitted equations* from the public parameters in
``identified.json`` (model ``m6`` / ``servo_p``):

* pure-P firmware: kp = 15, ki = kd = 0, +/-2 count deadband, no minimum drive
  (fitted_control.SERVO_FIRMWARE);
* internal target slew limited to ``max_velocity`` rad/s;
* voltage-limited DC motor: V = clip(kp * error_gain_ratio * error, +-V_max),
  I = (V - kt * w) / R, torque = kt * I, rotor inertia ``armature``;
* Stribeck + viscous + quadratic load friction written into MuJoCo's
  ``dof_frictionloss`` / ``dof_damping`` (constraint-based, as upstream does);
* the fitted internal command delay as a first-order lag of tau = command_delay;
* the training command delay of 3..6 physics steps (``DELAY_STEPS_OFFICIAL``),
  sampled once per episode.

This is a parameter-faithful port, not the bit-exact BAM implementation. The
deployability claim rests on the *contract* (obs/action/rates/gains/delay); the
actuator approximation is a documented sim-to-real risk to close on hardware.
"""

from __future__ import annotations

import numpy as np

from . import contract as C


def friction_static(params=None):
    """Constant part of the fitted friction (static + Stribeck at zero speed).

    Written once into ``dof_frictionloss`` at model setup: MuJoCo solves
    Coulomb friction as a constraint, which is stable at 200 Hz, and a constant
    value keeps the shared mjModel immutable so worker threads can share it.
    """
    p = C.MOTOR if params is None else params
    static = p["friction_base"] + p["load_friction_motor"] + p["load_friction_external"]
    stribeck_coeff = (
        p["friction_stribeck"]
        + p["load_friction_motor_stribeck"]
        + p["load_friction_external_stribeck"]
    )
    return static + stribeck_coeff


def damping_coeff(params=None):
    """Linear damping coefficient from the fitted viscous + quadratic terms."""
    p = C.MOTOR if params is None else params
    return p["friction_viscous"] + p["load_friction_motor_quad"] + p["load_friction_external_quad"]


def friction_terms(omega, params=None):
    """Split the fitted friction into MuJoCo-native terms.

    The official reference (training/.../servo_controller.py) writes the motor
    and load friction into ``dof_frictionloss`` / ``dof_damping`` rather than
    applying them as explicit forces: MuJoCo solves Coulomb friction as a
    constraint, which is stable at 200 Hz, whereas an explicit sign(w) force
    chatters and pumps energy into the joints. Static + Stribeck therefore
    become a (velocity-independent) friction loss, and the viscous + quadratic
    terms become a speed-dependent damping.
    """
    p = C.MOTOR if params is None else params
    a = np.abs(np.asarray(omega, dtype=np.float64))
    static = p["friction_base"] + p["load_friction_motor"] + p["load_friction_external"]
    stribeck_coeff = (
        p["friction_stribeck"]
        + p["load_friction_motor_stribeck"]
        + p["load_friction_external_stribeck"]
    )
    quad_coeff = p["load_friction_motor_quad"] + p["load_friction_external_quad"]
    frictionloss = static + stribeck_coeff          # Stribeck at ~0 speed
    damping = p["friction_viscous"] + quad_coeff * a
    return frictionloss, damping


class FittedServo:
    """One servo law instance shared by all driven joints (vectorised)."""

    def __init__(self, n, kp=None, delay_steps=None, seed=None):
        self.n = n
        self.kp = C.KP_FW if kp is None else float(kp)
        self.rng = np.random.default_rng(seed)
        self.delay_steps = C.DELAY_STEPS_OFFICIAL if delay_steps is None else delay_steps
        self._delay = self._sample_delay()
        self._buffer = None
        self._smooth = None
        self._voltage = None
        self._rotor_vel = None

    def _sample_delay(self):
        lo, hi = self.delay_steps
        return int(self.rng.integers(lo, hi + 1))

    def reset(self, q0):
        q0 = np.asarray(q0, dtype=np.float64).reshape(self.n)
        self._delay = self._sample_delay()
        self._buffer = np.repeat(q0[None, :], self._delay + 1, axis=0)
        self._smooth = q0.copy()          # slew-limited target
        self._voltage = np.zeros(self.n)  # first-order lagged command voltage
        self._rotor_vel = np.zeros(self.n)

    def command(self, targets, q, dq, dt):
        """Torque to write into ``data.ctrl`` for this physics step.

        targets: absolute joint targets (radians, after home+action and clipping).
        q, dq: current joint positions / velocities.
        """
        p = C.MOTOR
        # 1) training command delay: the actuator sees an old target.
        delayed = self._buffer[0]
        self._buffer[:-1] = self._buffer[1:]
        self._buffer[-1] = targets
        # 2) internal slew limit (firmware target smoothing).
        step = p["max_velocity"] * dt
        delta = np.clip(delayed - self._smooth, -step, step)
        self._smooth = self._smooth + delta
        # 3) pure-P firmware with deadband and fitted error gain.
        error = self._smooth - q
        dead = np.abs(error) < C.DEADBAND_RAD
        v_cmd = self.kp * p["error_gain_ratio"] * np.where(dead, 0.0, error)
        v_cmd = np.clip(v_cmd, -C.V_MAX, C.V_MAX)
        # 4) fitted internal command delay as a first-order lag.
        tau = max(p["command_delay"], 1e-4)
        alpha = 1.0 - np.exp(-dt / tau)
        self._voltage = self._voltage + alpha * (v_cmd - self._voltage)
        # 5) DC motor with back-EMF (direct drive: rotor velocity == joint).
        current = (self._voltage - p["kt"] * dq) / p["R"]
        motor_torque = p["kt"] * current
        self._rotor_vel = dq
        return motor_torque + self.damping_force(dq)

    def setup_friction(self, model, dof_indexes):
        """One-time: constant Coulomb friction into the (shared) model."""
        model.dof_frictionloss[dof_indexes] = friction_static()

    def damping_force(self, dq):
        """Velocity-dependent damping, applied explicitly (linear, chatter-free)."""
        return -damping_coeff() * dq
