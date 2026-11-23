"""The on-robot policy contract, transcribed from the official robotd reference.

Sources (read-only snapshots in vendor/official-handoff-20261113):
  reference/robotd/duck-control/src/obs.rs    61-D observation layout
  reference/robotd/duck-control/src/model.rs joint order, home pose, mouth, limits
  reference/robotd/robotd/src/control.rs      tick order and target construction
  reference/snapshots/robotd.toml             installed tuning on the robot
  resources/robot/identified/mujoco/actuators/identified.json  fitted motor model

Nothing here may be "improved": a wrong offset does not fail loudly, it produces a
plausible-looking robot that falls over.
"""

from __future__ import annotations

import math

import numpy as np

# ---------------------------------------------------------------- joints

NUM_JOINTS = 15
MOUTH_INDEX = 9  # the mouth is absent from every policy on both sides
JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll", "mouth",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]
JOINT_IDS = [20, 21, 22, 23, 24, 30, 31, 32, 33, 34, 10, 11, 12, 13, 14]
POLICY_JOINTS = [i for i in range(NUM_JOINTS) if i != MOUTH_INDEX]

# Home pose (radians), JOINT_NAMES order. robotd model.rs DEFAULT_POSITION.
DEFAULT_POSITION = np.array([
    0.0, -0.0873, -0.4579, -0.0049, 0.4530,
    0.3491, 0.3491, 0.0, 0.0, 0.0,
    0.0, 0.0873, 0.4579, 0.0049, -0.4530,
])

# Mechanical travel, radians (robotd model.rs JOINT_LIMITS). Used for safety
# clipping; the MuJoCo model carries its own (near-identical) jnt ranges.
JOINT_LIMITS = np.array([
    (-0.4538, 0.5236), (-0.6981, 0.4538), (-1.9199, 0.7505), (-1.2217, 1.2915),
    (-1.4486, 1.2217), (-2.2689, 0.9599), (-1.5708, 1.5708), (-2.0944, 2.0944),
    (-0.6109, 0.6109), (0.0, 0.6869),  # mouth: 0 closed .. 39.375 deg open
    (-0.4538, 0.5236), (-0.4538, 0.6981), (-1.9199, 0.7505), (-1.2915, 1.2217),
    (-1.4486, 1.2217),
])

# ---------------------------------------------------------------- mouth (bus servo 34)

MOUTH_CLOSED = 0.0
MOUTH_OPEN = math.radians(39.375)
COUNTS_PER_REV = 4096.0
HX_JOINT_DIRECTIONS = [-1] * NUM_JOINTS  # persistent encoder sign, JOINT_NAMES order
MOUTH_ENCODER_CLOSED = 2048
MOUTH_ENCODER_OPEN = 1600  # 448 counts == 39.375 deg, opening lowers the count

# Physical home encoder targets (raw counts), for on-robot verification only.
PHYSICAL_HOME_ENCODER_TARGETS = np.array([
    2055, 2096, 2300, 2048, 1796, 1780, 1820, 2048, 2048,
    2048, 2041, 2000, 1796, 2048, 2300,
])

# ---------------------------------------------------------------- policy contract

OBS_LEN = 61
ACTION_LEN = 14
COMMAND_LEN = 13
CONTROL_HZ = 50
PHYSICS_HZ = 200
PHYSICS_DT = 1.0 / PHYSICS_HZ
SUBSTEPS = PHYSICS_HZ // CONTROL_HZ  # 4

ACTION_SCALE = 1.0            # robotd.toml: action_scale = 1.0
STANDING_ACTION_SCALE = 1.0
GROUND_PICK_ACTION_SCALE = 1.0
HEAD_LOWPASS = 1.0            # 1.0 == no filtering, matching the installed config
LEGS_LOWPASS = 1.0
VOLTAGE_ADAPT = False

# Servo firmware (reference/training fitted_control.py SERVO_FIRMWARE).
KP_FW = 15.0
KI_FW = 0.0
KD_FW = 0.0
DEADBAND_COUNTS = 2
DEADBAND_RAD = math.tau * DEADBAND_COUNTS / COUNTS_PER_REV
VIN = 7.7
MAX_PWM = 1.0
V_MAX = VIN * MAX_PWM

# Fitted motor parameters (identified.json, sha256 dc24ebc6..., model m6 / servo_p).
MOTOR = dict(
    kt=0.37865889463616276,
    R=2.693919963478532,
    armature=0.0006311692946518165,
    error_gain_ratio=0.21832355753825228,
    max_velocity=100.0,          # rad/s internal target slew
    firmware_d_gain=8.020455542842399e-07,
    q_offset=-0.058722808482383586,
    command_delay=0.003034389286993531,   # s, fitted internal command delay
    friction_base=0.011422072103829199,
    friction_stribeck=2.1093956920384148e-07,
    load_friction_motor=0.25382862086655317,
    load_friction_external=2.5051782684146206e-06,
    load_friction_motor_stribeck=0.0012258838189231087,
    load_friction_external_stribeck=0.2504883219068844,
    load_friction_motor_quad=0.006000771033187639,
    load_friction_external_quad=0.0099950784209090788,
    dtheta_stribeck=0.9802298940666753,
    alpha=0.6551807241305062,
    friction_viscous=0.0017253682947783327,
)

# Training command delay (fitted_control.DELAY_STEPS['official']): the actuator sees
# the target 3..6 physics steps (15..30 ms) late. Sampled per episode.
DELAY_STEPS_OFFICIAL = (3, 6)

# Command magnitude limits from the robot launch line
# (--max-linear 0.3 --max-angular 1).
MAX_LINEAR = 0.3     # m/s
MAX_ANGULAR = 1.0    # rad/s

BASE_HOME_HEIGHT = 0.111848115018488  # scene.xml keyframe 'home' root height


def build_observation(gyro, gravity, q_policy, dq_policy, last_action, command):
    """Assemble the 61-D observation exactly as obs.rs does.

    gyro, gravity: (3,) trunk frame. q/dq: 14 policy joints (mouth excluded),
    radians / rad/s. last_action: previous 14-D action. command: 13-D block.
    """
    gyro = np.asarray(gyro, dtype=np.float64).reshape(3)
    gravity = np.asarray(gravity, dtype=np.float64).reshape(3)
    q = np.asarray(q_policy, dtype=np.float64).reshape(ACTION_LEN)
    dq = np.asarray(dq_policy, dtype=np.float64).reshape(ACTION_LEN)
    last = np.asarray(last_action, dtype=np.float64).reshape(ACTION_LEN)
    cmd = np.asarray(command, dtype=np.float64).reshape(COMMAND_LEN)
    obs = np.concatenate([
        gyro,                                             # 0..3
        gravity,                                          # 3..6
        q - DEFAULT_POSITION[POLICY_JOINTS],              # 6..20
        dq,                                               # 20..34
        last,                                             # 34..48
        cmd,                                              # 48..61
    ])
    assert obs.shape == (OBS_LEN,)
    return obs


def build_command(twist, head, body=(0.0, 0.0, 0.0)):
    """13-D command block: twist(3), head targets(4), body x/y always zero,
    then body z, roll, pitch, and yaw (zero for the standard policies)."""
    twist = np.asarray(twist, dtype=np.float64).reshape(3)
    head = np.asarray(head, dtype=np.float64).reshape(4)
    body = np.asarray(body, dtype=np.float64).reshape(3)
    return np.concatenate([twist, head, [0.0, 0.0], body, [0.0]])


def scatter_action(action):
    """Scatter 14 policy actions over 15 joints, skipping the mouth."""
    action = np.asarray(action, dtype=np.float64).reshape(ACTION_LEN)
    full = np.zeros(NUM_JOINTS)
    full[POLICY_JOINTS] = action
    return full  # mouth stays 0.0 — the beak is never policy-driven


def targets_from_action(action):
    """control.rs: targets = home pose + action_scale * action (mouth untouched)."""
    return DEFAULT_POSITION + ACTION_SCALE * scatter_action(action)


def clip_targets(targets):
    """Safety clip to mechanical travel (mirrors robotd's Safety layer)."""
    lo = JOINT_LIMITS[:, 0]
    hi = JOINT_LIMITS[:, 1]
    return np.clip(targets, lo, hi)


def mouth_angle_to_encoder(angle):
    """Positive opening lowers the encoder count (HX direction is negative)."""
    counts = MOUTH_ENCODER_CLOSED - angle / (math.tau / COUNTS_PER_REV)
    return int(round(np.clip(counts, MOUTH_ENCODER_OPEN, MOUTH_ENCODER_CLOSED)))


def encoder_to_mouth_angle(counts):
    return (MOUTH_ENCODER_CLOSED - counts) * (math.tau / COUNTS_PER_REV)


def joint_angle_to_encoder(index, angle):
    """Raw HX count for a policy joint at ``angle`` (radians), for on-robot checks."""
    counts = PHYSICAL_HOME_ENCODER_TARGETS[index] + HX_JOINT_DIRECTIONS[index] * (
        angle - DEFAULT_POSITION[index]
    ) / (math.tau / COUNTS_PER_REV)
    return int(round(np.clip(counts, 0, 4095)))
