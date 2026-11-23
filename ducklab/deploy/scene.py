"""Build the deployable training model from the official identified robot.xml.

The official simulator handoff (vendor/official-handoff-20261113) ships the
real robot's identified MuJoCo model: 14 hinge joints in wire order, 124 meshes,
200 Hz implicitfast physics, ground friction 1 0.005 0.0001. This module:

1. swaps the XML's position actuators (kp=50) for plain motors so the fitted
   servo law can write torque directly into ``data.ctrl``;
2. adds the mouth hinge + beak pads. The real robot's jaw is bus servo 34 and
   sits *outside* the policy; the official XML carries the mouth as a rigid
   body, so the joint is added here and driven by the separate beak controller;
3. appends the tossed/placed object as a free body (ball, cube or LINLEE duck
   proxy) with the scene ground;
4. installs a ``home`` keyframe at robotd's DEFAULT_POSITION.

The mouth joint, pads and object are the only additions; every official number
otherwise stays as shipped.
"""

from __future__ import annotations

import re
from pathlib import Path

import mujoco
import numpy as np

from . import contract as C

HANDOFF = (
    Path(__file__).resolve().parents[2]
    / "vendor/official-handoff-20261113/wonderduck-simulator-handoff-20261113-144034"
)
IDENTIFIED = HANDOFF / "resources/robot/identified"
ROBOT_XML = IDENTIFIED / "mujoco/robot.xml"

# Beak geometry (metres, head_roll frame). Hinge sits 18 mm behind the bill tip;
# the 39.375 deg travel drops the lower pad ~18 mm, opening a ~22 mm gap that
# captures a 24 mm ball. Spheres (not thin boxes) keep the contact frames
# well-defined: MuJoCo raises a fatal error on degenerate box-box frames, and a
# training run cannot survive that.
MOUTH_HINGE = (0.018, 0.0, 0.001)
PAD_RADIUS = 0.006
UPPER_PAD_POS = (0.050, 0.0, 0.008)
LOWER_PAD_POS = (0.050, 0.0, -0.008)


def _object_xml(kind):
    if kind == "ball":
        r, mass = 0.012, 0.004
        inertia = 0.4 * mass * r * r
        geom = f'<geom name="object_geom" type="sphere" size="{r}" mass="{mass}" friction="1 0.005 0.0001" condim="3" rgba="0.85 0.2 0.2 1"/>'
    elif kind == "cube":
        h, mass = 0.011, 0.004
        inertia = mass / 12.0 * ((2 * h) ** 2 + (2 * h) ** 2)
        geom = f'<geom name="object_geom" type="box" size="{h} {h} {h}" mass="{mass}" friction="1 0.005 0.0001" condim="3" rgba="0.2 0.4 0.85 1"/>'
    elif kind == "linlee_duck":
        # LINLEE rubber duck proxy: pinched by the bill, ~45 mm body.
        hx, hy, hz, mass = 0.024, 0.017, 0.017, 0.012
        inertia = mass / 5.0 * (hy * hy + hz * hz)
        geom = (
            f'<geom name="object_geom" type="ellipsoid" size="{hx} {hy} {hz}" '
            f'mass="{mass}" friction="1.2 0.005 0.0001" condim="3" rgba="0.95 0.75 0.1 1"/>'
        )
    else:
        raise ValueError(f"unknown object kind {kind!r}")
    return (
        '<body name="toss_object" pos="0 0 0.3">\n'
        '  <freejoint name="object_free"/>\n'
        f'  <inertial mass="{mass}" pos="0 0 0" diaginertia="{inertia} {inertia} {inertia}"/>\n'
        f'  {geom}\n'
        '</body>\n'
    )


def build_model_xml(object_kind="ball", with_mouth=True, lightweight=True):
    xml = ROBOT_XML.read_text()

    # absolute mesh directory so the composed model compiles from anywhere.
    # NOTE: discardvisual/maxhullvert were tried and rejected: this model marks
    # most collision geoms contype=0 (filtered via contact excludes), so
    # discardvisual deletes real collision geometry. Memory is instead handled
    # by sharing one mjModel across workers (see ducklab/deploy/env.py).
    meshdir = (ROBOT_XML.parent / "../meshes").resolve()
    extra = ''  # NB: discardvisual removes most collision geoms in this model
    xml = re.sub(
        r'<compiler[^>]*/>',
        f'<compiler angle="radian" meshdir="{meshdir}" autolimits="true" '
        f'inertiafromgeom="false" fusestatic="false"{extra}/>',
        xml,
        count=1,
    )

    # 1) torque motors for the 14 policy joints (+ the mouth when modelled)
    motors = []
    for name in C.JOINT_NAMES:
        if C.JOINT_NAMES.index(name) == C.MOUTH_INDEX:
            continue
        motors.append(
            f'<motor name="m_{name}" joint="{name}" gear="1" ctrllimited="false"/>'
        )
    if with_mouth:
        motors.append('<motor name="m_mouth" joint="mouth" gear="1" ctrllimited="false"/>')
    actuator_section = "<actuator>\n" + "\n".join(motors) + "\n</actuator>"
    xml, n = re.subn(r"<actuator>.*?</actuator>", lambda _m: actuator_section, xml, flags=re.S)
    assert n == 1, "actuator section not found"

    if with_mouth:
        # 2a) mouth hinge inside the (otherwise rigid) mouth body. MJCF body
        # element order is inertial -> joint -> geom, so inject after inertial.
        joint = (
            f'<joint name="mouth" type="hinge" pos="{MOUTH_HINGE[0]} {MOUTH_HINGE[1]} {MOUTH_HINGE[2]}" '
            f'axis="0 1 0" range="0 {C.MOUTH_OPEN:.6f}" limited="true" '
            f'armature="0.0002" damping="0.002" frictionloss="0.008"/>'
        )
        lower = (
            f'<geom name="beak_pad_lower" type="sphere" size="{PAD_RADIUS}" '
            f'pos="{LOWER_PAD_POS[0]} {LOWER_PAD_POS[1]} {LOWER_PAD_POS[2]}" '
            f'friction="1.2 0.005 0.0001" condim="3" contype="2" conaffinity="1" rgba="0.9 0.6 0.1 1"/>'
        )
        xml, n = re.subn(
            r'(<body name="mouth"[^>]*>\s*<inertial[^>]*/>)',
            lambda m: m.group(1) + "\n    " + joint + "\n    " + lower,
            xml,
            count=1,
        )
        assert n == 1, "mouth body not found"
        # note: the shipped mouth collision mesh already carries contype=0
        # conaffinity=0 (visual only in the official model); the two pads above
        # are what actually touches the object.
        # 2b) upper mandible pad on head_roll, inserted after its joint element
        upper = (
            f'<geom name="beak_pad_upper" type="sphere" size="{PAD_RADIUS}" '
            f'pos="{UPPER_PAD_POS[0]} {UPPER_PAD_POS[1]} {UPPER_PAD_POS[2]}" '
            f'friction="1.2 0.005 0.0001" condim="3" contype="2" conaffinity="1" rgba="0.9 0.6 0.1 1"/>'
        )
        xml, n = re.subn(
            r'(<joint name="head_roll" type="hinge"[^>]*/>)',
            lambda m: m.group(1) + "\n    " + upper,
            xml,
            count=1,
        )
        assert n == 1, "head_roll body not found"

    # 3) scene furniture + the object, appended into worldbody
    home = " ".join(f"{v:.9f}" for v in C.DEFAULT_POSITION)
    world = (
        '\n<geom name="ground" type="plane" size="2 2 0.1" friction="1 0.005 0.0001" condim="3"/>\n'
        '<light name="key_light" pos="0 -1 2" directional="true" diffuse="0.9 0.9 0.9"/>\n'
        '<light name="fill_light" pos="1 1 2" directional="true" diffuse="0.4 0.4 0.4"/>\n'
        + _object_xml(object_kind)
    )
    keyframe = (
        "\n<keyframe>\n"
        f'  <key name="home" qpos="0 0 {C.BASE_HOME_HEIGHT} 1 0 0 0 {home}" '
        f'ctrl="{" ".join("0" for _ in range(15 if with_mouth else 14))}"/>\n'
        "</keyframe>\n"
    )
    xml = xml.replace("</worldbody>", world + "</worldbody>")
    xml = xml.replace("</mujoco>", keyframe + "</mujoco>")
    return xml


def build(object_kind="ball", with_mouth=True, lightweight=True):
    """Compile the deployable model; returns (model, data, layout)."""
    xml = build_model_xml(object_kind=object_kind, with_mouth=with_mouth,
                          lightweight=lightweight)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    policy_act = np.array(
        [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"m_{n}")
         for n in C.JOINT_NAMES if n != "mouth"]
    )
    layout = dict(
        policy_act=policy_act,
        mouth_act=(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "m_mouth")
            if with_mouth else -1
        ),
        mouth_qpos=int(model.jnt_qposadr[model.joint("mouth").id]) if with_mouth else -1,
        mouth_dof=int(model.jnt_dofadr[model.joint("mouth").id]) if with_mouth else -1,
        object_qpos=int(model.jnt_qposadr[model.joint("object_free").id]),
        object_dof=int(model.jnt_dofadr[model.joint("object_free").id]),
        object_geom=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "object_geom"),
        ground_geom=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground"),
        pad_upper=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "beak_pad_upper") if with_mouth else -1,
        pad_lower=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "beak_pad_lower") if with_mouth else -1,
        trunk=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base"),
        head=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "head_roll"),
        mouth_tip=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip"),
        qpos_joints=np.array(
            [model.jnt_qposadr[model.joint(n).id] for n in C.JOINT_NAMES if n != "mouth"]
        ),
        dof_joints=np.array(
            [model.jnt_dofadr[model.joint(n).id] for n in C.JOINT_NAMES if n != "mouth"]
        ),
        nq=model.nq,
    )
    return model, data, layout
