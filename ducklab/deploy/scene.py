"""Build the deployable training model.

Base model: the OFFICIAL simulator robot
(``official-space/app/public/robot/mjlab/robot_allcollisions.xml`` from the
handoff package), NOT the ``identified`` candidate.

Why: the six official policies (which run on the real robot) stand and walk for
minutes on the official model but fall within ~2 s on the identified model in
every actuator configuration tried (fitted servo, kp=50 native, official
kp=0.55 regime, gyro-sensor obs). The handoff itself flags the identified XML
as an unverified candidate; the official model is the one the policies were
trained against, so it is the correct physics base for training.

This module keeps every official number (joint axes, ranges, actuator gains
kp=0.55/kv=0/forcerange=0.96, joint damping/frictionloss/armature, sensors,
timestep) and adds only what the tasks need:

* the ground, lights and the tossed/placed object as a free body;
* the mouth hinge + two beak pads. The real robot's jaw is bus servo 34 and
  sits outside the policy; the official model carries the head as a rigid
  assembly (``jaw_soft``), so the joint is added here and driven by the
  separate beak controller, exactly as on the robot;
* a ``home`` keyframe at robotd's DEFAULT_POSITION.
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
MJLAB = HANDOFF / "official-space/app/public/robot/mjlab"
ROBOT_XML = MJLAB / "robot_allcollisions.xml"

# The official XML declares meshdir="assets"; the meshes live in meshes/.
MESH_DIR = (MJLAB / "meshes").resolve()

# Beak geometry (metres, jaw_soft frame). The official head assembly ends at
# jaw_soft; the bill pokes forward from it. Hinge 12 mm behind the pads,
# 39.375 deg travel drops the lower pad ~18 mm (matching the real servo's
# measured 448 counts).
PAD_RADIUS = 0.006
UPPER_PAD_POS = (0.030, 0.0, 0.008)
LOWER_PAD_POS = (0.030, 0.0, -0.008)
MOUTH_HINGE = (0.012, 0.0, 0.0)


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


def build_model_xml(object_kind="ball", with_mouth=True):
    xml = ROBOT_XML.read_text()

    # absolute mesh directory + the documented 5 ms control timestep
    # (constants.js: TIMESTEP=0.005, DECIMATION=4 -> 50 Hz)
    xml = re.sub(
        r'<compiler[^>]*/>',
        f'<compiler angle="radian" meshdir="{MESH_DIR}" autolimits="true" '
        f'inertiafromgeom="false"/>',
        xml,
        count=1,
    )
    if '<option' in xml:
        xml = re.sub(
            r'<option[^>]*/>',
            f'<option timestep="{C.PHYSICS_DT}" integrator="implicitfast"/>',
            xml,
            count=1,
        )
    else:
        xml = xml.replace(
            '<mujoco',
            f'<mujoco><option timestep="{C.PHYSICS_DT}" integrator="implicitfast"/>',
            1,
        )

    if with_mouth:
        # mouth hinge inside a child of jaw_soft (the official head assembly),
        # plus the two beak pads. Contypes isolate the pads from the ground so a
        # head-down poke can never produce a degenerate contact frame.
        upper = (
            f'<geom name="beak_pad_upper" type="sphere" size="{PAD_RADIUS}" '
            f'pos="{UPPER_PAD_POS[0]} {UPPER_PAD_POS[1]} {UPPER_PAD_POS[2]}" '
            f'friction="1.2 0.005 0.0001" condim="3" contype="2" conaffinity="1" '
            f'rgba="0.9 0.6 0.1 1"/>'
        )
        mouth_body = (
            '<body name="mouth" pos="0 0 0">\n'
            f'  <joint name="mouth" type="hinge" pos="{MOUTH_HINGE[0]} {MOUTH_HINGE[1]} {MOUTH_HINGE[2]}" '
            f'axis="0 1 0" range="0 {C.MOUTH_OPEN:.6f}" limited="true" '
            f'armature="0.0002" damping="0.002" frictionloss="0.008"/>\n'
            f'  <geom name="beak_pad_lower" type="sphere" size="{PAD_RADIUS}" '
            f'pos="{LOWER_PAD_POS[0]} {LOWER_PAD_POS[1]} {LOWER_PAD_POS[2]}" '
            f'friction="1.2 0.005 0.0001" condim="3" contype="2" conaffinity="1" '
            f'rgba="0.9 0.6 0.1 1"/>\n'
            '  <inertial mass="0.005" pos="0.02 0 0" diaginertia="1e-5 1e-5 1e-5"/>\n'
            '</body>\n'
        )
        # insert the upper pad and the mouth child at the end of jaw_soft,
        # anchored on the official mouth_tip site (the last element there).
        anchor = re.search(r'<site group="3" name="mouth_tip"[^>]*/>', xml)
        assert anchor, "mouth_tip site not found"
        xml = xml[:anchor.end()] + "\n    " + upper + "\n    " + mouth_body + xml[anchor.end():]

        # a motor for the mouth (policy never drives it; the beak controller does)
        xml = xml.replace(
            "</actuator>",
            '<position name="m_mouth" joint="mouth" kp="2.0" kv="0.2" '
            'forcerange="-0.5 0.5" ctrllimited="false"/>\n</actuator>',
        )

    # scene furniture + the object
    home = " ".join(f"{v:.9f}" for v in C.DEFAULT_POSITION[:C.MOUTH_INDEX])
    world = (
        '\n<geom name="ground" type="plane" size="2 2 0.1" friction="1 0.005 0.0001" condim="3"/>\n'
        '<light name="key_light" pos="0 -1 2" directional="true" diffuse="0.9 0.9 0.9"/>\n'
        '<light name="fill_light" pos="1 1 2" directional="true" diffuse="0.4 0.4 0.4"/>\n'
        '<camera name="overview" pos="0.9 -1.1 0.65" xyaxes="0.72 0.69 0  -0.32 0.33 0.89"/>\n'
        + _object_xml(object_kind)
    )
    xml = xml.replace("</worldbody>", world + "</worldbody>")
    keyframe = (
        "\n<keyframe>\n"
        f'  <key name="home" qpos="0 0 {C.BASE_HOME_HEIGHT} 1 0 0 0 {home}" '
        f'ctrl="{" ".join("0" for _ in range(15 if with_mouth else 14))}"/>\n'
        "</keyframe>\n"
    )
    xml = xml.replace("</mujoco>", keyframe + "</mujoco>")
    return xml


def build(object_kind="ball", with_mouth=True):
    """Compile the deployable model; returns (model, data, layout)."""
    xml = build_model_xml(object_kind=object_kind, with_mouth=with_mouth)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    policy_joints = [n for n in C.JOINT_NAMES if n != "mouth"]
    layout = dict(
        policy_act=np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
             for n in policy_joints]
        ),
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
        home_height=float(model.body_pos[mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "trunk_base")][2]),
        head=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "jaw_soft"),
        gyro=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, "angular-velocity"),
        mouth_tip_site=mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mouth_tip"),
        qpos_joints=np.array([model.jnt_qposadr[model.joint(n).id] for n in policy_joints]),
        dof_joints=np.array([model.jnt_dofadr[model.joint(n).id] for n in policy_joints]),
    )
    return model, data, layout
