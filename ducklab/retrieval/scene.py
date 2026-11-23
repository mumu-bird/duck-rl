"""Scenes shared by head-camera supervision and the free-standing robot referee."""

import json
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from ..assets import ROOT, node, scene_xml
from ..bench import view_options

OBJECTS = json.loads((ROOT / "configs/retrieval-objects.json").read_text())["objects"]
HEIGHT, WIDTH = 160, 224
BIN = (0.34, 0.19)
BIN_HALF = np.array([0.084, 0.087])  # Inner half-extents of the open tray.
BASE_Z, WALL_H = 0.002, 0.018
CLUTTER = ("old_ball", "old_pen")


def add_item(xml, item):
    world, asset = xml.find("worldbody"), xml.find("asset")
    family, identity, length = item["family"], item["id"], item["length"]
    body_rgba = item.get("body_rgba", ".64 .32 .10 1")
    head_rgba = item.get("head_rgba", ".90 .70 .11 1")
    bill_rgba = item.get("bill_rgba", "1 .35 .06 1")
    friction = item.get("friction", ".85 .007 .004")
    contact_friction = item.get(
        "contact_friction", ".9 .9 .005 .0005 .0005"
    )
    solref = item.get("solref")
    solimp = item.get("solimp")
    body = node(world, "body", name="parcel", pos=(-2, -2, 0.2))
    node(body, "freejoint", name="parcel_free")
    if "source" in item:
        directory = ROOT / "vendor/retrieval" / identity
        if not (directory / "half.json").is_file():
            raise FileNotFoundError("Run tools/retrieval_assets.py first")
        hx, hy, hz = json.loads((directory / "half.json").read_text())
        node(asset, "mesh", name="scan", file=directory / "object.obj")
        node(
            asset, "texture", name="scan_tex", type="2d", file=directory / "texture.png"
        )
        node(asset, "material", name="scan_mat", texture="scan_tex", specular=0.03)
        node(
            body,
            "geom",
            name="appearance",
            type="mesh",
            mesh="scan",
            material="scan_mat",
            group=2,
            contype=0,
            conaffinity=0,
            density=0,
        )
    elif "half_m" in item:
        hx, hy, hz = item["half_m"]
    else:
        hx = length / 2
        hy, hz = (0.004, 0.004) if family == "pen" else (0.31 * length, 0.34 * length)
    mass = float(item.get("mass_kg", 0.004))
    node(
        body,
        "inertial",
        pos=(0, 0, 0),
        mass=mass,
        diaginertia=(
            mass * (hy * hy + hz * hz) / 3,
            mass * (hx * hx + hz * hz) / 3,
            mass * (hx * hx + hy * hy) / 3,
        ),
    )
    if family == "pen":
        rad = max(0.003, (hy + hz) / 2)
        size = dict(type="capsule", fromto=(-hx + rad, 0, 0, hx - rad, 0, 0), size=rad)
    elif family in ("curved", "utensil", "handled") or item.get("collision") == "scan":
        size = dict(type="mesh", mesh="scan")
    elif family == "brick":
        size = dict(type="box", size=(hx, hy, hz))
    else:
        size = dict(type="ellipsoid", size=(hx, hy, hz))
    collision_attrs = dict(
        name="parcel_hull",
        **size,
        mass=mass,
        condim=6,
        friction=friction,
        rgba=body_rgba,
        group=3 if "source" in item else 2,
    )
    if solref is not None:
        collision_attrs["solref"] = solref
    if solimp is not None:
        collision_attrs["solimp"] = solimp
    collision = node(body, "geom", **collision_attrs)
    if item.get("source") and family == "duck":
        # A single convex hull fills the notch between the chest, bill, and head.
        # Form overlapping convex sections from the actual scaled scan vertices.
        vertices = np.unique(
            np.array(
                [
                    [float(v) for v in line.split()[1:4]]
                    for line in (directory / "object.obj").read_text().splitlines()
                    if line.startswith("v ")
                ]
            ),
            axis=0,
        )
        edges = [-1.01, -0.18, 0.22, 0.57, 1.01]
        base_attrs = dict(collision.attrib)
        body.remove(collision)
        for index, (low, high) in enumerate(zip(edges, edges[1:])):
            points = vertices[
                (vertices[:, 2] >= (low - 0.035) * hz)
                & (vertices[:, 2] <= (high + 0.035) * hz)
            ]
            mesh_name = f"scan_hull_{index}"
            node(
                asset,
                "mesh",
                name=mesh_name,
                vertex=" ".join(f"{v:.8g}" for v in points.flat),
            )
            attrs = {
                k: v
                for k, v in base_attrs.items()
                if k not in ("type", "size", "name", "mass")
            }
            node(
                body,
                "geom",
                name="parcel_hull" if index == 0 else f"parcel_part_{index}",
                type="mesh",
                mesh=mesh_name,
                density=0,
                **attrs,
            )
    if "source" not in item and family == "pen":
        node(
            body,
            "geom",
            name="accent",
            type="cylinder",
            size=(hy * 1.15, hx * 0.18),
            pos=(hx * 0.62, 0, 0),
            quat=(0.707107, 0, 0.707107, 0),
            rgba=".15 .16 .19 1",
            contype=0,
            conaffinity=0,
            density=0,
        )
    if "source" not in item and family == "duck":
        collision.set("size", f"{hx} {hy} {hz * 0.76}")
        node(
            body,
            "geom",
            name="bird_head",
            type="ellipsoid",
            size=(hx * 0.56, hy * 0.84, hz * 0.58),
            pos=(hx * 0.45, 0, hz * 0.78),
            rgba=head_rgba,
            mass=0.0008,
            condim=6,
            friction=friction,
            **({"solref": solref} if solref is not None else {}),
            **({"solimp": solimp} if solimp is not None else {}),
        )
        eye_radius = max(0.0018, min(hx, hy, hz) * 0.16)
        for side, y in (("left", hy * 0.68), ("right", -hy * 0.68)):
            node(
                body,
                "geom",
                name="eye_" + side,
                type="sphere",
                size=eye_radius,
                pos=(hx * 0.56, y, hz * 0.99),
                rgba=".015 .015 .012 1",
                contype=0,
                conaffinity=0,
                group=2,
                density=0,
            )
        node(
            body,
            "geom",
            name="accent",
            type="ellipsoid",
            size=(hx * 0.40, hy * 0.58, hz * 0.18),
            pos=(hx * 1.04, 0, hz * 0.71),
            rgba=bill_rgba,
            mass=0.0001,
            friction=friction,
        )
    return np.array((hx, hy, hz))


def make_scene(item, bin_xy=BIN, moving_jaw=False):
    xml = scene_xml(None)
    world = xml.find("worldbody")
    eq = node(xml, "equality")
    contact_friction = item.get(
        "contact_friction", ".9 .9 .005 .0005 .0005"
    )
    solref = item.get("solref")
    solimp = item.get("solimp")
    if moving_jaw:
        # Official runtime documents travel. Axis through the CAD mouth body's origin,
        # damping, strength, and contact friction are lab assumptions pending calibration.
        xml.find("option").attrib.update(
            cone="elliptic",
            impratio="10",
            noslip_iterations="3",
            solver="Newton",
            tolerance="1e-10",
        )
        mouth = xml.find(".//body[@name='mouth']")
        node(
            mouth,
            "joint",
            name="bill_open",
            type="hinge",
            axis=(0, 1, 0),
            range=(0, math.radians(39.375)),
            damping=0.012,
            armature=0.00009,
        )
        node(
            xml.find("actuator"),
            "position",
            name="bill_servo",
            joint="bill_open",
            kp=1.4,
            ctrlrange=(0, math.radians(39.375)),
            forcerange=(-0.09, 0.09),
        )
    half = add_item(xml, item)
    for name, typ, sz in [
        ("old_ball", "sphere", (0.013,)),
        ("old_pen", "box", (0.032, 0.005, 0.006)),
    ]:
        body = node(world, "body", name=name, pos=(-1.4, 0, 0.01))
        node(body, "freejoint", name=name + "_free")
        node(
            body,
            "inertial",
            pos=(0, 0, 0),
            mass=0.003,
            diaginertia=(4.0e-7, 8.0e-7, 8.0e-7),
        )
        node(
            body,
            "geom",
            name=name + "_hull",
            type=typ,
            size=sz,
            mass=0.003,
            rgba=".23 .48 .72 1",
            condim=6,
            friction="1 .01 .006",
        )
    tip = xml.find(".//site[@name='mouth_tip']").get("pos")
    for name in ("parcel", *CLUTTER):
        node(
            eq,
            "connect",
            name="hold_" + name,
            body1="head_roll",
            body2=name,
            anchor=tip,
            active="false",
            solref=".008 1",
        )
    # Only the published bill's collision mesh can initiate the assisted grip.
    bx, by = map(float, bin_xy)
    sx, sy = BIN_HALF
    thick = 0.0025
    tray = node(world, "body", name="tray", pos=(bx, by, 0))
    for name, pos, dims in [
        ("base", (0, 0, BASE_Z / 2), (sx + 2 * thick, sy + 2 * thick, BASE_Z / 2)),
        ("north", (0, sy + thick, WALL_H / 2), (sx + 2 * thick, thick, WALL_H / 2)),
        ("south", (0, -sy - thick, WALL_H / 2), (sx + 2 * thick, thick, WALL_H / 2)),
        ("east", (sx + thick, 0, WALL_H / 2), (thick, sy, WALL_H / 2)),
        ("west", (-sx - thick, 0, WALL_H / 2), (thick, sy, WALL_H / 2)),
    ]:
        node(
            tray,
            "geom",
            name="tray_" + name,
            type="box",
            pos=pos,
            size=dims,
            rgba=".16 .52 .49 1",
            friction="1 .009 .004",
            condim=6,
        )
    if moving_jaw:
        # The archived head hull is convex and fills the mouth cavity. For item
        # collisions use a separate skull and two short bill surfaces, while the
        # original hulls still support the original learned ground motion.
        head = xml.find(".//body[@name='head_roll']")
        for name in ("upper_lip", "lower_lip"):
            parent = head if name == "upper_lip" else mouth
            z = -0.005 if name == "upper_lip" else -0.013
            node(
                parent,
                "geom",
                name=name,
                type="box",
                size=(0.012, 0.016, 0.0025),
                pos=(0.051, 0, z),
                contype=0,
                conaffinity=8,
                group=3,
                density=0,
                friction=".9 .005 .001",
                condim=4,
                solref=".012 1",
            )
        node(
            head,
            "geom",
            name="jaw_skull",
            type="ellipsoid",
            size=(0.046, 0.044, 0.043),
            pos=(-0.006, 0, 0.023),
            contype=0,
            conaffinity=8,
            group=3,
            density=0,
        )
        contacts = xml.find("contact")
        if contacts is None:
            contacts = node(xml, "contact")
        parts = [
            geom.get("name")
            for geom in world.find("body[@name='parcel']").findall("geom")
            if geom.get("name", "").startswith("parcel_part_")
        ]
        for lip in ("upper_lip", "lower_lip"):
            for obj in ("parcel_hull", *parts, "old_ball_hull", "old_pen_hull"):
                pair_attrs = dict(
                    geom1=lip,
                    geom2=obj,
                    condim=6,
                    friction=contact_friction,
                    solref=solref or ".012 1",
                    solreffriction=".008 1",
                    solimp=solimp or ".97 .995 .001",
                )
                node(contacts, "pair", **pair_attrs)
        reserved = (
            "head_bottom_ground_collision",
            "head_roll_ground_collision",
            "mouth_ground_collision",
        )
        for geom in world.iter("geom"):
            # Keep every other originally collidable prop, leg and surface.
            if geom.get("name") in reserved or geom.get("name") in (
                "upper_lip",
                "lower_lip",
                "jaw_skull",
            ):
                continue
            if geom.get("contype") == "0" and geom.get("conaffinity") == "0":
                continue
            if geom in list(world.find("body[@name='parcel']")) or geom.get("name") in (
                n + "_hull" for n in CLUTTER
            ):
                geom.set("contype", "8")
                geom.set("conaffinity", "8")
            else:
                geom.set("conaffinity", str(int(geom.get("conaffinity", "1")) | 8))
    return mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode")), half


def place(data, name, xyz, yaw=0.0):
    j = data.joint(name + "_free")
    j.qpos[:] = [*xyz, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
    j.qvel[:] = 0.0


def recolor(model, rng):
    for name in (*[x + "_hull" for x in CLUTTER], "parcel_hull"):
        model.geom(name).rgba[:3] = rng.uniform(0.08, 0.95, 3)
    bird_head = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "bird_head")
    if bird_head >= 0:
        # Keep procedural variants recognisably toy-duck-like while varying the albedo.
        palette = np.asarray(
            (
                (0.95, 0.72, 0.12),
                (0.98, 0.83, 0.30),
                (0.72, 0.88, 0.78),
                (0.94, 0.66, 0.78),
                (0.84, 0.77, 0.55),
            )
        )
        base = palette[int(rng.integers(len(palette)))]
        body = np.clip(base * rng.uniform(0.92, 1.06), 0.05, 0.98)
        model.geom("parcel_hull").rgba[:3] = body
        model.geom("bird_head").rgba[:3] = np.clip(body * 1.03, 0.05, 1.0)
        model.geom("accent").rgba[:3] = np.clip(
            np.asarray((0.95, 0.28, 0.05)) * rng.uniform(0.88, 1.08),
            0.03,
            1.0,
        )
    model.light_diffuse[0] = rng.uniform(0.45, 0.85, 3)


def camera_frame(renderer, data):
    renderer.update_scene(data, camera="head_rgb", scene_option=view_options())
    return renderer.render().copy()


def target_mask(renderer, model, data):
    renderer.enable_segmentation_rendering()
    try:
        seg = camera_frame(renderer, data)
    finally:
        renderer.disable_segmentation_rendering()
    ids = np.flatnonzero(model.geom_bodyid == model.body("parcel").id)
    return (
        (seg[:, :, 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & np.isin(seg[:, :, 0], ids)
    ).astype(np.uint8)
