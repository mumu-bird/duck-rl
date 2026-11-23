#!/usr/bin/env python3
"""Fetch the declared scans, retain source notices, and write scaled textured OBJ files."""

import hashlib
import io
import json
import os
import struct
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "vendor/retrieval"
KHRONOS = (
    "https://raw.githubusercontent.com/KhronosGroup/glTF-Sample-Assets/"
    "e124e8d8d7330a5372bc2b4d00d39a4a9d7997ea/"
)


def fetch(url, path, expected=None):
    if not path.exists():
        with urllib.request.urlopen(url, timeout=45) as resp:
            data = resp.read()
        if expected and hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f"hash mismatch: {url}")
        tmp = path.with_suffix(f".{os.getpid()}.part")
        tmp.write_bytes(data)
        tmp.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected and digest != expected:
        raise ValueError(f"hash mismatch: {path}")
    return digest


def ycb_mesh(archive, identity):
    with tarfile.open(archive) as tar:

        def read(name):
            return tar.extractfile(f"{identity}/google_16k/{name}").read()

        lines = read("textured.obj").decode().splitlines()
        img = read("texture_map.png")
    verts = np.array(
        [[float(x) for x in s.split()[1:4]] for s in lines if s.startswith("v ")]
    )
    uv = [s for s in lines if s.startswith("vt ")]
    faces = [
        "f " + " ".join("/".join(t.split("/")[:2]) for t in s.split()[1:])
        for s in lines
        if s.startswith("f ")
    ]
    # Rotate principal axes into a lying, lengthwise-X orientation. Preserve handedness.
    _, _, axes = np.linalg.svd(verts - verts.mean(0), full_matrices=False)
    if np.linalg.det(axes) < 0:
        axes[-1] *= -1
    return verts @ axes.T, uv, faces, img


def duck_mesh(path):
    data = path.read_bytes()
    size = struct.unpack_from("<I", data, 12)[0]
    meta = json.loads(data[20 : 20 + size])
    buf = data[28 + size :]

    def accessor(i):
        acc = meta["accessors"][i]
        view = meta["bufferViews"][acc["bufferView"]]
        cols = {"VEC3": 3, "VEC2": 2, "SCALAR": 1}[acc["type"]]
        dt = np.dtype({5126: "<f4", 5123: "<u2", 5125: "<u4"}[acc["componentType"]])
        offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        return np.ndarray(
            (acc["count"], cols),
            dtype=dt,
            buffer=buf,
            offset=offset,
            strides=(view.get("byteStride", dt.itemsize * cols), dt.itemsize),
        )

    prim = meta["meshes"][0]["primitives"][0]
    pos = accessor(prim["attributes"]["POSITION"])
    verts = np.column_stack((pos[:, 0], -pos[:, 2], pos[:, 1]))
    uv = [f"vt {u} {1 - v}" for u, v in accessor(prim["attributes"]["TEXCOORD_0"])]
    faces = [
        "f " + " ".join(f"{n}/{n}" for n in tri)
        for tri in accessor(prim["indices"]).reshape(-1, 3).astype(int) + 1
    ]
    view = meta["bufferViews"][meta["images"][0]["bufferView"]]
    start = view.get("byteOffset", 0)
    return verts, uv, faces, buf[start : start + view["byteLength"]]


def main():
    items = json.loads((ROOT / "configs/retrieval-objects.json").read_text())["objects"]
    manifest = {}
    pins = ROOT / "configs/retrieval-sha256.json"
    hashes = json.loads(pins.read_text()) if pins.exists() else {}
    for item in items:
        source = item.get("source")
        identity = item["id"]
        if not source:
            continue
        dest = ASSETS / identity
        dest.mkdir(parents=True, exist_ok=True)
        if source == "ycb":
            url = f"https://ycb-benchmarks.s3.amazonaws.com/data/google/{identity}_google_16k.tgz"
            raw = dest / "source.tgz"
        else:
            url = KHRONOS + "Models/Duck/glTF-Binary/Duck.glb"
            raw = dest / "source.glb"
            for remote, local in [
                ("LICENSES/SCEA.txt", "LICENSE.txt"),
                ("Models/Duck/README.md", "SOURCE.md"),
            ]:
                fetch(KHRONOS + remote, dest / local)
        digest = fetch(url, raw, hashes.get(identity))
        v, uv, faces, image = (
            ycb_mesh(raw, identity) if source == "ycb" else duck_mesh(raw)
        )
        v = v - (v.min(0) + v.max(0)) / 2
        v *= item["length"] / np.ptp(v, axis=0).max()
        with (dest / "object.obj").open("w") as f:
            np.savetxt(f, v, fmt="v %.9g %.9g %.9g")
            f.write("\n".join(uv + faces) + "\n")
        tex = Image.open(io.BytesIO(image)).convert("RGB")
        tex.thumbnail((256, 256))
        tex.save(dest / "texture.png")
        half = (np.ptp(v, axis=0) / 2).tolist()
        (dest / "half.json").write_text(json.dumps(half))
        manifest[identity] = dict(
            url=url, sha256=digest, split=item["split"], half_extents_m=half
        )
        print(identity, digest, flush=True)
    (ASSETS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
