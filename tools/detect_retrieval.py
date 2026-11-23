#!/usr/bin/env python3
"""Locate the newly visible item in two real or simulated camera photos with CPU ONNX Runtime."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import onnxruntime as ort
from PIL import Image, ImageDraw, ImageOps
from ducklab.retrieval.pixels import component

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("before", type=Path)
    p.add_argument("after", type=Path)
    p.add_argument(
        "--weights", type=Path, default=ROOT / "runs/fetch/vision/change.onnx"
    )
    p.add_argument("--annotated", type=Path)
    a = p.parse_args()
    with Image.open(a.before) as f:
        before = ImageOps.exif_transpose(f).convert("RGB")
    with Image.open(a.after) as f:
        after = ImageOps.exif_transpose(f).convert("RGB")
    if before.size != after.size:
        p.error("before and after must have the same dimensions and viewpoint")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    sess = ort.InferenceSession(
        str(a.weights), opts, providers=["CPUExecutionProvider"]
    )
    input_ = sess.get_inputs()[0]
    h, w = input_.shape[2:]
    planes = [
        np.asarray(im.resize((w, h), Image.Resampling.BILINEAR))
        for im in (before, after)
    ]
    pair = np.concatenate(planes, axis=2).transpose(2, 0, 1)[None]
    logits = sess.run(None, {input_.name: pair})[0][0]
    prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -70, 70)))
    region = component(prob)
    result = dict(detected=region is not None)
    if region is not None:
        yy, xx = region
        sx, sy = after.width / w, after.height / h
        box = [
            int(xx.min() * sx),
            int(yy.min() * sy),
            int(np.ceil((xx.max() + 1) * sx)),
            int(np.ceil((yy.max() + 1) * sy)),
        ]
        result.update(
            bbox_xyxy=box,
            center_xy=[
                round(float((np.median(xx) + 0.5) * sx), 1),
                round(float((np.median(yy) + 0.5) * sy), 1),
            ],
        )
        if a.annotated:
            ImageDraw.Draw(after).rectangle(
                box, outline="#00ffff", width=max(2, after.width // 200)
            )
    if a.annotated:
        a.annotated.parent.mkdir(parents=True, exist_ok=True)
        after.save(a.annotated)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
