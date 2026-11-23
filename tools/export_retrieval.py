#!/usr/bin/env python3
"""Export the paired-RGB model and verify every held-out sample with CPU ONNX Runtime."""

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import onnx
import onnxruntime as ort
import torch
from ducklab.retrieval.scene import ROOT, HEIGHT, WIDTH, OBJECTS
from ducklab.retrieval.vision import ChangeNet


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, default=ROOT / "runs/fetch/vision/change.pt")
    p.add_argument("--frames", type=Path, default=ROOT / "runs/fetch/frames")
    p.add_argument("--out", type=Path, default=ROOT / "runs/fetch/vision/change.onnx")
    a = p.parse_args()
    net = ChangeNet().eval()
    net.load_state_dict(torch.jit.load(str(a.weights), map_location="cpu").state_dict())
    torch.set_num_threads(2)
    dummy = torch.zeros(1, 6, HEIGHT, WIDTH, dtype=torch.uint8)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    temp = a.out.with_suffix(".partial.onnx")
    torch.onnx.export(
        net,
        dummy,
        temp,
        input_names=["before_after_rgb"],
        output_names=["foreground_logits"],
        opset_version=17,
        dynamo=False,
    )
    onnx.checker.check_model(str(temp))
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 2
    session = ort.InferenceSession(str(temp), opts, providers=["CPUExecutionProvider"])
    n = 0
    max_abs = 0.0
    flips = 0
    with torch.inference_mode():
        for item in (item for item in OBJECTS if item["split"] == "test"):
            with np.load(a.frames / (item["id"] + ".npz")) as frames:
                for before, after in zip(
                    frames["before"], frames["after"], strict=True
                ):
                    pair = (
                        np.concatenate((before, after), axis=-1)
                        .transpose(2, 0, 1)
                        .copy()[None]
                    )
                    ref = net(torch.from_numpy(pair)).numpy()
                    cand = session.run(None, {"before_after_rgb": pair})[0]
                    max_abs = max(max_abs, float(np.max(abs(ref - cand))))
                    flips += int(np.sum((ref >= 0) != (cand >= 0)))
                    n += 1
    if not n or max_abs > 1e-3 or flips > 5:
        raise ValueError(
            f"ONNX parity: {n} images, {max_abs} logit error, {flips} threshold crossings"
        )
    temp.replace(a.out)
    result = dict(
        pairs=n,
        maximum_logit_error=max_abs,
        flipped_pixels=flips,
        total_pixels=n * HEIGHT * WIDTH,
        opset=17,
        input="uint8[1,6,160,224]",
        output="float32[1,160,224] logits",
    )
    a.out.with_suffix(".parity.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result)


if __name__ == "__main__":
    main()
