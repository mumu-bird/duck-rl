#!/usr/bin/env python3
"""Train on GPU, choose an epoch on disjoint validation identities, then evaluate test once."""

import argparse
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.nn import functional as F
from ducklab.assets import ROOT
from ducklab.retrieval.scene import OBJECTS
from ducklab.retrieval.vision import ChangeNet
from ducklab.retrieval.pixels import component


def load(split, directory, device):
    ids = []
    frames = []
    masks = []
    for item in OBJECTS:
        if item["split"] != split:
            continue
        with np.load(directory / (item["id"] + ".npz")) as f:
            frames.append(np.concatenate((f["before"], f["after"]), axis=-1))
            masks.append(f["mask"])
            ids += [item["id"]] * len(masks[-1])
    return (
        torch.from_numpy(np.concatenate(frames)).permute(0, 3, 1, 2).to(device),
        torch.from_numpy(np.concatenate(masks)).to(device),
        np.array(ids),
    )


@torch.inference_mode()
def evaluate(net, dataset, detailed=False):
    images, labels, ids = dataset
    net.eval()
    masks = []
    scores = []
    for im, lab in zip(images.split(24), labels.split(24)):
        prob = net(im).sigmoid()
        pred = prob > 0.5
        real = lab.bool()
        positive = real.any((1, 2))
        iou = (pred & real).sum((1, 2)) / (pred | real).sum((1, 2)).clamp_min(1)
        scores.extend(iou[positive].tolist())
        if detailed:
            masks.extend(prob.cpu().numpy())
    if not detailed:
        return float(np.mean(scores))
    stats = {}
    for identity in dict.fromkeys(ids):
        pos = hit = neg = false = 0
        ious = []
        for idx in np.flatnonzero(ids == identity):
            real = labels[idx].cpu().numpy().astype(bool)
            region = component(masks[idx])
            found = np.zeros_like(real)
            if region is not None:
                found[region] = True
            if not real.any():
                neg += 1
                false += int(found.any())
                continue
            pos += 1
            iou = (real & found).sum() / (real | found).sum()
            ious.append(iou)
            hit += int(iou >= 0.5)
        stats[identity] = dict(
            visible=pos,
            localized_iou50=hit,
            empty=neg,
            false_triggers=false,
            mean_mask_iou=round(float(np.mean(ious)), 4),
        )
    return stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frames", type=Path, default=ROOT / "runs/fetch/frames")
    p.add_argument("--out", type=Path, default=ROOT / "runs/fetch/vision")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--seed", type=int, default=4102)
    p.add_argument("--device", default="cuda")
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = False
    train, val = [load(s, a.frames, a.device) for s in ("train", "val")]
    images, labels, _ = train
    net = ChangeNet().to(device=a.device, memory_format=torch.channels_last)
    opt = torch.optim.AdamW(net.parameters(), lr=0.002)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=a.epochs, eta_min=0.0002
    )
    device = torch.cuda.get_device_name() if a.device.startswith("cuda") else "CPU"
    best = -1.0
    history = []
    start = time.monotonic()
    for epoch in range(1, a.epochs + 1):
        net.train()
        for idx in torch.randperm(len(images), device=a.device).split(a.batch):
            x, y = images[idx], labels[idx].float()
            logits = net(x)
            prob = logits.sigmoid()
            bce = F.binary_cross_entropy_with_logits(logits, y, weight=1 + 19 * y)
            # A soft Dice term supplies a useful gradient for very thin silhouettes.
            dice = 1 - (2 * (prob * y).sum() + 1) / (prob.sum() + y.sum() + 1)
            opt.zero_grad(set_to_none=True)
            (bce + dice).backward()
            opt.step()
        schedule.step()
        score = evaluate(net, val)
        history.append(score)
        if score > best:
            best = score
            chosen = epoch
            torch.jit.trace(net.eval(), images[:1]).save(str(a.out / "change.pt"))
        print(f"epoch={epoch:02} val_mask_iou={score:.4f}", flush=True)
    elapsed = time.monotonic() - start
    best_net = torch.jit.load(str(a.out / "change.pt"), map_location=a.device).eval()
    result = dict(
        gpu=device,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        epochs=a.epochs,
        batch=a.batch,
        seed=a.seed,
        seconds=round(elapsed, 2),
        selected_epoch=chosen,
        val_history=history,
        train_pairs=len(images),
        val_pairs=len(val[0]),
        test=evaluate(best_net, load("test", a.frames, a.device), True),
    )
    if a.device.startswith("cuda"):
        result["peak_allocated_mib"] = round(
            torch.cuda.max_memory_allocated() / 2**20, 1
        )
    (a.out / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(result["test"])


if __name__ == "__main__":
    main()
