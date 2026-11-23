"""Class-agnostic segmentation of a newly arrived object in two RGB views."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .pixels import component


def silhouette_features(yy, xx, extent=None):
    if extent is None:
        extent = np.array(
            [xx.max() - xx.min() + 1, yy.max() - yy.min() + 1], dtype=np.float32
        )
    width, height = map(float, extent)
    elongation = np.clip(
        np.log(max(width, height) / max(1.0, min(width, height))) / np.log(8.0),
        0.0,
        1.0,
    )
    fill = np.clip(len(xx) / max(1.0, width * height), 0.0, 1.0)
    return float(elongation), float(fill)


class ChangeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(9, 12, 3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 12, 3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
        )
        self.context = nn.Sequential(
            nn.AvgPool2d(2),
            nn.Conv2d(12, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=2, dilation=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.out = nn.Sequential(
            nn.Conv2d(28, 12, 3, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.Conv2d(12, 1, 1),
        )

    def forward(self, frames):
        pair = frames.float() / 255.0
        before, after = pair[:, :3], pair[:, 3:]
        difference = (after - before) * 4
        detail = self.local(torch.cat((after, difference, difference.abs()), dim=1))
        context = F.interpolate(
            self.context(detail), size=detail.shape[2:], mode="nearest"
        )
        return self.out(torch.cat((detail, context), dim=1))[:, 0]


class Detector:
    def __init__(self, path, device):
        self.device = device
        self.net = torch.jit.load(str(path), map_location=device).eval()

    @torch.inference_mode()
    def locate(self, before, after, roi=None):
        return self.locate_batch([(before, after, roi)])[0]

    @torch.inference_mode()
    def locate_batch(self, samples):
        """Locate several independently rendered scenes in one device call."""
        samples = list(samples)
        if not samples:
            return []
        pairs = np.stack(
            [
                np.concatenate((before, after), axis=-1)
                .transpose(2, 0, 1)
                .copy()
                for before, after, _ in samples
            ]
        )
        probabilities = (
            self.net(torch.from_numpy(pairs).to(self.device))
            .sigmoid()
            .cpu()
            .numpy()
        )
        results = []
        for (_, _, roi), prob in zip(samples, probabilities, strict=True):
            if roi is not None:
                prob = np.where(roi, prob, 0.0)
            results.append(self._features(prob))
        return results

    @staticmethod
    def _features(prob):
        region = component(prob)
        if region is None:
            return None
        yy, xx = region
        # Horizontal centre and lower silhouette reduce ground-projection bias.
        variances, axes = np.linalg.eigh(np.cov(xx, yy))
        minor, major = np.sqrt(np.maximum(0.0, variances))
        center = np.array([np.median(xx), np.median(yy)])
        direction = major * axes[:, -1] if major > 2.6 * max(0.8, minor) else None
        extent = np.array(
            [xx.max() - xx.min() + 1, yy.max() - yy.min() + 1],
            dtype=np.float32,
        )
        elongation, fill = silhouette_features(yy, xx, extent)
        return (
            center + (0.0, min(1.45 * minor, np.quantile(yy, 0.92) - center[1])),
            center,
            direction,
            extent,
            elongation,
            fill,
        )
