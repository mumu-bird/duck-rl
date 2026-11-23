"""A small trainable motion schedule layered over the published ground-pick policy."""

from dataclasses import dataclass
import math
import json
from pathlib import Path


@dataclass(frozen=True)
class GraspProfile:
    reach: float = 0.072
    close: int = 47
    rise: int = 82
    ramp: int = 0
    retry_radius: float = 0.0
    retry_yaw: float = 0.0
    retry_reach: float = 0.0
    retry_side: float = 0.0
    scoop_pitch: float = 0.0
    scoop_neck: float = 0.0

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("grasp configuration must be finite")
        if self.reach <= 0 or self.ramp < 0 or self.retry_radius < 0:
            raise ValueError("invalid grasp distance, ramp or retry radius")

    def stroke(self, tick):
        """Lower and sweep on approach, then ease out once the bill has closed."""
        gain = max(0.0, min(1.0, (tick - 20) / 20, (120 - tick) / 25))
        sweep = gain * max(-1.0, min(1.0, (tick - self.close + 8) / 12))
        neck = self.scoop_neck * sweep
        return neck, self.scoop_pitch * gain - neck

    def command(self, tick):
        opening = max(0.0, min(1.0, tick / 12.0, (self.close + 6 - tick) / 6.0))
        # Shorten the dwell near the floor. The ONNX still produces all leg targets.
        advance = max(0, 107 - self.rise)
        if self.ramp > 0:
            fraction = max(0.0, min(1.0, (tick - self.rise) / self.ramp))
            # A cubic ramp has zero extra phase velocity at both boundaries.
            advance *= fraction * fraction * (3.0 - 2.0 * fraction)
        phase_tick = tick if tick < self.rise else min(198, tick + advance)
        phase = phase_tick * math.pi / 100
        return (math.cos(phase), math.sin(phase)), opening * math.radians(39.375)


@dataclass(frozen=True)
class DeliveryProfile:
    reach: float = 0.106  # Base-to-tray offset during approach, metres
    side: float = 0.0
    neck: float = -0.28
    pitch: float = -0.12
    release_tick: int = 28
    open_ticks: int = 10
    retain_until_open: bool = False
    lower_to_tray: bool = False

    def __post_init__(self):
        if not all(math.isfinite(v) for v in vars(self).values()):
            raise ValueError("delivery configuration must be finite")
        if self.open_ticks <= 0 or not 1 <= self.release_tick < (
            104 if self.lower_to_tray else 78
        ):
            raise ValueError(
                "release must occur before the end of the jaw-opening phase"
            )


def load_profiles(path=None):
    if path is None:
        path = Path(__file__).resolve().parents[2] / "configs/retrieval-grasps.json"
    profiles = json.loads(Path(path).read_text())
    result = {name: GraspProfile(**profiles[name]) for name in ("general", "thin")}
    if "retry" in profiles:
        result["retry"] = GraspProfile(**profiles["retry"])
    if "delivery" in profiles:
        result["delivery"] = DeliveryProfile(**profiles["delivery"])
    return result
