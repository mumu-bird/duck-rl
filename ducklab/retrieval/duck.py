"""Small-duck domain randomization and grasp-profile helpers."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..assets import ROOT
from .grasp import GraspProfile, load_profiles


VARIANT_CONFIG = ROOT / "configs/linlee-duck-variants.json"
PROFILE_CONFIG = ROOT / "configs/linlee-duck-profiles.json"


def load_duck_variants(path: Path = VARIANT_CONFIG) -> list[dict[str, Any]]:
    """Load procedural duck templates without mutating the JSON objects."""
    data = json.loads(Path(path).read_text())
    variants = data.get("variants")
    if not isinstance(variants, list) or not variants:
        raise ValueError("duck variant config must contain a non-empty variants list")
    return [dict(item) for item in variants]


def sample_duck(template: dict[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    """Sample a physical duck instance for a paired simulation trial."""
    scale_lo, scale_hi = template.get("scale", (0.94, 1.06))
    scale = float(rng.uniform(scale_lo, scale_hi))
    half = np.asarray(template["half_m"], dtype=float) * scale
    mass_lo, mass_hi = template.get("mass_grams", (12.0, 34.0))
    friction_lo, friction_hi = template.get("friction", (0.55, 0.90))
    friction = float(rng.uniform(friction_lo, friction_hi))
    sampled = dict(template)
    sampled.update(
        {
            "half_m": half.tolist(),
            "mass_kg": float(rng.uniform(mass_lo, mass_hi)) / 1000.0,
            "friction": f"{friction:.5f} .012 .004",
            "contact_friction": (
                f"{min(1.1, friction + 0.24):.5f} "
                f"{min(1.1, friction + 0.24):.5f} .005 .0005 .0005"
            ),
            "sample_scale": scale,
        }
    )
    return sampled


def duck_profile_candidates() -> list[GraspProfile]:
    """Return a small, paired grid for the first duck-specific profile search."""
    candidates: list[GraspProfile] = []
    for reach in (0.068, 0.074, 0.080):
        for close in (43, 49, 55):
            for rise in (78, 86, 94):
                candidates.append(
                    GraspProfile(
                        reach=reach,
                        close=close,
                        rise=rise,
                        ramp=45,
                        retry_radius=0.12,
                        retry_yaw=0.32,
                        retry_reach=-0.005,
                        scoop_pitch=-0.06,
                        scoop_neck=0.02,
                    )
                )
    return candidates


def profiles_for(grasp: GraspProfile, path: Path = PROFILE_CONFIG) -> dict[str, Any]:
    """Overlay a candidate grasp while retaining the tested delivery settings."""
    profiles = load_profiles(path)
    profiles["general"] = grasp
    profiles["retry"] = replace(
        grasp,
        reach=min(0.086, grasp.reach + 0.006),
        close=min(70, grasp.close + 5),
        scoop_pitch=-0.12,
    )
    return profiles


def serializable_profiles(profiles: dict[str, Any]) -> dict[str, Any]:
    """Convert loaded dataclasses back to the JSON profile format."""
    return {
        name: {
            key: value
            for key, value in vars(profile).items()
        }
        for name, profile in profiles.items()
    }
