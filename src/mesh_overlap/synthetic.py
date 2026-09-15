from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .registration import apply_transform


@dataclass(frozen=True, slots=True)
class SyntheticPair:
    source: NDArray[np.float64]
    target: NDArray[np.float64]
    source_to_target: NDArray[np.float64]


def _axis_angle(axis: NDArray[np.float64], angle: float) -> NDArray[np.float64]:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    cosine, sine = np.cos(angle), np.sin(angle)
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return cosine * np.eye(3) + (1 - cosine) * np.outer(axis, axis) + sine * cross


def make_synthetic_pair(
    point_count: int = 1500,
    noise: float = 0.003,
    seed: int = 42,
) -> SyntheticPair:
    """Create an asymmetric surface and a transformed noisy copy with known ground truth."""
    if point_count < 20 or noise < 0:
        raise ValueError("point_count must be at least 20 and noise cannot be negative.")
    rng = np.random.default_rng(seed)
    azimuth = rng.uniform(0, 2 * np.pi, point_count)
    cosine_polar = rng.uniform(-1, 1, point_count)
    sine_polar = np.sqrt(1 - cosine_polar**2)
    radius = 1 + 0.12 * np.sin(3 * azimuth) * sine_polar + 0.08 * cosine_polar
    target = np.column_stack(
        [
            1.0 * radius * sine_polar * np.cos(azimuth),
            0.75 * radius * sine_polar * np.sin(azimuth),
            1.25 * radius * cosine_polar,
        ]
    )
    rotation = _axis_angle(np.array([0.3, 0.7, 0.2]), angle=np.deg2rad(28))
    forward = np.eye(4)
    forward[:3, :3] = rotation
    forward[:3, 3] = [0.45, -0.30, 0.25]
    source = apply_transform(target, forward)
    source += rng.normal(scale=noise, size=source.shape)
    return SyntheticPair(source=source, target=target, source_to_target=np.linalg.inv(forward))
