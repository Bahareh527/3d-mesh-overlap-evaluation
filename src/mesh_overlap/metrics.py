from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.spatial import cKDTree

from .registration import _points


@dataclass(frozen=True, slots=True)
class AlignmentMetrics:
    mean_symmetric_surface_distance: float
    symmetric_hausdorff_distance: float
    voxel_dice: float
    voxel_jaccard: float
    source_voxels: int
    target_voxels: int
    intersecting_voxels: int


def nearest_neighbor_distances(source: ArrayLike, target: ArrayLike) -> NDArray[np.float64]:
    """Compute the Euclidean distance from each source point to its nearest target point."""
    source_points = _points(source, "source")
    target_points = _points(target, "target")
    distances, _ = cKDTree(target_points).query(source_points, k=1, workers=-1)
    return np.asarray(distances, dtype=np.float64)


def voxel_overlap(
    source: ArrayLike,
    target: ArrayLike,
    *,
    voxel_size: float,
    origin: ArrayLike | None = None,
) -> tuple[float, float, int, int, int]:
    """Compute Dice and Jaccard overlap from a shared occupied-voxel grid."""
    if voxel_size <= 0 or not np.isfinite(voxel_size):
        raise ValueError("voxel_size must be finite and positive.")
    source_points = _points(source, "source")
    target_points = _points(target, "target")
    grid_origin = (
        np.minimum(source_points.min(axis=0), target_points.min(axis=0))
        if origin is None
        else np.asarray(origin, dtype=np.float64)
    )
    if grid_origin.shape != (3,) or not np.isfinite(grid_origin).all():
        raise ValueError("origin must contain three finite coordinates.")

    def occupied(points: NDArray[np.float64]) -> set[tuple[int, int, int]]:
        indices = np.floor((points - grid_origin) / voxel_size).astype(np.int64)
        return {tuple(row) for row in indices}

    source_voxels = occupied(source_points)
    target_voxels = occupied(target_points)
    intersection = len(source_voxels & target_voxels)
    union = len(source_voxels | target_voxels)
    dice = 2 * intersection / (len(source_voxels) + len(target_voxels))
    jaccard = intersection / union
    return dice, jaccard, len(source_voxels), len(target_voxels), intersection


def evaluate_alignment(
    source: ArrayLike,
    target: ArrayLike,
    *,
    voxel_size: float,
) -> AlignmentMetrics:
    """Evaluate symmetric surface distances and shared-grid voxel overlap."""
    forward = nearest_neighbor_distances(source, target)
    backward = nearest_neighbor_distances(target, source)
    dice, jaccard, source_count, target_count, intersection = voxel_overlap(
        source, target, voxel_size=voxel_size
    )
    return AlignmentMetrics(
        mean_symmetric_surface_distance=float((forward.mean() + backward.mean()) / 2),
        symmetric_hausdorff_distance=float(max(forward.max(), backward.max())),
        voxel_dice=float(dice),
        voxel_jaccard=float(jaccard),
        source_voxels=source_count,
        target_voxels=target_count,
        intersecting_voxels=intersection,
    )
