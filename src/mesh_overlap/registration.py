from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _points(values: ArrayLike, name: str, minimum: int = 1) -> NDArray[np.float64]:
    points = np.asarray(values, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3 or len(points) < minimum:
        raise ValueError(f"{name} must have shape (N, 3) with at least {minimum} points.")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite coordinates.")
    return points


def rigid_transform_svd(source: ArrayLike, target: ArrayLike) -> NDArray[np.float64]:
    """Return the least-squares rigid transform mapping source correspondences to target."""
    source_points = _points(source, "source", minimum=3)
    target_points = _points(target, "target", minimum=3)
    if source_points.shape != target_points.shape:
        raise ValueError("source and target correspondences must have the same shape.")

    source_centroid = source_points.mean(axis=0)
    target_centroid = target_points.mean(axis=0)
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    if np.linalg.matrix_rank(source_centered) < 2 or np.linalg.matrix_rank(target_centered) < 2:
        raise ValueError("At least three non-collinear correspondences are required.")

    covariance = source_centered.T @ target_centered
    left, _, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0:
        right_transpose[-1] *= -1
        rotation = right_transpose.T @ left.T
    translation = target_centroid - rotation @ source_centroid

    transformation = np.eye(4, dtype=np.float64)
    transformation[:3, :3] = rotation
    transformation[:3, 3] = translation
    return transformation


def apply_transform(points: ArrayLike, transformation: ArrayLike) -> NDArray[np.float64]:
    """Apply a homogeneous 4x4 transformation to an array of 3D points."""
    point_array = _points(points, "points")
    matrix = np.asarray(transformation, dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("transformation must be a finite 4x4 matrix.")
    if not np.allclose(matrix[3], [0, 0, 0, 1]):
        raise ValueError("transformation must use homogeneous coordinates.")
    return point_array @ matrix[:3, :3].T + matrix[:3, 3]
