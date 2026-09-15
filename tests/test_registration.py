from __future__ import annotations

import numpy as np
import pytest

from mesh_overlap import apply_transform, rigid_transform_svd


def test_svd_recovers_a_known_rigid_transform() -> None:
    rng = np.random.default_rng(4)
    source = rng.normal(size=(30, 3))
    angle = np.deg2rad(32)
    rotation = np.array(
        [[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
    )
    expected = np.eye(4)
    expected[:3, :3] = rotation
    expected[:3, 3] = [0.5, -0.2, 1.1]
    target = apply_transform(source, expected)
    estimated = rigid_transform_svd(source, target)
    assert np.allclose(estimated, expected, atol=1e-10)


def test_svd_never_returns_a_reflection() -> None:
    source = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    target = source.copy()
    target[:, 0] *= -1
    estimated = rigid_transform_svd(source, target)
    assert np.linalg.det(estimated[:3, :3]) == pytest.approx(1.0)


def test_collinear_correspondences_are_rejected() -> None:
    points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    with pytest.raises(ValueError, match="non-collinear"):
        rigid_transform_svd(points, points)


def test_invalid_homogeneous_transform_is_rejected() -> None:
    transformation = np.eye(4)
    transformation[3, 0] = 1
    with pytest.raises(ValueError, match="homogeneous"):
        apply_transform([[0, 0, 0]], transformation)
