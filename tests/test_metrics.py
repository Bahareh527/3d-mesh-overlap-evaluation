from __future__ import annotations

import numpy as np
import pytest

from mesh_overlap import evaluate_alignment, nearest_neighbor_distances, voxel_overlap


def test_identical_clouds_have_perfect_overlap() -> None:
    points = np.array([[0, 0, 0], [1, 1, 1], [2, 0, 1]], dtype=float)
    metrics = evaluate_alignment(points, points, voxel_size=0.2)
    assert metrics.mean_symmetric_surface_distance == pytest.approx(0)
    assert metrics.symmetric_hausdorff_distance == pytest.approx(0)
    assert metrics.voxel_dice == pytest.approx(1)
    assert metrics.voxel_jaccard == pytest.approx(1)


def test_shared_grid_detects_disjoint_voxels() -> None:
    source = np.array([[0, 0, 0], [0.1, 0, 0]])
    target = source + 5
    dice, jaccard, _, _, intersection = voxel_overlap(source, target, voxel_size=0.5)
    assert dice == 0
    assert jaccard == 0
    assert intersection == 0


def test_nearest_neighbor_distances_have_source_length() -> None:
    source = np.array([[0, 0, 0], [2, 0, 0]], dtype=float)
    target = np.array([[1, 0, 0]], dtype=float)
    distances = nearest_neighbor_distances(source, target)
    assert distances.tolist() == pytest.approx([1, 1])


def test_voxel_size_must_be_positive() -> None:
    with pytest.raises(ValueError, match="voxel_size"):
        voxel_overlap([[0, 0, 0]], [[0, 0, 0]], voxel_size=0)
