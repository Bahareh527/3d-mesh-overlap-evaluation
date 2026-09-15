from __future__ import annotations

import numpy as np

from mesh_overlap import (
    apply_transform,
    evaluate_alignment,
    make_synthetic_pair,
    rigid_transform_svd,
)


def test_synthetic_alignment_improves_surface_distance_and_overlap() -> None:
    pair = make_synthetic_pair(point_count=600, noise=0.001, seed=11)
    before = evaluate_alignment(pair.source, pair.target, voxel_size=0.08)
    estimated = rigid_transform_svd(pair.source, pair.target)
    aligned = apply_transform(pair.source, estimated)
    after = evaluate_alignment(aligned, pair.target, voxel_size=0.08)
    assert after.mean_symmetric_surface_distance < before.mean_symmetric_surface_distance * 0.05
    assert after.voxel_dice > before.voxel_dice
    assert np.allclose(estimated, pair.source_to_target, atol=0.003)
