"""Privacy-safe tools for rigid 3D registration and overlap evaluation."""

from .metrics import AlignmentMetrics, evaluate_alignment, nearest_neighbor_distances, voxel_overlap
from .registration import apply_transform, rigid_transform_svd
from .synthetic import SyntheticPair, make_synthetic_pair

__all__ = [
    "AlignmentMetrics",
    "SyntheticPair",
    "apply_transform",
    "evaluate_alignment",
    "make_synthetic_pair",
    "nearest_neighbor_distances",
    "rigid_transform_svd",
    "voxel_overlap",
]
