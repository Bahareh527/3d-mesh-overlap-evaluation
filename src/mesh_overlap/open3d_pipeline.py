from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Open3DRegistrationResult:
    transformation: NDArray[np.float64]
    global_fitness: float
    global_inlier_rmse: float
    refined_fitness: float
    refined_inlier_rmse: float


def _open3d():
    try:
        import open3d as o3d
    except ImportError as error:
        message = "Install the optional dependency with: pip install -e '.[open3d]'"
        raise ImportError(message) from error
    return o3d


def load_point_cloud(path: str | Path, *, sample_points: int = 200_000):
    """Load a point cloud or sample points uniformly from a triangle mesh."""
    o3d = _open3d()
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Geometry file not found: {source}")
    cloud = o3d.io.read_point_cloud(str(source))
    if cloud.is_empty():
        mesh = o3d.io.read_triangle_mesh(str(source))
        if mesh.is_empty() or not mesh.has_triangles():
            raise ValueError(f"Could not read points or triangles from {source}")
        cloud = mesh.sample_points_uniformly(number_of_points=sample_points)
    return cloud


def _preprocess(cloud, voxel_size: float):
    o3d = _open3d()
    downsampled = cloud.voxel_down_sample(voxel_size)
    downsampled.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel_size, max_nn=30)
    )
    features = o3d.pipelines.registration.compute_fpfh_feature(
        downsampled,
        o3d.geometry.KDTreeSearchParamHybrid(radius=5 * voxel_size, max_nn=100),
    )
    return downsampled, features


def register_geometry_files(
    source_path: str | Path,
    target_path: str | Path,
    *,
    voxel_size: float,
    global_method: str = "ransac",
) -> Open3DRegistrationResult:
    """Run FPFH global registration followed by point-to-plane ICP refinement."""
    if voxel_size <= 0:
        raise ValueError("voxel_size must be positive and use the same units as the geometry.")
    if global_method not in {"ransac", "fgr"}:
        raise ValueError("global_method must be 'ransac' or 'fgr'.")
    o3d = _open3d()
    source = load_point_cloud(source_path)
    target = load_point_cloud(target_path)
    source_down, source_features = _preprocess(source, voxel_size)
    target_down, target_features = _preprocess(target, voxel_size)
    global_threshold = 1.5 * voxel_size
    if global_method == "ransac":
        global_result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_features,
            target_features,
            True,
            global_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            4,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(global_threshold),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100_000, 0.999),
        )
    else:
        global_result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
            source_down,
            target_down,
            source_features,
            target_features,
            o3d.pipelines.registration.FastGlobalRegistrationOption(
                maximum_correspondence_distance=global_threshold
            ),
        )
    source.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel_size, max_nn=30)
    )
    target.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=2 * voxel_size, max_nn=30)
    )
    refined = o3d.pipelines.registration.registration_icp(
        source,
        target,
        0.4 * voxel_size,
        global_result.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return Open3DRegistrationResult(
        transformation=np.asarray(refined.transformation),
        global_fitness=float(global_result.fitness),
        global_inlier_rmse=float(global_result.inlier_rmse),
        refined_fitness=float(refined.fitness),
        refined_inlier_rmse=float(refined.inlier_rmse),
    )
