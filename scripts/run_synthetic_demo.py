from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from mesh_overlap import (
    apply_transform,
    evaluate_alignment,
    make_synthetic_pair,
    rigid_transform_svd,
)

ROOT = Path(__file__).resolve().parents[1]


def run_demo() -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    pair = make_synthetic_pair(point_count=1800, noise=0.003, seed=42)
    landmark_indices = np.linspace(0, len(pair.source) - 1, 16, dtype=int)
    estimated = rigid_transform_svd(
        pair.source[landmark_indices],
        pair.target[landmark_indices],
    )
    aligned = apply_transform(pair.source, estimated)
    before = evaluate_alignment(pair.source, pair.target, voxel_size=0.09)
    after = evaluate_alignment(aligned, pair.target, voxel_size=0.09)
    summary = {
        "synthetic_only": True,
        "landmark_correspondences": len(landmark_indices),
        "before": asdict(before),
        "after": asdict(after),
    }
    return pair.source, pair.target, aligned, summary


def _scatter(axis, source: np.ndarray, target: np.ndarray, title: str) -> None:
    axis.scatter(*target[::4].T, s=3, alpha=0.45, label="target", color="#4C78A8")
    axis.scatter(*source[::4].T, s=3, alpha=0.45, label="source", color="#F58518")
    axis.set_title(title)
    axis.set_box_aspect((1, 1, 1))
    axis.legend(frameon=False, loc="upper left")


def main() -> None:
    source, target, aligned, summary = run_demo()
    figure = plt.figure(figsize=(12.5, 4.2))
    before_axis = figure.add_subplot(1, 3, 1, projection="3d")
    after_axis = figure.add_subplot(1, 3, 2, projection="3d")
    metric_axis = figure.add_subplot(1, 3, 3)
    _scatter(before_axis, source, target, "Before registration")
    _scatter(after_axis, aligned, target, "After SVD registration")
    before = summary["before"]
    after = summary["after"]
    metric_axis.bar(
        ["before", "after"],
        [before["mean_symmetric_surface_distance"], after["mean_symmetric_surface_distance"]],
        color=["#E45756", "#54A24B"],
    )
    metric_axis.set_title("Surface distance\n(synthetic only)")
    metric_axis.set_ylabel("mean symmetric distance")
    figure.tight_layout()
    output = ROOT / "docs" / "figures" / "synthetic_registration.png"
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(figure)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
