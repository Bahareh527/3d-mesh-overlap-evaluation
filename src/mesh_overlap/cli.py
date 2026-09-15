from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from .open3d_pipeline import register_geometry_files


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Register two local mesh or point-cloud files with Open3D."
    )
    command.add_argument("source", help="Source geometry (.ply, .obj, .stl, or supported format)")
    command.add_argument("target", help="Target geometry")
    command.add_argument(
        "--voxel-size",
        type=float,
        required=True,
        help="Downsampling voxel size in the geometry's coordinate units",
    )
    command.add_argument("--global-method", choices=["ransac", "fgr"], default="ransac")
    return command


def main() -> None:
    arguments = parser().parse_args()
    result = register_geometry_files(
        arguments.source,
        arguments.target,
        voxel_size=arguments.voxel_size,
        global_method=arguments.global_method,
    )
    payload = asdict(result)
    payload["transformation"] = result.transformation.tolist()
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
