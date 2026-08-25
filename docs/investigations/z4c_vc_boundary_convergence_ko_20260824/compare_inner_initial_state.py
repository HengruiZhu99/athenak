#!/usr/bin/env python3
"""Compare Rout=16 and Rout=128 native-VC initialization inside the old box."""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_reader(source: Path):
    path = source / "vis/python/bin_convert.py"
    spec = importlib.util.spec_from_file_location("boundary_ko_bin_convert", path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_binary


def canonical_inner(data: dict, limit: float) -> tuple[dict[tuple[float, float], np.ndarray], dict]:
    names = list(data["var_names"])
    require(len(names) == 25 and all(name.startswith("z4c_") for name in names),
            "expected the 25 evolved Z4c variables")
    points: dict[tuple[float, float], tuple[int, np.ndarray]] = {}
    same_level_spread = 0.0
    cross_level_spread = 0.0
    axis_coordinates: set[float] = set()
    for block, bounds in enumerate(np.asarray(data["mb_geometry"], dtype=float)):
        level = int(data["mb_logical"][block, 3])
        shape = np.asarray(data["mb_data"][names[0]][block]).shape
        require(shape[0] == 1, "expected collapsed Cartoon data")
        rho = np.linspace(bounds[0], bounds[1], shape[2])
        zed = np.linspace(bounds[2], bounds[3], shape[1])
        arrays = [np.asarray(data["mb_data"][name][block][0], dtype=float) for name in names]
        for j, z in enumerate(zed):
            if z < -limit - 1.0e-13 or z > limit + 1.0e-13:
                continue
            for i, r in enumerate(rho):
                if r < -1.0e-13 or r > limit + 1.0e-13:
                    continue
                key = (float(r), float(z))
                value = np.asarray([array[j, i] for array in arrays])
                if r == 0.0:
                    axis_coordinates.add(float(z))
                previous = points.get(key)
                if previous is not None:
                    spread = float(np.max(np.abs(previous[1] - value)))
                    if previous[0] == level:
                        same_level_spread = max(same_level_spread, spread)
                    else:
                        cross_level_spread = max(cross_level_spread, spread)
                if previous is None or level > previous[0]:
                    points[key] = (level, value)
    require(points, "empty inner point set")
    return ({key: value for key, (_, value) in points.items()},
            {"names": names, "same_level_spread": same_level_spread,
             "cross_level_spread": cross_level_spread,
             "axis_coordinates": sorted(axis_coordinates)})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--small", type=Path, required=True)
    parser.add_argument("--large", type=Path, required=True)
    parser.add_argument("--limit", type=float, default=16.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reader = load_reader(args.source)
    small, small_meta = canonical_inner(reader(str(args.small)), args.limit)
    large, large_meta = canonical_inner(reader(str(args.large)), args.limit)
    small_keys, large_keys = set(small), set(large)
    require(small_keys == large_keys,
            f"inner vertex sets differ: small-only={len(small_keys-large_keys)}, "
            f"large-only={len(large_keys-small_keys)}")
    require(small_meta["axis_coordinates"] == large_meta["axis_coordinates"],
            "axis vertex coordinates differ")
    names = small_meta["names"]
    require(names == large_meta["names"], "state variable ordering differs")
    maximum = {name: 0.0 for name in names}
    rms_sum = {name: 0.0 for name in names}
    finite = True
    for key in sorted(small):
        difference = small[key] - large[key]
        finite = finite and bool(np.isfinite(difference).all())
        for index, name in enumerate(names):
            value = float(abs(difference[index]))
            maximum[name] = max(maximum[name], value)
            rms_sum[name] += value * value
    count = len(small)
    result = {
        "schema": "z4c_vc_inner_initial_equivalence_v1",
        "small": str(args.small.resolve()),
        "large": str(args.large.resolve()),
        "limit": args.limit,
        "vertices": count,
        "axis_vertices": len(small_meta["axis_coordinates"]),
        "coordinates_identical": True,
        "axis_placement_identical": True,
        "all_differences_finite": finite,
        "small_shared_spread": {
            "same_level": small_meta["same_level_spread"],
            "cross_level": small_meta["cross_level_spread"],
        },
        "large_shared_spread": {
            "same_level": large_meta["same_level_spread"],
            "cross_level": large_meta["cross_level_spread"],
        },
        "maximum_absolute_difference": maximum,
        "rms_difference": {name: math.sqrt(value / count)
                           for name, value in rms_sum.items()},
        "maximum_over_all_variables": max(maximum.values()),
    }
    require(finite, "nonfinite state difference")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
