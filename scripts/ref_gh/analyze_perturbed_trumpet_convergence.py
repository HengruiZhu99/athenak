#!/usr/bin/env python3
"""Analyze 64/96/128 Ref-GH perturbed-trumpet self-convergence.

The AthenaK coarsened-binary files are read at coarsen_factor=1, assembled into
uniform global arrays, and interpolated with sixth-order tensor-product
Lagrange interpolation to a fixed cell-centered sample grid.  This keeps the
comparison operator higher order than the nominal fourth-order evolution.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
from pathlib import Path

import numpy as np


def _header_value(line: bytes) -> str:
    return line.decode("ascii").split("=", 1)[1].strip()


def read_cbin(path: Path) -> dict:
    """Read one uniform-mesh AthenaK cbin v1.1 file."""
    with path.open("rb") as stream:
        first = stream.readline()
        if first.strip() != b"Athena binary output version=1.1":
            raise ValueError(f"{path}: unsupported cbin header {first!r}")
        metadata: dict[str, object] = {}
        variables: list[str] = []
        input_bytes = None
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"{path}: truncated text header")
            stripped = line.strip()
            if stripped.startswith(b"time="):
                metadata["time"] = float(_header_value(stripped))
            elif stripped.startswith(b"cycle="):
                metadata["cycle"] = int(_header_value(stripped))
            elif stripped.startswith(b"size of location="):
                metadata["location_size"] = int(_header_value(stripped))
            elif stripped.startswith(b"size of variable="):
                metadata["variable_size"] = int(_header_value(stripped))
            elif stripped.startswith(b"number of variables="):
                metadata["nvar"] = int(_header_value(stripped))
            elif stripped.startswith(b"variables:"):
                variables = stripped.split(b":", 1)[1].decode("ascii").split()
            elif stripped.startswith(b"header offset="):
                input_bytes = int(_header_value(stripped))
                break
        if input_bytes is None:
            raise ValueError(f"{path}: missing input-header size")
        stream.read(input_bytes)
        if metadata.get("location_size") != 8 or metadata.get("variable_size") != 4:
            raise ValueError(f"{path}: expected binary64 locations and binary32 fields")
        nvar = int(metadata["nvar"])
        if len(variables) != nvar:
            raise ValueError(f"{path}: variable label/count mismatch")

        blocks = []
        record_prefix = struct.Struct("<10i6d")
        while True:
            prefix = stream.read(record_prefix.size)
            if not prefix:
                break
            if len(prefix) != record_prefix.size:
                raise ValueError(f"{path}: truncated MeshBlock prefix")
            values = record_prefix.unpack(prefix)
            ois, oie, ojs, oje, oks, oke, lx1, lx2, lx3, level = values[:10]
            bounds = values[10:]
            shape = (oke - oks + 1, oje - ojs + 1, oie - ois + 1)
            count = nvar*math.prod(shape)
            raw = stream.read(4*count)
            if len(raw) != 4*count:
                raise ValueError(f"{path}: truncated MeshBlock field data")
            data = np.frombuffer(raw, dtype="<f4").astype(np.float64)
            data = data.reshape((nvar,) + shape)
            blocks.append(((lx1, lx2, lx3, level), bounds, data))

    if not blocks or any(block[0][3] != 0 for block in blocks):
        raise ValueError(f"{path}: analyzer requires a non-AMR level-0 mesh")
    local_shape = blocks[0][2].shape[1:]
    roots = tuple(max(block[0][axis] for block in blocks) + 1 for axis in range(3))
    expected_blocks = roots[0]*roots[1]*roots[2]
    if len(blocks) != expected_blocks:
        raise ValueError(f"{path}: missing blocks ({len(blocks)} != {expected_blocks})")
    nz, ny, nx = local_shape
    global_data = np.empty((nvar, roots[2]*nz, roots[1]*ny, roots[0]*nx))
    mins = [min(block[1][2*axis] for block in blocks) for axis in range(3)]
    maxs = [max(block[1][2*axis + 1] for block in blocks) for axis in range(3)]
    for (lx1, lx2, lx3, _), _, data in blocks:
        global_data[:, lx3*nz:(lx3 + 1)*nz,
                    lx2*ny:(lx2 + 1)*ny,
                    lx1*nx:(lx1 + 1)*nx] = data
    metadata.update(variables=variables, data=global_data,
                    bounds=tuple(zip(mins, maxs)))
    return metadata


def interpolation_matrix(n_source: int, n_target: int,
                         lower: float, upper: float, width: int = 6) -> np.ndarray:
    if n_source < width:
        raise ValueError("source grid is too small for requested interpolation")
    source = lower + (np.arange(n_source) + 0.5)*(upper - lower)/n_source
    target = lower + (np.arange(n_target) + 0.5)*(upper - lower)/n_target
    matrix = np.zeros((n_target, n_source), dtype=np.float64)
    dx = (upper - lower)/n_source
    for row, coordinate in enumerate(target):
        nearest = int(round((coordinate - lower)/dx - 0.5))
        start = max(0, min(n_source - width, nearest - width//2 + 1))
        indices = np.arange(start, start + width)
        nodes = source[indices]
        weights = np.ones(width, dtype=np.float64)
        for a in range(width):
            for b in range(width):
                if a != b:
                    weights[a] *= (coordinate - nodes[b])/(nodes[a] - nodes[b])
        matrix[row, indices] = weights
    return matrix


def interpolate(data: np.ndarray, bounds, target_n: int) -> np.ndarray:
    nz, ny, nx = data.shape[-3:]
    wx = interpolation_matrix(nx, target_n, *bounds[0])
    wy = interpolation_matrix(ny, target_n, *bounds[1])
    wz = interpolation_matrix(nz, target_n, *bounds[2])
    along_x = np.einsum("vzyx,ax->vzya", data, wx, optimize=True)
    along_y = np.einsum("vzya,by->vzba", along_x, wy, optimize=True)
    return np.einsum("vzba,cz->vcba", along_y, wz, optimize=True)


def effective_self_order(ratio: float, resolutions: tuple[int, int, int]) -> float:
    """Solve unequal-ratio Richardson self-convergence for p."""
    h0, h1, h2 = (1.0/resolution for resolution in resolutions)
    if not math.isfinite(ratio) or ratio <= 0.0:
        return math.nan

    def predicted(order: float) -> float:
        return (h0**order - h1**order)/(h1**order - h2**order)

    lo, hi = 0.01, 12.0
    flo = predicted(lo) - ratio
    fhi = predicted(hi) - ratio
    if flo*fhi > 0.0:
        return math.nan
    for _ in range(100):
        mid = 0.5*(lo + hi)
        fmid = predicted(mid) - ratio
        if flo*fmid <= 0.0:
            hi = mid
            fhi = fmid
        else:
            lo = mid
            flo = fmid
    return 0.5*(lo + hi)


def norm_summary(coarse: np.ndarray, medium: np.ndarray, fine: np.ndarray,
                 mask: np.ndarray, resolutions: tuple[int, int, int]) -> dict:
    first = (coarse - medium)[:, mask]
    second = (medium - fine)[:, mask]
    first_l2 = float(np.sqrt(np.mean(first*first)))
    second_l2 = float(np.sqrt(np.mean(second*second)))
    first_linf = float(np.max(np.abs(first)))
    second_linf = float(np.max(np.abs(second)))
    ratio_l2 = first_l2/second_l2 if second_l2 > 0.0 else math.nan
    ratio_linf = first_linf/second_linf if second_linf > 0.0 else math.nan
    return {
        "difference_64_96_L2": first_l2,
        "difference_96_128_L2": second_l2,
        "ratio_L2": ratio_l2,
        "order_L2": effective_self_order(ratio_l2, resolutions),
        "difference_64_96_Linf": first_linf,
        "difference_96_128_Linf": second_linf,
        "ratio_Linf": ratio_linf,
        "order_Linf": effective_self_order(ratio_linf, resolutions),
    }


def load_triplet(paths: list[Path], target_n: int):
    loaded = [read_cbin(path) for path in paths]
    variables = loaded[0]["variables"]
    bounds = loaded[0]["bounds"]
    for item in loaded[1:]:
        if item["variables"] != variables or item["bounds"] != bounds:
            raise ValueError("triplet variable labels or physical domains differ")
    return variables, bounds, [interpolate(item["data"], bounds, target_n)
                               for item in loaded], loaded


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", nargs=3, type=Path, required=True,
                        metavar=("N64", "N96", "N128"))
    parser.add_argument("--constraint", nargs=3, type=Path, required=True,
                        metavar=("N64", "N96", "N128"))
    parser.add_argument("--target-n", type=int, default=32)
    parser.add_argument("--analysis-radius", type=float, default=1.0)
    parser.add_argument("--resolutions", nargs=3, type=int, default=(64, 96, 128),
                        metavar=("N0", "N1", "N2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    field_names, bounds, fields, field_meta = load_triplet(args.field, args.target_n)
    constraint_names, constraint_bounds, constraints, constraint_meta = load_triplet(
        args.constraint, args.target_n)
    if bounds != constraint_bounds:
        raise ValueError("field and constraint domains differ")
    coordinates = [lower + (np.arange(args.target_n) + 0.5)*(upper - lower)
                   /args.target_n for lower, upper in bounds]
    z, y, x = np.meshgrid(coordinates[2], coordinates[1], coordinates[0],
                          indexing="ij")
    mask = x*x + y*y + z*z < args.analysis_radius*args.analysis_radius

    # Psi contains an O(1) background and cbin stores binary32. Dynamic Pi/Phi
    # components retain roughly seven significant digits of the perturbation,
    # so they are the primary field-convergence measure.
    dynamic_fields = [array[2:] for array in fields]
    result = {
        "method": "sixth-order interpolation to fixed cell-centered grid",
        "target_n": args.target_n,
        "analysis_radius": args.analysis_radius,
        "resolutions": args.resolutions,
        "field_variables": field_names,
        "primary_field_variables": field_names[2:],
        "constraint_variables": constraint_names,
        "times": {
            "field": [item["time"] for item in field_meta],
            "constraint": [item["time"] for item in constraint_meta],
        },
        "dynamic_field": norm_summary(
            *dynamic_fields, mask, tuple(args.resolutions)),
        "native_constraint": norm_summary(
            *constraints, mask, tuple(args.resolutions)),
        "psi_binary32_secondary": norm_summary(
            fields[0][:2], fields[1][:2], fields[2][:2], mask,
            tuple(args.resolutions)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
