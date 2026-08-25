#!/usr/bin/env python3
"""Analyze 64/96/128 Ref-GH perturbed-trumpet self-convergence.

The AthenaK coarsened-binary files are read at coarsen_factor=1, assembled into
uniform global arrays, and interpolated with sixth-order tensor-product
Lagrange interpolation to a fixed cell-centered sample grid.  This keeps the
comparison operator higher order than the nominal fourth-order evolution.
"""

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
        metadata = {}
        variables = []
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
        variable_size = metadata.get("variable_size")
        if metadata.get("location_size") != 8 or variable_size not in (4, 8):
            raise ValueError(
                f"{path}: expected binary64 locations and binary32/binary64 fields")
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
            count = nvar*shape[0]*shape[1]*shape[2]
            raw = stream.read(variable_size*count)
            if len(raw) != variable_size*count:
                raise ValueError(f"{path}: truncated MeshBlock field data")
            dtype = "<f8" if variable_size == 8 else "<f4"
            data = np.frombuffer(raw, dtype=dtype).astype(np.float64)
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


def effective_self_order(ratio: float, resolutions) -> float:
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
                 mask: np.ndarray, resolutions) -> dict:
    first = (coarse - medium)[:, mask]
    second = (medium - fine)[:, mask]
    first_l2 = float(np.sqrt(np.mean(first*first)))
    second_l2 = float(np.sqrt(np.mean(second*second)))
    first_linf = float(np.max(np.abs(first)))
    second_linf = float(np.max(np.abs(second)))
    ratio_l2 = first_l2/second_l2 if second_l2 > 0.0 else math.nan
    ratio_linf = first_linf/second_linf if second_linf > 0.0 else math.nan
    n0, n1, n2 = resolutions
    return {
        f"difference_{n0}_{n1}_L2": first_l2,
        f"difference_{n1}_{n2}_L2": second_l2,
        "ratio_L2": ratio_l2,
        "order_L2": effective_self_order(ratio_l2, resolutions),
        f"difference_{n0}_{n1}_Linf": first_linf,
        f"difference_{n1}_{n2}_Linf": second_linf,
        "ratio_Linf": ratio_linf,
        "order_Linf": effective_self_order(ratio_linf, resolutions),
    }


def maximum_summary(coarse: np.ndarray, medium: np.ndarray, fine: np.ndarray,
                    mask: np.ndarray, resolutions, variable_names,
                    coordinates) -> dict:
    """Locate masked inter-resolution Linf differences on the target grid."""
    result = {}
    for left, right, n_left, n_right in (
            (coarse, medium, resolutions[0], resolutions[1]),
            (medium, fine, resolutions[1], resolutions[2])):
        difference = np.where(mask[None, ...], np.abs(left - right), -np.inf)
        variable, k, j, i = np.unravel_index(
            int(np.argmax(difference)), difference.shape)
        x = float(coordinates[0][i])
        y = float(coordinates[1][j])
        z = float(coordinates[2][k])
        result[f"difference_{n_left}_{n_right}"] = {
            "Linf": float(difference[variable, k, j, i]),
            "variable": variable_names[variable],
            "coordinate": [x, y, z],
            "radius": math.sqrt(x*x + y*y + z*z),
            "target_index_ijk": [int(i), int(j), int(k)],
        }
    return result


def load_triplet(paths, target_n: int):
    loaded = [read_cbin(path) for path in paths]
    variables = loaded[0]["variables"]
    bounds = loaded[0]["bounds"]
    for item in loaded[1:]:
        if (item["variables"] != variables or item["bounds"] != bounds
                or item["variable_size"] != loaded[0]["variable_size"]):
            raise ValueError(
                "triplet variable labels, precision, or physical domains differ")
    return variables, bounds, [interpolate(item["data"], bounds, target_n)
                               for item in loaded], loaded


def puncture_clear_mask(metadata: dict, target_n: int, stencil_radius: int,
                        puncture) -> np.ndarray:
    """Mask target samples whose complete source support avoids the puncture.

    A target interpolation depends on a tensor-product set of source cells.
    It is rejected if any contributing source cell has an axis-aligned
    finite-difference support box containing the puncture.
    """
    source_shape = metadata["data"].shape[-3:][::-1]
    support_near = []
    for axis, (lower, upper) in enumerate(metadata["bounds"]):
        n_source = source_shape[axis]
        source = lower + (np.arange(n_source) + 0.5)*(upper - lower)/n_source
        weights = interpolation_matrix(n_source, target_n, lower, upper)
        reach = stencil_radius*(upper - lower)/n_source
        contributes = np.abs(weights) > 32.0*np.finfo(np.float64).eps
        support_near.append(np.any(
            contributes & (np.abs(source[None, :] - puncture[axis]) <= reach),
            axis=1))
    invalid = (support_near[2][:, None, None]
               & support_near[1][None, :, None]
               & support_near[0][None, None, :])
    return ~invalid


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--field", nargs=3, type=Path, required=True,
                        metavar=("N64", "N96", "N128"))
    parser.add_argument("--constraint", nargs=3, type=Path, required=True,
                        metavar=("N64", "N96", "N128"))
    parser.add_argument("--target-n", type=int, default=32)
    parser.add_argument("--analysis-radius", type=float, default=1.0)
    parser.add_argument("--analysis-inner-radius", type=float, default=0.0)
    parser.add_argument(
        "--fd-stencil-radius", type=int, required=True,
        help=("maximum cell radius of every relevant stencil; for fourth-order "
              "Ref-GH with nonzero KO dissipation this is 3, not 2"))
    parser.add_argument("--puncture", nargs=3, type=float, default=(0.0, 0.0, 0.0),
                        metavar=("X", "Y", "Z"))
    parser.add_argument("--expected-time", type=float)
    parser.add_argument("--resolutions", nargs=3, type=int, default=(64, 96, 128),
                        metavar=("N0", "N1", "N2"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not (0.0 <= args.analysis_inner_radius < args.analysis_radius):
        raise ValueError("require 0 <= analysis-inner-radius < analysis-radius")
    if args.fd_stencil_radius < 1:
        raise ValueError("fd-stencil-radius must be positive")

    field_names, bounds, fields, field_meta = load_triplet(args.field, args.target_n)
    constraint_names, constraint_bounds, constraints, constraint_meta = load_triplet(
        args.constraint, args.target_n)
    if bounds != constraint_bounds:
        raise ValueError("field and constraint domains differ")
    output_times = [item["time"] for item in field_meta + constraint_meta]
    if args.expected_time is not None and any(
            abs(time - args.expected_time) > 1.0e-12 for time in output_times):
        raise ValueError(
            f"output times {output_times} do not match {args.expected_time}")
    coordinates = [lower + (np.arange(args.target_n) + 0.5)*(upper - lower)
                   /args.target_n for lower, upper in bounds]
    z, y, x = np.meshgrid(coordinates[2], coordinates[1], coordinates[0],
                          indexing="ij")
    radius2 = x*x + y*y + z*z
    mask = ((radius2 >= args.analysis_inner_radius*args.analysis_inner_radius)
            & (radius2 < args.analysis_radius*args.analysis_radius))
    clear_masks = [puncture_clear_mask(
        item, args.target_n, args.fd_stencil_radius, args.puncture)
        for item in field_meta + constraint_meta]
    for clear in clear_masks:
        mask &= clear
    if not np.any(mask):
        raise ValueError("puncture-support and radius masks remove every sample")

    variable_size = field_meta[0]["variable_size"]
    constraint_variable_size = constraint_meta[0]["variable_size"]
    binary64 = variable_size == 8 and constraint_variable_size == 8
    dynamic_fields = [array[2:] for array in fields]
    result = {
        "method": "sixth-order interpolation to fixed cell-centered grid",
        "field_precision": "binary64" if binary64 else "binary32",
        "target_n": args.target_n,
        "analysis_radius": args.analysis_radius,
        "analysis_inner_radius": args.analysis_inner_radius,
        "analysis_region": {
            "inner_radius_inclusive": args.analysis_inner_radius,
            "outer_radius_exclusive": args.analysis_radius,
        },
        "analysis_sample_count": int(np.count_nonzero(mask)),
        "puncture_stencil_mask": {
            "enabled": True,
            "fd_stencil_radius": args.fd_stencil_radius,
            "puncture": args.puncture,
            "rule": ("reject a target if any tensor-interpolation source cell "
                     "has a finite-difference support box containing the puncture"),
            "scope": ("caller-supplied maximum footprint across diagnostic and "
                      "evolution operators, including KO dissipation"),
        },
        "resolutions": args.resolutions,
        "field_variables": field_names,
        "primary_field_variables": field_names if binary64 else field_names[2:],
        "constraint_variables": constraint_names,
        "times": {
            "field": [item["time"] for item in field_meta],
            "constraint": [item["time"] for item in constraint_meta],
        },
        "field": norm_summary(*fields, mask, tuple(args.resolutions)),
        "field_difference_maxima": maximum_summary(
            *fields, mask, tuple(args.resolutions), field_names, coordinates),
        "dynamic_field": norm_summary(
            *dynamic_fields, mask, tuple(args.resolutions)),
        "dynamic_field_difference_maxima": maximum_summary(
            *dynamic_fields, mask, tuple(args.resolutions), field_names[2:],
            coordinates),
        "native_constraint": norm_summary(
            *constraints, mask, tuple(args.resolutions)),
        "native_constraint_difference_maxima": maximum_summary(
            *constraints, mask, tuple(args.resolutions), constraint_names,
            coordinates),
        "psi": norm_summary(
            fields[0][:2], fields[1][:2], fields[2][:2], mask,
            tuple(args.resolutions)),
        "field_per_variable": {
            name: norm_summary(
                fields[0][index:index + 1], fields[1][index:index + 1],
                fields[2][index:index + 1], mask, tuple(args.resolutions))
            for index, name in enumerate(field_names)
        },
        "constraint_per_variable": {
            name: norm_summary(
                constraints[0][index:index + 1],
                constraints[1][index:index + 1],
                constraints[2][index:index + 1], mask,
                tuple(args.resolutions))
            for index, name in enumerate(constraint_names)
        },
    }
    if not binary64:
        result["precision_limitation"] = (
            "Psi is secondary because the cbin payload is binary32")
        result["psi_binary32_secondary"] = result["psi"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
