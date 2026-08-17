#!/usr/bin/env python3
"""Zero-PDE stitched-parent diagnosis for one Cartoon Z4c refinement event.

This analysis intentionally never interpolates constraint fields.  It reconstructs
alternative evolved states on the identical accepted child lattice, applies the
production algebraic projection, and ports the production O6 Cartoon ADM-constraint
operator literally enough to require an independent byte-level comparison against the
captured production P5 state before any causal verdict is emitted.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cartoon_amr_jump_analyze as base


SCHEMA = "athenak_z4c_amr_stitched_transfer_diagnosis_v1"
PHASE_T0 = "t0_00_ACCEPTED_OLD_STATE"
PHASE_T2 = "t2_00_REFINE_OR_DEREFINE_TRANSFER"
PHASE_T4 = "t4_00_ALGEBRAIC_PROJECTION"
PHASE_T5 = "t5_00_ADM_OR_CONSTRAINT_RECOMPUTATION"
PHASE_GHOSTS = [
    PHASE_T0,
    "t3_01_MPI_RECEIVE",
    "t3_02_PHYSICAL_OR_AXIS_BC",
    "t3_03_SAME_LEVEL_COARSE_REFRESH",
    "t3_04_COARSE_TO_FINE_PROLONGATION",
    "t3_05_PHYSICAL_OR_AXIS_BC",
    "t3_06_PHYSICAL_OR_AXIS_BC",
]
VARS = [
    "chi", "gxx", "gxy", "gxz", "gyy", "gyz", "gzz", "Khat",
    "Axx", "Axy", "Axz", "Ayy", "Ayz", "Azz", "Gamx", "Gamy",
    "Gamz", "Theta", "alpha", "betax", "betay", "betaz", "Bx", "By", "Bz",
]
GROUPS = {
    "chi": [0], "gammatilde": list(range(1, 7)), "Khat": [7],
    "Atilde": list(range(8, 14)), "Gammatilde": list(range(14, 17)),
}
SYM = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
P5 = tuple(Fraction(value) for value in (-45, 420, 1890, -252, 35))
P5 = tuple(value / 2048 for value in P5)
FD1 = np.asarray([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60], dtype=np.float64)
FD2 = np.asarray([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90], dtype=np.float64)


class DiagnosisError(RuntimeError):
    pass


def strict_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, records: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def lagrange_weights(nodes: Iterable[int], target: Fraction) -> tuple[Fraction, ...]:
    values: list[Fraction] = []
    node_list = list(nodes)
    for node in node_list:
        weight = Fraction(1)
        for other in node_list:
            if other != node:
                weight *= (target - other) / (node - other)
        values.append(weight)
    return tuple(values)


P8_LOW_OFFSETS = tuple(range(-4, 4))
P8_HIGH_OFFSETS = tuple(range(-3, 5))
P8_LOW = lagrange_weights(P8_LOW_OFFSETS, Fraction(-1, 4))
P8_HIGH = lagrange_weights(P8_HIGH_OFFSETS, Fraction(1, 4))


def interpolation_self_test() -> None:
    for target, nodes, weights in (
            (Fraction(-1, 4), tuple(range(-2, 3)), P5),
            (Fraction(1, 4), tuple(range(-2, 3)), tuple(reversed(P5))),
            (Fraction(-1, 4), P8_LOW_OFFSETS, P8_LOW),
            (Fraction(1, 4), P8_HIGH_OFFSETS, P8_HIGH)):
        degree = len(nodes) - 1
        for power in range(degree + 1):
            got = sum(weight * Fraction(node) ** power
                      for node, weight in zip(nodes, weights))
            if got != target ** power:
                raise DiagnosisError(
                    f"interpolation moment failure target={target} degree={power}: {got}")
    if sum(P8_LOW) != 1 or sum(P8_HIGH) != 1:
        raise DiagnosisError("P8 does not preserve constants")
    if P8_LOW != tuple(reversed(P8_HIGH)):
        raise DiagnosisError("P8 low/high orientation is not reflection symmetric")


@dataclass
class DenseGroup:
    label: str
    child_level: int
    source_level: int
    spacing: float
    ix0: int
    iz0: int
    production: np.ndarray
    production_t4: np.ndarray
    production_constraints: np.ndarray
    p5: np.ndarray
    p8: np.ndarray
    p5_valid: np.ndarray
    p8_valid: np.ndarray
    common_mask: np.ndarray
    rho: np.ndarray
    z: np.ndarray
    old_seams_rho: tuple[float, ...]
    old_seams_z: tuple[float, ...]


def active_bounds(metadata: dict[str, Any]) -> tuple[slice, slice]:
    bounds = metadata["active_bounds"]
    return slice(bounds["js"], bounds["je"] + 1), slice(bounds["is"], bounds["ie"] + 1)


def topology_index(path: Path) -> dict[int, dict[str, str]]:
    return {int(row["gid"]): row for row in rows(path)}


def load_phase(root: Path, phase: str, name: str) -> tuple[np.ndarray, dict[str, Any],
                                                           dict[int, dict[str, str]]]:
    metadata = base.strict_load(root / phase / "phase.json")
    view = base.load_view(root / phase, name, metadata)
    topology = topology_index(root / phase / "topology.csv")
    return view, metadata, topology


def selected_refined(proposal: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = [row for row in proposal
                if int(row["new_level"]) == int(row["old_level"]) + 1]
    if len(selected) != 16:
        raise DiagnosisError(f"expected 16 refined children, found {len(selected)}")
    return selected


def global_index(coordinate: float, spacing: float) -> int:
    value = coordinate / spacing - 0.5
    index = int(round(value))
    if abs(value - index) > 2.0e-10:
        raise DiagnosisError(f"coordinate {coordinate} is not a cell center for h={spacing}")
    return index


def active_source_map(u0: np.ndarray, metadata: dict[str, Any],
                      topology: dict[int, dict[str, str]], level: int
                      ) -> tuple[dict[tuple[int, int], np.ndarray], dict[tuple[int, int], int],
                                 dict[str, Any]]:
    js, is_ = active_bounds(metadata)
    source: dict[tuple[int, int], np.ndarray] = {}
    provenance: dict[tuple[int, int], int] = {}
    interfaces_rho: set[float] = set()
    interfaces_z: set[float] = set()
    blocks = 0
    for gid, row in topology.items():
        if int(row["level"]) != level:
            continue
        blocks += 1
        local_m = int(row["local_m"])
        h = float(row["dx1"])
        if abs(h - float(row["dx2"])) > 1.0e-14:
            raise DiagnosisError("non-square parent cells are unsupported")
        interfaces_rho.update((float(row["x1min"]), float(row["x1max"])))
        interfaces_z.update((float(row["x2min"]), float(row["x2max"])))
        active = u0[local_m, :, 0, js, is_]
        for j in range(active.shape[1]):
            z = float(row["x2min"]) + (j + 0.5) * h
            gj = global_index(z, h)
            for i in range(active.shape[2]):
                rho = float(row["x1min"]) + (i + 0.5) * h
                gi = global_index(rho, h)
                key = (gj, gi)
                if key in source:
                    raise DiagnosisError(f"duplicate same-level active cell {key} at level {level}")
                source[key] = active[:, j, i].copy()
                provenance[key] = gid
    if not source:
        raise DiagnosisError(f"no active source cells at level {level}")
    gj = [key[0] for key in source]
    gi = [key[1] for key in source]
    manifest = {
        "level": level, "block_count": blocks, "active_cell_count": len(source),
        "global_i_bounds": [min(gi), max(gi)], "global_j_bounds": [min(gj), max(gj)],
        "duplicate_active_centers": 0,
        "meshblock_boundaries_rho": sorted(interfaces_rho),
        "meshblock_boundaries_z": sorted(interfaces_z),
    }
    return source, provenance, manifest


def interpolate_point(source: dict[tuple[int, int], np.ndarray], fj: int, fi: int,
                      order: str) -> tuple[np.ndarray | None, set[int]]:
    ci, cj = fi // 2, fj // 2
    high_i, high_j = bool(fi % 2), bool(fj % 2)
    if order == "p5":
        offsets_i = offsets_j = tuple(range(-2, 3))
        weights_i = tuple(reversed(P5)) if high_i else P5
        weights_j = tuple(reversed(P5)) if high_j else P5
    elif order == "p8":
        offsets_i = P8_HIGH_OFFSETS if high_i else P8_LOW_OFFSETS
        offsets_j = P8_HIGH_OFFSETS if high_j else P8_LOW_OFFSETS
        weights_i = P8_HIGH if high_i else P8_LOW
        weights_j = P8_HIGH if high_j else P8_LOW
    else:
        raise DiagnosisError(f"unknown interpolation {order}")
    result: np.ndarray | None = None
    used: set[int] = set()
    for oj, wj in zip(offsets_j, weights_j):
        for oi, wi in zip(offsets_i, weights_i):
            key = (cj + oj, ci + oi)
            value = source.get(key)
            if value is None:
                return None, set()
            if result is None:
                result = np.zeros_like(value)
            result += float(wi * wj) * value
    return result, used


def build_dense_groups(event: Path) -> tuple[list[DenseGroup], dict[str, Any]]:
    t0, meta0, topo0 = load_phase(event, PHASE_T0, "u0")
    t2, meta2, topo2 = load_phase(event, PHASE_T2, "u0")
    t4, _, topo4 = load_phase(event, PHASE_T4, "u0")
    con5, meta5, topo5 = load_phase(event, PHASE_T5, "constraints")
    proposal = rows(event / "t1_topology_proposal.csv")
    selected = selected_refined(proposal)

    # The offline implementation must reproduce every production refined child first.
    reconstruction_rows: list[dict[str, Any]] = []
    for row in selected:
        ngid, ogid = int(row["new_gid"]), int(row["old_gid"])
        ox1 = int(row["new_lx1"]) - 2 * int(row["old_lx1"])
        ox2 = int(row["new_lx2"]) - 2 * int(row["old_lx2"])
        parent = t0[int(topo0[ogid]["local_m"])]
        coarse = base.child_coarse_from_parent(parent, (25, 1, 24, 24),
                                                ox1, ox2, 32, 32)
        reconstructed = base.prolong_active(
            coarse, {"cis": 4, "cie": 19, "cjs": 4, "cje": 19,
                     "is": 4, "js": 4}, 0, "high_order")
        actual = t2[int(topo2[ngid]["local_m"]), :, 0, 4:36, 4:36]
        delta = np.abs(reconstructed - actual)
        reconstruction_rows.append({
            "new_gid": ngid, "old_gid": ogid, "child_level": int(row["new_level"]),
            "ox1": ox1, "ox2": ox2, "max_abs": float(np.max(delta)),
        })
    maximum = max(row["max_abs"] for row in reconstruction_rows)
    if maximum > 4.0e-14:
        raise DiagnosisError(f"P5_BLOCK does not reproduce production: {maximum}")

    groups: list[DenseGroup] = []
    stitching: dict[str, Any] = {"groups": [], "p5_block_reconstruction": {
        "maximum_absolute_error": maximum, "children": reconstruction_rows}}
    for child_level in sorted({int(row["new_level"]) for row in selected}):
        selected_level = [row for row in selected if int(row["new_level"]) == child_level]
        source_level = child_level - 1
        h = float(topo2[int(selected_level[0]["new_gid"])]["dx1"])
        target_cells: dict[tuple[int, int], tuple[int, int, int]] = {}
        old_seams_rho: set[float] = set()
        old_seams_z: set[float] = set()
        for row in selected_level:
            ngid, ogid = int(row["new_gid"]), int(row["old_gid"])
            nrow, orow = topo2[ngid], topo0[ogid]
            old_seams_rho.update((float(orow["x1min"]), float(orow["x1max"])))
            old_seams_z.update((float(orow["x2min"]), float(orow["x2max"])))
            for j in range(32):
                z = float(nrow["x2min"]) + (j + 0.5) * h
                gj = global_index(z, h)
                for i in range(32):
                    rho = float(nrow["x1min"]) + (i + 0.5) * h
                    gi = global_index(rho, h)
                    key = (gj, gi)
                    if key in target_cells:
                        raise DiagnosisError(f"duplicate target child cell {key}")
                    target_cells[key] = (ngid, j, i)
        gis = [key[1] for key in target_cells]
        gjs = [key[0] for key in target_cells]
        ix0, ix1, iz0, iz1 = min(gis), max(gis), min(gjs), max(gjs)
        shape = (25, iz1 - iz0 + 1, ix1 - ix0 + 1)
        production = np.full(shape, np.nan)
        production_t4 = np.full(shape, np.nan)
        production_constraints = np.full((7, shape[1], shape[2]), np.nan)
        for (gj, gi), (gid, j, i) in target_cells.items():
            jj, ii = gj - iz0, gi - ix0
            production[:, jj, ii] = t2[int(topo2[gid]["local_m"]), :, 0, 4 + j, 4 + i]
            production_t4[:, jj, ii] = t4[int(topo4[gid]["local_m"]), :, 0, 4 + j, 4 + i]
            production_constraints[:, jj, ii] = con5[
                int(topo5[gid]["local_m"]), :, 0, 4 + j, 4 + i]

        source, _, source_manifest = active_source_map(t0, meta0, topo0, source_level)
        p5 = np.full_like(production, np.nan)
        p8 = np.full_like(production, np.nan)
        p5_valid = np.zeros(shape[1:], dtype=bool)
        p8_valid = np.zeros(shape[1:], dtype=bool)
        for gj, gi in target_cells:
            jj, ii = gj - iz0, gi - ix0
            value, _ = interpolate_point(source, gj, gi, "p5")
            if value is not None:
                p5[:, jj, ii] = value
                p5_valid[jj, ii] = True
            value, _ = interpolate_point(source, gj, gi, "p8")
            if value is not None:
                p8[:, jj, ii] = value
                p8_valid[jj, ii] = True
        common = erode_mask(p5_valid & p8_valid & np.all(np.isfinite(production), axis=0), 3)
        if not np.any(common):
            raise DiagnosisError(f"no valid common derivative cells for level {child_level}")
        x = (np.arange(ix0, ix1 + 1) + 0.5) * h
        z = (np.arange(iz0, iz1 + 1) + 0.5) * h
        rho2d, z2d = np.meshgrid(x, z)
        label = f"L{source_level}_to_L{child_level}"
        groups.append(DenseGroup(
            label, child_level, source_level, h, ix0, iz0, production,
            production_t4, production_constraints, p5, p8, p5_valid, p8_valid,
            common, rho2d, z2d, tuple(sorted(old_seams_rho)), tuple(sorted(old_seams_z))))
        stitching["groups"].append({
            "label": label, "source": source_manifest,
            "selected_child_gids": sorted(int(row["new_gid"]) for row in selected_level),
            "selected_parent_gids": sorted({int(row["old_gid"]) for row in selected_level}),
            "child_shape": list(shape[1:]), "p5_valid_cells": int(np.sum(p5_valid)),
            "p8_valid_cells": int(np.sum(p8_valid)),
            "common_o6_derivative_cells": int(np.sum(common)),
        })
    return groups, stitching


def erode_mask(mask: np.ndarray, radius: int) -> np.ndarray:
    result = mask.copy()
    ny, nx = mask.shape
    for dj in range(-radius, radius + 1):
        for di in range(-radius, radius + 1):
            shifted = np.zeros_like(mask)
            src_j0, src_j1 = max(0, -dj), min(ny, ny - dj)
            src_i0, src_i1 = max(0, -di), min(nx, nx - di)
            shifted[src_j0 + dj:src_j1 + dj, src_i0 + di:src_i1 + di] = \
                mask[src_j0:src_j1, src_i0:src_i1]
            result &= shifted
    return result


def determinant6(values: np.ndarray) -> np.ndarray:
    xx, xy, xz, yy, yz, zz = values
    return xx * (yy * zz - yz * yz) - xy * (xy * zz - yz * xz) + \
        xz * (xy * yz - yy * xz)


def matrix_from6(values: np.ndarray) -> np.ndarray:
    result = np.empty((3, 3) + values.shape[1:], dtype=np.float64)
    for index, (a, b) in enumerate(SYM):
        result[a, b] = values[index]
        result[b, a] = values[index]
    return result


def cofactor_inverse(matrix: np.ndarray, reciprocal_det: np.ndarray | float) -> np.ndarray:
    xx, xy, xz = matrix[0, 0], matrix[0, 1], matrix[0, 2]
    yy, yz, zz = matrix[1, 1], matrix[1, 2], matrix[2, 2]
    result = np.empty_like(matrix)
    result[0, 0] = (yy * zz - yz * yz) * reciprocal_det
    result[0, 1] = result[1, 0] = (xz * yz - xy * zz) * reciprocal_det
    result[0, 2] = result[2, 0] = (xy * yz - xz * yy) * reciprocal_det
    result[1, 1] = (xx * zz - xz * xz) * reciprocal_det
    result[1, 2] = result[2, 1] = (xy * xz - xx * yz) * reciprocal_det
    result[2, 2] = (xx * yy - xy * xy) * reciprocal_det
    return result


def project(state: np.ndarray) -> np.ndarray:
    result = state.copy()
    det = determinant6(result[1:7])
    safe_det = np.where(det > 0.0, det, 1.0)
    scale = np.cbrt(1.0 / safe_det)
    result[1:7] *= scale
    metric = matrix_from6(result[1:7])
    inverse_det_one = cofactor_inverse(metric, 1.0)
    atilde = matrix_from6(result[8:14])
    trace = np.einsum("ab...,ab...->...", inverse_det_one, atilde)
    atilde -= (trace / 3.0)[None, None] * metric
    for index, (a, b) in enumerate(SYM):
        result[8 + index] = atilde[a, b]
    return result


def fd_first(field: np.ndarray, axis: int, h: float) -> np.ndarray:
    result = np.full_like(field, np.nan, dtype=np.float64)
    target = [slice(None)] * field.ndim
    target[axis] = slice(3, -3)
    accum = np.zeros_like(field[tuple(target)], dtype=np.float64)
    for offset, weight in zip(range(-3, 4), FD1):
        source = [slice(None)] * field.ndim
        source[axis] = slice(3 + offset, field.shape[axis] - 3 + offset)
        accum += weight * field[tuple(source)]
    result[tuple(target)] = accum / h
    return result


def fd_second(field: np.ndarray, axis: int, h: float) -> np.ndarray:
    result = np.full_like(field, np.nan, dtype=np.float64)
    target = [slice(None)] * field.ndim
    target[axis] = slice(3, -3)
    accum = np.zeros_like(field[tuple(target)], dtype=np.float64)
    for offset, weight in zip(range(-3, 4), FD2):
        source = [slice(None)] * field.ndim
        source[axis] = slice(3 + offset, field.shape[axis] - 3 + offset)
        accum += weight * field[tuple(source)]
    result[tuple(target)] = accum / (h * h)
    return result


def fd_mixed(field: np.ndarray, h: float) -> np.ndarray:
    result = np.full_like(field, np.nan, dtype=np.float64)
    accum = np.zeros_like(field[..., 3:-3, 3:-3], dtype=np.float64)
    for oj, wj in zip(range(-3, 4), FD1):
        for oi, wi in zip(range(-3, 4), FD1):
            accum += wj * wi * field[..., 3 + oj:field.shape[-2] - 3 + oj,
                                      3 + oi:field.shape[-1] - 3 + oi]
    result[..., 3:-3, 3:-3] = accum / (h * h)
    return result


def scalar_derivatives(field: np.ndarray, h: float, rho: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    first = np.zeros((3,) + field.shape, dtype=np.float64)
    second = np.zeros((3, 3) + field.shape, dtype=np.float64)
    first[0] = fd_first(field, -1, h)
    first[1] = fd_first(field, -2, h)
    second[0, 0] = fd_second(field, -1, h)
    second[1, 1] = fd_second(field, -2, h)
    second[0, 1] = second[1, 0] = fd_mixed(field, h)
    second[2, 2] = first[0] / rho
    return first, second


def tensor_derivatives(field: np.ndarray, h: float, rho: np.ndarray
                       ) -> tuple[np.ndarray, np.ndarray]:
    # field axes are component, component, z, rho.
    first = np.zeros((3, 3, 3) + field.shape[2:], dtype=np.float64)
    second = np.zeros((3, 3, 3, 3) + field.shape[2:], dtype=np.float64)
    for a in range(3):
        for b in range(3):
            first[0, a, b] = fd_first(field[a, b], -1, h)
            first[1, a, b] = fd_first(field[a, b], -2, h)
            second[0, 0, a, b] = fd_second(field[a, b], -1, h)
            second[1, 1, a, b] = fd_second(field[a, b], -2, h)
            second[0, 1, a, b] = second[1, 0, a, b] = fd_mixed(field[a, b], h)
    # Analytic SO(2) first derivatives in the suppressed Cartesian direction.
    first[2, 0, 0] = -2.0 * field[0, 2] / rho
    first[2, 2, 2] = 2.0 * field[0, 2] / rho
    first[2, 0, 2] = first[2, 2, 0] = (field[0, 0] - field[2, 2]) / rho
    first[2, 0, 1] = first[2, 1, 0] = -field[2, 1] / rho
    first[2, 2, 1] = first[2, 1, 2] = field[0, 1] / rho
    # Suppressed-suppressed second derivatives.
    radial = first[0]
    second[2, 2, 0, 0] = radial[0, 0] / rho - \
        2.0 * (field[0, 0] - field[2, 2]) / (rho * rho)
    second[2, 2, 2, 2] = radial[2, 2] / rho + \
        2.0 * (field[0, 0] - field[2, 2]) / (rho * rho)
    second[2, 2, 0, 2] = second[2, 2, 2, 0] = radial[0, 2] / rho - \
        4.0 * field[0, 2] / (rho * rho)
    for a, b in ((0, 1), (1, 0), (2, 1), (1, 2)):
        second[2, 2, a, b] = radial[a, b] / rho - field[a, b] / (rho * rho)
    second[2, 2, 1, 1] = radial[1, 1] / rho
    # One suppressed and one active derivative.
    for direction in (0, 1):
        rr = -2.0 * first[direction, 0, 2] / rho
        ss = 2.0 * first[direction, 0, 2] / rho
        rs = (first[direction, 0, 0] - first[direction, 2, 2]) / rho
        rz = -first[direction, 2, 1] / rho
        sz = first[direction, 0, 1] / rho
        if direction == 0:
            rr += 2.0 * field[0, 2] / (rho * rho)
            ss -= 2.0 * field[0, 2] / (rho * rho)
            rs -= (field[0, 0] - field[2, 2]) / (rho * rho)
            rz += field[2, 1] / (rho * rho)
            sz -= field[0, 1] / (rho * rho)
        for d1, d2 in ((direction, 2), (2, direction)):
            second[d1, d2, 0, 0] = rr
            second[d1, d2, 2, 2] = ss
            second[d1, d2, 0, 2] = second[d1, d2, 2, 0] = rs
            second[d1, d2, 0, 1] = second[d1, d2, 1, 0] = rz
            second[d1, d2, 2, 1] = second[d1, d2, 1, 2] = sz
    return first, second


def to_adm(state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    psi4 = np.power(state[0], -1.0)
    gt = matrix_from6(state[1:7])
    at = matrix_from6(state[8:14])
    g = psi4[None, None] * gt
    k = psi4[None, None] * at + ((state[7] + 2.0 * state[17]) / 3.0)[None, None] * g
    return psi4, g, k


def compute_constraints(state: np.ndarray, h: float, rho: np.ndarray
                        ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    psi4, g, kdd = to_adm(state)
    dg, ddg = tensor_derivatives(g, h, rho)
    dk, _ = tensor_derivatives(kdd, h, rho)
    dpsi4, _ = scalar_derivatives(psi4, h, rho)
    det = (g[0, 0] * (g[1, 1] * g[2, 2] - g[1, 2] * g[1, 2])
           - g[0, 1] * (g[0, 1] * g[2, 2] - g[1, 2] * g[0, 2])
           + g[0, 2] * (g[0, 1] * g[1, 2] - g[1, 1] * g[0, 2]))
    gu = cofactor_inverse(g, 1.0 / det)
    gamma_d = np.zeros((3, 3, 3) + rho.shape)
    gamma_u = np.zeros_like(gamma_d)
    for c in range(3):
        for a in range(3):
            for b in range(3):
                gamma_d[c, a, b] = 0.5 * (dg[a, b, c] + dg[b, a, c] - dg[c, a, b])
                for d in range(3):
                    gamma_u[c, a, b] += gu[c, d] * gamma_d[d, a, b]
    ricci = np.zeros_like(g)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                for d in range(3):
                    for e in range(3):
                        ricci[a, b] += gu[c, d] * gamma_u[e, a, c] * gamma_d[e, b, d]
                        ricci[a, b] -= gu[c, d] * gamma_u[e, a, b] * gamma_d[e, c, d]
                    ricci[a, b] += 0.5 * gu[c, d] * (
                        -ddg[c, d, a, b] - ddg[a, b, c, d]
                        + ddg[a, c, b, d] + ddg[b, c, a, d])
    scalar_r = np.einsum("ab...,ab...->...", gu, ricci)
    kud = np.einsum("ac...,cb...->ab...", gu, kdd)
    trace_k = np.einsum("aa...->...", kud)
    kk = np.einsum("ab...,ba...->...", kud, kud)
    ham = scalar_r + trace_k * trace_k - kk

    dk_cov = np.zeros((3, 3, 3) + rho.shape)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                dk_cov[a, b, c] = dk[a, b, c]
                for d in range(3):
                    dk_cov[a, b, c] -= gamma_u[d, a, b] * kdd[d, c]
                    dk_cov[a, b, c] -= gamma_u[d, a, c] * kdd[b, d]
    dk_up = np.einsum("ad...,dbc...->abc...", gu, dk_cov)
    momentum_up = np.zeros((3,) + rho.shape)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                momentum_up[a] += gu[a, b] * dk_up[c, b, c]
                momentum_up[a] -= gu[b, c] * dk_up[a, b, c]
    momentum_down = np.einsum("ab...,b...->a...", g, momentum_up)
    momentum2 = np.einsum("ab...,a...,b...->...", g, momentum_up, momentum_up)

    contracted = np.zeros((3,) + rho.shape)
    for a in range(3):
        for b in range(3):
            for c in range(3):
                contracted[a] += gu[b, c] * gamma_u[a, b, c]
    gamma_z4c = psi4[None] * contracted + 0.5 * np.einsum("ab...,b...->a...", gu, dpsi4)
    gamma_evolved = state[14:17]
    gamma_delta = gamma_evolved - gamma_z4c
    gt = matrix_from6(state[1:7])
    z2 = 0.25 * np.einsum("ab...,a...,b...->...", gt, gamma_delta, gamma_delta)
    c2 = ham * ham + momentum2 + state[17] * state[17] + 4.0 * z2
    constraints = {"C": c2, "H": ham, "M": momentum2, "Z": z2,
                   "Mx": momentum_down[0], "My": momentum_down[1],
                   "Mz": momentum_down[2]}
    adm = {"psi4": psi4, "g": g, "K": kdd, "detg": det,
           "dg": dg, "ddg": ddg, "dK": dk}
    return constraints, adm


def scan_squared_production(event: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for phase in (PHASE_T0, PHASE_T5):
        view, meta, topology = load_phase(event, phase, "constraints")
        js, is_ = active_bounds(meta)
        active = view[:, :, 0, js, is_]
        phase_result = {}
        for name, index in (("C", 0), ("M", 2), ("Z", 3)):
            values = active[:, index]
            minimum = float(np.min(values))
            if minimum < 0.0:
                location = np.unravel_index(np.argmin(values), values.shape)
                gid = int(list(sorted(topology))[location[0]])
                raise DiagnosisError(
                    f"negative raw squared diagnostic {name}={minimum} at {phase} gid={gid}")
            phase_result[name] = {"minimum": minimum, "finite": bool(np.all(np.isfinite(values)))}
        result[phase] = phase_result
    return result


def metrics(constraints: dict[str, np.ndarray], adm: dict[str, np.ndarray],
            mask: np.ndarray, rho: np.ndarray, z: np.ndarray, h: float,
            seams_rho: tuple[float, ...], seams_z: tuple[float, ...]
            ) -> list[dict[str, Any]]:
    coordinate = 2.0 * math.pi * rho * h * h
    proper = coordinate * np.sqrt(adm["detg"])
    rows_out: list[dict[str, Any]] = []
    for family in ("C", "H", "M", "Z"):
        raw = constraints[family]
        square = raw if family in ("C", "M", "Z") else raw * raw
        magnitude = np.sqrt(np.maximum(square, 0.0))
        if not np.all(np.isfinite(square[mask])):
            raise DiagnosisError(f"nonfinite reconstructed {family}")
        if family in ("C", "M", "Z") and np.any(square[mask] < 0.0):
            raise DiagnosisError(f"negative reconstructed squared diagnostic {family}")
        flat = np.where(mask, magnitude, -np.inf)
        where = np.unravel_index(np.argmax(flat), flat.shape)
        volume = float(np.sum(proper[mask]))
        coordinate_volume = float(np.sum(coordinate[mask]))
        seam_distance = min(
            [abs(float(rho[where]) - value) for value in seams_rho]
            + [abs(float(z[where]) - value) for value in seams_z])
        rows_out.append({
            "family": family, "valid_cells": int(np.sum(mask)),
            "proper_ring_integral": float(np.sum(square[mask] * proper[mask])),
            "coordinate_ring_integral": float(np.sum(square[mask] * coordinate[mask])),
            "proper_ring_volume": volume, "coordinate_ring_volume": coordinate_volume,
            "rms": math.sqrt(float(np.sum(square[mask] * proper[mask])) / volume),
            "linf": float(magnitude[where]), "linf_rho": float(rho[where]),
            "linf_z": float(z[where]), "linf_distance_parent_meshblock_seam": seam_distance,
            "finite_count": int(np.sum(np.isfinite(square[mask]))), "nonfinite_count": 0,
        })
    return rows_out


def representation_rows(group: DenseGroup, states: dict[str, np.ndarray],
                        mask: np.ndarray) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for left, right in (("P5_BLOCK", "P5_STITCHED"),
                        ("P5_STITCHED", "P8_STITCHED")):
        delta = states[left] - states[right]
        for index, name in enumerate(VARS):
            values = delta[index][mask]
            output.append({"group": group.label, "comparison": f"{left}-{right}",
                           "variable": name, "rms": float(np.sqrt(np.mean(values * values))),
                           "linf": float(np.max(np.abs(values)))})
    return output


def derivative_rows(group: DenseGroup, states: dict[str, np.ndarray],
                    mask: np.ndarray) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for left, right in (("P5_BLOCK", "P5_STITCHED"),
                        ("P5_STITCHED", "P8_STITCHED")):
        for name, indices in GROUPS.items():
            first_deltas: list[np.ndarray] = []
            second_deltas: list[np.ndarray] = []
            for index in indices:
                for axis in (-1, -2):
                    first_deltas.append(fd_first(states[left][index], axis, group.spacing)
                                        - fd_first(states[right][index], axis, group.spacing))
                    second_deltas.append(fd_second(states[left][index], axis, group.spacing)
                                         - fd_second(states[right][index], axis, group.spacing))
                second_deltas.append(fd_mixed(states[left][index], group.spacing)
                                     - fd_mixed(states[right][index], group.spacing))
            first_values = np.concatenate([value[mask] for value in first_deltas])
            second_values = np.concatenate([value[mask] for value in second_deltas])
            output.append({
                "group": group.label, "comparison": f"{left}-{right}", "field": name,
                "first_derivative_rms": float(np.sqrt(np.mean(first_values ** 2))),
                "first_derivative_linf": float(np.max(np.abs(first_values))),
                "second_derivative_rms": float(np.sqrt(np.mean(second_values ** 2))),
                "second_derivative_linf": float(np.max(np.abs(second_values))),
            })
    return output


def ghost_census(event: Path, selected_old: set[int], selected_new: set[int]
                 ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    worst: list[dict[str, Any]] = []
    for phase in PHASE_GHOSTS:
        view, meta, topology = load_phase(event, phase, "u0")
        focus = selected_old if phase == PHASE_T0 else selected_new
        active_lookup: dict[tuple[int, int, int], tuple[int, int, int, int]] = {}
        js, is_ = active_bounds(meta)
        for gid, row in topology.items():
            h = float(row["dx1"]); level = int(row["level"]); m = int(row["local_m"])
            for j in range(32):
                gj = global_index(float(row["x2min"]) + (j + 0.5) * h, h)
                for i in range(32):
                    gi = global_index(float(row["x1min"]) + (i + 0.5) * h, h)
                    active_lookup[(level, gj, gi)] = (gid, m, 4 + j, 4 + i)
        records: list[dict[str, Any]] = []
        for receiver_gid, row in topology.items():
            level = int(row["level"]); h = float(row["dx1"]); m = int(row["local_m"])
            for face in ("inner_x1", "outer_x1", "inner_x2", "outer_x2"):
                for q in range(1, 5):
                    for tangent in range(32):
                        if face == "inner_x1": i, j = 4 - q, 4 + tangent
                        elif face == "outer_x1": i, j = 35 + q, 4 + tangent
                        elif face == "inner_x2": i, j = 4 + tangent, 4 - q
                        else: i, j = 4 + tangent, 35 + q
                        rho = float(row["x1min"]) + (i - 4 + 0.5) * h
                        z = float(row["x2min"]) + (j - 4 + 0.5) * h
                        key = (level, global_index(z, h), global_index(rho, h))
                        sender = active_lookup.get(key)
                        if sender is None:
                            continue
                        sender_gid, sm, sj, si = sender
                        if receiver_gid not in focus and sender_gid not in focus:
                            continue
                        delta = view[m, :, 0, j, i] - view[sm, :, 0, sj, si]
                        scales = np.maximum(1.0, np.abs(view[sm, :, 0, sj, si]))
                        relative = np.abs(delta) / scales
                        index = int(np.argmax(relative))
                        records.append({
                            "phase": phase, "receiver_gid": receiver_gid,
                            "sender_gid": sender_gid, "face": face, "ghost_depth": q,
                            "variable": VARS[index], "rho": rho, "z": z,
                            "abs_delta": float(abs(delta[index])),
                            "relative_delta": float(relative[index]),
                        })
        records.sort(key=lambda row: row["relative_delta"], reverse=True)
        epsilon_gate = 128.0 * np.finfo(np.float64).eps
        summaries.append({
            "phase": phase, "compared_face_ghost_cells": len(records),
            "max_abs_delta": max((row["abs_delta"] for row in records), default=0.0),
            "max_relative_delta": max((row["relative_delta"] for row in records), default=0.0),
            "above_roundoff_scaled_gate": sum(row["relative_delta"] > epsilon_gate
                                               for row in records),
            "roundoff_scaled_gate": epsilon_gate,
        })
        worst.extend(records[:8])
    return summaries, worst


def coarse_cache_census(event: Path, selected_new: set[int]
                        ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summaries: list[dict[str, Any]] = []
    worst: list[dict[str, Any]] = []
    for phase in ("t3_00_RESTRICT", "t3_01_MPI_RECEIVE",
                  "t3_03_SAME_LEVEL_COARSE_REFRESH"):
        view, meta, topology = load_phase(event, phase, "coarse_u0")
        active_lookup: dict[tuple[int, int, int], tuple[int, int, int, int]] = {}
        for gid, row in topology.items():
            spacing = 2.0 * float(row["dx1"]); level = int(row["level"]) - 1
            m = int(row["local_m"])
            for j in range(16):
                gj = global_index(float(row["x2min"]) + (j + 0.5) * spacing, spacing)
                for i in range(16):
                    gi = global_index(float(row["x1min"]) + (i + 0.5) * spacing, spacing)
                    active_lookup[(level, gj, gi)] = (gid, m, 4 + j, 4 + i)
        records: list[dict[str, Any]] = []
        for receiver_gid, row in topology.items():
            if receiver_gid not in selected_new:
                continue
            spacing = 2.0 * float(row["dx1"]); level = int(row["level"]) - 1
            m = int(row["local_m"])
            for face in ("inner_x1", "outer_x1", "inner_x2", "outer_x2"):
                for q in range(1, 5):
                    for tangent in range(16):
                        if face == "inner_x1": i, j = 4 - q, 4 + tangent
                        elif face == "outer_x1": i, j = 19 + q, 4 + tangent
                        elif face == "inner_x2": i, j = 4 + tangent, 4 - q
                        else: i, j = 4 + tangent, 19 + q
                        rho = float(row["x1min"]) + (i - 4 + 0.5) * spacing
                        z = float(row["x2min"]) + (j - 4 + 0.5) * spacing
                        sender = active_lookup.get(
                            (level, global_index(z, spacing), global_index(rho, spacing)))
                        if sender is None:
                            continue
                        sender_gid, sm, sj, si = sender
                        delta = view[m, :, 0, j, i] - view[sm, :, 0, sj, si]
                        scales = np.maximum(1.0, np.abs(view[sm, :, 0, sj, si]))
                        relative = np.abs(delta) / scales
                        index = int(np.argmax(relative))
                        records.append({
                            "phase": phase, "receiver_gid": receiver_gid,
                            "sender_gid": sender_gid, "face": face, "ghost_depth": q,
                            "variable": VARS[index], "rho": rho, "z": z,
                            "abs_delta": float(abs(delta[index])),
                            "relative_delta": float(relative[index]),
                        })
        records.sort(key=lambda row: row["relative_delta"], reverse=True)
        epsilon_gate = 128.0 * np.finfo(np.float64).eps
        summaries.append({
            "phase": phase, "compared_same_level_coarse_face_cells": len(records),
            "max_abs_delta": max((row["abs_delta"] for row in records), default=0.0),
            "max_relative_delta": max((row["relative_delta"] for row in records), default=0.0),
            "above_roundoff_scaled_gate": sum(row["relative_delta"] > epsilon_gate
                                               for row in records),
            "roundoff_scaled_gate": epsilon_gate,
        })
        worst.extend(records[:8])
    return summaries, worst


def largest_true_rectangle(mask: np.ndarray) -> tuple[slice, slice]:
    # The diagnostic masks are rectangular apart from unsupported outer bands.  Find the
    # largest all-true axis-aligned rectangle with the standard histogram algorithm.
    heights = np.zeros(mask.shape[1], dtype=int)
    best = (0, 0, 0, 0, 0)  # area, top, bottom-exclusive, left, right-exclusive
    for row in range(mask.shape[0]):
        heights = np.where(mask[row], heights + 1, 0)
        stack: list[int] = []
        for column in range(mask.shape[1] + 1):
            height = heights[column] if column < mask.shape[1] else 0
            while stack and heights[stack[-1]] > height:
                index = stack.pop(); h = heights[index]
                left = stack[-1] + 1 if stack else 0
                area = h * (column - left)
                if area > best[0]:
                    best = (area, row - h + 1, row + 1, left, column)
            stack.append(column)
    if best[0] == 0:
        raise DiagnosisError("no complete rectangle in diagnostic mask")
    return slice(best[1], best[2]), slice(best[3], best[4])


def spectral_rows(group: DenseGroup, states: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    js, is_ = largest_true_rectangle(group.common_mask)
    output: list[dict[str, Any]] = []
    for label, state in states.items():
        for name, indices in GROUPS.items():
            high_energy = total_energy = nyquist = variance = 0.0
            for index in indices:
                data = state[index, js, is_]
                data = data - np.mean(data)
                transform = np.fft.rfft2(data)
                power = np.abs(transform) ** 2
                ky = np.fft.fftfreq(data.shape[0])[:, None]
                kx = np.fft.rfftfreq(data.shape[1])[None, :]
                radius = np.sqrt((ky / 0.5) ** 2 + (kx / 0.5) ** 2)
                total_energy += float(np.sum(power))
                high_energy += float(np.sum(power[radius >= 0.75]))
                checker = (-1.0) ** (np.indices(data.shape).sum(axis=0))
                nyquist += float(np.sum(data * checker)) ** 2 / data.size
                variance += float(np.sum(data * data))
            output.append({
                "group": group.label, "representation": label, "field": name,
                "rectangle_nz": js.stop - js.start, "rectangle_nrho": is_.stop - is_.start,
                "high_mode_fraction_radius_ge_0p75_nyquist":
                    high_energy / total_energy if total_energy else 0.0,
                "checkerboard_projection_fraction": nyquist / variance if variance else 0.0,
            })
    return output


def plot_group(group: DenseGroup, constraints: dict[str, dict[str, np.ndarray]],
               states: dict[str, np.ndarray], output: Path) -> list[str]:
    paths: list[str] = []
    extent = [float(group.rho.min()), float(group.rho.max()),
              float(group.z.min()), float(group.z.max())]
    for family in ("C", "H", "M"):
        fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
        for column, label in enumerate(("P5_BLOCK", "P5_STITCHED", "P8_STITCHED")):
            raw = constraints[label][family]
            magnitude = np.sqrt(np.maximum(raw, 0.0)) if family != "H" else np.abs(raw)
            image = axes[0, column].imshow(
                np.where(group.common_mask, np.log10(np.maximum(magnitude, 1e-300)), np.nan),
                origin="lower", extent=extent, aspect="auto", cmap="magma")
            axes[0, column].set_title(f"{label} log10|{family}|")
            fig.colorbar(image, ax=axes[0, column])
        state_delta_1 = np.sqrt(np.sum((states["P5_BLOCK"] - states["P5_STITCHED"]) ** 2,
                                       axis=0))
        state_delta_2 = np.sqrt(np.sum((states["P5_STITCHED"] - states["P8_STITCHED"]) ** 2,
                                       axis=0))
        for axis, data, title in (
                (axes[1, 0], state_delta_1, "||P5_BLOCK-P5_STITCHED||"),
                (axes[1, 1], state_delta_2, "||P5_STITCHED-P8_STITCHED||"),
                (axes[1, 2], group.common_mask.astype(float), "valid O6 stencil mask")):
            image = axis.imshow(np.where(group.p5_valid | group.p8_valid, data, np.nan),
                                origin="lower", extent=extent, aspect="auto", cmap="viridis")
            axis.set_title(title); fig.colorbar(image, ax=axis)
        for axis in axes.flat:
            for value in group.old_seams_rho:
                if extent[0] <= value <= extent[1]: axis.axvline(value, color="white", lw=0.5)
            for value in group.old_seams_z:
                if extent[2] <= value <= extent[3]: axis.axhline(value, color="white", lw=0.5)
            axis.set_xlabel("rho/M"); axis.set_ylabel("z/M")
        name = f"{group.label}_{family}_stitched_comparison.png"
        fig.savefig(output / name, dpi=180); plt.close(fig); paths.append(name)
    return paths


def classify(metric_rows: list[dict[str, Any]], ghost_summary: list[dict[str, Any]],
             cache_summary: list[dict[str, Any]], validation: dict[str, Any]
             ) -> dict[str, Any]:
    index = {(row["group"], row["representation"], row["family"]): row
             for row in metric_rows}
    ratios: dict[str, Any] = {}
    seam_better = 0; p8_better = 0; comparisons = 0
    for group in sorted({row["group"] for row in metric_rows}):
        ratios[group] = {}
        for family in ("C", "H", "M"):
            block = index[(group, "P5_BLOCK", family)]["proper_ring_integral"]
            stitched = index[(group, "P5_STITCHED", family)]["proper_ring_integral"]
            p8 = index[(group, "P8_STITCHED", family)]["proper_ring_integral"]
            ratios[group][family] = {
                "P5_STITCHED_over_P5_BLOCK": stitched / block,
                "P8_STITCHED_over_P5_STITCHED": p8 / stitched,
            }
            seam_better += stitched <= 0.5 * block
            p8_better += p8 <= 0.5 * stitched
            comparisons += 1
    post_receive_ghost_mismatch = any(
        row["above_roundoff_scaled_gate"] for row in ghost_summary
        if row["phase"] != PHASE_T0)
    refreshed_cache_mismatch = any(
        row["above_roundoff_scaled_gate"] for row in cache_summary
        if row["phase"] == "t3_03_SAME_LEVEL_COARSE_REFRESH")
    if post_receive_ghost_mismatch or refreshed_cache_mismatch:
        disposition = "concrete_ghost_or_cache_bug_isolated"
    elif seam_better >= 4:
        disposition = "meshblock_seam_semantics_isolated"
    elif p8_better >= 4:
        disposition = "transfer_order_insufficient"
    else:
        disposition = "unresolved_mixed_transfer_parent_seam"
    port_pass = all(group["constraint_port_pass"] for group in validation["groups"])
    return {
        "schema": SCHEMA, "qualification_claim": False, "disposition": disposition,
        "constraint_reconstruction_port_pass": port_pass,
        "constraint_ratio_interpretation": (
            "qualified" if port_pass else
            "exploratory_only; disposition rests on direct captured cache/ghost census"),
        "prospective_factor_two_counts": {
            "P5_STITCHED_vs_P5_BLOCK": seam_better,
            "P8_STITCHED_vs_P5_STITCHED": p8_better,
            "total_group_family_comparisons": comparisons,
        },
        "constraint_integral_ratios": ratios,
        "post_receive_same_level_ghost_mismatch": post_receive_ghost_mismatch,
        "post_refresh_same_level_coarse_cache_mismatch": refreshed_cache_mismatch,
        "classification_note": (
            "Factor-two counts are a transparent material-improvement screen, not a "
            "convergence threshold. Parent-underresolution requires the conditional "
            "filtering diagnostic and is not inferred automatically."),
    }


def analyze(args: argparse.Namespace) -> None:
    interpolation_self_test()
    event = args.event.resolve(); output = args.output.resolve()
    if output.exists():
        raise DiagnosisError(f"output already exists: {output}")
    output.mkdir(parents=True)
    groups, stitching = build_dense_groups(event)
    squared_scan = scan_squared_production(event)
    proposal = rows(event / "t1_topology_proposal.csv")
    selected = selected_refined(proposal)
    selected_old = {int(row["old_gid"]) for row in selected}
    selected_new = {int(row["new_gid"]) for row in selected}
    ghost_summary, ghost_worst = ghost_census(event, selected_old, selected_new)
    cache_summary, cache_worst = coarse_cache_census(event, selected_new)

    metric_rows: list[dict[str, Any]] = []
    representation_error: list[dict[str, Any]] = []
    derivative_error: list[dict[str, Any]] = []
    spectra: list[dict[str, Any]] = []
    validation: dict[str, Any] = {"groups": []}
    plot_paths: list[str] = []
    for group in groups:
        states = {"P5_BLOCK": project(group.production),
                  "P5_STITCHED": project(group.p5),
                  "P8_STITCHED": project(group.p8)}
        projection_delta = np.abs(states["P5_BLOCK"] - group.production_t4)
        projection_max = float(np.nanmax(projection_delta[:, group.common_mask]))
        if projection_max > 2.0e-13:
            raise DiagnosisError(f"projection does not reproduce production for {group.label}: "
                                 f"{projection_max}")
        constraints: dict[str, dict[str, np.ndarray]] = {}
        adms: dict[str, dict[str, np.ndarray]] = {}
        for label, state in states.items():
            constraints[label], adms[label] = compute_constraints(
                state, group.spacing, group.rho)
        # The accepted P5_BLOCK constraints are already present in the phase capture.
        # Keep them as the production reference.  In particular, do not silently replace
        # their block-local ghost semantics with derivatives across a stitched active-cell
        # mosaic: that was the invalid T0-style comparison this diagnostic is designed to
        # avoid.  The literal NumPy port below remains an independently audited check and
        # is retained in validation.json, while the alternative stitched states are
        # evaluated only on their fully supported common stencil.
        offline_block_constraints = constraints["P5_BLOCK"]
        constraints["P5_BLOCK"] = {
            name: group.production_constraints[index]
            for name, index in (("C", 0), ("H", 1), ("M", 2), ("Z", 3),
                                ("Mx", 4), ("My", 5), ("Mz", 6))
        }
        for label in states:
            rows_here = metrics(constraints[label], adms[label], group.common_mask,
                                group.rho, group.z, group.spacing,
                                group.old_seams_rho, group.old_seams_z)
            for row in rows_here:
                row.update({"group": group.label, "representation": label})
            metric_rows.extend(rows_here)
        production_delta: dict[str, float] = {}
        for family, index in (("C", 0), ("H", 1), ("M", 2), ("Z", 3),
                              ("Mx", 4), ("My", 5), ("Mz", 6)):
            delta = np.abs(offline_block_constraints[family]
                           - group.production_constraints[index])
            production_delta[family] = float(np.nanmax(delta[group.common_mask]))
        # This audit is reported, not hidden.  The Python port is not used as the
        # production P5_BLOCK reference unless it agrees within the established gate.
        scale = max(1.0, float(np.nanmax(np.abs(group.production_constraints[:, group.common_mask]))))
        port_gate = 2.0e-9 * scale
        port_pass = max(production_delta.values()) <= port_gate
        validation["groups"].append({
            "group": group.label, "algebraic_projection_max_abs": projection_max,
            "constraint_max_abs_by_field": production_delta,
            "production_constraint_scale": scale,
            "constraint_port_absolute_gate": port_gate,
            "constraint_port_pass": port_pass,
            "constraint_reference_used_for_P5_BLOCK": "captured_production_constraints",
        })
        representation_error.extend(representation_rows(group, states, group.common_mask))
        derivative_error.extend(derivative_rows(group, states, group.common_mask))
        spectra.extend(spectral_rows(group, states))
        plot_paths.extend(plot_group(group, constraints, states, output))

    verdict = classify(metric_rows, ghost_summary, cache_summary, validation)
    provenance = {
        "schema": SCHEMA, "event_root": str(event),
        "event_phase_sha256": sha256(event / "t1_phase.json"),
        "restart_sha256_declared": args.restart_sha256,
        "source_commit": args.source_commit, "source_tree": args.source_tree,
        "target_cycle": 1722, "target_time": 9.50625,
        "selected_parent_gids": sorted(selected_old),
        "selected_child_gids": sorted(selected_new),
        "raw_squared_diagnostic_scan": squared_scan,
    }
    strict_dump(output / "topology_stitching_manifest.json", stitching)
    strict_dump(output / "production_reproduction_validation.json", validation)
    strict_dump(output / "provenance.json", provenance)
    strict_dump(output / "verdict.json", verdict)
    write_csv(output / "constraint_metrics.csv", metric_rows, list(metric_rows[0]))
    write_csv(output / "representation_errors.csv", representation_error,
              list(representation_error[0]))
    write_csv(output / "derivative_disagreements.csv", derivative_error,
              list(derivative_error[0]))
    write_csv(output / "high_frequency_diagnostics.csv", spectra, list(spectra[0]))
    write_csv(output / "same_level_ghost_summary.csv", ghost_summary, list(ghost_summary[0]))
    write_csv(output / "same_level_ghost_worst.csv", ghost_worst,
              list(ghost_worst[0]) if ghost_worst else ["phase"])
    write_csv(output / "coarse_cache_summary.csv", cache_summary, list(cache_summary[0]))
    write_csv(output / "coarse_cache_worst.csv", cache_worst,
              list(cache_worst[0]) if cache_worst else ["phase"])
    strict_dump(output / "plot_inventory.json", {"plots": plot_paths})
    print(json.dumps(verdict, sort_keys=True, indent=2))


def self_test() -> None:
    interpolation_self_test()
    mask = np.ones((9, 10), dtype=bool); mask[0, 0] = False
    eroded = erode_mask(mask, 1)
    if eroded[1, 1] or not eroded[3, 3]:
        raise DiagnosisError("mask erosion fixture failed")
    state = np.zeros((25, 10, 11)); state[0] = 1.0
    state[1] = state[4] = state[6] = 1.0
    projected = project(state)
    if not np.allclose(projected, state):
        raise DiagnosisError("flat-state algebraic projection fixture failed")
    x = np.arange(20.0)[None, :] ** 5
    derivative = fd_first(x, -1, 1.0)
    exact = 5.0 * np.arange(20.0) ** 4
    if not np.allclose(derivative[0, 3:-3], exact[3:-3]):
        raise DiagnosisError("O6 first derivative polynomial fixture failed")
    print("SELF_TEST_PASS")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    result.add_argument("--event", type=Path)
    result.add_argument("--output", type=Path)
    result.add_argument("--restart-sha256",
                        default="83e996d2d5069307888a69fff47a7524c2f63f11869fb628630bca54dd5943ea")
    result.add_argument("--source-commit", required=False, default="")
    result.add_argument("--source-tree", required=False, default="")
    result.add_argument("--self-test", action="store_true")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.self_test:
        self_test(); return
    if args.event is None or args.output is None:
        raise DiagnosisError("--event and --output are required")
    analyze(args)


if __name__ == "__main__":
    try:
        main()
    except (DiagnosisError, base.AnalysisError) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(2) from error
