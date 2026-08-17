#!/usr/bin/env python3
"""Byte-backed localization of the cycle-1722 Brill AMR constraint jump.

This script consumes only the authenticated T0/T2/T3_06/T4/T5 phase bytes.
It does not reimplement the ADM constraints.  Constraint values are always the
captured production-C++ results.  The only source-level model here is the exact
O6 read support: centered first/second derivatives reach three cells in each
active direction and mixed derivatives use their tensor-product support.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np


SCHEMA = "athenak_z4c_amr_constraint_localization_existing_bytes_v1"
PHASES = (
    "t0_00_ACCEPTED_OLD_STATE",
    "t2_00_REFINE_OR_DEREFINE_TRANSFER",
    "t3_06_PHYSICAL_OR_AXIS_BC",
    "t4_00_ALGEBRAIC_PROJECTION",
    "t5_00_ADM_OR_CONSTRAINT_RECOMPUTATION",
)
Z4C_NAMES = (
    "chi", "gtilde_xx", "gtilde_xy", "gtilde_xz", "gtilde_yy",
    "gtilde_yz", "gtilde_zz", "Khat", "Atilde_xx", "Atilde_xy",
    "Atilde_xz", "Atilde_yy", "Atilde_yz", "Atilde_zz", "Gamma_x",
    "Gamma_y", "Gamma_z", "Theta", "alpha", "beta_x", "beta_y",
    "beta_z", "B_x", "B_y", "B_z",
)
ADM_NAMES = (
    "gamma_xx", "gamma_xy", "gamma_xz", "gamma_yy", "gamma_yz",
    "gamma_zz", "K_xx", "K_xy", "K_xz", "K_yy", "K_yz", "K_zz",
    "psi4",
)
CON_NAMES = ("C", "H", "M2", "Z2", "Mx", "My", "Mz")
METRICS = ("C", "H2", "M2", "Z2")
STENCIL_CLASSES = (
    "ACTIVE_ONLY", "SAME_LEVEL_SEAM", "COARSE_FINE", "AXIS",
    "PHYSICAL_BOUNDARY", "MIXED_CORNER",
)


class AnalysisError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_load(path: Path) -> Any:
    def reject(value: str) -> None:
        raise AnalysisError(f"non-finite JSON token {value} in {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject)


def strict_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_view(phase: Path, name: str, metadata: dict[str, Any]) -> np.ndarray:
    keys = {"u0": "u0_shape", "adm": "adm_shape", "constraints": "constraint_shape"}
    shape = metadata.get(keys[name])
    if not isinstance(shape, list) or len(shape) != 5 or not all(
            isinstance(value, int) and value > 0 for value in shape):
        raise AnalysisError(f"invalid {keys[name]} in {phase / 'phase.json'}")
    path = phase / f"{name}.bin"
    if path.stat().st_size != math.prod(shape) * 8:
        raise AnalysisError(f"binary size mismatch for {path}")
    data = np.fromfile(path, dtype="<f8").reshape(shape)
    if not np.all(np.isfinite(data)):
        raise AnalysisError(f"non-finite data in {path}")
    return data


def topology(phase: Path) -> dict[int, dict[str, str]]:
    rows = read_csv(phase / "topology.csv")
    result = {int(row["gid"]): row for row in rows}
    if len(result) != len(rows):
        raise AnalysisError(f"duplicate gid in {phase / 'topology.csv'}")
    return result


def active(view: np.ndarray, metadata: dict[str, Any], local_m: int) -> np.ndarray:
    bounds = metadata["active_bounds"]
    result = view[local_m, :, bounds["ks"]:bounds["ke"] + 1,
                  bounds["js"]:bounds["je"] + 1,
                  bounds["is"]:bounds["ie"] + 1]
    if result.shape[1] != 1:
        raise AnalysisError("expected collapsed x3 storage")
    return result[:, 0]


def det_gamma(adm: np.ndarray) -> np.ndarray:
    gxx, gxy, gxz, gyy, gyz, gzz = adm[:6]
    return (gxx * (gyy * gzz - gyz * gyz)
            - gxy * (gxy * gzz - gyz * gxz)
            + gxz * (gxy * gyz - gyy * gxz))


def metric_field(constraints: np.ndarray, metric: str) -> np.ndarray:
    if metric == "C":
        return constraints[0]
    if metric == "H2":
        return constraints[1] * constraints[1]
    if metric == "M2":
        return constraints[2]
    if metric == "Z2":
        return constraints[3]
    raise AnalysisError(f"unknown constraint metric {metric}")


def magnitude_field(constraints: np.ndarray, metric: str) -> np.ndarray:
    values = metric_field(constraints, metric)
    return np.sqrt(np.maximum(values, 0.0))


def cell_geometry(row: dict[str, str], nx: int, ny: int) -> tuple[np.ndarray, np.ndarray,
                                                                   float, float]:
    dx1 = (float(row["x1max"]) - float(row["x1min"])) / nx
    dx2 = (float(row["x2max"]) - float(row["x2min"])) / ny
    rho = float(row["x1min"]) + (np.arange(nx, dtype=np.float64) + 0.5) * dx1
    z = float(row["x2min"]) + (np.arange(ny, dtype=np.float64) + 0.5) * dx2
    return rho, z, dx1, dx2


def phase_integrals(root: Path, phase_name: str) -> dict[str, Any]:
    phase = root / phase_name
    metadata = strict_load(phase / "phase.json")
    topo = topology(phase)
    adm_all = load_view(phase, "adm", metadata)
    con_all = load_view(phase, "constraints", metadata)
    result: dict[str, Any] = {
        "phase": phase_name,
        "coordinate_ring_volume": 0.0,
        "proper_ring_volume": 0.0,
    }
    for measure in ("coordinate", "proper"):
        for metric in METRICS:
            result[f"{measure}_{metric}_integral"] = 0.0
            result[f"{measure}_{metric}_rms"] = None
    maxima = {metric: (-math.inf, None, None, None, None, None) for metric in METRICS}
    for gid, row in topo.items():
        local_m = int(row["local_m"])
        adm = active(adm_all, metadata, local_m)
        con = active(con_all, metadata, local_m)
        determinant = det_gamma(adm)
        if not np.all(np.isfinite(determinant)) or not np.all(determinant > 0.0):
            raise AnalysisError(f"invalid determinant in {phase_name}/gid={gid}")
        ny, nx = con.shape[-2:]
        rho, z, dx1, dx2 = cell_geometry(row, nx, ny)
        coordinate = 2.0 * math.pi * rho[np.newaxis, :] * dx1 * dx2
        coordinate = np.broadcast_to(coordinate, (ny, nx))
        proper = coordinate * np.sqrt(determinant)
        result["coordinate_ring_volume"] += float(np.sum(coordinate))
        result["proper_ring_volume"] += float(np.sum(proper))
        for metric in METRICS:
            squared = metric_field(con, metric)
            magnitude = magnitude_field(con, metric)
            for label, weight in (("coordinate", coordinate), ("proper", proper)):
                result[f"{label}_{metric}_integral"] += float(np.sum(squared * weight))
            index = np.unravel_index(int(np.argmax(magnitude)), magnitude.shape)
            value = float(magnitude[index])
            if value > maxima[metric][0]:
                maxima[metric] = (value, gid, int(index[1]), int(index[0]),
                                  float(rho[index[1]]), float(z[index[0]]))
    for measure, volume_key in (("coordinate", "coordinate_ring_volume"),
                                ("proper", "proper_ring_volume")):
        for metric in METRICS:
            result[f"{measure}_{metric}_rms"] = math.sqrt(
                result[f"{measure}_{metric}_integral"] / result[volume_key])
    for metric, values in maxima.items():
        result[f"{metric}_linf"] = values[0]
        result[f"{metric}_max_gid"] = values[1]
        result[f"{metric}_max_i"] = values[2]
        result[f"{metric}_max_j"] = values[3]
        result[f"{metric}_max_rho"] = values[4]
        result[f"{metric}_max_z"] = values[5]
    return result


def compare_arrays(before: np.ndarray, after: np.ndarray, names: tuple[str, ...],
                   comparison: str, scope: str) -> list[dict[str, Any]]:
    if before.shape != after.shape or before.shape[1] != len(names):
        raise AnalysisError(f"shape/name mismatch in {comparison}/{scope}")
    rows: list[dict[str, Any]] = []
    epsilon = np.finfo(np.float64).eps
    for component, name in enumerate(names):
        left, right = before[:, component], after[:, component]
        delta = right - left
        scale = np.maximum(np.maximum(np.abs(left), np.abs(right)), np.finfo(float).tiny)
        threshold = 64.0 * epsilon * max(1.0, float(np.max(scale)))
        absolute = np.abs(delta)
        index = np.unravel_index(int(np.argmax(absolute)), absolute.shape)
        rows.append({
            "comparison": comparison,
            "scope": scope,
            "field": name,
            "exact_byte_equal": bool(np.array_equal(left, right)),
            "max_abs_change": float(absolute[index]),
            "max_relative_change": float(np.max(absolute / scale)),
            "nonzero_changed_values": int(np.count_nonzero(delta)),
            "changed_values_above_roundoff": int(np.count_nonzero(absolute > threshold)),
            "roundoff_threshold": threshold,
            "max_change_flat_index": int(np.ravel_multi_index(index, absolute.shape)),
        })
    return rows


def phase_field_comparisons(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for before_name, after_name in ((PHASES[2], PHASES[3]), (PHASES[3], PHASES[4])):
        before_root, after_root = root / before_name, root / after_name
        before_meta, after_meta = (strict_load(before_root / "phase.json"),
                                   strict_load(after_root / "phase.json"))
        if before_meta["active_bounds"] != after_meta["active_bounds"]:
            raise AnalysisError("active bounds changed during T3/T4/T5")
        before_topo, after_topo = topology(before_root), topology(after_root)
        if before_topo != after_topo:
            raise AnalysisError("topology changed during T3/T4/T5")
        for view_name, names in (("u0", Z4C_NAMES), ("adm", ADM_NAMES),
                                 ("constraints", CON_NAMES)):
            left_all = load_view(before_root, view_name, before_meta)
            right_all = load_view(after_root, view_name, after_meta)
            rows.extend(compare_arrays(left_all, right_all, names,
                                       f"{before_name}->{after_name}", "stored"))
            left_active: list[np.ndarray] = []
            right_active: list[np.ndarray] = []
            for gid in sorted(before_topo):
                lm = int(before_topo[gid]["local_m"])
                left_active.append(active(left_all, before_meta, lm))
                right_active.append(active(right_all, after_meta, lm))
            rows.extend(compare_arrays(np.stack(left_active), np.stack(right_active), names,
                                       f"{before_name}->{after_name}", "active"))
        left_adm = load_view(before_root, "adm", before_meta)
        right_adm = load_view(after_root, "adm", after_meta)
        left_det, right_det = [], []
        for gid in sorted(before_topo):
            lm = int(before_topo[gid]["local_m"])
            left_det.append(det_gamma(active(left_adm, before_meta, lm))[None])
            right_det.append(det_gamma(active(right_adm, after_meta, lm))[None])
        rows.extend(compare_arrays(np.stack(left_det), np.stack(right_det),
                                   ("det_gamma",),
                                   f"{before_name}->{after_name}", "active"))
    return rows


def fraction(value: str) -> Fraction:
    return Fraction(value)


def intervals_overlap(a0: Fraction, a1: Fraction,
                      b0: Fraction, b1: Fraction) -> bool:
    return max(a0, b0) < min(a1, b1)


def face_kind(gid: int, face: str, topo: dict[int, dict[str, str]]) -> str:
    row = topo[gid]
    bc_key = {"inner_x1": "inner_x1_bc", "outer_x1": "outer_x1_bc",
              "inner_x2": "inner_x2_bc", "outer_x2": "outer_x2_bc"}[face]
    bc = int(row[bc_key])
    if bc != 0:
        if face == "inner_x1" and bc == 9:
            return "AXIS"
        return "PHYSICAL_BOUNDARY"
    x10, x11 = fraction(row["x1min"]), fraction(row["x1max"])
    x20, x21 = fraction(row["x2min"]), fraction(row["x2max"])
    neighbors: list[dict[str, str]] = []
    for other_gid, other in topo.items():
        if other_gid == gid:
            continue
        ox10, ox11 = fraction(other["x1min"]), fraction(other["x1max"])
        ox20, ox21 = fraction(other["x2min"]), fraction(other["x2max"])
        touches = False
        if face == "inner_x1":
            touches = ox11 == x10 and intervals_overlap(x20, x21, ox20, ox21)
        elif face == "outer_x1":
            touches = ox10 == x11 and intervals_overlap(x20, x21, ox20, ox21)
        elif face == "inner_x2":
            touches = ox21 == x20 and intervals_overlap(x10, x11, ox10, ox11)
        else:
            touches = ox20 == x21 and intervals_overlap(x10, x11, ox10, ox11)
        if touches:
            neighbors.append(other)
    if not neighbors:
        raise AnalysisError(f"no neighbor found for gid={gid} face={face}")
    level = int(row["level"])
    differences = {int(other["level"]) - level for other in neighbors}
    if differences == {0}:
        return "SAME_LEVEL_SEAM"
    if all(abs(value) == 1 for value in differences):
        return "COARSE_FINE"
    raise AnalysisError(f"ambiguous neighbor levels for gid={gid} face={face}: {differences}")


def classify_cell(gid: int, i: int, j: int, nx: int, ny: int,
                  topo: dict[int, dict[str, str]], radius: int = 3
                  ) -> tuple[str, tuple[str, ...], tuple[str, ...]]:
    crossed: list[str] = []
    if i < radius:
        crossed.append("inner_x1")
    if nx - 1 - i < radius:
        crossed.append("outer_x1")
    if j < radius:
        crossed.append("inner_x2")
    if ny - 1 - j < radius:
        crossed.append("outer_x2")
    if not crossed:
        return "ACTIVE_ONLY", (), ()
    kinds = tuple(face_kind(gid, face, topo) for face in crossed)
    if len(crossed) > 1:
        return "MIXED_CORNER", tuple(crossed), kinds
    return kinds[0], tuple(crossed), kinds


def origin_by_new_gid(proposal: list[dict[str, str]]) -> dict[int, str]:
    result: dict[int, str] = {}
    for row in proposal:
        new_level, old_level = int(row["new_level"]), int(row["old_level"])
        if new_level == old_level + 1:
            label = ("EXPLICIT_REFINED_CHILD" if int(row["requested_flag"]) == 1
                     else "BALANCE_INDUCED_CHILD")
        elif new_level == old_level:
            label = "UNCHANGED_LEAF"
        elif new_level < old_level:
            label = "DEREFINED_LEAF"
        else:
            raise AnalysisError(f"unsupported topology transition in {row}")
        result[int(row["new_gid"])] = label
    return result


def active_byte_comparison(root: Path, proposal: list[dict[str, str]]) -> list[dict[str, Any]]:
    t2, t3 = root / PHASES[1], root / PHASES[2]
    meta2, meta3 = strict_load(t2 / "phase.json"), strict_load(t3 / "phase.json")
    topo2, topo3 = topology(t2), topology(t3)
    if topo2 != topo3:
        raise AnalysisError("T2/T3 topology mismatch")
    u2, u3 = load_view(t2, "u0", meta2), load_view(t3, "u0", meta3)
    origins = origin_by_new_gid(proposal)
    accum: dict[tuple[str, str], dict[str, Any]] = {}
    for origin in sorted(set(origins.values())):
        for field in Z4C_NAMES:
            accum[(origin, field)] = {
                "origin": origin, "field": field, "active_value_count": 0,
                "nonzero_changed_values": 0, "changed_values_above_roundoff": 0,
                "max_abs_change": 0.0, "max_relative_change": 0.0,
                "max_gid": None, "max_i": None, "max_j": None,
            }
    for gid in sorted(topo2):
        lm = int(topo2[gid]["local_m"])
        left, right = active(u2, meta2, lm), active(u3, meta3, lm)
        origin = origins[gid]
        for component, field in enumerate(Z4C_NAMES):
            delta = right[component] - left[component]
            absolute = np.abs(delta)
            scale = np.maximum(np.maximum(np.abs(left[component]),
                                          np.abs(right[component])),
                               np.finfo(float).tiny)
            threshold = 64.0 * np.finfo(float).eps * max(1.0, float(np.max(scale)))
            item = accum[(origin, field)]
            item["active_value_count"] += int(delta.size)
            item["nonzero_changed_values"] += int(np.count_nonzero(delta))
            item["changed_values_above_roundoff"] += int(np.count_nonzero(
                absolute > threshold))
            relative = absolute / scale
            index = np.unravel_index(int(np.argmax(absolute)), absolute.shape)
            if float(absolute[index]) > item["max_abs_change"]:
                item["max_abs_change"] = float(absolute[index])
                item["max_relative_change"] = float(relative[index])
                item["max_gid"], item["max_j"], item["max_i"] = gid, index[0], index[1]
    return list(accum.values())


def constraint_decomposition(root: Path, proposal: list[dict[str, str]]) -> tuple[
        list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    phase = root / PHASES[2]
    metadata, topo = strict_load(phase / "phase.json"), topology(phase)
    adm_all = load_view(phase, "adm", metadata)
    con_all = load_view(phase, "constraints", metadata)
    origins = origin_by_new_gid(proposal)
    cells: list[dict[str, Any]] = []
    accum: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"cell_count": 0, "coordinate_volume": 0.0, "proper_volume": 0.0,
                 "coordinate_integral": 0.0, "proper_integral": 0.0,
                 "linf": -math.inf, "max_gid": None, "max_i": None,
                 "max_j": None, "max_rho": None, "max_z": None})
    for gid, row in topo.items():
        lm = int(row["local_m"])
        adm, con = active(adm_all, metadata, lm), active(con_all, metadata, lm)
        determinant = det_gamma(adm)
        ny, nx = con.shape[-2:]
        rho, z, dx1, dx2 = cell_geometry(row, nx, ny)
        face_cache = {face: face_kind(gid, face, topo) for face in
                      ("inner_x1", "outer_x1", "inner_x2", "outer_x2")}
        same_faces = [face for face, kind in face_cache.items()
                      if kind == "SAME_LEVEL_SEAM"]
        cf_faces = [face for face, kind in face_cache.items() if kind == "COARSE_FINE"]
        for j in range(ny):
            for i in range(nx):
                stencil, faces, kinds = classify_cell(gid, i, j, nx, ny, topo)
                coordinate = 2.0 * math.pi * float(rho[i]) * dx1 * dx2
                proper = coordinate * math.sqrt(float(determinant[j, i]))
                distance = lambda face: {
                    "inner_x1": i + 0.5, "outer_x1": nx - i - 0.5,
                    "inner_x2": j + 0.5, "outer_x2": ny - j - 0.5,
                }[face]
                cell: dict[str, Any] = {
                    "gid": gid, "level": int(row["level"]), "i": i, "j": j,
                    "rho": float(rho[i]), "z": float(z[j]),
                    "hierarchy_origin": origins[gid], "stencil_class": stencil,
                    "crossed_faces": ";".join(faces),
                    "crossed_face_kinds": ";".join(kinds),
                    "distance_to_axis_cells": (i + 0.5 if face_cache["inner_x1"] == "AXIS"
                                               else None),
                    "nearest_same_level_seam_cells": (min(map(distance, same_faces))
                                                       if same_faces else None),
                    "nearest_coarse_fine_cells": (min(map(distance, cf_faces))
                                                   if cf_faces else None),
                    "coordinate_weight": coordinate, "proper_weight": proper,
                }
                for metric in METRICS:
                    squared = float(metric_field(con, metric)[j, i])
                    magnitude = math.sqrt(max(0.0, squared))
                    cell[f"{metric}_squared"] = squared
                    cell[f"{metric}_coordinate_contribution"] = squared * coordinate
                    cell[f"{metric}_proper_contribution"] = squared * proper
                    key = (origins[gid], stencil, metric)
                    item = accum[key]
                    item["cell_count"] += 1
                    item["coordinate_volume"] += coordinate
                    item["proper_volume"] += proper
                    item["coordinate_integral"] += squared * coordinate
                    item["proper_integral"] += squared * proper
                    if magnitude > item["linf"]:
                        item["linf"] = magnitude
                        item["max_gid"], item["max_i"], item["max_j"] = gid, i, j
                        item["max_rho"], item["max_z"] = float(rho[i]), float(z[j])
                cells.append(cell)
    totals: dict[str, dict[str, float]] = {}
    for metric in METRICS:
        totals[metric] = {
            "coordinate": sum(item["coordinate_integral"] for key, item in accum.items()
                              if key[2] == metric),
            "proper": sum(item["proper_integral"] for key, item in accum.items()
                          if key[2] == metric),
        }
    crosstab: list[dict[str, Any]] = []
    for (origin, stencil, metric), item in sorted(accum.items()):
        item["coordinate_rms"] = math.sqrt(item["coordinate_integral"] /
                                           item["coordinate_volume"])
        item["proper_rms"] = math.sqrt(item["proper_integral"] /
                                       item["proper_volume"])
        item["coordinate_fraction_of_t3"] = (item["coordinate_integral"] /
                                              totals[metric]["coordinate"])
        item["proper_fraction_of_t3"] = item["proper_integral"] / totals[metric]["proper"]
        crosstab.append({"hierarchy_origin": origin, "stencil_class": stencil,
                         "metric": metric, **item})
    by_stencil: list[dict[str, Any]] = []
    for stencil in STENCIL_CLASSES:
        for metric in METRICS:
            chosen = [row for row in crosstab if row["stencil_class"] == stencil and
                      row["metric"] == metric]
            cell_count = sum(row["cell_count"] for row in chosen)
            coordinate_volume = sum(row["coordinate_volume"] for row in chosen)
            proper_volume = sum(row["proper_volume"] for row in chosen)
            coordinate_integral = sum(row["coordinate_integral"] for row in chosen)
            proper_integral = sum(row["proper_integral"] for row in chosen)
            maxima = max(chosen, key=lambda row: row["linf"]) if chosen else None
            by_stencil.append({
                "stencil_class": stencil, "metric": metric, "cell_count": cell_count,
                "coordinate_volume": coordinate_volume, "proper_volume": proper_volume,
                "coordinate_integral": coordinate_integral,
                "proper_integral": proper_integral,
                "coordinate_fraction_of_t3": (coordinate_integral /
                                               totals[metric]["coordinate"]),
                "proper_fraction_of_t3": proper_integral / totals[metric]["proper"],
                "coordinate_rms": (math.sqrt(coordinate_integral / coordinate_volume)
                                   if coordinate_volume else None),
                "proper_rms": (math.sqrt(proper_integral / proper_volume)
                               if proper_volume else None),
                "linf": maxima["linf"] if maxima else None,
                "max_gid": maxima["max_gid"] if maxima else None,
                "max_i": maxima["max_i"] if maxima else None,
                "max_j": maxima["max_j"] if maxima else None,
                "max_rho": maxima["max_rho"] if maxima else None,
                "max_z": maxima["max_z"] if maxima else None,
            })
    return cells, crosstab, by_stencil


def integrate_block(con: np.ndarray, adm: np.ndarray, row: dict[str, str]
                    ) -> dict[str, float]:
    determinant = det_gamma(adm)
    ny, nx = con.shape[-2:]
    rho, _, dx1, dx2 = cell_geometry(row, nx, ny)
    coordinate = np.broadcast_to(2.0 * math.pi * rho[np.newaxis, :] * dx1 * dx2,
                                 (ny, nx))
    proper = coordinate * np.sqrt(determinant)
    result = {"coordinate_volume": float(np.sum(coordinate)),
              "proper_volume": float(np.sum(proper))}
    for metric in METRICS:
        squared = metric_field(con, metric)
        result[f"coordinate_{metric}"] = float(np.sum(squared * coordinate))
        result[f"proper_{metric}"] = float(np.sum(squared * proper))
    return result


def parent_region_table(root: Path, proposal: list[dict[str, str]],
                        phase_totals: dict[str, dict[str, Any]]) -> tuple[
                            list[dict[str, Any]], dict[int, dict[str, Any]]]:
    old_root, new_root = root / PHASES[0], root / PHASES[2]
    old_meta, new_meta = strict_load(old_root / "phase.json"), strict_load(
        new_root / "phase.json")
    old_topo, new_topo = topology(old_root), topology(new_root)
    old_adm_all, old_con_all = (load_view(old_root, "adm", old_meta),
                                load_view(old_root, "constraints", old_meta))
    new_adm_all, new_con_all = (load_view(new_root, "adm", new_meta),
                                load_view(new_root, "constraints", new_meta))
    refined: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in proposal:
        if int(row["new_level"]) == int(row["old_level"]) + 1:
            refined[int(row["old_gid"])].append(row)
    rows: list[dict[str, Any]] = []
    allocation: dict[int, dict[str, Any]] = {}
    for old_gid, children in sorted(refined.items()):
        if len(children) != 4:
            raise AnalysisError(f"parent {old_gid} has {len(children)} child blocks")
        old_row = old_topo[old_gid]
        old_lm = int(old_row["local_m"])
        old_values = integrate_block(active(old_con_all, old_meta, old_lm),
                                     active(old_adm_all, old_meta, old_lm), old_row)
        new_values = {key: 0.0 for key in old_values}
        child_gids = []
        for child in children:
            gid = int(child["new_gid"])
            child_gids.append(gid)
            new_row = new_topo[gid]
            lm = int(new_row["local_m"])
            values = integrate_block(active(new_con_all, new_meta, lm),
                                     active(new_adm_all, new_meta, lm), new_row)
            for key, value in values.items():
                new_values[key] += value
        classification = ("EXPLICIT_REFINEMENT" if any(
            int(child["requested_flag"]) == 1 for child in children)
                          else "BALANCE_INDUCED_REFINEMENT")
        record: dict[str, Any] = {
            "old_gid": old_gid, "child_gids": ";".join(map(str, sorted(child_gids))),
            "classification": classification, "old_level": int(old_row["level"]),
            "new_level": int(children[0]["new_level"]),
            "x1min": float(old_row["x1min"]), "x1max": float(old_row["x1max"]),
            "x2min": float(old_row["x2min"]), "x2max": float(old_row["x2max"]),
            "old_proper_volume": old_values["proper_volume"],
            "new_proper_volume": new_values["proper_volume"],
        }
        for metric in METRICS:
            for measure in ("coordinate", "proper"):
                old_value = old_values[f"{measure}_{metric}"]
                new_value = new_values[f"{measure}_{metric}"]
                global_jump = (phase_totals[PHASES[2]][f"{measure}_{metric}_integral"] -
                               phase_totals[PHASES[0]][f"{measure}_{metric}_integral"])
                record[f"old_{measure}_{metric}"] = old_value
                record[f"new_{measure}_{metric}"] = new_value
                record[f"ratio_{measure}_{metric}"] = (new_value / old_value
                                                       if old_value != 0.0 else None)
                record[f"delta_{measure}_{metric}"] = new_value - old_value
                record[f"fraction_global_jump_{measure}_{metric}"] = (
                    (new_value - old_value) / global_jump if global_jump != 0.0 else None)
        rows.append(record)
        allocation[old_gid] = {"children": set(child_gids), "old": old_values,
                               "new": new_values}
    return rows, allocation


def attribute_constraint_jumps(
        cells: list[dict[str, Any]], proposal: list[dict[str, str]],
        parent_alloc: dict[int, dict[str, Any]], root: Path
        ) -> list[dict[str, Any]]:
    # This is a conservative region attribution, not interpolation of the old
    # constraint field.  Each old parent integral is distributed over its new
    # child cells in proportion to their T3 proper-volume weights.
    by_new_gid = {int(row["new_gid"]): row for row in proposal}
    new_cells: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for cell in cells:
        new_cells[cell["gid"]].append(cell)
    for old_gid, info in parent_alloc.items():
        selected = [cell for gid in info["children"] for cell in new_cells[gid]]
        for measure in ("coordinate", "proper"):
            total_weight = sum(cell[f"{measure}_weight"] for cell in selected)
            for metric in METRICS:
                old_integral = info["old"][f"{measure}_{metric}"]
                for cell in selected:
                    old_share = old_integral * cell[f"{measure}_weight"] / total_weight
                    cell[f"{metric}_{measure}_jump_attributed"] = (
                        cell[f"{metric}_{measure}_contribution"] - old_share)
    old_phase = root / PHASES[0]
    old_meta, old_topo = strict_load(old_phase / "phase.json"), topology(old_phase)
    old_adm_all = load_view(old_phase, "adm", old_meta)
    old_con_all = load_view(old_phase, "constraints", old_meta)
    for gid, selected in new_cells.items():
        if "C_proper_jump_attributed" in selected[0]:
            continue
        proposal_row = by_new_gid[gid]
        old_gid = int(proposal_row["old_gid"])
        if int(proposal_row["new_level"]) != int(proposal_row["old_level"]):
            for cell in selected:
                for measure in ("coordinate", "proper"):
                    for metric in METRICS:
                        cell[f"{metric}_{measure}_jump_attributed"] = math.nan
            continue
        old_row = old_topo[old_gid]
        old_lm = int(old_row["local_m"])
        old_adm = active(old_adm_all, old_meta, old_lm)
        old_con = active(old_con_all, old_meta, old_lm)
        determinant = det_gamma(old_adm)
        ny, nx = old_con.shape[-2:]
        rho, _, dx1, dx2 = cell_geometry(old_row, nx, ny)
        old_coordinate = np.broadcast_to(
            2.0 * math.pi * rho[np.newaxis, :] * dx1 * dx2, (ny, nx))
        old_weights = {"coordinate": old_coordinate,
                       "proper": old_coordinate * np.sqrt(determinant)}
        for cell in selected:
            for measure, old_weight in old_weights.items():
                for metric in METRICS:
                    old_contribution = metric_field(old_con, metric) * old_weight
                    cell[f"{metric}_{measure}_jump_attributed"] = (
                        cell[f"{metric}_{measure}_contribution"] -
                        float(old_contribution[cell["j"], cell["i"]]))

    accum: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"cell_count": 0, "coordinate_jump": 0.0, "proper_jump": 0.0})
    for cell in cells:
        for metric in METRICS:
            key = (cell["hierarchy_origin"], cell["stencil_class"], metric)
            item = accum[key]
            item["cell_count"] += 1
            for measure in ("coordinate", "proper"):
                value = cell[f"{metric}_{measure}_jump_attributed"]
                if not math.isfinite(value):
                    raise AnalysisError("non-finite attributed jump")
                item[f"{measure}_jump"] += value
    totals = {(metric, measure): sum(
        item[f"{measure}_jump"] for key, item in accum.items() if key[2] == metric)
              for metric in METRICS for measure in ("coordinate", "proper")}
    rows: list[dict[str, Any]] = []
    for (origin, stencil, metric), item in sorted(accum.items()):
        record = {"hierarchy_origin": origin, "stencil_class": stencil,
                  "metric": metric, **item}
        for measure in ("coordinate", "proper"):
            total = totals[(metric, measure)]
            record[f"fraction_global_{measure}_jump"] = (
                item[f"{measure}_jump"] / total if total != 0.0 else None)
        rows.append(record)
    return rows


def render_maps(cells: list[dict[str, Any]], output: Path) -> None:
    rho = np.asarray([row["rho"] for row in cells])
    z = np.asarray([row["z"] for row in cells])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for axis, metric in zip(axes, ("C", "H2", "M2")):
        values = np.asarray([math.sqrt(max(0.0, row[f"{metric}_squared"]))
                             for row in cells])
        positive = values[values > 0.0]
        floor = max(float(np.min(positive)) if positive.size else 1.0e-30, 1.0e-30)
        image = axis.scatter(rho, z, c=np.log10(np.maximum(values, floor)), s=1,
                             linewidths=0, cmap="magma")
        axis.set_title(f"T3_06 log10 {metric.replace('2', '')} magnitude")
        axis.set_xlabel(r"$\rho/M$")
        axis.set_ylabel(r"$z/M$")
        axis.set_aspect("equal")
        fig.colorbar(image, ax=axis)
    fig.savefig(output / "t3_constraint_maps.png", dpi=180)
    plt.close(fig)

    colors = {name: index for index, name in enumerate(STENCIL_CLASSES)}
    origins = {name: index for index, name in enumerate(sorted(
        {row["hierarchy_origin"] for row in cells}))}
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for axis, values, title, labels in (
            (axes[0], [colors[row["stencil_class"]] for row in cells],
             "Exact O6 stencil provenance", colors),
            (axes[1], [origins[row["hierarchy_origin"]] for row in cells],
             "Hierarchy origin", origins)):
        image = axis.scatter(rho, z, c=values, s=1, linewidths=0, cmap="tab10")
        axis.set_title(title)
        axis.set_xlabel(r"$\rho/M$")
        axis.set_ylabel(r"$z/M$")
        axis.set_aspect("equal")
        handles = [plt.Line2D([], [], marker="s", linestyle="", color=image.cmap(
            image.norm(index)), label=label) for label, index in labels.items()]
        axis.legend(handles=handles, fontsize=7, loc="upper right")
    fig.savefig(output / "stencil_and_hierarchy_masks.png", dpi=180)
    plt.close(fig)

    values = np.asarray([row["C_proper_jump_attributed"] for row in cells])
    limit = float(np.max(np.abs(values)))
    nonzero = np.abs(values[values != 0.0])
    linear = max(float(np.quantile(nonzero, 0.25)) if nonzero.size else 1.0e-16,
                 limit * 1.0e-8)
    normalization = SymLogNorm(linthresh=linear, linscale=0.5,
                               vmin=-limit, vmax=limit, base=10)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5), constrained_layout=True)
    image = None
    for axis, size, title in (
            (axes[0], 1, "full domain"),
            (axes[1], 8, r"explicit-refinement region near $z=0$")):
        image = axis.scatter(rho, z, c=values, s=size, linewidths=0,
                             cmap="coolwarm", norm=normalization)
        axis.set_title(title)
        axis.set_xlabel(r"$\rho/M$")
        axis.set_ylabel(r"$z/M$")
        axis.set_aspect("equal")
    axes[1].set_xlim(4.75, 6.25)
    axes[1].set_ylim(-1.1, 1.1)
    fig.suptitle("Conservative region-attributed C integral jump per cell")
    colorbar = fig.colorbar(
        image, ax=axes, label="attributed proper-ring contribution",
        ticks=[-1.0, -1.0e-2, -1.0e-4, -1.0e-6, 0.0,
               1.0e-6, 1.0e-4, 1.0e-2, 1.0])
    colorbar.ax.minorticks_off()
    fig.savefig(output / "constraint_jump_contribution_map.png", dpi=180)
    plt.close(fig)


def verify_source_manifest(raw: Path, manifest: Path) -> dict[str, Any]:
    suffix = "event_c00001722_l2_to_l3/"
    expected: dict[str, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, remote = line.partition("  ")
        if separator and suffix in remote:
            expected[remote.split(suffix, 1)[1]] = digest
    checked: list[dict[str, Any]] = []
    for path in sorted(item for item in raw.rglob("*") if item.is_file()):
        relative = path.relative_to(raw).as_posix()
        # schema.json lives one directory above the event in the remote ledger.
        if relative == "schema.json":
            continue
        digest = sha256(path)
        if expected.get(relative) != digest:
            raise AnalysisError(f"source manifest mismatch for {relative}")
        checked.append({"path": relative, "sha256": digest,
                        "size_bytes": path.stat().st_size})
    return {"source_manifest": str(manifest), "source_manifest_sha256": sha256(manifest),
            "verified_file_count": len(checked), "files": checked}


def analyze(args: argparse.Namespace) -> None:
    raw, output = args.raw.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest_verification = verify_source_manifest(raw, args.source_manifest.resolve())
    schema = strict_load(raw / "schema.json")
    if schema.get("schema") != "athenak_z4c_amr_jump_diagnostic_v1" or \
            schema.get("real_bytes") != 8 or schema.get("amr_transfer") != "high_order":
        raise AnalysisError("raw schema does not identify the production high-order event")
    proposal = read_csv(raw / "t1_topology_proposal.csv")
    phase_totals = {phase: phase_integrals(raw, phase) for phase in
                    (PHASES[0], PHASES[2], PHASES[3], PHASES[4])}
    comparison_rows = phase_field_comparisons(raw)
    active_rows = active_byte_comparison(raw, proposal)
    cells, crosstab, by_stencil = constraint_decomposition(raw, proposal)
    parent_rows, parent_alloc = parent_region_table(raw, proposal, phase_totals)
    jump_rows = attribute_constraint_jumps(cells, proposal, parent_alloc, raw)
    render_maps(cells, output)

    phase_fields = [
        "phase", "coordinate_ring_volume", "proper_ring_volume",
        *[f"{measure}_{metric}_{quantity}" for measure in ("coordinate", "proper")
          for metric in METRICS for quantity in ("integral", "rms")],
        *[f"{metric}_{suffix}" for metric in METRICS for suffix in
          ("linf", "max_gid", "max_i", "max_j", "max_rho", "max_z")],
    ]
    write_csv(output / "phase_comparison.csv", list(phase_totals.values()), phase_fields)
    write_csv(output / "phase_field_changes.csv", comparison_rows, list(comparison_rows[0]))
    write_csv(output / "active_byte_t2_to_t3.csv", active_rows, list(active_rows[0]))
    write_csv(output / "stencil_provenance_cells.csv", cells, list(cells[0]))
    write_csv(output / "constraint_contributions_by_origin_and_stencil.csv", crosstab,
              list(crosstab[0]))
    write_csv(output / "constraint_contributions_by_stencil.csv", by_stencil,
              list(by_stencil[0]))
    write_csv(output / "parent_region_native_constraint_comparison.csv", parent_rows,
              list(parent_rows[0]))
    write_csv(output / "constraint_jump_by_origin_and_stencil.csv", jump_rows,
              list(jump_rows[0]))

    t3, t4, t5 = (phase_totals[PHASES[index]] for index in (2, 3, 4))
    projection_fraction = {
        metric: ((t4[f"proper_{metric}_integral"] - t3[f"proper_{metric}_integral"]) /
                 t5[f"proper_{metric}_integral"])
        for metric in METRICS
    }
    recomputation_exact = all(row["exact_byte_equal"] for row in comparison_rows
                              if row["comparison"].startswith(PHASES[3] + "->"))
    active_unchanged = all(row["nonzero_changed_values"] == 0 for row in active_rows)
    fractions = {metric: {row["stencil_class"]: row["proper_fraction_of_t3"]
                          for row in by_stencil if row["metric"] == metric}
                 for metric in METRICS}
    jump_fractions = {metric: {
        stencil: sum(row["proper_jump"] for row in jump_rows
                     if row["metric"] == metric and row["stencil_class"] == stencil) /
                 sum(row["proper_jump"] for row in jump_rows if row["metric"] == metric)
        for stencil in STENCIL_CLASSES} for metric in METRICS}
    summary = {
        "schema": SCHEMA,
        "source": {
            "diagnostic_schema": schema,
            "manifest_verification": manifest_verification,
            "raw_root": str(raw),
        },
        "target": {"cycle": 1722, "time": 9.5062499999999073,
                   "old_meshblocks": 74, "new_meshblocks": 86,
                   "old_max_level": 2, "new_max_level": 3},
        "phase_totals": phase_totals,
        "projection_fraction_of_final_proper_integral": projection_fraction,
        "t4_to_t5_all_captured_fields_exact": recomputation_exact,
        "t2_to_t3_all_active_z4c_values_exact": active_unchanged,
        "t3_proper_constraint_fraction_by_stencil": fractions,
        "proper_constraint_jump_fraction_by_stencil": jump_fractions,
        "refined_parent_count": len(parent_rows),
        "qualification_claim": False,
    }
    strict_dump(output / "existing_byte_summary.json", summary)
    print(json.dumps({
        "projection_fraction": projection_fraction,
        "t4_to_t5_exact": recomputation_exact,
        "t2_to_t3_active_exact": active_unchanged,
        "fractions": fractions,
        "jump_fractions": jump_fractions,
    }, sort_keys=True, indent=2, allow_nan=False))


def self_test() -> None:
    # O6 radius-three support and exact face classification smoke checks.
    if classify_cell.__defaults__ != (3,):
        raise AnalysisError("O6 stencil radius changed")
    simple = {
        0: {"gid": "0", "level": "0", "x1min": "0", "x1max": "1",
            "x2min": "0", "x2max": "1", "inner_x1_bc": "9",
            "outer_x1_bc": "1", "inner_x2_bc": "3", "outer_x2_bc": "1"},
    }
    if classify_cell(0, 3, 3, 8, 8, simple)[0] != "ACTIVE_ONLY":
        raise AnalysisError("active-only self-test failed")
    if classify_cell(0, 0, 4, 8, 8, simple)[0] != "AXIS":
        raise AnalysisError("axis self-test failed")
    if classify_cell(0, 0, 0, 8, 8, simple)[0] != "MIXED_CORNER":
        raise AnalysisError("mixed-corner self-test failed")
    print("analyze_existing_event self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--source-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.raw is None or args.source_manifest is None or args.output is None:
        parser.error("--raw, --source-manifest, and --output are required")
    analyze(args)


if __name__ == "__main__":
    main()
