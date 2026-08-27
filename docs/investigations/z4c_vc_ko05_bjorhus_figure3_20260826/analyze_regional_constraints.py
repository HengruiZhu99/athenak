#!/usr/bin/env python3
"""Common-lattice regional constraint analysis for the KO=0.5 campaign.

This is deliberately a sampling diagnostic, not a replacement for the native
leaf history inventory.  It samples each hierarchy on the same h=0.25 vertex
lattice, choosing the finest leaf representation at every coordinate.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASES = ("n128", "n256", "n512")
KINDS = ("state", "constraints", "curvature")
RADII = (4.0, 8.0, 12.0)
TARGET_TIMES = tuple(float(value) for value in range(15))
H = 0.25
RHO_MAX = 12.0
Z_MAX = 12.0
NR = int(round(RHO_MAX / H)) + 1
NZ = int(round(2.0 * Z_MAX / H)) + 1
COLORS = {"n128": "#377eb8", "n256": "#4daf4a", "n512": "#e41a1c"}
CONSTRAINT_FIELDS = ("con_C", "con_H", "con_M", "con_Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_reader(source: Path):
    path = source / "vis/python/bin_convert.py"
    spec = importlib.util.spec_from_file_location("ko05_bin_convert", path)
    require(spec is not None and spec.loader is not None, f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_binary


def binary_header(path: Path) -> tuple[float, int]:
    with path.open("rb") as stream:
        require(stream.readline().startswith(b"Athena binary output version=1.1"),
                f"bad binary {path}")
        count = int(stream.readline().split(b"=")[-1])
        values: dict[str, str] = {}
        for _ in range(count - 1):
            key, value = stream.readline().decode().split("=", 1)
            values[key.strip()] = value.strip()
    return float(values["time"]), int(values["cycle"])


def catalog(roots: list[Path], kind: str) -> dict[float, Path]:
    result: dict[float, Path] = {}
    for root in roots:
        for path in sorted((root / "bin/rank_00000000").glob(f"*.{kind}.*.bin")):
            result[binary_header(path)[0]] = path
    require(result, f"no {kind} files in {roots}")
    return result


def lagrange(nodes: np.ndarray, target: float) -> np.ndarray:
    weights = np.ones(len(nodes), dtype=float)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                weights[i] *= (target - nodes[j]) / (nodes[i] - nodes[j])
    return weights


def select(files: dict[float, Path], target: float) -> tuple[list[Path], np.ndarray]:
    items = sorted(files.items())
    exact = [item for item in items if abs(item[0] - target) <= 2.0e-13]
    if exact:
        return [exact[0][1]], np.ones(1)
    require(len(items) >= 4 and items[0][0] <= target <= items[-1][0],
            f"target {target} lacks temporal support")
    times = np.asarray([item[0] for item in items])
    position = int(np.searchsorted(times, target))
    start = max(0, min(position - 2, len(items) - 4))
    chosen = items[start:start + 4]
    nodes = np.asarray([item[0] for item in chosen])
    return [item[1] for item in chosen], lagrange(nodes, target)


def canonical(data: dict[str, Any]) -> dict[str, np.ndarray]:
    names = list(data["var_names"])
    values = np.full((len(names), NZ, NR), np.nan)
    owner = np.full((NZ, NR), -1, dtype=np.int16)
    geometries = np.asarray(data["mb_geometry"], dtype=float)
    logical = np.asarray(data["mb_logical"], dtype=int)
    for block, bounds in enumerate(geometries):
        shape = np.asarray(data["mb_data"][names[0]][block]).shape
        require(shape[0] == 1, "Cartoon output is not collapsed")
        rho = np.linspace(bounds[0], bounds[1], shape[2])
        zed = np.linspace(bounds[2], bounds[3], shape[1])
        qi = rho / H
        qj = (zed + Z_MAX) / H
        local_i = np.flatnonzero(np.abs(qi - np.rint(qi)) <= 2.0e-10)
        local_j = np.flatnonzero(np.abs(qj - np.rint(qj)) <= 2.0e-10)
        global_i = np.rint(qi[local_i]).astype(int)
        global_j = np.rint(qj[local_j]).astype(int)
        keep_i = (global_i >= 0) & (global_i < NR)
        keep_j = (global_j >= 0) & (global_j < NZ)
        local_i, global_i = local_i[keep_i], global_i[keep_i]
        local_j, global_j = local_j[keep_j], global_j[keep_j]
        level = int(logical[block, 3])
        arrays = [np.asarray(data["mb_data"][name][block][0], dtype=float)
                  for name in names]
        for jl, jg in zip(local_j, global_j):
            for il, ig in zip(local_i, global_i):
                if level > owner[jg, ig]:
                    owner[jg, ig] = level
                    values[:, jg, ig] = [array[jl, il] for array in arrays]
    require(np.all(owner >= 0), "common lattice is incomplete")
    require(np.isfinite(values).all(), "common lattice contains nonfinite data")
    return {name: values[index] for index, name in enumerate(names)}


def interpolate_kind(files: dict[float, Path], target: float, reader) -> tuple[
        dict[str, np.ndarray], list[dict]]:
    paths, weights = select(files, target)
    result: dict[str, np.ndarray] | None = None
    records: list[dict] = []
    for path, weight in zip(paths, weights):
        sampled = canonical(reader(str(path)))
        if result is None:
            result = {name: np.zeros_like(value) for name, value in sampled.items()}
        require(result.keys() == sampled.keys(), "binary field inventory changed")
        for name, value in sampled.items():
            result[name] += weight * value
        time, cycle = binary_header(path)
        records.append({"path": str(path), "time": time, "cycle": cycle,
                        "weight": float(weight)})
    assert result is not None
    return result, records


def physical_sqrt_gamma(state: dict[str, np.ndarray]) -> np.ndarray:
    names = ("z4c_gxx", "z4c_gxy", "z4c_gxz", "z4c_gyy", "z4c_gyz", "z4c_gzz",
             "z4c_chi")
    require(not set(names) - state.keys(), "state lacks conformal metric or chi")
    gxx, gxy, gxz, gyy, gyz, gzz, chi = (state[name] for name in names)
    det_tilde = (gxx * (gyy * gzz - gyz * gyz)
                 - gxy * (gxy * gzz - gyz * gxz)
                 + gxz * (gxy * gyz - gyy * gxz))
    require(np.all(chi > 0.0) and np.all(det_tilde > 0.0),
            "sampled metric is inadmissible")
    return np.sqrt(det_tilde) * chi ** (-1.5)


def squared_constraint(name: str, value: np.ndarray) -> np.ndarray:
    return value * value if name == "con_H" else value


def amplitude_order(coarse: float, fine: float) -> float:
    return 0.5 * math.log2(coarse / fine) if coarse > 0.0 and fine > 0.0 else math.nan


def write_csv(path: Path, rows: list[dict]) -> None:
    require(rows, f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    for case in CASES:
        parser.add_argument(f"--{case}", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {case: getattr(args, case) for case in CASES}
    files = {case: {kind: catalog(roots[case], kind) for kind in KINDS} for case in CASES}
    common_end = min(max(files[case]["constraints"]) for case in CASES)
    targets = [value for value in TARGET_TIMES if value <= common_end + 2.0e-11]
    require(targets and targets[-1] >= 14.0, "insufficient common output support")
    reader = load_reader(args.source)
    rho_1d = np.linspace(0.0, RHO_MAX, NR)
    zed_1d = np.linspace(-Z_MAX, Z_MAX, NZ)
    rho, zed = np.meshgrid(rho_1d, zed_1d)
    radius = np.hypot(rho, zed)
    coordinate_weight = 2.0 * math.pi * rho * H * H
    rows: list[dict] = []
    health_rows: list[dict] = []
    provenance: dict[str, Any] = {}
    for target in targets:
        sampled: dict[str, dict[str, float]] = {}
        provenance[str(target)] = {}
        for case in CASES:
            state, state_records = interpolate_kind(files[case]["state"], target, reader)
            constraints, constraint_records = interpolate_kind(
                files[case]["constraints"], target, reader)
            curvature, curvature_records = interpolate_kind(
                files[case]["curvature"], target, reader)
            require("z4c_Kretschmann" in curvature, "curvature output lacks Kretschmann")
            sqrt_gamma = physical_sqrt_gamma(state)
            chi_min_index = np.unravel_index(np.argmin(state["z4c_chi"]),
                                             state["z4c_chi"].shape)
            kret_abs = np.abs(curvature["z4c_Kretschmann"])
            kret_max_index = np.unravel_index(np.argmax(kret_abs), kret_abs.shape)
            health_rows.append({
                "time": target,
                "resolution": case,
                "common_lattice_min_chi": float(state["z4c_chi"][chi_min_index]),
                "min_chi_rho": float(rho[chi_min_index]),
                "min_chi_z": float(zed[chi_min_index]),
                "minimum_sqrt_gamma": float(np.min(sqrt_gamma)),
                "maximum_sqrt_gamma": float(np.max(sqrt_gamma)),
                "common_lattice_max_abs_kretschmann": float(kret_abs[kret_max_index]),
                "max_abs_kretschmann_rho": float(rho[kret_max_index]),
                "max_abs_kretschmann_z": float(zed[kret_max_index]),
                "common_lattice_spacing": H,
            })
            sampled[case] = {}
            provenance[str(target)][case] = {
                "state": state_records,
                "constraints": constraint_records,
                "curvature": curvature_records,
            }
            for region_radius in RADII:
                mask = radius <= region_radius + 1.0e-12
                coord_volume = float(np.sum(coordinate_weight[mask]))
                proper_weight = coordinate_weight * sqrt_gamma
                proper_volume = float(np.sum(proper_weight[mask]))
                for field in CONSTRAINT_FIELDS:
                    square = squared_constraint(field, constraints[field])
                    coord_inventory = float(np.sum(coordinate_weight[mask] * square[mask]))
                    proper_inventory = float(np.sum(proper_weight[mask] * square[mask]))
                    key = f"{region_radius:g}:{field}"
                    sampled[case][key] = proper_inventory
                    rows.append({
                        "time": target,
                        "resolution": case,
                        "radius": region_radius,
                        "field": field,
                        "coordinate_inventory": coord_inventory,
                        "proper_inventory": proper_inventory,
                        "coordinate_volume": coord_volume,
                        "proper_volume": proper_volume,
                        "proper_rms_amplitude": math.sqrt(proper_inventory / proper_volume),
                        "common_lattice_spacing": H,
                    })
        for region_radius in RADII:
            for field in CONSTRAINT_FIELDS:
                key = f"{region_radius:g}:{field}"
                for row in rows[-len(CASES) * len(RADII) * len(CONSTRAINT_FIELDS):]:
                    if row["time"] == target and row["radius"] == region_radius and row["field"] == field:
                        row["q128_256_amplitude"] = amplitude_order(
                            sampled["n128"][key], sampled["n256"][key])
                        row["q256_512_amplitude"] = amplitude_order(
                            sampled["n256"][key], sampled["n512"][key])

    output = args.output
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(output / "regional_constraint_inventories.csv", rows)
    write_csv(output / "common_lattice_state_health.csv", health_rows)
    (output / "regional_sampling_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fig, axes = plt.subplots(3, 4, figsize=(15.0, 10.5), constrained_layout=True)
    for axis, (region_radius, field) in zip(
            axes.flat, ((radius_value, field_value) for radius_value in RADII
                        for field_value in CONSTRAINT_FIELDS)):
        for case in CASES:
            selected = [row for row in rows if row["resolution"] == case
                        and row["radius"] == region_radius and row["field"] == field]
            axis.semilogy([row["time"] for row in selected],
                          [row["proper_inventory"] for row in selected], "o-",
                          color=COLORS[case], markersize=3, label=case.upper())
        axis.set_title(f"{field}, r <= {region_radius:g}")
        axis.set_xlabel("coordinate time")
        axis.grid(alpha=0.25, which="both")
    axes[0, 0].legend(fontsize=8)
    fig.savefig(figures / "regional_constraint_inventories.png", dpi=220)
    fig.savefig(figures / "regional_constraint_inventories.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(3, 4, figsize=(15.0, 10.5), constrained_layout=True)
    for axis, (region_radius, field) in zip(
            axes.flat, ((radius_value, field_value) for radius_value in RADII
                        for field_value in CONSTRAINT_FIELDS)):
        selected = [row for row in rows if row["resolution"] == "n128"
                    and row["radius"] == region_radius and row["field"] == field]
        axis.plot([row["time"] for row in selected],
                  [row["q128_256_amplitude"] for row in selected], "o-",
                  color=COLORS["n128"], markersize=3, label="q128-256")
        axis.plot([row["time"] for row in selected],
                  [row["q256_512_amplitude"] for row in selected], "o-",
                  color=COLORS["n512"], markersize=3, label="q256-512")
        axis.axhline(4.0, color="black", linestyle="--", linewidth=0.8)
        axis.axhline(0.0, color="gray", linewidth=0.7)
        axis.set_title(f"{field}, r <= {region_radius:g}")
        axis.set_xlabel("coordinate time")
        axis.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(figures / "regional_constraint_orders.png", dpi=220)
    fig.savefig(figures / "regional_constraint_orders.pdf")
    plt.close(fig)

    summary = {
        "schema": "z4c_vc_ko05_common_lattice_regional_constraints_v1",
        "coordinate_times": targets,
        "common_lattice_spacing": H,
        "regions": list(RADII),
        "measure": "2*pi*rho*sqrt(gamma)*drho*dz on common h=0.25 vertex lattice",
        "limitation": (
            "This is a matched common-lattice sampling diagnostic. It filters scales below "
            "h=0.25 and is not the native leaf quadrature used by the production history."
        ),
    }
    (output / "regional_constraint_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
