#!/usr/bin/env python3
"""Common-coordinate-time, common-vertex analysis of the VC AMR discriminator."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASES = ("n128", "n256", "n512")
KINDS = ("state", "constraints", "curvature")
TARGET_TIMES = (0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
HEADER = re.compile(r"\[(\d+)\]=([^\s]+)")
H = 0.25
NR, NZ = 65, 129


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_reader(source: Path):
    path = source / "vis/python/bin_convert.py"
    spec = importlib.util.spec_from_file_location("vc_figure3_bin_convert", path)
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


def lagrange(nodes: np.ndarray, target: float) -> np.ndarray:
    result = np.ones(len(nodes), dtype=float)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                result[i] *= (target - nodes[j]) / (nodes[i] - nodes[j])
    return result


def select(paths: list[Path], target: float) -> tuple[list[Path], np.ndarray]:
    catalog = sorted((binary_header(path)[0], path) for path in paths)
    exact = [item for item in catalog if abs(item[0] - target) <= 2e-13]
    if exact:
        return [exact[0][1]], np.ones(1)
    require(len(catalog) >= 4 and catalog[0][0] <= target <= catalog[-1][0],
            f"target {target} lacks temporal support")
    times = np.asarray([item[0] for item in catalog])
    position = int(np.searchsorted(times, target))
    start = max(0, min(position - 2, len(catalog) - 4))
    chosen = catalog[start:start + 4]
    nodes = np.asarray([item[0] for item in chosen])
    return [item[1] for item in chosen], lagrange(nodes, target)


def canonical(data: dict[str, Any]) -> dict[str, Any]:
    names = list(data["var_names"])
    values = np.full((len(names), NZ, NR), np.nan)
    owner = np.full((NZ, NR), -1, dtype=np.int16)
    seam = np.zeros((NZ, NR), dtype=bool)
    same_level_spread = 0.0
    coarse_fine_spread = 0.0
    geometries = np.asarray(data["mb_geometry"], dtype=float)
    logical = np.asarray(data["mb_logical"], dtype=int)
    for block, bounds in enumerate(geometries):
        shape = np.asarray(data["mb_data"][names[0]][block]).shape
        require(shape[0] == 1, "Cartoon output is not collapsed")
        rho = np.linspace(bounds[0], bounds[1], shape[2])
        zed = np.linspace(bounds[2], bounds[3], shape[1])
        qi = rho / H
        qj = (zed + 16.0) / H
        ii_local = np.flatnonzero(np.abs(qi - np.rint(qi)) <= 2e-10)
        jj_local = np.flatnonzero(np.abs(qj - np.rint(qj)) <= 2e-10)
        ii_global = np.rint(qi[ii_local]).astype(int)
        jj_global = np.rint(qj[jj_local]).astype(int)
        keep_i = (ii_global >= 0) & (ii_global < NR)
        keep_j = (jj_global >= 0) & (jj_global < NZ)
        ii_local, ii_global = ii_local[keep_i], ii_global[keep_i]
        jj_local, jj_global = jj_local[keep_j], jj_global[keep_j]
        level = int(logical[block, 3])
        arrays = [np.asarray(data["mb_data"][name][block][0], dtype=float) for name in names]
        for jl, jg in zip(jj_local, jj_global):
            for il, ig in zip(ii_local, ii_global):
                point = np.asarray([array[jl, il] for array in arrays])
                previous = owner[jg, ig]
                if previous == level:
                    same_level_spread = max(same_level_spread,
                                            float(np.max(np.abs(values[:, jg, ig] - point))))
                elif previous >= 0:
                    coarse_fine_spread = max(coarse_fine_spread,
                                             float(np.max(np.abs(values[:, jg, ig] - point))))
                if level > previous:
                    owner[jg, ig] = level
                    values[:, jg, ig] = point
                    seam[jg, ig] = il in (0, shape[2] - 1) or jl in (0, shape[1] - 1)
    require(np.all(owner >= 0) and np.isfinite(values).all(), "common VC lattice incomplete")
    cf = np.zeros_like(seam)
    cf[:, 1:] |= owner[:, 1:] != owner[:, :-1]
    cf[:, :-1] |= owner[:, 1:] != owner[:, :-1]
    cf[1:, :] |= owner[1:, :] != owner[:-1, :]
    cf[:-1, :] |= owner[1:, :] != owner[:-1, :]
    # Include one common-lattice layer around each identified interface.
    cf = (cf | np.roll(cf, 1, axis=0) | np.roll(cf, -1, axis=0) |
          np.roll(cf, 1, axis=1) | np.roll(cf, -1, axis=1))
    return {"names": names, "values": values, "owner": owner, "seam": seam,
            "coarse_fine": cf, "same_level_spread": same_level_spread,
            "coarse_fine_spread": coarse_fine_spread}


def fourth_derivative(array: np.ndarray, spacing: float, axis: int) -> np.ndarray:
    moved = np.moveaxis(array, axis, -1)
    result = np.empty_like(moved)
    result[..., 2:-2] = (moved[..., :-4] - 8.0 * moved[..., 1:-3] +
                         8.0 * moved[..., 3:-1] - moved[..., 4:]) / (12.0 * spacing)
    result[..., 0] = (-25.0 * moved[..., 0] + 48.0 * moved[..., 1] -
                      36.0 * moved[..., 2] + 16.0 * moved[..., 3] -
                      3.0 * moved[..., 4]) / (12.0 * spacing)
    result[..., 1] = (-3.0 * moved[..., 0] - 10.0 * moved[..., 1] +
                      18.0 * moved[..., 2] - 6.0 * moved[..., 3] +
                      moved[..., 4]) / (12.0 * spacing)
    result[..., -2] = -(-3.0 * moved[..., -1] - 10.0 * moved[..., -2] +
                        18.0 * moved[..., -3] - 6.0 * moved[..., -4] +
                        moved[..., -5]) / (12.0 * spacing)
    result[..., -1] = -(-25.0 * moved[..., -1] + 48.0 * moved[..., -2] -
                        36.0 * moved[..., -3] + 16.0 * moved[..., -4] -
                        3.0 * moved[..., -5]) / (12.0 * spacing)
    return np.moveaxis(result, -1, axis)


def gamma_incompatibility(names: list[str], values: np.ndarray) -> dict[str, np.ndarray]:
    index = {name: number for number, name in enumerate(names)}
    required = ("z4c_gxx", "z4c_gxy", "z4c_gxz", "z4c_gyy", "z4c_gyz",
                "z4c_gzz", "z4c_Gamx", "z4c_Gamy", "z4c_Gamz")
    require(not set(required) - index.keys(), "state lacks Gamma/metric fields")
    metric = np.empty((NZ, NR, 3, 3), dtype=float)
    metric[..., 0, 0] = values[index["z4c_gxx"]]
    metric[..., 0, 1] = metric[..., 1, 0] = values[index["z4c_gxy"]]
    metric[..., 0, 2] = metric[..., 2, 0] = values[index["z4c_gxz"]]
    metric[..., 1, 1] = values[index["z4c_gyy"]]
    metric[..., 1, 2] = metric[..., 2, 1] = values[index["z4c_gyz"]]
    metric[..., 2, 2] = values[index["z4c_gzz"]]
    inverse = np.linalg.inv(metric)
    dr = fourth_derivative(inverse, H, axis=1)
    dz = fourth_derivative(inverse, H, axis=0)
    rho = np.linspace(0.0, 16.0, NR)[None, :]
    gamma_metric = np.empty((3, NZ, NR), dtype=float)
    gamma_metric[0, :, 1:] = -(dr[:, 1:, 0, 0] +
                                (inverse[:, 1:, 0, 0] - inverse[:, 1:, 1, 1]) /
                                rho[:, 1:] + dz[:, 1:, 0, 2])
    gamma_metric[0, :, 0] = 0.0
    gamma_metric[1] = -(dr[..., 1, 0] + dz[..., 1, 2])
    gamma_metric[2, :, 1:] = -(dr[:, 1:, 2, 0] + inverse[:, 1:, 2, 0] /
                                rho[:, 1:] + dz[:, 1:, 2, 2])
    gamma_metric[2, :, 0] = -(2.0 * dr[:, 0, 2, 0] + dz[:, 0, 2, 2])
    return {f"Gamma_evolved_minus_metric_{label}":
            values[index[f"z4c_Gam{label}"]] - gamma_metric[component]
            for component, label in enumerate(("x", "y", "z"))}


def interpolate_kind(paths: list[Path], target: float, reader, cache: dict[Path, dict]) -> dict:
    chosen, weights = select(paths, target)
    records = []
    result = None
    nearest = int(np.argmin([abs(binary_header(path)[0] - target) for path in chosen]))
    topology = None
    for number, (path, weight) in enumerate(zip(chosen, weights)):
        if path not in cache:
            cache[path] = canonical(reader(str(path)))
        item = cache[path]
        if result is None:
            result = np.zeros_like(item["values"])
        result += weight * item["values"]
        if number == nearest:
            topology = item
        time, cycle = binary_header(path)
        records.append({"path": str(path), "time": time, "cycle": cycle,
                        "weight": float(weight)})
    assert result is not None and topology is not None
    return {"names": cache[chosen[0]]["names"], "values": result,
            "owner": topology["owner"], "seam": topology["seam"],
            "coarse_fine": topology["coarse_fine"], "records": records,
            "same_level_spread": max(cache[path]["same_level_spread"] for path in chosen),
            "coarse_fine_spread": max(cache[path]["coarse_fine_spread"] for path in chosen)}


def causal_trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as stream:
        raw = list(csv.DictReader(stream))
    by_time = {float(row["time"]): float(row["max_coordinate_speed"]) for row in raw}
    time = np.asarray(sorted(by_time)); speed = np.asarray([by_time[t] for t in time])
    distance = np.zeros_like(time)
    distance[1:] = np.cumsum(0.5 * (speed[1:] + speed[:-1]) * np.diff(time))
    return time, distance


def rms(array: np.ndarray, mask: np.ndarray, rho: np.ndarray) -> tuple[float, float]:
    selected = mask & np.isfinite(array)
    require(np.any(selected), "empty comparison region")
    weight = rho[selected]
    if np.sum(weight) > 0.0:
        value = math.sqrt(float(np.sum(weight * array[selected] ** 2) / np.sum(weight)))
    else:
        value = math.sqrt(float(np.mean(array[selected] ** 2)))
    return value, float(np.max(np.abs(array[selected])))


def metric_pivots(fields: dict[str, np.ndarray]) -> tuple[np.ndarray, ...]:
    """Sylvester leading minors used by the production admissibility gate."""
    gxx = fields["z4c_gxx"]; gxy = fields["z4c_gxy"]; gxz = fields["z4c_gxz"]
    gyy = fields["z4c_gyy"]; gyz = fields["z4c_gyz"]; gzz = fields["z4c_gzz"]
    pivot0 = gxx
    pivot1 = gxx * gyy - gxy * gxy
    pivot2 = (gxx * (gyy * gzz - gyz * gyz) -
              gxy * (gxy * gzz - gyz * gxz) +
              gxz * (gxy * gyz - gyy * gxz))
    return pivot0, pivot1, pivot2


def finite_min(values: list[float]) -> float | None:
    finite = [value for value in values if math.isfinite(value)]
    return float(min(finite)) if finite else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    for case in CASES:
        parser.add_argument(f"--{case}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    roots = {case: getattr(args, case) for case in CASES}
    reader = load_reader(args.source)
    catalogs = {case: {kind: sorted((roots[case] / "bin/rank_00000000").glob(
        f"{case}.{kind}.*.bin")) for kind in KINDS} for case in CASES}
    require(all(len(catalogs[case][kind]) >= 4 for case in CASES for kind in KINDS),
            "insufficient binary inventory")
    support = min(binary_header(catalogs[case]["state"][-1])[0] for case in CASES)
    targets = [time for time in TARGET_TIMES if time <= support + 1e-12]
    require(len(targets) >= 3, "too little common field support")
    causal = {case: causal_trace(roots[case] / "z4c_timestep_contract.csv") for case in CASES}
    rho_1d = np.linspace(0.0, 16.0, NR)
    zed_1d = np.linspace(-16.0, 16.0, NZ)
    rho, zed = np.meshgrid(rho_1d, zed_1d)
    radius = np.hypot(rho, zed)
    outer_distance = np.minimum(16.0 - rho, 16.0 - np.abs(zed))
    cache: dict[Path, dict] = {}
    rows: list[dict] = []
    core_constraint_rows: list[dict] = []
    health_rows: list[dict] = []
    provenance: dict[str, Any] = {}
    for time in targets:
        snapshots = {case: {kind: interpolate_kind(catalogs[case][kind], time,
                                                    reader, cache)
                            for kind in KINDS} for case in CASES}
        trusted_radius = min(16.0 - float(np.interp(time, *causal[case]))
                             for case in CASES)
        trusted = radius <= max(0.0, trusted_radius - H)
        seam = np.logical_or.reduce([snapshots[case]["state"]["seam"] for case in CASES])
        cf = np.logical_or.reduce([snapshots[case]["state"]["coarse_fine"] for case in CASES])
        regions = {
            "trusted_core": trusted,
            "axis_trusted": trusted & np.isclose(rho, 0.0),
            "block_interiors_trusted": trusted & ~seam & ~cf,
            "same_level_seams_trusted": trusted & seam & ~cf,
            "coarse_fine_neighborhood_trusted": trusted & cf,
            "full_domain": np.ones_like(trusted),
            "outer_two_layers": outer_distance <= 2.0 * H + 1e-12,
        }
        fields: dict[str, dict[str, np.ndarray]] = {case: {} for case in CASES}
        for case in CASES:
            state = snapshots[case]["state"]
            for name, value in zip(state["names"], state["values"]):
                fields[case][name] = value
            fields[case]["derived_K"] = (fields[case]["z4c_Khat"] +
                                          2.0 * fields[case]["z4c_Theta"])
            fields[case].update(gamma_incompatibility(state["names"], state["values"]))
            for kind in ("constraints", "curvature"):
                for name, value in zip(snapshots[case][kind]["names"],
                                       snapshots[case][kind]["values"]):
                    fields[case][name] = value
        names = list(fields["n128"])
        require(all(list(fields[case]) == names for case in CASES), "field inventory changed")
        for case in CASES:
            for constraint in ("con_C", "con_H", "con_M", "con_Z"):
                value, linf = rms(fields[case][constraint], trusted, rho)
                core_constraint_rows.append({"coordinate_time": time,
                                             "trusted_radius": trusted_radius,
                                             "resolution": case,
                                             "field": constraint,
                                             "ring_rms": value,
                                             "linf": linf,
                                             "samples": int(np.sum(trusted))})
            pivots = metric_pivots(fields[case])
            for region_name, mask in (("trusted_core", trusted),
                                      ("full_domain", np.ones_like(trusted))):
                health_rows.append({"coordinate_time": time,
                                    "resolution": case,
                                    "region": region_name,
                                    "min_chi": float(np.min(fields[case]["z4c_chi"][mask])),
                                    "min_metric_spd_pivot": float(min(
                                        np.min(pivot[mask]) for pivot in pivots)),
                                    "same_level_shared_node_spread": float(
                                        snapshots[case]["state"]["same_level_spread"]),
                                    "coarse_fine_shared_node_spread": float(
                                        snapshots[case]["state"]["coarse_fine_spread"])})
        for name in names:
            coarse = fields["n128"][name] - fields["n256"][name]
            fine = fields["n256"][name] - fields["n512"][name]
            for region_name, mask in regions.items():
                if not np.any(mask):
                    continue
                e1, l1 = rms(coarse, mask, rho)
                e2, l2 = rms(fine, mask, rho)
                q = e1 / e2 if e2 > 0.0 else math.nan
                rows.append({"coordinate_time": time, "trusted_radius": trusted_radius,
                             "field": name, "region": region_name,
                             "E_128_256_rms": e1, "E_256_512_rms": e2,
                             "Q_rms": q, "p_rms": math.log2(q) if q > 0.0 else math.nan,
                             "E_128_256_linf": l1, "E_256_512_linf": l2,
                             "p_linf": math.log2(l1 / l2) if l1 > 0.0 and l2 > 0.0 else math.nan,
                             "samples": int(np.sum(mask))})
        provenance[str(time)] = {"trusted_radius": trusted_radius,
                                 "cases": {case: {kind: {
                                     "records": snapshots[case][kind]["records"],
                                     "same_level_spread": snapshots[case][kind]["same_level_spread"],
                                     "coarse_fine_spread": snapshots[case][kind]["coarse_fine_spread"]}
                                     for kind in KINDS} for case in CASES}}

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    with (output / "field_convergence.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    for filename, table in (("constraints_vs_time_core.csv", core_constraint_rows),
                            ("state_health.csv", health_rows)):
        with (output / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(table[0]), lineterminator="\n")
            writer.writeheader(); writer.writerows(table)
    (output / "field_sampling_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    selected_fields = ("z4c_chi", "z4c_alpha", "derived_K", "z4c_Theta",
                       "z4c_Gamx", "Gamma_evolved_minus_metric_x", "con_C",
                       "con_H", "con_M", "con_Z", "z4c_Kretschmann")
    fig, axes = plt.subplots(3, 4, figsize=(15.0, 10.5), constrained_layout=True)
    for axis, name in zip(axes.flat, selected_fields):
        selected = [row for row in rows if row["field"] == name and
                    row["region"] == "trusted_core"]
        axis.plot([row["coordinate_time"] for row in selected],
                  [row["p_rms"] for row in selected], "o-")
        axis.axhline(4.0, color="black", linestyle="--", linewidth=0.8)
        axis.axhline(0.0, color="gray", linewidth=0.7)
        axis.set_title(name); axis.set_xlabel("coordinate time")
        axis.set_ylabel("effective order p"); axis.grid(alpha=0.25)
    axes.flat[-1].axis("off")
    figures = output / "figures"; figures.mkdir(exist_ok=True)
    fig.savefig(figures / "trusted_core_field_convergence.png", dpi=220)
    fig.savefig(figures / "trusted_core_field_convergence.pdf")
    plt.close(fig)

    colors = {"n128": "#377eb8", "n256": "#4daf4a", "n512": "#e41a1c"}
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    for axis, name in zip(axes.flat, ("con_C", "con_H", "con_M", "con_Z")):
        for case in CASES:
            selected = [row for row in core_constraint_rows
                        if row["field"] == name and row["resolution"] == case]
            axis.semilogy([row["coordinate_time"] for row in selected],
                          [row["ring_rms"] for row in selected], "o-",
                          color=colors[case], label=case.upper())
        axis.set_title(name); axis.set_xlabel("coordinate time")
        axis.grid(alpha=0.25, which="both")
    axes[0, 0].legend()
    fig.savefig(figures / "constraints_vs_time_core.png", dpi=220)
    fig.savefig(figures / "constraints_vs_time_core.pdf")
    plt.close(fig)

    summary = {"schema": "z4c_vc_figure3_field_analysis_v1",
               "common_coordinate_times": targets,
               "common_lattice_spacing": H,
               "fields": names,
               "regions": sorted({row["region"] for row in rows}),
               "trusted_core_minimum_order": {
                   name: finite_min([row["p_rms"] for row in rows
                                     if row["field"] == name and
                                     row["region"] == "trusted_core"])
                   for name in names}}
    (output / "field_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
