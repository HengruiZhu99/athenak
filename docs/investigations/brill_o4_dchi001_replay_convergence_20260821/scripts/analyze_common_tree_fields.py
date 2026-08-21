#!/usr/bin/env python3
"""Sample common-tree Z4c fields and measure trusted-window self convergence."""

from __future__ import annotations

import argparse
import csv
import gzip
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
TARGETS = (5.0, 7.5, 9.0)
BASE_FIELDS = {
    "chi": "z4c_chi", "alpha": "z4c_alpha", "Khat": "z4c_Khat",
    "Theta": "z4c_Theta", "Axx": "z4c_Axx", "Axy": "z4c_Axy",
    "Ayy": "z4c_Ayy", "Gamx": "z4c_Gamx", "Gamy": "z4c_Gamy",
}
OUTPUT_FIELDS = ("chi", "alpha", "K", "Theta", "Axx", "Axy", "Ayy",
                 "Gamx", "Gamy")


def fail(message: str) -> None:
    raise RuntimeError(message)


def load_reader(source: Path):
    path = source / "vis/python/bin_convert.py"
    spec = importlib.util.spec_from_file_location("campaign_bin_convert", path)
    if spec is None or spec.loader is None:
        fail(f"cannot import binary reader: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_binary


def binary_header(path: Path) -> tuple[float, int]:
    with path.open("rb") as stream:
        if not stream.readline().startswith(b"Athena binary output version=1.1"):
            fail(f"bad Athena binary header: {path}")
        count = int(stream.readline().split(b"=")[-1])
        values: dict[str, str] = {}
        for _ in range(count - 1):
            key, value = stream.readline().decode().split("=", 1)
            values[key.strip()] = value.strip()
    return float(values["time"]), int(values["cycle"])


def read_history(path: Path) -> dict[str, np.ndarray]:
    labels: dict[str, int] = {}
    rows = []
    import re
    pattern = re.compile(r"\[(\d+)\]=([^\s]+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            labels.update({name: int(index) - 1 for index, name in pattern.findall(line)})
        elif line.strip():
            rows.append([float(value) for value in line.split()])
    for required in ("time", "axisTau"):
        if required not in labels:
            fail(f"history lacks {required}: {path}")
    array = np.asarray(rows, dtype=float)
    if array.size == 0 or not np.isfinite(array).all():
        fail(f"invalid history: {path}")
    return {name: array[:, index] for name, index in labels.items()}


def target_time(history: dict[str, np.ndarray], tau: float) -> float:
    values = history["axisTau"]
    if np.any(np.diff(values) <= 0.0) or not values[0] <= tau <= values[-1]:
        fail(f"target tau={tau} is outside monotone history support")
    return float(np.interp(tau, values, history["time"]))


def lagrange_weights(nodes: np.ndarray, target: float) -> np.ndarray:
    weights = np.ones(len(nodes), dtype=float)
    for i in range(len(nodes)):
        for j in range(len(nodes)):
            if i != j:
                weights[i] *= (target - nodes[j]) / (nodes[i] - nodes[j])
    return weights


def select_snapshots(paths: list[Path], target: float) -> tuple[list[Path], np.ndarray]:
    catalog = sorted((binary_header(path)[0], path) for path in paths)
    times = np.asarray([item[0] for item in catalog])
    if len(times) < 4 or not times[0] <= target <= times[-1]:
        fail(f"target time {target} lacks four-snapshot support")
    position = int(np.searchsorted(times, target))
    start = max(0, min(position - 2, len(times) - 4))
    chosen = catalog[start:start + 4]
    nodes = np.asarray([item[0] for item in chosen])
    return [item[1] for item in chosen], lagrange_weights(nodes, target)


def interpolation_indices(q: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    start = max(0, min(int(math.floor(q - 0.5)) - 2, count - 5))
    indices = np.arange(start, start + 5)
    nodes = indices.astype(float) + 0.5
    return indices, lagrange_weights(nodes, q)


def block_bounds(data: dict[str, Any]) -> np.ndarray:
    """Return active-cell face bounds from Athena binary MeshBlock geometry.

    ``mb_geometry`` stores the center of the first output cell followed by the
    cell spacing: ``x1i,x2i,x3i,dx1,dx2,dx3``.  It does not store face bounds.
    """
    geometry = np.asarray(data["mb_geometry"], dtype=float)
    nx = int(data["nx1_out_mb"])
    ny = int(data["nx2_out_mb"])
    if geometry.ndim != 2 or geometry.shape[1] != 6 or nx < 5 or ny < 5:
        fail("invalid MeshBlock geometry or output extent")
    x1i, x2i = geometry[:, 0], geometry[:, 1]
    dx1, dx2 = geometry[:, 3], geometry[:, 4]
    if np.any(dx1 <= 0.0) or np.any(dx2 <= 0.0):
        fail("nonpositive MeshBlock cell spacing")
    return np.column_stack((x1i - 0.5 * dx1,
                            x1i + (nx - 0.5) * dx1,
                            x2i - 0.5 * dx2,
                            x2i + (ny - 0.5) * dx2))


def coarse_fine_sides(data: dict[str, Any]) -> list[set[str]]:
    geometry = block_bounds(data)
    levels = np.asarray(data["mb_logical"], dtype=int)[:, 3]
    sides = [set() for _ in range(len(geometry))]
    tolerance = 128.0 * np.finfo(float).eps * 32.0
    for i, a in enumerate(geometry):
        for j, b in enumerate(geometry):
            if i == j or levels[i] == levels[j]:
                continue
            z_overlap = min(a[3], b[3]) - max(a[2], b[2])
            r_overlap = min(a[1], b[1]) - max(a[0], b[0])
            if z_overlap > tolerance and abs(a[0] - b[1]) <= tolerance:
                sides[i].add("rlo")
            if z_overlap > tolerance and abs(a[1] - b[0]) <= tolerance:
                sides[i].add("rhi")
            if r_overlap > tolerance and abs(a[2] - b[3]) <= tolerance:
                sides[i].add("zlo")
            if r_overlap > tolerance and abs(a[3] - b[2]) <= tolerance:
                sides[i].add("zhi")
    return sides


def sample_snapshot(data: dict[str, Any], rho: np.ndarray, zed: np.ndarray
                    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    geometry = block_bounds(data)
    nx = int(data["nx1_out_mb"])
    ny = int(data["nx2_out_mb"])
    values = {name: np.full(rho.shape, np.nan) for name in BASE_FIELDS}
    block_distance = np.full(rho.shape, np.inf)
    cf_distance = np.full(rho.shape, np.inf)
    sides = coarse_fine_sides(data)
    for block, bounds in enumerate(geometry):
        rlo, rhi, zlo, zhi = bounds[:4]
        mask = ((rho >= rlo) & (rho < rhi) & (zed >= zlo) & (zed < zhi))
        if not np.any(mask):
            continue
        dx = (rhi - rlo) / nx
        dy = (zhi - zlo) / ny
        positions = np.flatnonzero(mask)
        for flat in positions:
            qx = (rho.flat[flat] - rlo) / dx
            qy = (zed.flat[flat] - zlo) / dy
            ii, wx = interpolation_indices(qx, nx)
            jj, wy = interpolation_indices(qy, ny)
            tensor = wy[:, None] * wx[None, :]
            for name, source in BASE_FIELDS.items():
                array = np.asarray(data["mb_data"][source][block][0], dtype=float)
                values[name].flat[flat] = float(np.sum(array[np.ix_(jj, ii)] * tensor))
            distances = {
                "rlo": (rho.flat[flat] - rlo) / dx,
                "rhi": (rhi - rho.flat[flat]) / dx,
                "zlo": (zed.flat[flat] - zlo) / dy,
                "zhi": (zhi - zed.flat[flat]) / dy,
            }
            block_distance.flat[flat] = min(distances.values())
            if sides[block]:
                cf_distance.flat[flat] = min(distances[side] for side in sides[block])
    if any(not np.isfinite(array).all() for array in values.values()):
        fail("common lattice is not covered by a finite leaf field")
    values["K"] = values["Khat"] + 2.0 * values["Theta"]
    return ({name: values[name] for name in OUTPUT_FIELDS},
            {"block_distance": block_distance, "cf_distance": cf_distance})


def sample_at_time(paths: list[Path], target: float, reader, rho: np.ndarray,
                   zed: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], list[dict[str, Any]]]:
    chosen, weights = select_snapshots(paths, target)
    accumulated = {name: np.zeros(rho.shape) for name in OUTPUT_FIELDS}
    nearest = int(np.argmin([abs(binary_header(path)[0] - target) for path in chosen]))
    masks = None
    provenance = []
    for index, (path, weight) in enumerate(zip(chosen, weights)):
        data = reader(str(path))
        sampled, snapshot_masks = sample_snapshot(data, rho, zed)
        for name in OUTPUT_FIELDS:
            accumulated[name] += weight * sampled[name]
        if index == nearest:
            masks = snapshot_masks
        provenance.append({"path": str(path), "time": data["time"],
                           "cycle": data["cycle"], "temporal_weight": float(weight)})
    assert masks is not None
    return accumulated, masks, provenance


def rms(difference: np.ndarray, weights: np.ndarray, mask: np.ndarray) -> float:
    selected = mask & np.isfinite(difference) & np.isfinite(weights)
    if not np.any(selected) or not np.sum(weights[selected]) > 0.0:
        return math.nan
    return float(np.sqrt(np.sum(weights[selected] * difference[selected] ** 2)
                         / np.sum(weights[selected])))


def write_profile(path: Path, rho: np.ndarray, zed: np.ndarray,
                  fields: dict[str, dict[str, np.ndarray]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    names = [f"{case}_{field}" for case in CASES for field in OUTPUT_FIELDS]
    with gzip.open(path, "wt", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["rho", "z", *names])
        arrays = [fields[case][field] for case in CASES for field in OUTPUT_FIELDS]
        for index in range(rho.size):
            writer.writerow([rho.flat[index], zed.flat[index],
                             *(array.flat[index] for array in arrays)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--n128", type=Path, required=True)
    parser.add_argument("--n256", type=Path, required=True)
    parser.add_argument("--n512", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    reader = load_reader(args.source)
    roots = {case: getattr(args, case) for case in CASES}
    histories = {case: read_history(root / f"{case}.z4c.user.hst")
                 for case, root in roots.items()}
    paths = {case: sorted((root / "bin/rank_00000000").glob(f"{case}.z4c.*.bin"))
             for case, root in roots.items()}
    if any(len(items) < 4 for items in paths.values()):
        fail("insufficient binary snapshots")

    radial = np.arange(0.125, 16.0, 0.25)
    axial = np.arange(-15.875, 16.0, 0.25)
    rho, zed = np.meshgrid(radial, axial, indexing="xy")
    coordinate_weight = rho.copy()
    rows: list[dict[str, Any]] = []
    provenance: dict[str, Any] = {}
    output = args.output
    for tau in TARGETS:
        fields: dict[str, dict[str, np.ndarray]] = {}
        masks: dict[str, dict[str, np.ndarray]] = {}
        provenance[str(tau)] = {}
        for case in CASES:
            time = target_time(histories[case], tau)
            fields[case], masks[case], records = sample_at_time(
                paths[case], time, reader, rho, zed)
            provenance[str(tau)][case] = {"coordinate_time": time,
                                          "snapshots": records}
        regions = {
            "entire": np.ones(rho.shape, dtype=bool),
            "axis_rho_lt_0.5": rho < 0.5,
            "block_interiors": np.logical_and.reduce([
                masks[case]["block_distance"] > 4.0 for case in CASES]),
            "coarse_fine_neighborhood": np.logical_or.reduce([
                masks[case]["cf_distance"] <= 4.0 for case in CASES]),
        }
        for field in OUTPUT_FIELDS:
            coarse = fields["n128"][field] - fields["n256"][field]
            fine = fields["n256"][field] - fields["n512"][field]
            for region, mask in regions.items():
                e_coarse = rms(coarse, coordinate_weight, mask)
                e_fine = rms(fine, coordinate_weight, mask)
                ratio = e_coarse / e_fine if e_fine > 0.0 else math.nan
                order = math.log2(ratio) if ratio > 0.0 else math.nan
                rows.append({"axisTau": tau, "field": field, "region": region,
                             "E_128_256": e_coarse, "E_256_512": e_fine,
                             "Q": ratio, "p": order, "samples": int(mask.sum())})
        write_profile(output / "data/common_tau_profiles" / f"tau_{tau:g}.csv.gz",
                      rho, zed, fields)

    data = output / "data"
    data.mkdir(parents=True, exist_ok=True)
    with (data / "field_convergence.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    (data / "field_sampling_provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    figure, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True)
    for axis, field in zip(axes.flat, OUTPUT_FIELDS):
        selected = [row for row in rows if row["field"] == field and row["region"] == "entire"]
        axis.plot([row["axisTau"] for row in selected], [row["p"] for row in selected], "o-")
        axis.axhline(4.0, color="black", linestyle="--", linewidth=0.8)
        axis.axhline(0.0, color="gray", linewidth=0.6)
        axis.set_title(field); axis.grid(alpha=0.25)
    for axis in axes[-1]: axis.set_xlabel(r"central proper time $\tau_c/M$")
    for axis in axes[:, 0]: axis.set_ylabel("effective order p")
    figure.suptitle("Common-tree Z4c field self-convergence (ring-coordinate RMS)")
    figure.tight_layout()
    figures = output / "figures"; figures.mkdir(parents=True, exist_ok=True)
    figure.savefig(figures / "field_convergence_order.png", dpi=180)
    plt.close(figure)
    print("COMMON_TREE_FIELD_ANALYSIS_PASS")


if __name__ == "__main__":
    main()
