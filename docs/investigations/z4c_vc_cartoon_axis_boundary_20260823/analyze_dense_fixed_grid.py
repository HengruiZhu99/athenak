#!/usr/bin/env python3
"""Common-time, common-vertex convergence audit for fixed-grid VC Brill data."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np


TAUS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 3.0)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_time(path: Path) -> tuple[float, int]:
    with path.open("rb") as stream:
        require(stream.readline().split()[0] == b"Athena", f"bad binary {path}")
        count = int(stream.readline().split(b"=")[-1])
        header = {}
        for _ in range(count - 1):
            key, value = stream.readline().decode().split("=")
            header[key.strip()] = value.strip()
    return float(header["time"]), int(header["cycle"])


def read_history(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = next(line for line in lines if line.startswith("#  [1]="))
    labels = [part.split("=")[-1].strip()
              for part in header[2:].split("[") if "]=" in part]
    rows = np.asarray([[float(value) for value in line.split()]
                       for line in lines if line and not line.startswith("#")])
    return labels, rows


def target_times(run: Path) -> dict[float, float]:
    history = next(run.glob("*.z4c.user.hst"))
    labels, rows = read_history(history)
    tau = rows[:, labels.index("axisTau")]
    time = rows[:, labels.index("time")]
    return {value: float(np.interp(value, tau, time)) for value in TAUS}


def canonical(data: dict) -> tuple[list[str], np.ndarray, float]:
    names = list(data["var_names"])
    nr, nz = data["Nx1"] + 1, data["Nx2"] + 1
    values = np.full((len(names), nz, nr), np.nan)
    spread = 0.0
    hits = np.zeros((nz, nr), dtype=np.int16)
    hr = (data["x1max"] - data["x1min"]) / data["Nx1"]
    hz = (data["x2max"] - data["x2min"]) / data["Nx2"]
    for block, geometry in enumerate(data["mb_geometry"]):
        shape = data["mb_data"][names[0]][block].shape
        require(shape[0] == 1, "Cartoon binary has noncollapsed x3")
        rho = np.linspace(geometry[0], geometry[1], shape[2])
        zed = np.linspace(geometry[2], geometry[3], shape[1])
        ii = np.rint((rho - data["x1min"]) / hr).astype(int)
        jj = np.rint((zed - data["x2min"]) / hz).astype(int)
        for j_local, j_global in enumerate(jj):
            for i_local, i_global in enumerate(ii):
                point = np.asarray([data["mb_data"][name][block][0, j_local, i_local]
                                    for name in names])
                if hits[j_global, i_global]:
                    spread = max(spread, float(np.max(np.abs(
                        values[:, j_global, i_global] - point))))
                else:
                    values[:, j_global, i_global] = point
                hits[j_global, i_global] += 1
    require(np.all(hits > 0) and np.all(np.isfinite(values)),
            "canonical grid is incomplete or nonfinite")
    return names, values, spread


def interpolated_snapshot(run: Path, time: float, bin_convert) -> dict:
    paths = sorted((run / "bin").glob("*.state.*.bin"))
    samples = [(read_time(path)[0], path) for path in paths]
    if time <= samples[0][0] + 1.0e-14:
        lo = hi = samples[0]
    else:
        hi_index = next(index for index, sample in enumerate(samples)
                        if sample[0] >= time)
        lo, hi = samples[hi_index - 1], samples[hi_index]
    lo_data = bin_convert.read_binary(str(lo[1]))
    lo_names, lo_values, lo_spread = canonical(lo_data)
    if lo[1] == hi[1]:
        values, hi_spread, weight = lo_values, lo_spread, 0.0
    else:
        hi_data = bin_convert.read_binary(str(hi[1]))
        hi_names, hi_values, hi_spread = canonical(hi_data)
        require(lo_names == hi_names, "variable inventory changed in time")
        weight = (time - lo[0]) / (hi[0] - lo[0])
        values = (1.0 - weight) * lo_values + weight * hi_values
    return {"names": lo_names, "values": values,
            "bracket": [lo[0], hi[0]], "weight": weight,
            "shared_spread": max(lo_spread, hi_spread)}


def masks(nr: int, nz: int) -> dict[str, np.ndarray]:
    rho = np.linspace(0.0, 16.0, nr)
    zed = np.linspace(-16.0, 16.0, nz)
    rr, zz = np.meshgrid(rho, zed)
    h = 16.0 / (nr - 1)
    seam_r = np.isclose(np.mod(rr, 4.0), 0.0, atol=1.0e-12)
    seam_z = np.isclose(np.mod(zz + 16.0, 4.0), 0.0, atol=1.0e-12)
    edge_distance = np.minimum(16.0 - rr, 16.0 - np.abs(zz))
    result = {
        "full": np.ones_like(rr, dtype=bool),
        "axis": np.isclose(rr, 0.0),
        "z0_seam": np.isclose(zz, 0.0),
        "same_level_seams": seam_r | seam_z,
        "core_r8": np.hypot(rr, zz) <= 8.0 + 1.0e-12,
        "core_r8_mb_interior": ((np.hypot(rr, zz) <= 8.0 + 1.0e-12) &
                                ~seam_r & ~seam_z & (rr > 4*h)),
    }
    for layer in range(1, 5):
        result[f"radial_layer_{layer}"] = np.isclose(rr, layer * h)
    for layer in range(5):
        result[f"outer_layer_{layer}"] = np.isclose(edge_distance, layer * h)
    return result


def metric(values: np.ndarray, mask: np.ndarray, rr: np.ndarray) -> dict[str, float]:
    selected = values[mask]
    weights = rr[mask]
    ring = (float(np.sqrt(np.sum(weights * selected**2) / np.sum(weights)))
            if np.sum(weights) > 0.0 else None)
    return {"rms": float(np.sqrt(np.mean(selected**2))),
            "ring_rms": ring, "linf": float(np.max(np.abs(selected)))}


def order(first: float | None, second: float | None) -> float | None:
    if first is None or second is None or first <= 0.0 or second <= 0.0:
        return None
    return math.log(first / second, 2.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs=2, action="append", required=True,
                        metavar=("N", "PATH"))
    parser.add_argument("--vis-python", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.vis_python.resolve()))
    import bin_convert  # pylint: disable=import-error,import-outside-toplevel

    runs = {int(n): Path(path).resolve() for n, path in args.run}
    require(sorted(runs) == [128, 256, 512], "requires N128/N256/N512")
    common_times = target_times(runs[512])
    rows = []
    provenance = []
    earliest = None
    for tau, time in common_times.items():
        snapshots = {n: interpolated_snapshot(run, time, bin_convert)
                     for n, run in runs.items()}
        require(snapshots[128]["names"] == snapshots[256]["names"] ==
                snapshots[512]["names"], "state inventory differs")
        # Restrict both pairwise differences to the same N128 physical vertices.
        coarse = snapshots[128]["values"]
        middle = snapshots[256]["values"][:, ::2, ::2]
        fine = snapshots[512]["values"][:, ::4, ::4]
        require(coarse.shape == middle.shape == fine.shape, "nested VC grids mismatch")
        regions = masks(coarse.shape[2], coarse.shape[1])
        rho = np.linspace(0.0, 16.0, coarse.shape[2])
        zed = np.linspace(-16.0, 16.0, coarse.shape[1])
        rr, _ = np.meshgrid(rho, zed)
        for index, name in enumerate(snapshots[128]["names"]):
            d1 = middle[index] - coarse[index]
            d2 = fine[index] - middle[index]
            for region_name, mask in regions.items():
                first, second = metric(d1, mask, rr), metric(d2, mask, rr)
                record = {"axis_tau": tau, "coordinate_time": time,
                          "variable": name, "region": region_name}
                for kind in ("rms", "ring_rms", "linf"):
                    record[f"difference_128_256_{kind}"] = first[kind]
                    record[f"difference_256_512_{kind}"] = second[kind]
                    record[f"observed_order_{kind}"] = order(first[kind], second[kind])
                rows.append(record)
                if (tau > 0.0 and record["observed_order_rms"] is not None and
                        record["observed_order_rms"] < 0.0):
                    candidate = (tau, name, region_name,
                                 record["observed_order_rms"])
                    if earliest is None or candidate[0] < earliest[0]:
                        earliest = candidate
        provenance.append({"axis_tau": tau, "coordinate_time": time,
                           "snapshots": {str(n): {
                               "bracket": snapshots[n]["bracket"],
                               "weight": snapshots[n]["weight"],
                               "shared_spread": snapshots[n]["shared_spread"]}
                               for n in snapshots}})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "state_region_convergence.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {"schema": "z4c_vc_dense_fixed_grid_v1",
               "common_times": provenance,
               "earliest_negative_rms_order": earliest,
               "negative_order_count": sum(
                   row["observed_order_rms"] is not None and
                   row["observed_order_rms"] < 0.0 for row in rows)}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
