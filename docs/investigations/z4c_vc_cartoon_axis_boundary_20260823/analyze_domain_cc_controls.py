#!/usr/bin/env python3
"""Analyze large-domain VC and matched cell-centered fixed-grid controls."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
import sys

import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical(data: dict, vertex: bool) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    names = list(data["var_names"])
    nr = data["Nx1"] + int(vertex)
    nz = data["Nx2"] + int(vertex)
    values = np.full((len(names), nz, nr), np.nan)
    hits = np.zeros((nz, nr), dtype=np.int16)
    hr = (data["x1max"] - data["x1min"]) / data["Nx1"]
    hz = (data["x2max"] - data["x2min"]) / data["Nx2"]
    rho_global = (np.linspace(data["x1min"], data["x1max"], nr) if vertex else
                  data["x1min"] + (np.arange(nr) + 0.5) * hr)
    z_global = (np.linspace(data["x2min"], data["x2max"], nz) if vertex else
                data["x2min"] + (np.arange(nz) + 0.5) * hz)
    for block, geometry in enumerate(data["mb_geometry"]):
        shape = data["mb_data"][names[0]][block].shape
        if vertex:
            rho = np.linspace(geometry[0], geometry[1], shape[2])
            zed = np.linspace(geometry[2], geometry[3], shape[1])
        else:
            rho = geometry[0] + (np.arange(shape[2]) + 0.5) * (geometry[1]-geometry[0])/shape[2]
            zed = geometry[2] + (np.arange(shape[1]) + 0.5) * (geometry[3]-geometry[2])/shape[1]
        ii = np.rint((rho - rho_global[0]) / hr).astype(int)
        jj = np.rint((zed - z_global[0]) / hz).astype(int)
        for j_local, j_global in enumerate(jj):
            for i_local, i_global in enumerate(ii):
                if not hits[j_global, i_global]:
                    values[:, j_global, i_global] = [
                        data["mb_data"][name][block][0, j_local, i_local]
                        for name in names]
                hits[j_global, i_global] += 1
    require(np.all(hits > 0) and np.all(np.isfinite(values)), "canonical grid invalid")
    return names, values, rho_global, z_global


def lagrange_resample_axis(values: np.ndarray, source: np.ndarray,
                           target: np.ndarray, axis: int) -> np.ndarray:
    result_shape = list(values.shape)
    result_shape[axis] = len(target)
    result = np.empty(result_shape, dtype=float)
    for destination, coordinate in enumerate(target):
        insertion = int(np.searchsorted(source, coordinate))
        start = max(0, min(insertion - 2, len(source) - 4))
        indices = np.arange(start, start + 4)
        nodes = source[indices]
        weights = np.ones(4)
        for a in range(4):
            for b in range(4):
                if a != b:
                    weights[a] *= (coordinate - nodes[b]) / (nodes[a] - nodes[b])
        selected = np.take(values, indices, axis=axis)
        moved = np.moveaxis(selected, axis, -1)
        interpolated = np.tensordot(moved, weights, axes=([-1], [0]))
        destination_slice = [slice(None)] * result.ndim
        destination_slice[axis] = destination
        result[tuple(destination_slice)] = interpolated
    return result


def resample(values: np.ndarray, rho: np.ndarray, zed: np.ndarray,
             target_rho: np.ndarray, target_z: np.ndarray) -> np.ndarray:
    radial = lagrange_resample_axis(values, rho, target_rho, axis=2)
    return lagrange_resample_axis(radial, zed, target_z, axis=1)


def observed(first: float, second: float) -> float | None:
    return math.log(first / second, 2.0) if first > 0.0 and second > 0.0 else None


def region_masks(rho: np.ndarray, zed: np.ndarray, domain: float) -> dict[str, np.ndarray]:
    rr, zz = np.meshgrid(rho, zed)
    h = rho[1] - rho[0]
    seams = ((np.abs(np.remainder(rr, 4.0)) < 1e-12) |
             (np.abs(np.remainder(zz, 4.0)) < 1e-12))
    nearest_axis = float(np.min(rho))
    nearest_outer = min(float(domain - np.max(rho)),
                        float(domain - np.max(np.abs(zed))))
    return {"full": np.ones_like(rr, dtype=bool),
            "axis_or_nearest": np.abs(rr - nearest_axis) < 1e-12,
            "core_r8": np.hypot(rr, zz) <= 8.0 + 1e-12,
            "core_r8_mb_interior": ((np.hypot(rr, zz) <= 8.0 + 1e-12) &
                                    ~seams & (rr > 4*h)),
            "outer_or_nearest": ((np.abs((domain-rr)-nearest_outer) < 1e-12) |
                                 (np.abs((domain-np.abs(zz))-nearest_outer) < 1e-12))}


def convergence(cases: list[dict], vertex: bool, domain: float) -> list[dict]:
    reference_rho, reference_z = cases[0]["rho"], cases[0]["zed"]
    common = []
    for case in cases:
        common.append(case["values"] if case is cases[0] else
                      resample(case["values"], case["rho"], case["zed"],
                               reference_rho, reference_z))
    masks = region_masks(reference_rho, reference_z, domain)
    rows = []
    for index, name in enumerate(cases[0]["names"]):
        for region, mask in masks.items():
            first = common[1][index][mask] - common[0][index][mask]
            second = common[2][index][mask] - common[1][index][mask]
            e1 = float(np.sqrt(np.mean(first**2)))
            e2 = float(np.sqrt(np.mean(second**2)))
            rows.append({"variable": name, "region": region,
                         "difference_128_256_rms": e1,
                         "difference_256_512_rms": e2,
                         "observed_order_rms": observed(e1, e2)})
    return rows


def load_binary_case(root: Path, geometry: str, resolution: int, variable: str,
                     vertex: bool, bin_convert) -> dict:
    paths = sorted((root / geometry / f"N{resolution}" / "bin").glob(
        f"*.{variable}.*.bin"))
    require(paths, f"missing {geometry} N{resolution} {variable} binary")
    path = paths[-1]
    data = bin_convert.read_binary(str(path))
    names, values, rho, zed = canonical(data, vertex)
    return {"names": names, "values": values, "rho": rho, "zed": zed,
            "time": data["time"], "cycle": data["cycle"]}


def load_base_csv(path: Path, family: str) -> tuple[list[str], dict]:
    prefix = "state_z4c_" if family == "state" else "con_"
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream)
                if int(row["rk_stage"]) == 1 and int(row["canonical_owner"]) == 1]
    columns = ([name for name in rows[0] if name.startswith(prefix)] if family == "state"
               else ["con_C", "con_H", "con_M", "con_Z", "con_Mx", "con_My", "con_Mz"])
    names = [name.removeprefix("state_") for name in columns]
    return names, {(round(float(row["rho"]), 13), round(float(row["x2"]), 13)):
                   np.asarray([float(row[name]) for name in columns]) for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--base-exact-root", required=True, type=Path)
    parser.add_argument("--vis-python", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.vis_python))
    import bin_convert  # pylint: disable=import-error,import-outside-toplevel

    all_rows = []
    summary = {"schema": "z4c_vc_domain_cc_controls_v1"}
    for geometry, vertex, domain in (("vc_large", True, 32.0),
                                      ("cc_base", False, 16.0)):
        summary[geometry] = {}
        for variable in ("state", "constraints"):
            cases = [load_binary_case(args.root, geometry, n, variable, vertex,
                                      bin_convert) for n in (128, 256, 512)]
            require(cases[0]["names"] == cases[1]["names"] == cases[2]["names"],
                    "variable inventory differs")
            rows = convergence(cases, vertex, domain)
            for row in rows:
                row.update({"geometry": geometry, "family": variable})
            all_rows.extend(rows)
            minima = {}
            for region in sorted({row["region"] for row in rows}):
                significant = [row["observed_order_rms"] for row in rows
                               if row["region"] == region and
                               row["observed_order_rms"] is not None and
                               row["difference_256_512_rms"] > 1e-12]
                minima[region] = min(significant) if significant else None
            summary[geometry][variable] = {"minimum_order_by_region": minima}

    base_large = []
    for resolution in (128, 256, 512):
        stride = resolution // 128
        for family, variable in (("state", "state"), ("constraints", "constraints")):
            base_path = (args.base_exact_root / f"N{resolution}" / "tau1250" /
                         "diagnostic" / "rhs.rank000000.csv.gz")
            base_names, base = load_base_csv(base_path, family)
            large = load_binary_case(args.root, "vc_large", resolution, variable,
                                     True, bin_convert)
            require(base_names == large["names"], "base/large variable inventory differs")
            rho_indices = np.arange(0, resolution + 1, stride)
            z_indices = np.arange(resolution, 3 * resolution + 1, stride)
            sampled = large["values"][:, z_indices[:, None], rho_indices[None, :]]
            rho = large["rho"][rho_indices]
            zed = large["zed"][z_indices]
            rr, zz = np.meshgrid(rho, zed)
            for index, name in enumerate(base_names):
                delta = np.asarray([sampled[index, j, i] - base[(round(float(r),13),
                                                                  round(float(z),13))][index]
                                    for j, z in enumerate(zed) for i, r in enumerate(rho)])
                core = (np.hypot(rr, zz) <= 8.0 + 1e-12).ravel()
                base_large.append({"resolution": resolution, "family": family,
                                   "variable": name,
                                   "core_r8_rms": float(np.sqrt(np.mean(delta[core]**2))),
                                   "core_r8_linf": float(np.max(np.abs(delta[core])))})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for filename, rows in (("convergence.csv", all_rows),
                           ("base_large_core_difference.csv", base_large)):
        with (args.output_dir / filename).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    summary["base_large_core_max_linf"] = max(row["core_r8_linf"] for row in base_large)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
