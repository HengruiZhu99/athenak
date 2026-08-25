#!/usr/bin/env python3
"""Production-matched radial and shell constraint integrals for native VC output.

The production history entries named ``*-norm2`` are extensive integrals of
the squared constraint magnitudes.  They are not square roots and are not
volume-normalized RMS values.  For Cartoon SO(2), this script reproduces the
native-VC leaf quadrature

    2*pi*rho*dx1*dx2*w1*w2*sqrt(det(gamma))

on every leaf MeshBlock.  Shared block-endpoint copies remain present with
their local trapezoidal endpoint weights, matching
``Z4cDiagnosticVertexMeasure``.  The rho=0 axis has zero ring measure.
"""

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
FIELDS = ("C", "H", "M", "Z")
CONSTRAINT_VARIABLE = {"C": "con_C", "H": "con_H", "M": "con_M", "Z": "con_Z"}
COLORS = {"n128": "#377eb8", "n256": "#4daf4a", "n512": "#e41a1c"}
HEADER = re.compile(r"\[(\d+)\]=([^\s]+)")


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


def read_history(path: Path) -> dict[str, np.ndarray]:
    labels: dict[str, int] = {}
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            labels.update({name: int(index) - 1 for index, name in HEADER.findall(line)})
        elif line.strip():
            rows.append([float(value) for value in line.split()])
    required = {"time", "axisTau", "C-norm2", "H-norm2", "M-norm2", "Z-norm2"}
    require(not required - labels.keys(), f"{path}: missing history columns")
    array = np.asarray(rows, dtype=float)
    require(array.ndim == 2 and len(array) > 1 and np.isfinite(array).all(),
            f"{path}: invalid history")
    by_time = {float(row[labels["time"]]): row for row in array}
    array = np.asarray([by_time[key] for key in sorted(by_time)], dtype=float)
    return {name: array[:, index] for name, index in labels.items()}


def parameter(header: list[str], block: str, key: str, default: float) -> float:
    active = ""
    for line in header:
        if line.startswith("<"):
            active = line.strip("<>")
            continue
        if active == block and "=" in line:
            name, value = line.split("=", 1)
            if name.strip() == key:
                return float(value.strip())
    return default


def catalogs(root: Path) -> tuple[dict[float, Path], dict[float, Path]]:
    binary_root = root / "bin/rank_00000000"
    state_paths = sorted(binary_root.glob("*.state.*.bin"))
    constraint_paths = sorted(binary_root.glob("*.constraints.*.bin"))
    require(state_paths and constraint_paths, f"missing binary outputs under {binary_root}")
    state = {round(binary_header(path)[0], 13): path for path in state_paths}
    constraints = {round(binary_header(path)[0], 13): path for path in constraint_paths}
    common = sorted(set(state) & set(constraints))
    require(common, f"no state/constraint time pairs under {root}")
    return ({time: state[time] for time in common},
            {time: constraints[time] for time in common})


def metric_determinant(fields: dict[str, np.ndarray]) -> np.ndarray:
    gxx = fields["z4c_gxx"]
    gxy = fields["z4c_gxy"]
    gxz = fields["z4c_gxz"]
    gyy = fields["z4c_gyy"]
    gyz = fields["z4c_gyz"]
    gzz = fields["z4c_gzz"]
    return (gxx * (gyy * gzz - gyz * gyz)
            - gxy * (gxy * gzz - gyz * gxz)
            + gxz * (gxy * gyz - gyy * gxz))


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def finite_maximum(values) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return max(finite) if finite else None


def integrate_snapshot(state_path: Path, constraint_path: Path, reader,
                       cutoffs: tuple[float, ...],
                       shell_edges: tuple[float, ...]) -> tuple[list[dict], dict]:
    state = reader(str(state_path))
    constraints = reader(str(constraint_path))
    require(abs(float(state["time"]) - float(constraints["time"])) <= 2.0e-12,
            "state/constraint time mismatch")
    require(int(state["cycle"]) == int(constraints["cycle"]),
            "state/constraint cycle mismatch")
    require(np.array_equal(state["mb_logical"], constraints["mb_logical"]),
            "state/constraint logical trees differ")
    require(np.array_equal(state["mb_geometry"], constraints["mb_geometry"]),
            "state/constraint geometries differ")
    required_state = {"z4c_chi", "z4c_gxx", "z4c_gxy", "z4c_gxz",
                      "z4c_gyy", "z4c_gyz", "z4c_gzz"}
    require(not required_state - set(state["var_names"]), "state lacks metric fields")
    require(not set(CONSTRAINT_VARIABLE.values()) - set(constraints["var_names"]),
            "constraint output lacks C/H/M/Z")
    chi_power = parameter(state["header"], "z4c", "chi_psi_power", -4.0)
    require(chi_power != 0.0, "zero chi_psi_power")

    region_names = [f"R{value:g}" for value in cutoffs] + ["full"]
    totals = {(region, field): 0.0 for region in region_names for field in FIELDS}
    volumes = {region: 0.0 for region in region_names}
    shell_names = [f"shell_{shell_edges[index]:g}_{shell_edges[index + 1]:g}"
                   for index in range(len(shell_edges) - 1)]
    shell_names.append(f"shell_gt_{shell_edges[-1]:g}")
    for shell in shell_names:
        for field in FIELDS:
            totals[(shell, field)] = 0.0
        volumes[shell] = 0.0
    maxima = {(region, field): {"weighted": -1.0, "magnitude": -1.0,
                                "weighted_rho": math.nan, "weighted_z": math.nan,
                                "magnitude_rho": math.nan, "magnitude_z": math.nan}
              for region in region_names + shell_names for field in FIELDS}

    minimum_chi = math.inf
    minimum_conformal_spd_pivot = math.inf
    minimum_det_tilde = math.inf
    minimum_det_gamma = math.inf
    for block, bounds in enumerate(np.asarray(state["mb_geometry"], dtype=float)):
        state_fields = {name: np.asarray(state["mb_data"][name][block], dtype=float)
                        for name in required_state}
        constraint_fields = {
            field: np.asarray(constraints["mb_data"][name][block], dtype=float)
            for field, name in CONSTRAINT_VARIABLE.items()
        }
        shape = state_fields["z4c_chi"].shape
        require(shape == next(iter(constraint_fields.values())).shape,
                "state/constraint block shape mismatch")
        require(shape[0] == 1 and shape[1] >= 2 and shape[2] >= 2,
                "expected collapsed native-VC block")
        ni, nj = shape[2], shape[1]
        rho_1d = np.linspace(bounds[0], bounds[1], ni)
        z_1d = np.linspace(bounds[2], bounds[3], nj)
        rho, zed = np.meshgrid(rho_1d, z_1d)
        radius = np.hypot(rho, zed)
        dx1 = (bounds[1] - bounds[0]) / (ni - 1)
        dx2 = (bounds[3] - bounds[2]) / (nj - 1)
        weight1 = np.ones(ni); weight1[[0, -1]] = 0.5
        weight2 = np.ones(nj); weight2[[0, -1]] = 0.5
        trapezoid = weight2[:, None] * weight1[None, :]

        chi = state_fields["z4c_chi"][0]
        det_tilde = metric_determinant({name: value[0]
                                        for name, value in state_fields.items()})
        pivot0 = state_fields["z4c_gxx"][0]
        pivot1 = (state_fields["z4c_gxx"][0] * state_fields["z4c_gyy"][0]
                  - state_fields["z4c_gxy"][0] ** 2)
        psi4 = np.power(chi, 4.0 / chi_power)
        det_gamma = np.power(psi4, 3) * det_tilde
        require(np.isfinite(chi).all() and np.all(chi > 0.0), "invalid chi in snapshot")
        require(np.isfinite(det_tilde).all() and np.all(det_tilde > 0.0),
                "invalid conformal determinant in snapshot")
        require(np.isfinite(pivot0).all() and np.all(pivot0 > 0.0)
                and np.isfinite(pivot1).all() and np.all(pivot1 > 0.0),
                "invalid conformal SPD pivot in snapshot")
        require(np.isfinite(det_gamma).all() and np.all(det_gamma > 0.0),
                "invalid physical determinant in snapshot")
        minimum_chi = min(minimum_chi, float(np.min(chi)))
        minimum_conformal_spd_pivot = min(
            minimum_conformal_spd_pivot,
            float(np.min(pivot0)), float(np.min(pivot1)), float(np.min(det_tilde)))
        minimum_det_tilde = min(minimum_det_tilde, float(np.min(det_tilde)))
        minimum_det_gamma = min(minimum_det_gamma, float(np.min(det_gamma)))
        measure = 2.0 * math.pi * rho * dx1 * dx2 * trapezoid * np.sqrt(det_gamma)
        squared: dict[str, np.ndarray] = {}
        for field, raw3 in constraint_fields.items():
            raw = raw3[0]
            require(np.isfinite(raw).all(), f"nonfinite {field}")
            if field == "H":
                squared[field] = raw * raw
            else:
                tolerance = 128.0 * np.finfo(float).eps * max(1.0, float(np.max(np.abs(raw))))
                require(float(np.min(raw)) >= -tolerance,
                        f"negative squared constraint inventory for {field}")
                squared[field] = np.maximum(raw, 0.0)

        masks = {f"R{value:g}": radius <= value + 4.0e-13 for value in cutoffs}
        masks["full"] = np.ones_like(radius, dtype=bool)
        for index in range(len(shell_edges) - 1):
            lower, upper = shell_edges[index:index + 2]
            masks[shell_names[index]] = ((radius > lower + 4.0e-13)
                                         & (radius <= upper + 4.0e-13))
        masks[shell_names[-1]] = radius > shell_edges[-1] + 4.0e-13

        for region, mask in masks.items():
            volumes[region] += float(np.sum(measure[mask]))
            for field in FIELDS:
                contribution = measure * squared[field]
                totals[(region, field)] += float(np.sum(contribution[mask]))
                magnitude = np.sqrt(squared[field])
                weighted_view = np.where(mask, contribution, -1.0)
                magnitude_view = np.where(mask, magnitude, -1.0)
                iw = int(np.argmax(weighted_view)); im = int(np.argmax(magnitude_view))
                weighted_value = float(weighted_view.flat[iw])
                magnitude_value = float(magnitude_view.flat[im])
                record = maxima[(region, field)]
                if weighted_value > record["weighted"]:
                    jw, kw = np.unravel_index(iw, weighted_view.shape)
                    record.update(weighted=weighted_value,
                                  weighted_rho=float(rho[jw, kw]),
                                  weighted_z=float(zed[jw, kw]))
                if magnitude_value > record["magnitude"]:
                    jm, km = np.unravel_index(im, magnitude_view.shape)
                    record.update(magnitude=magnitude_value,
                                  magnitude_rho=float(rho[jm, km]),
                                  magnitude_z=float(zed[jm, km]))

    rows: list[dict] = []
    for region in region_names + shell_names:
        for field in FIELDS:
            rows.append({"region": region, "field": field,
                         "norm2": totals[(region, field)],
                         "proper_volume": volumes[region],
                         **maxima[(region, field)]})
    return rows, {
        "time": float(state["time"]),
        "cycle": int(state["cycle"]),
        "leaf_count": int(state["n_mbs"]),
        "minimum_chi": minimum_chi,
        "minimum_conformal_spd_pivot": minimum_conformal_spd_pivot,
        "minimum_conformal_determinant": minimum_det_tilde,
        "minimum_physical_determinant": minimum_det_gamma,
        "chi_psi_power": chi_power,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    for case in CASES:
        parser.add_argument(f"--{case}", type=Path, required=True)
    parser.add_argument("--cutoffs", default="4,8,12,14")
    parser.add_argument("--shell-edges", default="0,4,8,12,16")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cutoffs = tuple(float(value) for value in args.cutoffs.split(","))
    shell_edges = tuple(float(value) for value in args.shell_edges.split(","))
    require(all(b > a for a, b in zip(shell_edges, shell_edges[1:])),
            "shell edges must increase")
    roots = {case: getattr(args, case) for case in CASES}
    reader = load_reader(args.source)

    radial_rows: list[dict] = []
    health_rows: list[dict] = []
    histories: dict[str, dict[str, np.ndarray]] = {}
    for case, root in roots.items():
        hst_files = list(root.glob("*.z4c.user.hst"))
        require(len(hst_files) == 1, f"{root}: expected one Z4c history")
        histories[case] = read_history(hst_files[0])
        states, constraints = catalogs(root)
        for key in states:
            rows, health = integrate_snapshot(states[key], constraints[key], reader,
                                              cutoffs, shell_edges)
            axis_tau = float(np.interp(health["time"], histories[case]["time"],
                                       histories[case]["axisTau"]))
            for row in rows:
                radial_rows.append({"resolution": case, "time": health["time"],
                                    "axisTau": axis_tau, **row})
            full = {row["field"]: row["norm2"] for row in rows if row["region"] == "full"}
            history_full = {field: float(np.interp(
                health["time"], histories[case]["time"],
                histories[case][f"{field}-norm2"])) for field in FIELDS}
            health_rows.append({"resolution": case, "axisTau": axis_tau, **health,
                                **{f"full_reconstructed_{field}": full[field] for field in FIELDS},
                                **{f"full_history_{field}": history_full[field] for field in FIELDS},
                                **{f"full_relative_difference_{field}":
                                   ((full[field] - history_full[field]) / history_full[field]
                                    if history_full[field] != 0.0 else math.nan)
                                   for field in FIELDS}})

    output = args.output
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(output / "radial_constraint_integrals.csv", radial_rows)
    write_csv(output / "radial_reconstruction_health.csv", health_rows)

    regions = [f"R{value:g}" for value in cutoffs] + ["full"]
    common_low = max(min(row["axisTau"] for row in radial_rows
                         if row["resolution"] == case) for case in CASES)
    common_high = min(max(row["axisTau"] for row in radial_rows
                          if row["resolution"] == case) for case in CASES)
    tau = np.linspace(common_low, common_high, 401)
    convergence_rows: list[dict] = []
    for region in regions:
        for field in FIELDS:
            series: dict[str, np.ndarray] = {}
            for case in CASES:
                selected = sorted((row for row in radial_rows
                                   if row["resolution"] == case
                                   and row["region"] == region
                                   and row["field"] == field),
                                  key=lambda row: row["axisTau"])
                series[case] = np.interp(tau,
                                         [row["axisTau"] for row in selected],
                                         [row["norm2"] for row in selected])
            for index, proper_time in enumerate(tau):
                n128, n256, n512 = (float(series[case][index]) for case in CASES)
                p12 = math.log2(n128 / n256) if n128 > 0.0 and n256 > 0.0 else math.nan
                p23 = math.log2(n256 / n512) if n256 > 0.0 and n512 > 0.0 else math.nan
                e12, e23 = abs(n128 - n256), abs(n256 - n512)
                pself = math.log2(e12 / e23) if e12 > 0.0 and e23 > 0.0 else math.nan
                convergence_rows.append({"axisTau": float(proper_time), "region": region,
                                         "field": field, "n128": n128, "n256": n256,
                                         "n512": n512, "p_128_256": p12,
                                         "p_256_512": p23, "p_self": pself})
    write_csv(output / "radial_constraint_convergence.csv", convergence_rows)

    shell_rows: list[dict] = []
    shell_names = [f"shell_{shell_edges[index]:g}_{shell_edges[index + 1]:g}"
                   for index in range(len(shell_edges) - 1)] + [f"shell_gt_{shell_edges[-1]:g}"]
    for row in radial_rows:
        if row["region"] not in shell_names:
            continue
        full = next(item["norm2"] for item in radial_rows
                    if item["resolution"] == row["resolution"]
                    and item["time"] == row["time"] and item["region"] == "full"
                    and item["field"] == row["field"])
        shell_rows.append({**row, "fraction_of_full": row["norm2"] / full if full > 0.0 else math.nan})
    write_csv(output / "radial_shell_budget.csv", shell_rows)

    for region in regions:
        fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), constrained_layout=True)
        for axis, field in zip(axes.flat, FIELDS):
            for case in CASES:
                selected = sorted((row for row in radial_rows
                                   if row["resolution"] == case
                                   and row["region"] == region and row["field"] == field),
                                  key=lambda row: row["axisTau"])
                axis.semilogy([row["axisTau"] for row in selected],
                              [row["norm2"] for row in selected],
                              "o-", color=COLORS[case], label=case.upper(), markersize=3)
            axis.set_title(field); axis.set_xlabel(r"central proper time $\tau_c/M$")
            axis.grid(alpha=0.24, which="both")
        axes[0, 0].legend()
        fig.suptitle(f"Constraint integrals: {region}")
        fig.savefig(figures / f"constraints_{region}.png", dpi=240)
        fig.savefig(figures / f"constraints_{region}.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(len(FIELDS), len(regions), figsize=(4.0 * len(regions), 3.0 * len(FIELDS)),
                             constrained_layout=True, squeeze=False)
    for row_index, field in enumerate(FIELDS):
        for column, region in enumerate(regions):
            axis = axes[row_index, column]
            selected = [row for row in convergence_rows
                        if row["field"] == field and row["region"] == region]
            axis.plot([row["axisTau"] for row in selected],
                      [row["p_128_256"] for row in selected], label="N128-N256")
            axis.plot([row["axisTau"] for row in selected],
                      [row["p_256_512"] for row in selected], label="N256-N512")
            axis.axhline(4.0, color="black", linestyle="--", linewidth=0.7)
            axis.axhline(0.0, color="black", linewidth=0.5)
            axis.set_title(f"{field}, {region}"); axis.set_ylim(-2.0, 10.0)
            axis.grid(alpha=0.22)
    axes[0, 0].legend(fontsize=7)
    fig.savefig(figures / "radial_pairwise_orders.png", dpi=220)
    fig.savefig(figures / "radial_pairwise_orders.pdf")
    plt.close(fig)

    terminal_rows = [row for row in convergence_rows
                     if abs(row["axisTau"] - common_high) <= 1.0e-12]
    summary = {
        "schema": "z4c_vc_production_matched_radial_constraints_v1",
        "quadrature": "native VC leaf trapezoid times 2*pi*rho*sqrt(det(gamma))",
        "history_semantics": "extensive squared-constraint integrals; no square root or RMS normalization",
        "common_axis_tau": [common_low, common_high],
        "terminal": terminal_rows,
        "maximum_full_history_reconstruction_relative_difference": {
            field: finite_maximum(abs(row[f"full_relative_difference_{field}"])
                                  for row in health_rows)
            for field in FIELDS
        },
    }
    (output / "radial_constraint_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
