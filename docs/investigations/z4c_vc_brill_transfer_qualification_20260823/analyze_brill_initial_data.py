#!/usr/bin/env python3
"""Analyze directly sampled native-VC Brill initial-data diagnostics."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np


STATE = "state_z4c_"
CONSTRAINTS = ("con_C", "con_H", "con_M", "con_Z", "con_Mx", "con_My", "con_Mz")
DISCRETE_DERIVATIVE_FIELDS = {"Gamx", "Gamy", "Gamz"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_rows(run: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    paths = sorted(run.glob("rhs.rank*.csv")) + sorted(run.glob("rhs.rank*.csv.gz"))
    for path in paths:
        if path.suffix == ".gz":
            stream_context = gzip.open(path, "rt", newline="", encoding="utf-8")
        else:
            stream_context = path.open(newline="", encoding="utf-8")
        with stream_context as stream:
            rows.extend(csv.DictReader(stream))
    require(rows and all(row["schema"] == "z4c_vc_rhs_field_v2" for row in rows),
            f"invalid native-VC diagnostic in {run}")
    require(all(float(row["time"]) == 0.0 and int(row["cycle"]) == 0 and
                int(row["rk_stage"]) == 1 for row in rows),
            f"initial-data diagnostic in {run} is not the first production RHS")
    return rows


def state_names(row: dict[str, str]) -> list[str]:
    names = [name.removeprefix(STATE) for name in row if name.startswith(STATE)]
    require(len(names) == 25, "initial-data diagnostic lacks 25 evolved fields")
    return names


def canonical_key(row: dict[str, str]) -> tuple[int, int, int]:
    return int(row["key1"]), int(row["key2"]), int(row["key3"])


def physical_key(row: dict[str, str]) -> tuple[float, float, float]:
    return tuple(round(float(row[name]), 13) for name in ("rho", "x2", "x3"))


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values)))


def run_metrics(resolution: int, rows: list[dict[str, str]]) -> dict[str, object]:
    names = state_names(rows[0])
    groups: dict[tuple[int, int, int], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[canonical_key(row)].append(row)
    shared = [group for group in groups.values() if len(group) > 1]
    require(shared, "uniform multiblock Brill audit found no shared vertices")
    shared_state_spread = 0.0
    for group in shared:
        for name in names:
            values = [float(row[STATE + name]) for row in group]
            shared_state_spread = max(shared_state_spread, max(values) - min(values))

    owners = [row for row in rows if int(row["canonical_owner"]) == 1]
    require(owners, "Brill audit found no canonical vertices")
    arrays = {name: np.asarray([float(row[STATE + name]) for row in owners])
              for name in names}
    gxx, gxy, gxz = arrays["gxx"], arrays["gxy"], arrays["gxz"]
    gyy, gyz, gzz = arrays["gyy"], arrays["gyz"], arrays["gzz"]
    determinant = (gxx * (gyy * gzz - gyz * gyz) -
                   gxy * (gxy * gzz - gxz * gyz) +
                   gxz * (gxy * gyz - gxz * gyy))
    pivot0 = gxx
    pivot1 = gyy - gxy * gxy / pivot0
    pivot2 = determinant / (pivot0 * pivot1)
    inverse = np.empty((len(owners), 3, 3))
    inverse[:, 0, 0] = (gyy * gzz - gyz * gyz) / determinant
    inverse[:, 0, 1] = inverse[:, 1, 0] = (gxz * gyz - gxy * gzz) / determinant
    inverse[:, 0, 2] = inverse[:, 2, 0] = (gxy * gyz - gxz * gyy) / determinant
    inverse[:, 1, 1] = (gxx * gzz - gxz * gxz) / determinant
    inverse[:, 1, 2] = inverse[:, 2, 1] = (gxy * gxz - gxx * gyz) / determinant
    inverse[:, 2, 2] = (gxx * gyy - gxy * gxy) / determinant
    a = np.zeros_like(inverse)
    a[:, 0, 0], a[:, 0, 1], a[:, 0, 2] = arrays["Axx"], arrays["Axy"], arrays["Axz"]
    a[:, 1, 0], a[:, 1, 1], a[:, 1, 2] = arrays["Axy"], arrays["Ayy"], arrays["Ayz"]
    a[:, 2, 0], a[:, 2, 1], a[:, 2, 2] = arrays["Axz"], arrays["Ayz"], arrays["Azz"]
    trace_a = np.einsum("nij,nij->n", inverse, a)

    axis = [row for row in owners if row["role"] == "axis"]
    require(axis, "Brill audit found no rho=0 axis vertices")
    def axis_max(expression) -> float:
        return max(abs(expression(row)) for row in axis)
    axis_metrics = {
        "gxx_minus_gyy": axis_max(lambda row: float(row[STATE + "gxx"]) -
                                              float(row[STATE + "gyy"])),
        "gxy": axis_max(lambda row: float(row[STATE + "gxy"])),
        "gxz": axis_max(lambda row: float(row[STATE + "gxz"])),
        "gyz": axis_max(lambda row: float(row[STATE + "gyz"])),
        "Axx_minus_Ayy": axis_max(lambda row: float(row[STATE + "Axx"]) -
                                              float(row[STATE + "Ayy"])),
        "Axy": axis_max(lambda row: float(row[STATE + "Axy"])),
        "Axz": axis_max(lambda row: float(row[STATE + "Axz"])),
        "Ayz": axis_max(lambda row: float(row[STATE + "Ayz"])),
        "Gamx": axis_max(lambda row: float(row[STATE + "Gamx"])),
        "Gamy": axis_max(lambda row: float(row[STATE + "Gamy"])),
        "betax": axis_max(lambda row: float(row[STATE + "betax"])),
        "betay": axis_max(lambda row: float(row[STATE + "betay"])),
        "Bx": axis_max(lambda row: float(row[STATE + "Bx"])),
        "By": axis_max(lambda row: float(row[STATE + "By"])),
    }
    constraint_metrics = {}
    for name in CONSTRAINTS:
        values = np.asarray([float(row[name]) for row in owners])
        constraint_metrics[name] = {"rms": rms(values),
                                    "linf": float(np.max(np.abs(values)))}
    return {
        "resolution": resolution,
        "diagnostic_stride": int(rows[0]["diagnostic_stride"]),
        "sampled_canonical_vertices": len(owners),
        "shared_groups": len(shared),
        "shared_state_max_spread": shared_state_spread,
        "min_chi": float(np.min(arrays["chi"])),
        "min_alpha": float(np.min(arrays["alpha"])),
        "max_abs_det_gtilde_minus_one": float(np.max(np.abs(determinant - 1.0))),
        "max_abs_trace_Atilde": float(np.max(np.abs(trace_a))),
        "minimum_spd_pivots": [float(np.min(pivot0)), float(np.min(pivot1)),
                               float(np.min(pivot2))],
        "axis": axis_metrics,
        "constraints": constraint_metrics,
        "field_map": {physical_key(row):
                      [float(row[STATE + name]) for name in names]
                      for row in owners},
        "constraint_map": {physical_key(row):
                           [float(row[name]) for name in CONSTRAINTS]
                           for row in owners},
        "field_names": names,
    }


def constraint_summary(run: Path, resolution: int) -> list[dict[str, object]]:
    path = run / "z4c_vc_brill_direct_fixed.constraints.dat"
    require(path.is_file(), f"missing production constraint summary for N{resolution}")
    rows = [line.split() for line in path.read_text(encoding="utf-8").splitlines()
            if line and not line.startswith("#")]
    require(len(rows) == 4, f"invalid constraint summary for N{resolution}")
    output = []
    for row in rows:
        require(len(row) == 14, "constraint summary schema changed")
        output.append({
            "resolution": resolution,
            "nx1": int(row[0]), "nx2": int(row[1]), "nx3": int(row[2]),
            "cycle": int(row[3]),
            "region": row[4], "weighting": row[5],
            "measure": float(row[6]), "points": int(row[7]),
            "C_rms": float(row[8]), "H_rms": float(row[9]),
            "M_rms": float(row[10]), "Z_rms": float(row[11]),
            "H_linf": float(row[12]), "M_linf": float(row[13]),
        })
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs=2, action="append", metavar=("N", "PATH"),
                        required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    metrics = []
    summaries: list[dict[str, object]] = []
    for n_text, path_text in args.run:
        resolution = int(n_text)
        run = Path(path_text).resolve()
        metrics.append(run_metrics(resolution, load_rows(run)))
        summaries.extend(constraint_summary(run, resolution))
    metrics.sort(key=lambda item: int(item["resolution"]))
    field_names = list(metrics[0]["field_names"])
    direct_field_indices = [index for index, name in enumerate(field_names)
                            if name not in DISCRETE_DERIVATIVE_FIELDS]
    gamma_field_indices = [index for index, name in enumerate(field_names)
                           if name in DISCRETE_DERIVATIVE_FIELDS]
    field_comparisons = []
    constraint_comparisons = []
    for coarse, fine in zip(metrics, metrics[1:]):
        require(fine["field_names"] == field_names,
                "initial-data diagnostic field ordering changed")
        shared_keys = coarse["field_map"].keys() & fine["field_map"].keys()
        require(len(shared_keys) == len(coarse["field_map"]),
                f"N{fine['resolution']} does not cover every common sample")
        differences = np.asarray([
            np.asarray(fine["field_map"][key]) -
            np.asarray(coarse["field_map"][key]) for key in shared_keys])
        per_field = {
            name: {
                "rms": rms(differences[:, index]),
                "linf": float(np.max(np.abs(differences[:, index]))),
            }
            for index, name in enumerate(field_names)
        }
        direct_differences = differences[:, direct_field_indices]
        gamma_differences = differences[:, gamma_field_indices]
        field_comparisons.append({
            "coarse_resolution": coarse["resolution"],
            "fine_resolution": fine["resolution"],
            "common_vertices": len(shared_keys),
            "all_field_rms": rms(differences.ravel()),
            "all_field_linf": float(np.max(np.abs(differences))),
            "direct_initialized_field_rms": rms(direct_differences.ravel()),
            "direct_initialized_field_linf":
                float(np.max(np.abs(direct_differences))),
            "discrete_gamma_field_rms": rms(gamma_differences.ravel()),
            "discrete_gamma_field_linf":
                float(np.max(np.abs(gamma_differences))),
            "per_field": per_field,
        })
        constraint_differences = np.asarray([
            np.asarray(fine["constraint_map"][key]) -
            np.asarray(coarse["constraint_map"][key]) for key in shared_keys])
        constraint_comparisons.append({
            "coarse_resolution": coarse["resolution"],
            "fine_resolution": fine["resolution"],
            "common_vertices": len(shared_keys),
            "per_constraint": {
                name: {
                    "rms": rms(constraint_differences[:, index]),
                    "linf": float(np.max(np.abs(
                        constraint_differences[:, index]))),
                }
                for index, name in enumerate(CONSTRAINTS)
            },
        })
    common_node_difference_orders = {}
    if len(field_comparisons) == 2:
        common_node_difference_orders["fields"] = {
            name: {
                metric: math.log(
                    field_comparisons[0]["per_field"][name][metric] /
                    field_comparisons[1]["per_field"][name][metric], 2.0)
                if field_comparisons[0]["per_field"][name][metric] > 0.0 and
                field_comparisons[1]["per_field"][name][metric] > 0.0 else None
                for metric in ("rms", "linf")
            }
            for name in field_names
        }
        common_node_difference_orders["constraints"] = {
            name: {
                metric: math.log(
                    constraint_comparisons[0]["per_constraint"][name][metric] /
                    constraint_comparisons[1]["per_constraint"][name][metric], 2.0)
                if constraint_comparisons[0]["per_constraint"][name][metric] > 0.0 and
                constraint_comparisons[1]["per_constraint"][name][metric] > 0.0
                else None
                for metric in ("rms", "linf")
            }
            for name in CONSTRAINTS
        }
    for metric in metrics:
        del metric["field_map"]
        del metric["constraint_map"]
        del metric["field_names"]
    proper_box = sorted(
        (row for row in summaries if row["region"] == "box" and
         row["weighting"] == "proper"),
        key=lambda row: int(row["resolution"]))
    constraint_convergence = []
    for coarse, fine in zip(proper_box, proper_box[1:]):
        record: dict[str, object] = {
            "coarse_resolution": coarse["resolution"],
            "fine_resolution": fine["resolution"],
        }
        ratio = int(fine["resolution"]) / int(coarse["resolution"])
        for name in ("C_rms", "H_rms", "M_rms", "Z_rms"):
            coarse_value, fine_value = float(coarse[name]), float(fine[name])
            record[name + "_order"] = (
                math.log(coarse_value / fine_value, ratio)
                if coarse_value > 0.0 and fine_value > 0.0 else None)
        constraint_convergence.append(record)
    write_csv(output / "constraint_summary.csv", summaries)
    compact = []
    for metric in metrics:
        compact.append({
            "resolution": metric["resolution"],
            "diagnostic_stride": metric["diagnostic_stride"],
            "sampled_canonical_vertices": metric["sampled_canonical_vertices"],
            "shared_groups": metric["shared_groups"],
            "shared_state_max_spread": metric["shared_state_max_spread"],
            "min_chi": metric["min_chi"], "min_alpha": metric["min_alpha"],
            "max_abs_det_gtilde_minus_one": metric["max_abs_det_gtilde_minus_one"],
            "max_abs_trace_Atilde": metric["max_abs_trace_Atilde"],
            "minimum_spd_pivot": min(metric["minimum_spd_pivots"]),
            "maximum_axis_regularity_residual": max(metric["axis"].values()),
            "production_Z_rms": metric["constraints"]["con_Z"]["rms"],
            "production_Z_linf": metric["constraints"]["con_Z"]["linf"],
        })
    write_csv(output / "initial_metrics.csv", compact)
    result = {
        "schema": "z4c_vc_brill_initial_data_v2",
        "metrics": metrics,
        "common_node_field_comparisons": field_comparisons,
        "common_node_constraint_comparisons": constraint_comparisons,
        "common_node_difference_orders": common_node_difference_orders,
        "proper_box_constraint_convergence": constraint_convergence,
        "constraint_summary_rows": len(summaries),
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
