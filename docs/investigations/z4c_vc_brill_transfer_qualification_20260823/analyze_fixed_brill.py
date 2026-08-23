#!/usr/bin/env python3
"""Analyze fixed-grid native-VC Brill convergence at t=5 and common axis time."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_binary(path: Path, vis_python: Path) -> dict:
    sys.path.insert(0, str(vis_python))
    import bin_convert  # pylint: disable=import-error,import-outside-toplevel
    data = bin_convert.read_binary(str(path))
    require(data["Nx3"] == 1 and data["n_mbs"] == 32,
            f"unexpected fixed-grid topology in {path}")
    require(abs(data["time"] - 5.0) < 1.0e-12,
            f"terminal binary is not at t=5: {path}")
    return data


def canonical_map(data: dict) -> tuple[list[str], dict[tuple[float, float], np.ndarray], float]:
    names = list(data["var_names"])
    values: dict[tuple[float, float], np.ndarray] = {}
    maximum_shared_spread = 0.0
    for block in range(data["n_mbs"]):
        geometry = data["mb_geometry"][block]
        shape = data["mb_data"][names[0]][block].shape
        require(shape[0] == 1, "fixed Cartoon binary gained an active x3 extent")
        rho = np.linspace(geometry[0], geometry[1], shape[2])
        zed = np.linspace(geometry[2], geometry[3], shape[1])
        for j, z_value in enumerate(zed):
            for i, rho_value in enumerate(rho):
                key = (round(float(rho_value), 13), round(float(z_value), 13))
                point = np.asarray([data["mb_data"][name][block][0, j, i]
                                    for name in names], dtype=float)
                if key in values:
                    maximum_shared_spread = max(
                        maximum_shared_spread,
                        float(np.max(np.abs(values[key] - point))))
                else:
                    values[key] = point
    expected = (data["Nx1"] + 1) * (data["Nx2"] + 1)
    require(len(values) == expected,
            f"canonical VC grid has {len(values)} points, expected {expected}")
    return names, values, maximum_shared_spread


def history(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = next(line for line in lines if line.startswith("#  [1]="))
    labels = [part.split("=")[-1].strip()
              for part in header[2:].split("[") if "]=" in part]
    rows = np.asarray([[float(value) for value in line.split()]
                       for line in lines if line and not line.startswith("#")])
    require(rows.ndim == 2 and rows.shape[1] == len(labels),
            f"invalid history table {path}")
    return labels, rows


def interpolate_history(labels: list[str], rows: np.ndarray,
                        tau: float) -> dict[str, float]:
    tau_values = rows[:, labels.index("axisTau")]
    require(tau_values[0] <= tau <= tau_values[-1],
            f"history does not cover axisTau={tau}")
    return {name: float(np.interp(tau, tau_values, rows[:, index]))
            for index, name in enumerate(labels)}


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values)))


def ring_rms(difference: np.ndarray, keys: list[tuple[float, float]]) -> float:
    rho = np.asarray([key[0] for key in keys])
    weights = rho.copy()
    require(float(np.sum(weights)) > 0.0, "ring measure has zero support")
    return float(np.sqrt(np.sum(weights * difference * difference) /
                         np.sum(weights)))


def observed_order(coarse: float, fine: float) -> float | None:
    return math.log(coarse / fine, 2.0) if coarse > 0.0 and fine > 0.0 else None


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs=2, action="append", metavar=("N", "PATH"),
                        required=True)
    parser.add_argument("--vis-python", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    cases = []
    for n_text, path_text in args.run:
        resolution = int(n_text)
        root = Path(path_text).resolve()
        state_paths = sorted((root / "bin").glob("*.state.*.bin"))
        constraint_paths = sorted((root / "bin").glob("*.constraints.*.bin"))
        history_paths = sorted(root.glob("*.z4c.user.hst"))
        require(state_paths and constraint_paths and len(history_paths) == 1,
                f"N{resolution} fixed-grid outputs are incomplete")
        state_data = load_binary(state_paths[-1], args.vis_python.resolve())
        constraint_data = load_binary(constraint_paths[-1], args.vis_python.resolve())
        state_names, state_map, state_shared = canonical_map(state_data)
        constraint_names, constraint_map, constraint_shared = canonical_map(
            constraint_data)
        labels, history_rows = history(history_paths[0])
        cases.append({
            "resolution": resolution, "state_names": state_names,
            "state_map": state_map, "constraint_names": constraint_names,
            "constraint_map": constraint_map,
            "state_shared_max_spread": state_shared,
            "constraint_shared_max_spread": constraint_shared,
            "history_labels": labels, "history": history_rows,
            "cycle": state_data["cycle"], "time": state_data["time"],
        })
    cases.sort(key=lambda item: item["resolution"])
    require([case["resolution"] for case in cases] == [128, 256, 512],
            "fixed-grid analyzer requires N128/N256/N512")

    field_rows: list[dict[str, object]] = []
    pair_records = []
    for coarse, fine in zip(cases, cases[1:]):
        require(coarse["state_names"] == fine["state_names"] and
                coarse["constraint_names"] == fine["constraint_names"],
                "fixed-grid output variable inventory changed with resolution")
        keys = sorted(coarse["state_map"])
        require(all(key in fine["state_map"] for key in keys),
                "fine fixed grid does not contain every coarse vertex")
        record = {"coarse": coarse["resolution"], "fine": fine["resolution"],
                  "state": {}, "constraints": {}}
        for family, names, coarse_map, fine_map in (
                ("state", coarse["state_names"], coarse["state_map"],
                 fine["state_map"]),
                ("constraints", coarse["constraint_names"],
                 coarse["constraint_map"], fine["constraint_map"])):
            for index, name in enumerate(names):
                difference = np.asarray(
                    [fine_map[key][index] - coarse_map[key][index] for key in keys])
                metrics = {"rms": rms(difference),
                           "ring_rms": ring_rms(difference, keys),
                           "linf": float(np.max(np.abs(difference)))}
                record[family][name] = metrics
                field_rows.append({"family": family, "variable": name,
                                   "coarse_resolution": coarse["resolution"],
                                   "fine_resolution": fine["resolution"], **metrics})
        pair_records.append(record)

    orders = {family: {
        name: {metric: observed_order(pair_records[0][family][name][metric],
                                     pair_records[1][family][name][metric])
               for metric in ("rms", "ring_rms", "linf")}
        for name in pair_records[0][family]}
        for family in ("state", "constraints")}

    tau_rows: list[dict[str, object]] = []
    tau_records = {}
    for tau in (0.5, 1.0, 2.0, 3.0):
        samples = [interpolate_history(case["history_labels"], case["history"], tau)
                   for case in cases]
        tau_records[str(tau)] = samples
        common_labels = [name for name in cases[0]["history_labels"]
                         if name in cases[1]["history_labels"] and
                         name in cases[2]["history_labels"]]
        for name in common_labels:
            first = abs(samples[1][name] - samples[0][name])
            second = abs(samples[2][name] - samples[1][name])
            tau_rows.append({"axis_tau": tau, "variable": name,
                             "N128": samples[0][name], "N256": samples[1][name],
                             "N512": samples[2][name], "difference_128_256": first,
                             "difference_256_512": second,
                             "observed_order": observed_order(first, second)})

    write_csv(output / "terminal_field_differences.csv", field_rows)
    write_csv(output / "common_axis_time_history.csv", tau_rows)
    result = {
        "schema": "z4c_vc_fixed_brill_convergence_v1",
        "runs": [{key: case[key] for key in
                  ("resolution", "cycle", "time", "state_shared_max_spread",
                   "constraint_shared_max_spread")} for case in cases],
        "terminal_difference_orders": orders,
        "common_axis_time_samples": tau_records,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
