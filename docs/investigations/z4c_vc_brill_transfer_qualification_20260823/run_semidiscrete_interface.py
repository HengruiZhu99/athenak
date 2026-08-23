#!/usr/bin/env python3
"""Measure the actual native-VC Z4c RHS defect at a static AMR interface.

The static hierarchy is compared with two directly sampled uniform references:
the base-resolution reference supplies coarse-leaf values and the 2x reference
supplies fine-leaf values.  Matching is by physical MeshBlock bounds and local
vertex index, so bulk truncation differences between H and H/2 are not folded
into the interlevel-boundary defect.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
from typing import Iterable

import numpy as np


STATE_PREFIX = "state_z4c_"
RHS_PREFIX = "rhs_z4c_"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def configured_input(template: Path, destination: Path, transfer: str) -> None:
    text = template.read_text(encoding="utf-8")
    centering = "grid_centering = vertex"
    require(text.count(centering) == 1, "input lacks unique VC selector")
    text = text.replace(
        centering, centering + f"\nvertex_prolongation_order = {transfer}")
    amplitude = "amp = 1.0e-4"
    require(text.count(amplitude) == 1, "input lacks expected amplitude marker")
    text = text.replace(
        amplitude,
        amplitude + "\nwrite_period_error = false"
        "\nrhs_interface_manufactured_state = true")
    destination.write_text(text, encoding="utf-8")


def run_case(athena: Path, template: Path, root: Path, label: str,
             resolution: int, hierarchy: str, transfer: str) -> list[dict[str, str]]:
    work = root / label
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    input_path = work / template.name
    configured_input(template, input_path, transfer)
    meshblock = resolution // 2
    if hierarchy == "uniform_fine":
        nx1, nx2 = 2 * resolution, resolution
    else:
        nx1, nx2 = resolution, resolution // 2
    command = [
        str(athena), "-i", str(input_path), "-d", str(work),
        f"mesh/nx1={nx1}", f"mesh/nx2={nx2}",
        f"meshblock/nx1={meshblock}", f"meshblock/nx2={meshblock}",
        "mesh/ix1_bc=outflow", "mesh/ox1_bc=outflow",
        "time/nlim=1",
    ]
    if hierarchy != "static":
        command.append("mesh_refinement/refinement=none")
    environment = os.environ.copy()
    environment.setdefault("OMP_NUM_THREADS", "8")
    environment.setdefault("OMP_PROC_BIND", "false")
    environment["ATHENA_Z4C_VC_RHS_FIELD_DIAGNOSTIC"] = str(work / "rhs")
    result = subprocess.run(command, cwd=work, env=environment, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            check=False)
    (work / "stdout.log").write_text(result.stdout, encoding="utf-8")
    (work / "stderr.log").write_text(result.stderr, encoding="utf-8")
    require(result.returncode == 0,
            f"{label} failed ({result.returncode})\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}")
    paths = sorted(work.glob("rhs.rank*.csv"))
    require(paths, f"{label} produced no RHS field diagnostic")
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as stream:
            rows.extend(csv.DictReader(stream))
    require(rows and all(row["schema"] == "z4c_vc_rhs_field_v2" for row in rows),
            f"{label} has invalid RHS diagnostic schema")
    return rows


def rounded(value: str) -> float:
    return round(float(value), 14)


def block_key(row: dict[str, str]) -> tuple[float, ...]:
    return tuple(rounded(row[name]) for name in
                 ("x1min", "x1max", "x2min", "x2max", "x3min", "x3max"))


def point_key(row: dict[str, str]) -> tuple[object, ...]:
    return (*block_key(row), int(row["i"]), int(row["j"]), int(row["k"]))


def interface_distance(row: dict[str, str]) -> int | None:
    bounds = block_key(row)
    x1min, x1max = bounds[0], bounds[1]
    if not (abs(x1min) <= 1.0e-13 or abs(x1max) <= 1.0e-13):
        return None
    dx = (x1max - x1min) / int(row["nx1_intervals"])
    distance = int(round(abs(float(row["rho"])) / dx))
    if abs(x1max) <= 1.0e-13:
        return distance if float(row["rho"]) <= 0.0 else None
    if abs(x1min) <= 1.0e-13:
        return distance if float(row["rho"]) >= 0.0 else None
    return None


def categories(row: dict[str, str]) -> list[str]:
    result = ["all"]
    relative_level = int(row["relative_level"])
    result.append("fine_all" if relative_level > 0 else "coarse_all")
    role = row["role"]
    if role == "shared_coarse_fine_coincident":
        result.append("coincident_interface")
    elif role == "hanging_fine_interface":
        result.append("hanging_interface")
    distance = interface_distance(row)
    if distance is not None and 0 <= distance <= 4:
        result.append(f"interface_layer_{distance}")
        codimension = int(row["local_edge_codimension"])
        result.append("interface_face" if codimension <= 1 else
                      ("interface_edge" if codimension == 2 else
                       "interface_corner"))
    elif relative_level > 0:
        result.append("fine_interior")
    else:
        result.append("coarse_interior")
    return result


def rms(values: Iterable[float]) -> float:
    array = np.asarray(list(values), dtype=float)
    return float(np.sqrt(np.mean(array * array))) if array.size else math.nan


def compare(static: list[dict[str, str]], coarse: list[dict[str, str]],
            fine: list[dict[str, str]], resolution: int, transfer: str,
            metric_rows: list[dict[str, object]],
            state_metric_rows: list[dict[str, object]],
            spectrum_rows: list[dict[str, object]],
            spectrum_bin_rows: list[dict[str, object]]) -> dict[str, object]:
    coarse_map = {point_key(row): row for row in coarse}
    fine_map = {point_key(row): row for row in fine}
    variables = [name.removeprefix(RHS_PREFIX) for name in static[0]
                 if name.startswith(RHS_PREFIX)]
    require(len(variables) == 25, "RHS diagnostic does not contain 25 variables")
    defects: dict[tuple[str, str], list[float]] = {}
    state_defects: dict[tuple[str, str], list[float]] = {}
    state_max = 0.0
    matched = 0
    per_row_defect: list[tuple[dict[str, str], dict[str, float]]] = []
    for row in static:
        reference_map = fine_map if int(row["relative_level"]) > 0 else coarse_map
        reference = reference_map.get(point_key(row))
        require(reference is not None,
                f"missing level-matched uniform reference for {point_key(row)}")
        matched += 1
        row_defect: dict[str, float] = {}
        for variable in variables:
            state_value = (float(row[STATE_PREFIX + variable]) -
                           float(reference[STATE_PREFIX + variable]))
            state_max = max(state_max, abs(state_value))
            value = (float(row[RHS_PREFIX + variable]) -
                     float(reference[RHS_PREFIX + variable]))
            row_defect[variable] = value
            for category in categories(row):
                defects.setdefault((category, variable), []).append(value)
                state_defects.setdefault((category, variable), []).append(state_value)
        per_row_defect.append((row, row_defect))
    for (category, variable), values in sorted(defects.items()):
        metric_rows.append({
            "resolution": resolution,
            "transfer_order": transfer,
            "category": category,
            "variable": variable,
            "count": len(values),
            "rms": rms(values),
            "linf": max(abs(value) for value in values),
        })
    for (category, variable), values in sorted(state_defects.items()):
        state_metric_rows.append({
            "resolution": resolution,
            "transfer_order": transfer,
            "category": category,
            "variable": variable,
            "count": len(values),
            "rms": rms(values),
            "linf": max(abs(value) for value in values),
        })

    # Fourier audit along the fine side of x1=0.  Canonical owners remove
    # duplicate same-level copies without averaging away an RHS discrepancy.
    line = [(row, values) for row, values in per_row_defect
            if int(row["relative_level"]) > 0 and
            abs(float(row["x1min"])) <= 1.0e-13 and
            abs(float(row["rho"])) <= 1.0e-13]
    by_x2: dict[float, tuple[dict[str, str], dict[str, float]]] = {}
    for row, values in line:
        by_x2.setdefault(rounded(row["x2"]), (row, values))
    ordered = [by_x2[key] for key in sorted(by_x2)]
    require(len(ordered) >= resolution // 2,
            "fine interface line is too short for Fourier analysis")
    for variable in variables:
        values = np.asarray([item[1][variable] for item in ordered], dtype=float)
        values -= np.mean(values)
        spectrum = np.fft.rfft(values)
        power = np.abs(spectrum) ** 2
        if power.size:
            power[0] = 0.0
        total = float(np.sum(power))
        nyquist = max(1, len(values) // 2)
        high = float(np.sum(power[np.arange(power.size) >= 0.5 * nyquist]))
        dominant = int(np.argmax(power)) if total > 0.0 else 0
        spectrum_rows.append({
            "resolution": resolution,
            "transfer_order": transfer,
            "variable": variable,
            "samples": len(values),
            "total_power": total,
            "high_half_nyquist_fraction": high / total if total > 0.0 else 0.0,
            "dominant_k_over_nyquist": dominant / nyquist,
        })
        for mode, mode_power in enumerate(power):
            spectrum_bin_rows.append({
                "resolution": resolution,
                "transfer_order": transfer,
                "variable": variable,
                "mode": mode,
                "k_over_nyquist": mode / nyquist,
                "power": float(mode_power),
            })
    return {"matched_rows": matched, "maximum_state_mismatch": state_max}


def add_orders(rows: list[dict[str, object]]) -> None:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["transfer_order"]), str(row["category"]),
               str(row["variable"]))
        groups.setdefault(key, []).append(row)
    for group in groups.values():
        group.sort(key=lambda row: int(row["resolution"]))
        for index, row in enumerate(group):
            row["order_from_previous"] = ""
            if index == 0:
                continue
            previous = group[index - 1]
            coarse_error = float(previous["rms"])
            fine_error = float(row["rms"])
            if coarse_error > 0.0 and fine_error > 0.0:
                row["order_from_previous"] = math.log(
                    coarse_error / fine_error,
                    int(row["resolution"]) / int(previous["resolution"]))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    require(rows, f"refusing to write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--athena", required=True, type=Path)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resolutions", nargs="+", type=int,
                        default=(16, 32, 64, 128))
    parser.add_argument("--transfers", nargs="+", choices=("4", "6"),
                        default=("4", "6"))
    args = parser.parse_args()
    require(all(value >= 16 and value % 4 == 0 for value in args.resolutions),
            "resolutions must be multiples of four and at least 16")
    root = args.work_dir.resolve()
    output = args.output_dir.resolve()
    shutil.rmtree(root, ignore_errors=True)
    shutil.rmtree(output, ignore_errors=True)
    root.mkdir(parents=True)
    output.mkdir(parents=True)
    metric_rows: list[dict[str, object]] = []
    state_metric_rows: list[dict[str, object]] = []
    spectrum_rows: list[dict[str, object]] = []
    spectrum_bin_rows: list[dict[str, object]] = []
    comparisons: list[dict[str, object]] = []
    for resolution in args.resolutions:
        coarse = run_case(args.athena.resolve(), args.input.resolve(), root,
                          f"uniform_coarse_n{resolution}", resolution,
                          "uniform_coarse", "6")
        fine = run_case(args.athena.resolve(), args.input.resolve(), root,
                        f"uniform_fine_n{resolution}", resolution,
                        "uniform_fine", "6")
        for transfer in args.transfers:
            static = run_case(args.athena.resolve(), args.input.resolve(), root,
                              f"static_q{transfer}_n{resolution}", resolution,
                              "static", transfer)
            summary = compare(static, coarse, fine, resolution, transfer,
                              metric_rows, state_metric_rows, spectrum_rows,
                              spectrum_bin_rows)
            comparisons.append({"resolution": resolution,
                                "transfer_order": transfer, **summary})
    add_orders(metric_rows)
    add_orders(state_metric_rows)
    write_csv(output / "rhs_defect_metrics.csv", metric_rows)
    write_csv(output / "state_defect_metrics.csv", state_metric_rows)
    write_csv(output / "interface_spectra.csv", spectrum_rows)
    write_csv(output / "interface_spectrum_bins.csv", spectrum_bin_rows)
    summary = {
        "schema": "z4c_vc_semidiscrete_interface_v2",
        "resolutions": args.resolutions,
        "transfer_orders": args.transfers,
        "reference_policy": "uniform level-matched by physical block and local vertex",
        "comparisons": comparisons,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
