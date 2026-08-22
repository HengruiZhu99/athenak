#!/usr/bin/env python3
"""Characterize stationary-trumpet error growth versus time, cycles, and resolution."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from summarize_stationary_history import DEFAULT_TIMES, closest, read_history


def parse_case(text: str) -> tuple[str, int, Path, Path]:
    fields = text.split("=", 1)
    if len(fields) != 2:
        raise argparse.ArgumentTypeError(
            "--case must be LABEL,RESOLUTION=HISTORY,DAT")
    left, right = fields
    left_fields = left.split(",")
    right_fields = right.split(",")
    if len(left_fields) != 2 or len(right_fields) != 2:
        raise argparse.ArgumentTypeError(
            "--case must be LABEL,RESOLUTION=HISTORY,DAT")
    return left_fields[0], int(left_fields[1]), Path(right_fields[0]), Path(right_fields[1])


def read_final(path: Path) -> dict[str, float]:
    lines = [line.split() for line in path.read_text().splitlines()
             if line and not line.startswith("#")]
    if len(lines) != 1 or len(lines[0]) != 12:
        raise ValueError(f"{path}: expected one 12-column final row")
    values = list(map(float, lines[0]))
    return {
        "resolution": values[0], "cycles": values[1], "time": values[2],
        "field_Linf": values[3], "constraint_Linf": values[4],
        "initial_rhs_Linf": values[5],
    }


def rms(row: dict[str, float], name: str) -> float:
    return math.sqrt(row[f"{name}-L2sq"] / row["Volume"])


def log_slope(x_values: list[float], y_values: list[float]) -> float:
    selected = [(x, y) for x, y in zip(x_values, y_values)
                if x > 0.0 and y > 0.0 and math.isfinite(y)]
    if len(selected) < 2:
        return math.nan
    x, y = zip(*selected)
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def write_tsv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)

    metrics = ("GH", "Reduction", "Curl", "PsiError", "Pi", "Phi")
    matched_rows: list[dict[str, str]] = []
    fit_rows: list[dict[str, str]] = []
    final_rows: list[dict[str, str]] = []
    final_by_metric: dict[str, list[tuple[int, float]]] = {
        metric: [] for metric in metrics
    }

    for label, resolution, history_path, final_path in args.case:
        history = read_history(history_path)
        final = read_final(final_path)
        if int(final["resolution"]) != resolution or abs(final["time"] - 20.0) > 1.0e-12:
            raise ValueError(f"{final_path}: unexpected final resolution/time")
        nominal_dt = max(row["dt"] for row in history if row["dt"] > 0.0)
        for requested in DEFAULT_TIMES:
            row = closest(history, requested)
            item = {
                "case": label,
                "resolution": str(resolution),
                "requested_time": f"{requested:.17g}",
                "actual_time": f"{row['time']:.17g}",
                "estimated_cycles": str(int(round(row["time"] / nominal_dt))),
                "bad_state": f"{row['bad-state']:.17g}",
            }
            for metric in metrics:
                item[f"{metric}_RMS"] = f"{rms(row, metric):.17g}"
            matched_rows.append(item)

        fit_history = [row for row in history if row["time"] >= 1.0]
        for metric in metrics:
            errors = [rms(row, metric) for row in fit_history]
            times = [row["time"] for row in fit_history]
            cycles = [row["time"] / nominal_dt for row in fit_history]
            fit_rows.append({
                "scope": label,
                "metric": metric,
                "scaling_variable": "time",
                "log_log_slope": f"{log_slope(times, errors):.17g}",
            })
            fit_rows.append({
                "scope": label,
                "metric": metric,
                "scaling_variable": "estimated_cycles",
                "log_log_slope": f"{log_slope(cycles, errors):.17g}",
            })
            final_error = rms(closest(history, 20.0), metric)
            final_by_metric[metric].append((resolution, final_error))

        final_rows.append({
            "case": label,
            "resolution": str(resolution),
            "cycles": str(int(final["cycles"])),
            "time": f"{final['time']:.17g}",
            "field_Linf": f"{final['field_Linf']:.17g}",
            "constraint_Linf": f"{final['constraint_Linf']:.17g}",
            "initial_rhs_Linf": f"{final['initial_rhs_Linf']:.17g}",
            "maximum_bad_state": f"{max(row['bad-state'] for row in history):.17g}",
            "nominal_dt": f"{nominal_dt:.17g}",
        })

    for metric, values in final_by_metric.items():
        values.sort()
        resolution_slope = log_slope(
            [float(resolution) for resolution, _ in values],
            [error for _, error in values])
        fit_rows.append({
            "scope": "t20_across_resolution",
            "metric": metric,
            "scaling_variable": "resolution_N",
            "log_log_slope": f"{resolution_slope:.17g}",
        })

    prefix = args.output_prefix
    write_tsv(prefix.with_name(prefix.name + "_matched.tsv"), matched_rows)
    write_tsv(prefix.with_name(prefix.name + "_fits.tsv"), fit_rows)
    write_tsv(prefix.with_name(prefix.name + "_final.tsv"), final_rows)


if __name__ == "__main__":
    main()
