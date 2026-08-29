#!/usr/bin/env python3
"""Analyze a repaired Ref-GH STANDARD matched-trumpet evolution segment.

The script is deliberately compatible with Aurora's Python 3.6.  It reports
the old gauge-on Case-D exponential discriminator separately from the hard
finite/target-time gate.  Constraint and metric comparisons are evidence for
scientific review; they are not hidden behind adjustable pass thresholds.
"""

import argparse
import csv
import json
import math
import pathlib
import re
import sys


HISTORY_RMS = {
    "GH_RMS": (2, 11),
    "reduction_RMS": (3, 11),
    "curl_RMS": (4, 11),
    "metric_error_RMS": (5, 11),
    "lapse_error_RMS": (6, 11),
    "shift_error_RMS": (7, 11),
    "GH_near_RMS": (8, 20),
    "reduction_near_RMS": (9, 20),
    "curl_near_RMS": (10, 20),
}

HISTORY_DIRECT = {
    "regular_max": 14,
    "G_condition_max": 15,
    "coordinate_condition_max": 16,
    "characteristic_speed_max": 17,
    "effective_CFL": 18,
    "minus_detg_min": 19,
    "bad_state": 21,
    "Q_Linf": 22,
    "Delta_Linf": 23,
    "frame_Ricci_Linf": 24,
    "coordinate_Ricci_Linf": 25,
    "source_curvature_Linf": 26,
    "source_QQ_Linf": 27,
    "source_DeltaDelta_Linf": 28,
    "source_damping_Linf": 29,
    "source_frame_correction_Linf": 30,
    "q": 31,
    "q_dot": 32,
    "q_estimate": 33,
}

RHS_FAMILIES = (
    "Psi_RHS_Linf",
    "Pi_RHS_Linf",
    "Phi_RHS_Linf",
    "Hhat_RHS_Linf",
    "theta_RHS_Linf",
    "Upsilon_RHS_Linf",
)

COMPARE_METRICS = (
    "GH_RMS",
    "reduction_RMS",
    "curl_RMS",
    "metric_error_RMS",
    "lapse_error_RMS",
    "shift_error_RMS",
    "GH_near_RMS",
    "reduction_near_RMS",
    "curl_near_RMS",
    "Q_Linf",
    "Delta_Linf",
)

# Frozen Case-D fit from the committed discriminator evidence.  A result is
# classified as recurrence only when its GH fit lies within 25 percent of the
# old rate and remains strongly exponential.  The raw fit is always reported.
OLD_CASE_D_GH_SLOPE = 26.654903904216415
OLD_MODE_RATE_FRACTION = 0.25
OLD_MODE_MIN_R_SQUARED = 0.95


def read_numeric_rows(path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            rows.append([float(value) for value in line.split()])
    if not rows:
        raise ValueError("no numeric rows in {}".format(path))
    return rows


def read_status(path):
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if "=" in line:
                key, value = line.rstrip("\n").split("=", 1)
                result[key] = value
    return result


def rms(row, sum_index, volume_index):
    if row[volume_index] <= 0.0 or row[sum_index] < 0.0:
        return math.nan
    return math.sqrt(row[sum_index] / row[volume_index])


def metric_series(rows):
    result = {}
    for name, indices in HISTORY_RMS.items():
        result[name] = [(row[0], rms(row, indices[0], indices[1])) for row in rows]
    for name, index in HISTORY_DIRECT.items():
        result[name] = [(row[0], row[index]) for row in rows]
    return result


def log_fit(points, minimum_time):
    accepted = [
        (time, math.log(value))
        for time, value in points
        if time >= minimum_time and math.isfinite(value) and value > 0.0
    ]
    if len(accepted) < 3:
        return {"samples": len(accepted), "slope": None,
                "e_folding_time": None, "r_squared": None}
    mean_x = sum(item[0] for item in accepted) / len(accepted)
    mean_y = sum(item[1] for item in accepted) / len(accepted)
    denominator = sum((item[0] - mean_x) ** 2 for item in accepted)
    if denominator == 0.0:
        return {"samples": len(accepted), "slope": None,
                "e_folding_time": None, "r_squared": None}
    slope = sum((x - mean_x) * (y - mean_y) for x, y in accepted) / denominator
    intercept = mean_y - slope * mean_x
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in accepted)
    total = sum((y - mean_y) ** 2 for _, y in accepted)
    return {
        "samples": len(accepted),
        "minimum_time": minimum_time,
        "slope": slope,
        "e_folding_time": 1.0 / slope if slope > 0.0 else None,
        "r_squared": 1.0 - residual / total if total > 0.0 else None,
    }


def read_maxloc(path):
    records = []
    if not path.is_file():
        return records
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            records.append({
                "time": float(row["time"]),
                "cycle": int(row["cycle"]),
                "diagnostic": row["diagnostic"],
                "maximum": float(row["maximum"]),
                "radius": float(row["radius"]),
                "rank": int(row["rank"]),
                "gid": int(row["gid"]),
                "x": float(row["x"]),
                "y": float(row["y"]),
                "z": float(row["z"]),
            })
    return records


def summarize_run(directory, basename, target_time, minimum_fit_time):
    history_path = directory / (basename + ".ref_gh.hst")
    status_path = directory / "run_status.txt"
    rows = read_numeric_rows(history_path)
    status = read_status(status_path)
    series = metric_series(rows)
    maxloc = read_maxloc(directory / (basename + ".ref_gh_maxloc.tsv"))
    final = {name: values[-1][1] for name, values in series.items()}
    maximum = {
        name: max(value for _, value in values if math.isfinite(value))
        for name, values in series.items()
    }
    fits = {
        name: log_fit(values, minimum_fit_time)
        for name, values in series.items()
    }
    rhs = {}
    for family in RHS_FAMILIES:
        records = [row for row in maxloc if row["diagnostic"] == family]
        rhs[family] = {
            "fit": log_fit(
                ((row["time"], row["maximum"]) for row in records),
                minimum_fit_time,
            ),
            "final": records[-1] if records else None,
            "maximum": max((row["maximum"] for row in records), default=None),
        }

    all_values_finite = all(
        math.isfinite(value) for row in rows for value in row
    ) and all(math.isfinite(row["maximum"]) for row in maxloc)
    run_exit_status = int(status.get("run_exit_status", "-999"))
    last_time = rows[-1][0]
    reached_target = last_time >= target_time - 1.0e-10
    bad_state_clear = maximum["bad_state"] == 0.0
    valid_timestep = (
        final["effective_CFL"] > 0.0
        and math.isfinite(final["effective_CFL"])
        and rows[-1][1] > 0.0
        and math.isfinite(rows[-1][1])
    )
    gh_fit = fits["GH_RMS"]
    rate_low = OLD_CASE_D_GH_SLOPE * (1.0 - OLD_MODE_RATE_FRACTION)
    rate_high = OLD_CASE_D_GH_SLOPE * (1.0 + OLD_MODE_RATE_FRACTION)
    old_mode_recurrence = (
        gh_fit["slope"] is not None
        and gh_fit["r_squared"] is not None
        and rate_low <= gh_fit["slope"] <= rate_high
        and gh_fit["r_squared"] >= OLD_MODE_MIN_R_SQUARED
    )
    hard_gate_passed = (
        run_exit_status == 0
        and reached_target
        and all_values_finite
        and bad_state_clear
        and valid_timestep
        and not old_mode_recurrence
    )
    return {
        "basename": basename,
        "directory": str(directory),
        "run_exit_status": run_exit_status,
        "target_time": target_time,
        "last_time": last_time,
        "history_rows": len(rows),
        "maxloc_rows": len(maxloc),
        "all_values_finite": all_values_finite,
        "bad_state_clear": bad_state_clear,
        "valid_effective_timestep": valid_timestep,
        "reached_target": reached_target,
        "old_case_d_gh_slope": OLD_CASE_D_GH_SLOPE,
        "old_mode_rate_window": [rate_low, rate_high],
        "old_mode_min_r_squared": OLD_MODE_MIN_R_SQUARED,
        "old_mode_recurrence": old_mode_recurrence,
        "hard_gate_passed": hard_gate_passed,
        "final": final,
        "maximum": maximum,
        "growth_fits": fits,
        "rhs": rhs,
    }


def control_envelope(paths):
    envelope = {}
    for path in paths:
        rows = read_numeric_rows(path)
        series = metric_series(rows)
        for metric in COMPARE_METRICS:
            values = [value for _, value in series[metric] if math.isfinite(value)]
            envelope[metric] = max(envelope.get(metric, 0.0), max(values))
    return envelope


def safe_ratio(numerator, denominator):
    if denominator == 0.0:
        return 0.0 if numerator == 0.0 else math.inf
    return numerator / denominator


def write_table(path, summary):
    columns = [
        "target_time", "last_time", "exit_status", "hard_gate_passed",
        "old_mode_recurrence", "GH_RMS", "reduction_RMS", "curl_RMS",
        "metric_error_RMS", "GH_near_RMS", "Q_Linf", "Delta_Linf",
        "effective_CFL", "bad_state", "GH_slope", "GH_e_folding_time",
        "GH_r_squared",
    ]
    gh_fit = summary["growth_fits"]["GH_RMS"]
    values = {
        "target_time": summary["target_time"],
        "last_time": summary["last_time"],
        "exit_status": summary["run_exit_status"],
        "hard_gate_passed": int(summary["hard_gate_passed"]),
        "old_mode_recurrence": int(summary["old_mode_recurrence"]),
        "GH_slope": gh_fit["slope"],
        "GH_e_folding_time": gh_fit["e_folding_time"],
        "GH_r_squared": gh_fit["r_squared"],
    }
    for name in columns:
        if name in summary["final"]:
            values[name] = summary["final"][name]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=columns, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--basename", required=True)
    parser.add_argument("--target-time", type=float, required=True)
    parser.add_argument("--minimum-fit-time", type=float, default=0.2)
    parser.add_argument("--control-history", type=pathlib.Path,
                        action="append", default=[])
    parser.add_argument("--output-json", type=pathlib.Path, required=True)
    parser.add_argument("--output-tsv", type=pathlib.Path, required=True)
    parser.add_argument("--require-hard-gate", action="store_true")
    args = parser.parse_args()

    summary = summarize_run(
        args.run_dir, args.basename, args.target_time, args.minimum_fit_time
    )
    if args.control_history:
        envelope = control_envelope(args.control_history)
        summary["gauge_off_control_envelope"] = envelope
        summary["maximum_ratio_to_gauge_off_control_envelope"] = {
            name: safe_ratio(summary["maximum"][name], envelope[name])
            for name in COMPARE_METRICS
        }
        summary["control_comparison_is_descriptive_not_a_hard_gate"] = True
    summary["claim_boundary"] = (
        "One repaired 96^3 STANDARD evolution segment only; no resolution, "
        "20M robustness, long-time stability, or production-readiness claim."
    )

    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_table(args.output_tsv, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.require_hard_gate and not summary["hard_gate_passed"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
