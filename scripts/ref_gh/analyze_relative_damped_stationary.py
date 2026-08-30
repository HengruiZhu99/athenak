#!/usr/bin/env python3
"""Summarize a relative-damped stationary-trumpet evolution.

The script is compatible with Aurora's Python 3.6.  It reports the old
moving-driver exponential discriminator separately from endpoint/finite-state
checks and never treats target-time completion alone as stability evidence.
"""

import argparse
import csv
import json
import math
import pathlib
import sys


RMS_COLUMNS = {
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

DIRECT_COLUMNS = {
    "regular_max": 14,
    "relative_metric_condition_max": 15,
    "coordinate_condition_max": 16,
    "characteristic_speed_max": 17,
    "effective_CFL": 18,
    "minus_detg_min": 19,
    "bad_state": 21,
    "Q_Linf": 22,
    "Delta_Linf": 23,
    "source_curvature_Linf": 26,
    "source_QQ_Linf": 27,
    "source_DeltaDelta_Linf": 28,
    "source_damping_Linf": 29,
    "source_frame_correction_Linf": 30,
    "relative_D_Linf": 31,
    "relative_WD_Linf": 32,
    "relative_source_Linf": 33,
}

RHS_FAMILIES = (
    "Psi_RHS_Linf",
    "Pi_RHS_Linf",
    "Phi_RHS_Linf",
    "Hhat_RHS_Linf",
    "theta_RHS_Linf",
    "Upsilon_RHS_Linf",
)

OLD_MODE_SLOPE = 26.654903904216415
OLD_MODE_RATE_FRACTION = 0.25
OLD_MODE_MIN_R_SQUARED = 0.95


def read_rows(path):
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip() and not line.lstrip().startswith("#"):
                rows.append([float(value) for value in line.split()])
    if not rows:
        raise ValueError("no numeric rows in {}".format(path))
    if any(len(row) < 34 for row in rows):
        raise ValueError("relative-gauge history requires 34 columns")
    return rows


def read_status(path):
    result = {}
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if "=" in line:
                key, value = line.rstrip("\n").split("=", 1)
                result[key] = value
    return result


def make_series(rows):
    series = {}
    for name, indices in RMS_COLUMNS.items():
        values = []
        for row in rows:
            numerator = row[indices[0]]
            volume = row[indices[1]]
            valid = numerator >= 0.0 and volume > 0.0
            value = math.sqrt(numerator/volume) if valid else math.nan
            values.append((row[0], value))
        series[name] = values
    for name, index in DIRECT_COLUMNS.items():
        series[name] = [(row[0], row[index]) for row in rows]
    return series


def log_fit(points, minimum_time):
    accepted = [
        (time, math.log(value))
        for time, value in points
        if time >= minimum_time and math.isfinite(value) and value > 0.0
    ]
    if len(accepted) < 3:
        return {"samples": len(accepted), "slope": None,
                "e_folding_time": None, "r_squared": None}
    mean_x = sum(item[0] for item in accepted)/len(accepted)
    mean_y = sum(item[1] for item in accepted)/len(accepted)
    denominator = sum((item[0] - mean_x)**2 for item in accepted)
    if denominator == 0.0:
        return {"samples": len(accepted), "slope": None,
                "e_folding_time": None, "r_squared": None}
    slope = sum((x - mean_x)*(y - mean_y) for x, y in accepted)/denominator
    intercept = mean_y - slope*mean_x
    residual = sum((y - intercept - slope*x)**2 for x, y in accepted)
    total = sum((y - mean_y)**2 for _, y in accepted)
    return {
        "samples": len(accepted),
        "minimum_time": minimum_time,
        "slope": slope,
        "e_folding_time": 1.0/slope if slope > 0.0 else None,
        "r_squared": 1.0 - residual/total if total > 0.0 else None,
    }


def read_maxloc(path):
    if not path.is_file():
        return []
    records = []
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


def summarize(args):
    history = args.run_dir/(args.basename + ".ref_gh.hst")
    maxloc_path = args.run_dir/(args.basename + ".ref_gh_maxloc.tsv")
    rows = read_rows(history)
    status = read_status(args.run_dir/"run_status.txt")
    maxloc = read_maxloc(maxloc_path)
    series = make_series(rows)
    final = {name: values[-1][1] for name, values in series.items()}
    maximum = {
        name: max(value for _, value in values if math.isfinite(value))
        for name, values in series.items()
    }
    fits = {name: log_fit(values, args.minimum_fit_time)
            for name, values in series.items()}
    rhs = {}
    for family in RHS_FAMILIES:
        records = [record for record in maxloc
                   if record["diagnostic"] == family]
        rhs[family] = {
            "fit": log_fit(((record["time"], record["maximum"])
                            for record in records), args.minimum_fit_time),
            "final": records[-1] if records else None,
            "maximum": max((record["maximum"] for record in records),
                           default=None),
        }

    all_finite = (
        all(math.isfinite(value) for row in rows for value in row)
        and all(math.isfinite(record["maximum"]) for record in maxloc)
    )
    final_time = rows[-1][0]
    run_exit = int(status.get("run_exit_status", "-999"))
    gh_fit = fits["GH_RMS"]
    rate_low = OLD_MODE_SLOPE*(1.0 - OLD_MODE_RATE_FRACTION)
    rate_high = OLD_MODE_SLOPE*(1.0 + OLD_MODE_RATE_FRACTION)
    old_mode = (
        gh_fit["slope"] is not None
        and gh_fit["r_squared"] is not None
        and rate_low <= gh_fit["slope"] <= rate_high
        and gh_fit["r_squared"] >= OLD_MODE_MIN_R_SQUARED
    )
    reached_target = final_time >= args.target_time - 1.0e-10
    gate_passed = (
        run_exit == 0 and reached_target and all_finite
        and maximum["bad_state"] == 0.0
        and rows[-1][1] > 0.0 and final["effective_CFL"] > 0.0
        and not old_mode
    )
    return {
        "classification": (
            "PASS_STATIONARY_SEGMENT" if gate_passed
            else "FAIL_STATIONARY_SEGMENT"
        ),
        "claim_boundary": (
            "one 96^3 stationary segment only; "
            "no resolution or long-time claim"
        ),
        "basename": args.basename,
        "target_time": args.target_time,
        "final_time": final_time,
        "history_rows": len(rows),
        "maxloc_rows": len(maxloc),
        "run_exit_status": run_exit,
        "all_values_finite": all_finite,
        "reached_target": reached_target,
        "bad_state_clear": maximum["bad_state"] == 0.0,
        "old_mode_rate_window": [rate_low, rate_high],
        "old_mode_recurrence": old_mode,
        "gate_passed": gate_passed,
        "final": final,
        "maximum": maximum,
        "growth_fits": fits,
        "rhs": rhs,
    }


def write_tsv(path, result):
    fit = result["growth_fits"]["GH_RMS"]
    values = {
        "classification": result["classification"],
        "target_time": result["target_time"],
        "final_time": result["final_time"],
        "run_exit_status": result["run_exit_status"],
        "gate_passed": int(result["gate_passed"]),
        "old_mode_recurrence": int(result["old_mode_recurrence"]),
        "GH_RMS": result["final"]["GH_RMS"],
        "reduction_RMS": result["final"]["reduction_RMS"],
        "curl_RMS": result["final"]["curl_RMS"],
        "metric_error_RMS": result["final"]["metric_error_RMS"],
        "relative_D_Linf": result["final"]["relative_D_Linf"],
        "relative_source_Linf": result["final"]["relative_source_Linf"],
        "GH_slope": fit["slope"],
        "GH_e_folding_time": fit["e_folding_time"],
        "GH_r_squared": fit["r_squared"],
    }
    columns = list(values)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerow(values)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=pathlib.Path)
    parser.add_argument("--basename", required=True)
    parser.add_argument("--target-time", required=True, type=float)
    parser.add_argument("--minimum-fit-time", type=float, default=0.2)
    parser.add_argument("--output-json", required=True, type=pathlib.Path)
    parser.add_argument("--output-tsv", required=True, type=pathlib.Path)
    parser.add_argument("--require-pass", action="store_true")
    args = parser.parse_args()
    result = summarize(args)
    args.output_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    write_tsv(args.output_tsv, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 2 if args.require_pass and not result["gate_passed"] else 0


if __name__ == "__main__":
    sys.exit(main())
