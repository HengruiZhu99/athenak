#!/usr/bin/env python3
"""Compare failed matched-trumpet histories across MPI decompositions."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np


STREAMS = (
    "ref_gh.hst", "user.hst", "adm_common0.hst", "adm_common1.hst",
    "adm_common2.hst", "adm_common3.hst", "adm_common4.hst",
    "adm_common5.hst",
)


def rows(path: Path) -> np.ndarray:
    result = [
        [float(value) for value in line.split()]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not result:
        raise ValueError(f"{path}: no numerical rows")
    return np.asarray(result, dtype=np.float64)


def failure_point(path: Path) -> dict[str, float | int | bool]:
    last = None
    fatal = False
    pattern = re.compile(
        r"elapsed=\S+ cycle=(\d+) time=([0-9.eE+-]+) dt=([0-9.eE+-]+)")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.search(line)
        if match:
            last = {
                "cycle": int(match.group(1)),
                "time": float(match.group(2)),
                "dt": float(match.group(3)),
            }
        fatal = fatal or "reached an invalid effective timestep" in line
    if last is None:
        raise ValueError(f"{path}: no cycle telemetry")
    return {**last, "invalid_effective_timestep": fatal}


def log_slope(time: np.ndarray, values: np.ndarray) -> dict[str, float]:
    selected = (time >= 0.2) & (values > 0.0) & np.isfinite(values)
    if np.count_nonzero(selected) < 3:
        raise ValueError("insufficient positive samples for growth fit")
    slope, intercept = np.polyfit(time[selected], np.log(values[selected]), 1)
    return {
        "log_slope_per_M": float(slope),
        "e_folding_time_M": float(1.0/slope) if slope > 0.0 else math.inf,
        "fit_intercept": float(intercept),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance", type=float, default=5.0e-12)
    args = parser.parse_args()

    comparisons = {}
    global_maximum = 0.0
    for suffix in STREAMS:
        left_path = args.left / f"{args.left_label}.{suffix}"
        right_path = args.right / f"{args.right_label}.{suffix}"
        left = rows(left_path)
        right = rows(right_path)
        if left.shape != right.shape:
            raise ValueError(
                f"{suffix}: shape mismatch {left.shape} versus {right.shape}")
        both_nan = np.isnan(left) & np.isnan(right)
        comparable = np.isfinite(left) & np.isfinite(right)
        if not np.all(comparable | both_nan):
            raise ValueError(f"{suffix}: mismatched or unsupported nonfinite value")
        scale = np.maximum(1.0, np.maximum(np.abs(left), np.abs(right)))
        conditioned = np.where(both_nan, 0.0, np.abs(left - right)/scale)
        maximum = float(np.max(conditioned))
        where = np.unravel_index(int(np.argmax(conditioned)), conditioned.shape)
        comparisons[suffix] = {
            "rows": int(left.shape[0]),
            "columns": int(left.shape[1]),
            "conditioned_Linf": maximum,
            "maximum_row": int(where[0]),
            "maximum_column_zero_based": int(where[1]),
        }
        global_maximum = max(global_maximum, maximum)

    left_ref = rows(args.left / f"{args.left_label}.ref_gh.hst")
    volume = left_ref[:, 11]
    time = left_ref[:, 0]
    growth = {
        "GH_RMS": log_slope(time, np.sqrt(left_ref[:, 2]/volume)),
        "reduction_RMS": log_slope(time, np.sqrt(left_ref[:, 3]/volume)),
        "curl_RMS": log_slope(time, np.sqrt(left_ref[:, 4]/volume)),
        "physical_metric_error_RMS": log_slope(
            time, np.sqrt(left_ref[:, 5]/volume)),
        "Q_Linf": log_slope(time, left_ref[:, 22]),
        "Delta_Linf": log_slope(time, left_ref[:, 23]),
        "source_frame_correction_Linf": log_slope(time, left_ref[:, 30]),
    }
    left_failure = failure_point(args.left / "run.log")
    right_failure = failure_point(args.right / "run.log")
    same_failure = (
        left_failure["cycle"] == right_failure["cycle"]
        and abs(left_failure["time"] - right_failure["time"]) <= 5.0e-13
        and left_failure["invalid_effective_timestep"]
        and right_failure["invalid_effective_timestep"]
    )
    result = {
        "schema": "ref-gh-single-puncture-decomposition-comparison-v1",
        "left": str(args.left),
        "right": str(args.right),
        "tolerance": args.tolerance,
        "streams": comparisons,
        "global_conditioned_Linf": global_maximum,
        "histories_agree": global_maximum <= args.tolerance,
        "left_failure": left_failure,
        "right_failure": right_failure,
        "same_failure_cycle_and_time": same_failure,
        "decomposition_independent_failure": bool(
            same_failure and global_maximum <= args.tolerance),
        "growth_fits": growth,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["decomposition_independent_failure"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
