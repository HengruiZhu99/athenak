#!/usr/bin/env python3
"""Analyze the frozen Ref-GH ordering/gamma2/gauge A-D matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable


CASES = "abcd"
HISTORY_METRICS = {
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
    "characteristic_speed_max": 17,
    "effective_CFL": 18,
    "bad_state": 21,
    "Q_Linf": 22,
    "Delta_Linf": 23,
    "source_curvature_Linf": 26,
    "source_QQ_Linf": 27,
    "source_DeltaDelta_Linf": 28,
    "source_damping_Linf": 29,
    "source_frame_correction_Linf": 30,
}
RHS_FAMILIES = (
    "Psi_RHS_Linf",
    "Pi_RHS_Linf",
    "Phi_RHS_Linf",
    "Hhat_RHS_Linf",
    "theta_RHS_Linf",
    "Upsilon_RHS_Linf",
)
CHI_REGIONS = (
    "chi_beta",
    "chi_beta_near_r_lt_1",
    "chi_beta_inner_r_lt_0p5",
    "chi_beta_annulus_0p5_1",
    "chi_beta_annulus_1_1p5",
    "chi_beta_annulus_1p5_2",
    "chi_beta_outer_r_ge_2",
    "chi_beta_first_ge_1",
)


def read_numeric_rows(path: Path) -> list[list[float]]:
    rows = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            rows.append([float(value) for value in line.split()])
    if not rows:
        raise ValueError(f"no numeric rows in {path}")
    return rows


def read_status(path: Path) -> dict[str, str]:
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            result[key] = value
    return result


def rms(row: list[float], sum_index: int, volume_index: int) -> float:
    if row[volume_index] <= 0.0 or row[sum_index] < 0.0:
        return math.nan
    return math.sqrt(row[sum_index] / row[volume_index])


def log_fit(points: Iterable[tuple[float, float]], minimum_time: float) -> dict:
    accepted = [
        (time, math.log(value))
        for time, value in points
        if time >= minimum_time and math.isfinite(value) and value > 0.0
    ]
    if len(accepted) < 3:
        return {"samples": len(accepted), "slope": None, "e_folding_time": None}
    mean_x = sum(x for x, _ in accepted) / len(accepted)
    mean_y = sum(y for _, y in accepted) / len(accepted)
    denominator = sum((x - mean_x) ** 2 for x, _ in accepted)
    if denominator == 0.0:
        return {"samples": len(accepted), "slope": None, "e_folding_time": None}
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


def read_maxloc(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        records = []
        for row in reader:
            records.append(
                {
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
                }
            )
    return records


def nearest(records: list[dict], diagnostic: str, target: float) -> dict:
    candidates = [row for row in records if row["diagnostic"] == diagnostic]
    if not candidates:
        raise ValueError(f"missing {diagnostic}")
    return min(candidates, key=lambda row: (abs(row["time"] - target), row["time"]))


def analyze_case(root: Path, case_name: str, minimum_fit_time: float) -> dict:
    directory = root / f"case_{case_name}_t3"
    history = read_numeric_rows(directory / f"case_{case_name}_t3.ref_gh.hst")
    maxloc = read_maxloc(directory / f"case_{case_name}_t3.ref_gh_maxloc.tsv")
    status = read_status(directory / "run_status.txt")
    result = {
        "case": case_name,
        "exit_status": int(status["run_exit_status"]),
        "last_time": history[-1][0],
        "history_rows": len(history),
        "final": {},
        "growth_fits": {},
        "rhs_growth_fits": {},
        "rhs_final_maxima": {},
    }
    for name, (sum_index, volume_index) in HISTORY_METRICS.items():
        values = [(row[0], rms(row, sum_index, volume_index)) for row in history]
        result["final"][name] = values[-1][1]
        result["growth_fits"][name] = log_fit(values, minimum_fit_time)
    for name, index in HISTORY_DIRECT.items():
        values = [(row[0], row[index]) for row in history]
        result["final"][name] = values[-1][1]
        result["growth_fits"][name] = log_fit(values, minimum_fit_time)
    for family in RHS_FAMILIES:
        records = [row for row in maxloc if row["diagnostic"] == family]
        result["rhs_growth_fits"][family] = log_fit(
            ((row["time"], row["maximum"]) for row in records), minimum_fit_time
        )
        if records:
            result["rhs_final_maxima"][family] = records[-1]
    result["chi_beta_final"] = {
        diagnostic: nearest(maxloc, diagnostic, history[-1][0])
        for diagnostic in CHI_REGIONS
    }
    return result


def write_summary(path: Path, cases: dict[str, dict]) -> None:
    columns = [
        "case",
        "exit_status",
        "last_time",
        "GH_RMS",
        "reduction_RMS",
        "curl_RMS",
        "metric_error_RMS",
        "regular_max",
        "Q_Linf",
        "source_frame_correction_Linf",
        "GH_e_folding_time",
        "source_frame_e_folding_time",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, delimiter="\t")
        writer.writeheader()
        for name in CASES:
            case = cases[name]
            final = case["final"]
            writer.writerow(
                {
                    "case": name,
                    "exit_status": case["exit_status"],
                    "last_time": f"{case['last_time']:.17e}",
                    "GH_RMS": f"{final['GH_RMS']:.17e}",
                    "reduction_RMS": f"{final['reduction_RMS']:.17e}",
                    "curl_RMS": f"{final['curl_RMS']:.17e}",
                    "metric_error_RMS": f"{final['metric_error_RMS']:.17e}",
                    "regular_max": f"{final['regular_max']:.17e}",
                    "Q_Linf": f"{final['Q_Linf']:.17e}",
                    "source_frame_correction_Linf": (
                        f"{final['source_frame_correction_Linf']:.17e}"
                    ),
                    "GH_e_folding_time": case["growth_fits"]["GH_RMS"][
                        "e_folding_time"
                    ],
                    "source_frame_e_folding_time": case["growth_fits"][
                        "source_frame_correction_Linf"
                    ]["e_folding_time"],
                }
            )


def write_chi_profile(path: Path, root: Path, cases: dict[str, dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        fieldnames = ["case", "target_time", "time", "region", "maximum", "radius"]
        writer = csv.DictWriter(stream, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for case_name in CASES:
            maxloc = read_maxloc(
                root / f"case_{case_name}_t3" / f"case_{case_name}_t3.ref_gh_maxloc.tsv"
            )
            targets = (0.0, 0.2, 0.5, 1.0, cases[case_name]["last_time"])
            for target in targets:
                for region in CHI_REGIONS:
                    row = nearest(maxloc, region, target)
                    writer.writerow(
                        {
                            "case": case_name,
                            "target_time": f"{target:.17e}",
                            "time": f"{row['time']:.17e}",
                            "region": region,
                            "maximum": f"{row['maximum']:.17e}",
                            "radius": f"{row['radius']:.17e}",
                        }
                    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--minimum-fit-time", type=float, default=0.2)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = {
        name: analyze_case(args.artifact_root, name, args.minimum_fit_time)
        for name in CASES
    }
    payload = {
        "schema": "ref-gh-ordering-gauge-discriminator-v1",
        "minimum_fit_time": args.minimum_fit_time,
        "cases": cases,
    }
    (args.output_dir / "phase3_growth_fits.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_summary(args.output_dir / "phase3_summary.tsv", cases)
    write_chi_profile(args.output_dir / "chi_beta_regions.tsv", args.artifact_root, cases)


if __name__ == "__main__":
    main()
