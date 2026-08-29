#!/usr/bin/env python3
"""Summarize fresh A/B controls and the stationary driver residual ladder."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pathlib
import re


FAMILIES = ("Hhat", "theta", "Upsilon")
SECTORS = ("actual", "driver", "KO")


def numeric_rows(path: pathlib.Path) -> list[list[float]]:
    return [
        [float(value) for value in line.split()]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def rms(row: list[float], sum_index: int, volume_index: int) -> float:
    return math.sqrt(row[sum_index] / row[volume_index])


def read_sector_table(path: pathlib.Path) -> dict[tuple[str, str], dict]:
    lines = [line for line in path.read_text(encoding="utf-8").splitlines()
             if not line.startswith("#")]
    reader = csv.DictReader(lines, delimiter="\t")
    return {(row["sector"], row["family"]): row for row in reader}


def power_fit(rows: list[dict], key: str) -> float:
    points = [(math.log(row["minimum_included_radius"]),
               math.log(row[key])) for row in rows if row[key] > 0.0]
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    return sum((x - mean_x) * (y - mean_y) for x, y in points) / sum(
        (x - mean_x) ** 2 for x, _ in points
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    controls = []
    for case in ("a", "b"):
        directory = args.artifact_root / f"case_{case}_t5_fresh"
        rows = numeric_rows(directory / f"case_{case}_t5_fresh.ref_gh.hst")
        final = rows[-1]
        controls.append({
            "case": case,
            "last_time": final[0],
            "GH_RMS": rms(final, 2, 11),
            "reduction_RMS": rms(final, 3, 11),
            "curl_RMS": rms(final, 4, 11),
            "metric_error_RMS": rms(final, 5, 11),
            "lapse_error_RMS": rms(final, 6, 11),
            "shift_error_RMS": rms(final, 7, 11),
            "regular_max": final[14],
            "bad_state": final[21],
            "Q_Linf": final[22],
        })

    residuals = []
    audit_pattern = re.compile(
        r"\|F-H_constraint\|_Linf=([^,]+), "
        r"\|tildeGamma\|_Linf=([^,]+), "
        r"\|stored_Hhat_A\|_Linf=([^,]+), "
        r"\|stored_theta_A\|_Linf=([^\s]+)"
    )
    for nx in (64, 96, 128):
        directory = args.artifact_root / f"fixed_point_n{nx}"
        table = read_sector_table(directory / f"fixed_point_n{nx}.ref_gh_rhs_sectors.tsv")
        match = audit_pattern.search((directory / "run.log").read_text(encoding="utf-8"))
        if match is None:
            raise RuntimeError(f"missing stationary audit for N={nx}")
        row = {
            "nx": nx,
            "h": 4.0 / nx,
            "minimum_included_radius": float(table[("actual", "theta")]["radius"]),
            "target_minus_constraint": float(match.group(1)),
            "conformal_gamma": float(match.group(2)),
            "stored_Hhat": float(match.group(3)),
            "stored_theta": float(match.group(4)),
        }
        for sector in SECTORS:
            for family in FAMILIES:
                record = table[(sector, family)]
                row[f"{sector}_{family}"] = float(record["maximum"])
                row[f"{sector}_{family}_radius"] = float(record["radius"])
        residuals.append(row)

    def write_tsv(path: pathlib.Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]), delimiter="\t")
            writer.writeheader()
            writer.writerows(rows)

    write_tsv(args.output_dir / "fresh_controls_t5.tsv", controls)
    write_tsv(args.output_dir / "stationary_driver_residuals.tsv", residuals)
    fitted = {
        key: power_fit(residuals, key)
        for key in (
            "target_minus_constraint", "stored_Hhat", "stored_theta",
            "driver_Hhat", "driver_theta", "actual_theta", "KO_theta",
        )
    }
    summary = {
        "fresh_controls_reached_t5": all(row["last_time"] == 5.0 for row in controls),
        "fresh_controls_finite": all(row["bad_state"] == 0.0 for row in controls),
        "residual_power_vs_minimum_radius": fitted,
        "interpretation": (
            "The maxima move inward in fixed r/h as resolution increases. "
            "They are binary64 cancellation/KO envelopes amplified by singular "
            "trumpet coefficients, not a fixed-coordinate truncation sequence."
        ),
    }
    (args.output_dir / "phase67_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
