#!/usr/bin/env python3
"""Extract point locations behind negative exact-time self-convergence orders."""

from __future__ import annotations

import argparse
import csv
import gzip
import math
from pathlib import Path

import numpy as np


TARGETS = ("0125", "0250", "0500", "0750", "1000", "1250")
PREFIXES = ("state_z4c_", "rhs_z4c_", "con_")


def load(path: Path) -> dict[tuple[float, float], dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream)
                if row["rk_stage"] == "1" and row["canonical_owner"] == "1"]
    return {(round(float(row["rho"]), 13), round(float(row["x2"]), 13)): row
            for row in rows}


def region(point: tuple[float, float], name: str) -> bool:
    rho, zed = point
    h = 0.125
    outer_distance = min(16.0 - rho, 16.0 - abs(zed))
    seam_r = abs(math.remainder(rho, 4.0)) < 1.0e-12
    seam_z = abs(math.remainder(zed, 4.0)) < 1.0e-12
    if name == "full":
        return True
    if name == "outer_layer_0":
        return abs(outer_distance) < 1.0e-12
    if name == "outer_layer_1":
        return abs(outer_distance - h) < 1.0e-12
    if name == "axis_core_r8":
        return abs(rho) < 1.0e-12 and abs(zed) <= 8.0 + 1.0e-12
    if name == "core_r8_mb_interior":
        return (math.hypot(rho, zed) <= 8.0 + 1.0e-12 and rho > 4*h and
                not seam_r and not seam_z)
    raise ValueError(name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output_rows = []
    for tag in TARGETS:
        cases = []
        for resolution in (128, 256, 512):
            path = (args.root / f"N{resolution}" / f"tau{tag}" / "diagnostic" /
                    "rhs.rank000000.csv.gz")
            cases.append(load(path))
        shared = sorted(cases[0].keys() & cases[1].keys() & cases[2].keys())
        names = [name for name in next(iter(cases[0].values()))
                 if name.startswith(PREFIXES)]
        for region_name in ("full", "outer_layer_0", "outer_layer_1",
                            "axis_core_r8", "core_r8_mb_interior"):
            points = [point for point in shared if region(point, region_name)]
            for name in names:
                values = [np.asarray([float(case[point][name]) for point in points])
                          for case in cases]
                differences = [values[1] - values[0], values[2] - values[1]]
                rms = [float(np.sqrt(np.mean(delta**2))) for delta in differences]
                if min(rms) <= 0.0:
                    continue
                observed_order = math.log(rms[0] / rms[1], 2.0)
                if observed_order >= 0.0 or rms[1] <= 1.0e-12:
                    continue
                index = int(np.argmax(np.abs(differences[1])))
                point = points[index]
                family = ("state" if name.startswith("state_") else
                          "rhs" if name.startswith("rhs_") else "constraints")
                output_rows.append({
                    "axis_tau": int(tag) / 1000.0,
                    "region": region_name,
                    "family": family,
                    "variable": name,
                    "observed_order_rms": observed_order,
                    "difference_128_256_rms": rms[0],
                    "difference_256_512_rms": rms[1],
                    "worst_rho": point[0], "worst_z": point[1],
                    "N128_value": values[0][index],
                    "N256_value": values[1][index],
                    "N512_value": values[2][index],
                    "worst_abs_difference_256_512": abs(differences[1][index]),
                })
    output_rows.sort(key=lambda row: (row["axis_tau"], row["region"],
                                      row["observed_order_rms"]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output_rows[0]))
        writer.writeheader()
        writer.writerows(output_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
