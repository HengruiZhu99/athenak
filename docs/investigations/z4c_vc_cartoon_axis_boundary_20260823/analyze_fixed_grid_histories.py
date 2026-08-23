#!/usr/bin/env python3
"""Compare fixed-grid Z4c history norms at common axis-proper times."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


TARGETS = (0.0, 0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 3.0)
FIELDS = ("C-norm2", "H-norm2", "M-norm2", "Z-norm2")


def load_history(path: Path) -> tuple[list[str], np.ndarray]:
    lines = path.read_text(encoding="utf-8").splitlines()
    header = next(line for line in lines if line.startswith("#  [1]="))
    labels = re.findall(r"\[\d+\]=([^\s]+)", header)
    rows = np.asarray([[float(value) for value in line.split()]
                       for line in lines if line and not line.startswith("#")])
    if rows.ndim != 2 or rows.shape[1] != len(labels):
        raise RuntimeError(f"invalid history table {path}")
    if not np.all(np.isfinite(rows)):
        raise RuntimeError(f"nonfinite history row in {path}")
    return labels, rows


def interpolate(labels: list[str], rows: np.ndarray, target: float) -> dict[str, float]:
    tau = rows[:, labels.index("axisTau")]
    if target < tau[0] - 1.0e-14 or target > tau[-1] + 1.0e-14:
        raise RuntimeError(f"target axisTau={target} outside [{tau[0]}, {tau[-1]}]")
    result = {"axisTau": target}
    for field in ("time", *FIELDS):
        result[field] = float(np.interp(target, tau, rows[:, labels.index(field)]))
    return result


def order(delta_coarse: float, delta_fine: float) -> float | None:
    if delta_coarse <= 0.0 or delta_fine <= 0.0:
        return None
    return math.log(delta_coarse / delta_fine, 2.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    tables: dict[int, list[dict[str, float]]] = {}
    source_files = {}
    for resolution in (128, 256, 512):
        matches = sorted((args.root / "raw-histories" / f"N{resolution}").glob("*.z4c.user.hst"))
        if len(matches) != 1:
            raise RuntimeError(f"expected one N{resolution} history, found {matches}")
        labels, rows = load_history(matches[0])
        tables[resolution] = [interpolate(labels, rows, target) for target in TARGETS]
        source_files[str(resolution)] = str(matches[0])

    rows_out = []
    for target_index, target in enumerate(TARGETS):
        for field in FIELDS:
            values = [tables[resolution][target_index][field]
                      for resolution in (128, 256, 512)]
            d12, d23 = abs(values[1] - values[0]), abs(values[2] - values[1])
            rows_out.append({
                "axis_tau": target,
                "coordinate_time_N128": tables[128][target_index]["time"],
                "coordinate_time_N256": tables[256][target_index]["time"],
                "coordinate_time_N512": tables[512][target_index]["time"],
                "field": field,
                "N128": values[0], "N256": values[1], "N512": values[2],
                "difference_128_256": d12,
                "difference_256_512": d23,
                "observed_order": order(d12, d23),
                "monotone_decreasing_with_resolution":
                    values[2] <= values[1] <= values[0],
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "history_norm_convergence.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows_out[0]))
        writer.writeheader()
        writer.writerows(rows_out)
    summary = {
        "schema": "z4c_vc_fixed_grid_history_convergence_v1",
        "root": str(args.root),
        "source_files": source_files,
        "targets": list(TARGETS),
        "fields": list(FIELDS),
        "minimum_observed_order": min(
            row["observed_order"] for row in rows_out
            if row["observed_order"] is not None),
        "negative_order_count": sum(
            row["observed_order"] is not None and row["observed_order"] < 0.0
            for row in rows_out),
        "nonmonotone_norm_count": sum(
            not row["monotone_decreasing_with_resolution"] for row in rows_out),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
