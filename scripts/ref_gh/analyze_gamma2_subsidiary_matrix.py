#!/usr/bin/env python3
"""Check Ref-GH gamma2 reduction/curl decay with KO off and on."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


CASES = (
    (0.0, 0.0, "g0p0_d0p0"),
    (0.5, 0.0, "g0p5_d0p0"),
    (1.0, 0.0, "g1p0_d0p0"),
    (0.0, 0.02, "g0p0_d0p02"),
    (0.5, 0.02, "g0p5_d0p02"),
    (1.0, 0.02, "g1p0_d0p02"),
)


def read_history(path: Path) -> list[list[float]]:
    rows = []
    for line in path.read_text().splitlines():
        if line and not line.startswith("#"):
            rows.append([float(value) for value in line.split()])
    if len(rows) < 2:
        raise ValueError(f"history has fewer than two rows: {path}")
    if any(len(row) < 5 for row in rows):
        raise ValueError(f"history is missing native constraint columns: {path}")
    return rows


def l2_growth(rows: list[list[float]], column: int) -> list[float]:
    initial = rows[0][column]
    if not math.isfinite(initial) or not initial > 0.0:
        raise ValueError("initial squared norm must be finite and positive")
    result = []
    for row in rows:
        value = row[column]
        if not math.isfinite(value) or value < 0.0:
            raise ValueError("squared norm must be finite and nonnegative")
        result.append(math.sqrt(value/initial))
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--absolute-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    histories = {}
    for gamma2, dissipation, tag in CASES:
        path = args.root/tag/f"{tag}.ref_gh.hst"
        rows = read_history(path)
        histories[(gamma2, dissipation)] = {
            "path": str(path),
            "rows": rows,
            "time": [row[0] for row in rows],
            "gh": l2_growth(rows, 2),
            "reduction": l2_growth(rows, 3),
            "curl": l2_growth(rows, 4),
        }

    records = []
    all_pass = True
    for gamma2, dissipation, tag in CASES:
        item = histories[(gamma2, dissipation)]
        baseline = histories[(0.0, dissipation)]
        if item["time"] != baseline["time"]:
            raise ValueError(f"history times differ for {tag}")
        record = {
            "tag": tag,
            "gamma2": gamma2,
            "dissipation": dissipation,
            "final_time": item["time"][-1],
            "final_gh_growth": item["gh"][-1],
        }
        for constraint in ("reduction", "curl"):
            expected = [
                base*math.exp(-gamma2*time)
                for base, time in zip(baseline[constraint], item["time"])
            ]
            errors = [
                abs(observed-predicted)
                for observed, predicted in zip(item[constraint], expected)
            ]
            passed = max(errors) <= args.absolute_tolerance
            all_pass = all_pass and passed
            record[f"final_{constraint}_growth"] = item[constraint][-1]
            record[f"final_{constraint}_ko_factor"] = baseline[constraint][-1]
            record[f"final_{constraint}_predicted"] = expected[-1]
            record[f"max_{constraint}_absolute_error"] = max(errors)
            record[f"{constraint}_pass"] = passed
        records.append(record)

    result = {
        "schema": "ref-gh-gamma2-subsidiary-matrix-v1",
        "matrix": {
            "gamma2": [0.0, 0.5, 1.0],
            "dissipation": [0.0, 0.02],
        },
        "prediction": (
            "C_L2(gamma2,diss,t) = C_L2(0,diss,t) exp(-gamma2 t)"
        ),
        "history_columns": {
            "gh_l2_squared": 3,
            "reduction_l2_squared": 4,
            "curl_l2_squared": 5,
        },
        "absolute_tolerance": args.absolute_tolerance,
        "records": records,
        "reduction_and_curl_pass": all_pass,
        "gh_constraint_qualification": (
            "not assessed by this matrix; the random GH seed does not have the "
            "single exponential gamma2 subsidiary prediction"
        ),
    }
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    columns = tuple(records[0])
    with args.output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
