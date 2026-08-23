#!/usr/bin/env python3
"""Summarize the bounded Schwarzschild reference-only path scan."""

import argparse
import csv
import json
import math
from pathlib import Path


TIMES = (0.0, 0.5, 1.0, 1.25, 1.4, 1.5, 1.6, 1.7, 2.0)
PATHS = ("shrinking_width", "frozen_wormhole", "fixed_core")
MEASURES = (
    "Ricci", "Riemann", "spin", "spin_derivative", "matched_source",
    "dB_dr", "d2B_dr2",
)


def log_slope(points):
    xs = [math.log(x) for x, _ in points]
    ys = [math.log(y) for _, y in points]
    xbar = sum(xs) / len(xs)
    ybar = sum(ys) / len(ys)
    denominator = sum((x - xbar) ** 2 for x in xs)
    return sum((x - xbar) * (y - ybar) for x, y in zip(xs, ys)) / denominator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("scan", type=Path)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()

    with args.scan.open(newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    indexed = {}
    for row in rows:
        key = (row["path"], round(float(row["time"]), 12), row["measure"])
        if key in indexed:
            raise SystemExit(f"duplicate scan record: {key}")
        indexed[key] = row
    required = {
        (path, round(time, 12), measure)
        for path in PATHS for time in TIMES for measure in MEASURES
    }
    if set(indexed) != required:
        missing = sorted(required - set(indexed))
        extra = sorted(set(indexed) - required)
        raise SystemExit(f"scan matrix mismatch: missing={missing}, extra={extra}")

    shrinking = {
        measure: [
            (float(indexed[("shrinking_width", round(time, 12), measure)]["r_core"]),
             float(indexed[("shrinking_width", round(time, 12), measure)]["maximum"]))
            for time in TIMES[1:]
        ]
        for measure in ("dB_dr", "d2B_dr2")
    }
    slopes = {measure: log_slope(points) for measure, points in shrinking.items()}
    comparisons = []
    for time in (1.0, 1.5, 1.6, 2.0):
        for measure in ("Ricci", "Riemann", "spin_derivative", "matched_source"):
            values = {
                path: float(indexed[(path, round(time, 12), measure)]["maximum"])
                for path in PATHS
            }
            comparisons.append({
                "time": time,
                "measure": measure,
                **values,
                "shrinking_over_fixed_core": (
                    values["shrinking_width"] / values["fixed_core"]
                ),
            })

    result = {
        "input": str(args.scan),
        "records": len(rows),
        "samples_per_record": int(rows[0]["samples"]),
        "shrinking_width_log_slopes_vs_r_core": slopes,
        "expected_slopes": {"dB_dr": -1.0, "d2B_dr2": -2.0},
        "comparisons": comparisons,
    }
    args.json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with args.table.open("w", newline="") as stream:
        fields = (
            "time", "measure", "shrinking_width", "frozen_wormhole",
            "fixed_core", "shrinking_over_fixed_core",
        )
        writer = csv.DictWriter(
            stream, fieldnames=fields, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(comparisons)
    print(json.dumps({
        "records": len(rows),
        "dB_dr_slope": slopes["dB_dr"],
        "d2B_dr2_slope": slopes["d2B_dr2"],
    }, sort_keys=True))


if __name__ == "__main__":
    main()
