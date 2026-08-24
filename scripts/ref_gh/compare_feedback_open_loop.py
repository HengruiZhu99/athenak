#!/usr/bin/env python3
"""Compare controlled T4 and prescribed T5 histories on the common time window."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


COLUMNS = {
    "condition": 8,
    "relative_lapse_min": 9,
    "relative_lapse_max": 10,
    "v2": 11,
    "transition": 17,
    "xi": 19,
    "xi_dot": 20,
    "risk": 23,
    "gh_l2": 28,
    "reduction_l2": 29,
    "curl_l2": 30,
}


def read_rows(path: Path) -> list[list[float]]:
    by_time: dict[float, list[float]] = {}
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            row = [float(value) for value in line.split()]
            if len(row) < 34:
                raise ValueError(f"{path}: expected at least 34 columns")
            by_time[row[0]] = row
    if not by_time:
        raise ValueError(f"{path}: no history rows")
    return [by_time[time] for time in sorted(by_time)]


def interpolate(rows: list[list[float]], time: float) -> dict[str, float]:
    if time < rows[0][0] or time > rows[-1][0]:
        raise ValueError(f"time {time} outside [{rows[0][0]}, {rows[-1][0]}]")
    right = next((i for i, row in enumerate(rows) if row[0] >= time), None)
    assert right is not None
    if rows[right][0] == time or right == 0:
        return {key: rows[right][index] for key, index in COLUMNS.items()}
    left = right - 1
    fraction = (time - rows[left][0])/(rows[right][0] - rows[left][0])
    return {
        key: rows[left][index] + fraction*(rows[right][index] - rows[left][index])
        for key, index in COLUMNS.items()
    }


def first_crossing(rows: list[list[float]], column: str, threshold: float,
                   above: bool = True) -> float | None:
    index = COLUMNS[column]
    for row in rows:
        if (row[index] >= threshold) if above else (row[index] <= threshold):
            return row[0]
    return None


def maxima(rows: list[list[float]], end: float) -> dict[str, float]:
    selected = [row for row in rows if row[0] <= end + 1.0e-12]
    return {key: max(row[index] for row in selected)
            for key, index in COLUMNS.items()
            if key not in {"relative_lapse_min"}}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("controlled", type=Path)
    parser.add_argument("prescribed", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    controlled = read_rows(args.controlled)
    prescribed = read_rows(args.prescribed)
    common_end = min(controlled[-1][0], prescribed[-1][0])
    sample_times = [time for time in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0,
                                      3.5, common_end)
                    if controlled[0][0] <= time <= common_end]
    sample_times = sorted(set(sample_times))
    result = {
        "schema": "ref-gh-feedback-open-loop-comparison-v1",
        "common_end_time": common_end,
        "controlled_final_time": controlled[-1][0],
        "prescribed_final_time": prescribed[-1][0],
        "events": {
            label: {
                "risk_slow": first_crossing(rows, "risk", 0.70),
                "risk_stop": first_crossing(rows, "risk", 1.0),
                "v2_stop": first_crossing(rows, "v2", 0.20),
                "condition_stop": first_crossing(rows, "condition", 8.0),
                "gh_warning": first_crossing(rows, "gh_l2", 2.0e-2),
                "reduction_warning": first_crossing(rows, "reduction_l2", 5.0e-3),
                "curl_warning": first_crossing(rows, "curl_l2", 8.0e-2),
            }
            for label, rows in (("controlled", controlled),
                                ("prescribed", prescribed))
        },
        "maxima_on_common_window": {
            "controlled": maxima(controlled, common_end),
            "prescribed": maxima(prescribed, common_end),
        },
        "samples": [
            {
                "time": time,
                "controlled": interpolate(controlled, time),
                "prescribed": interpolate(prescribed, time),
            }
            for time in sample_times
        ],
        "finite": all(math.isfinite(value) for rows in (controlled, prescribed)
                      for row in rows for value in row),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
