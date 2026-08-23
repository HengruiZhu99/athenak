#!/usr/bin/env python3
"""Summarize Ref-GH feedback-continuation histories and causal reach."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


USER_COLUMNS = {
    "time": 0, "dt": 1, "delta_q": 2, "delta_p": 3,
    "condition": 8, "relative_lapse_min": 9, "relative_lapse_max": 10,
    "v2_max": 11, "minus_physical_lapse_min": 14,
    "characteristic_max": 16, "transition": 17, "generation": 18,
    "xi": 19, "xi_dot": 20, "xi_ddot": 21, "v_cmd": 22, "risk": 23,
    "risk_condition": 24, "risk_lapse_min": 25, "risk_lapse_max": 26,
    "risk_v2": 27, "gh_l2": 28, "reduction_l2": 29, "curl_l2": 30,
    "constraint_veto": 31, "frozen": 32, "completed": 33,
}


def rows(path: Path) -> list[list[float]]:
    by_time: dict[float, list[float]] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = [float(value) for value in line.split()]
        if len(values) < max(USER_COLUMNS.values()) + 1:
            raise ValueError(f"{path}: expected at least 34 columns, got {len(values)}")
        by_time[values[0]] = values
    return [by_time[key] for key in sorted(by_time)]


def first_time(data: list[list[float]], key: str, predicate) -> float | None:
    index = USER_COLUMNS[key]
    for row in data:
        if predicate(row[index]):
            return row[0]
    return None


def summarize(path: Path, outer_distance: float) -> dict[str, object]:
    data = rows(path)
    if not data:
        raise ValueError(f"{path}: no data rows")
    travel = 0.0
    for left, right in zip(data, data[1:]):
        travel += 0.5*(left[USER_COLUMNS["characteristic_max"]]
                      + right[USER_COLUMNS["characteristic_max"]])*(right[0]-left[0])
    maxima = {
        key: max(row[index] for row in data)
        for key, index in USER_COLUMNS.items()
        if key.startswith("risk") or key in {
            "condition", "relative_lapse_max", "v2_max", "gh_l2",
            "reduction_l2", "curl_l2"}
    }
    minima = {
        "relative_lapse_min": min(row[USER_COLUMNS["relative_lapse_min"]]
                                  for row in data),
        "physical_lapse_min": min(-row[USER_COLUMNS["minus_physical_lapse_min"]]
                                  for row in data),
    }
    monotone_xi = all(right[USER_COLUMNS["xi"]] + 1.0e-14
                      >= left[USER_COLUMNS["xi"]]
                      for left, right in zip(data, data[1:]))
    nonnegative_rate = all(row[USER_COLUMNS["xi_dot"]] >= -1.0e-14 for row in data)
    deltas_exact_zero = all(
        row[USER_COLUMNS[key]] == 0.0
        for row in data
        for key in ("delta_q", "delta_p")
    )
    return {
        "schema": "ref-gh-feedback-continuation-history-v1",
        "source": str(path),
        "rows": len(data),
        "initial_time": data[0][0],
        "final_time": data[-1][0],
        "final_xi": data[-1][USER_COLUMNS["xi"]],
        "final_xi_dot": data[-1][USER_COLUMNS["xi_dot"]],
        "final_transition": data[-1][USER_COLUMNS["transition"]],
        "reached_xi_one": any(row[USER_COLUMNS["xi"]] >= 1.0 for row in data),
        "time_to_xi_one": first_time(data, "xi", lambda value: value >= 1.0),
        "first_feedback_slowing": first_time(
            data, "v_cmd", lambda value: value < 0.25 - 1.0e-14),
        "first_freeze": first_time(data, "frozen", lambda value: value > 0.5),
        "first_constraint_veto": first_time(
            data, "constraint_veto", lambda value: value > 0.5),
        "monotone_xi": monotone_xi,
        "nonnegative_xi_dot": nonnegative_rate,
        "deltas_exact_zero": deltas_exact_zero,
        "maxima": maxima,
        "minima": minima,
        "accumulated_characteristic_distance": travel,
        "outer_coordinate_distance": outer_distance,
        "remaining_causal_distance": outer_distance - travel,
        "finite": all(math.isfinite(row[index]) for row in data
                      for index in USER_COLUMNS.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    parser.add_argument("--outer-distance", type=float, default=12.0)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = summarize(args.history, args.outer_distance)
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
