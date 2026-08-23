#!/usr/bin/env python3
"""Verify controller-history native norms against the Ref-GH history."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


USER_CONSTRAINT_COLUMNS = (28, 29, 30)
REF_CONSTRAINT_SQUARE_COLUMNS = (2, 3, 4)
REF_VOLUME_COLUMN = 11
NAMES = ("gh_l2", "reduction_l2", "curl_l2")


def load(path: Path) -> dict[float, list[float]]:
    result: dict[float, list[float]] = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        row = [float(value) for value in line.split()]
        result[row[0]] = row
    if not result:
        raise ValueError(f"{path}: no data rows")
    return result


def compare(user_path: Path, ref_path: Path, tolerance: float) -> dict[str, object]:
    user = load(user_path)
    ref = load(ref_path)
    common_times = sorted(set(user) & set(ref))
    if not common_times:
        raise ValueError("histories have no common output times")
    maxima = {name: 0.0 for name in NAMES}
    differences = {name: 0.0 for name in NAMES}
    for time in common_times:
        user_row = user[time]
        ref_row = ref[time]
        volume = ref_row[REF_VOLUME_COLUMN]
        if not volume > 0.0:
            raise ValueError(f"nonpositive volume at t={time}")
        for name, user_column, ref_column in zip(
                NAMES, USER_CONSTRAINT_COLUMNS, REF_CONSTRAINT_SQUARE_COLUMNS):
            expected = math.sqrt(max(ref_row[ref_column], 0.0)/volume)
            actual = user_row[user_column]
            maxima[name] = max(maxima[name], expected)
            differences[name] = max(differences[name], abs(actual - expected))
    passed = all(value <= tolerance for value in differences.values())
    return {
        "schema": "ref-gh-feedback-constraint-history-consistency-v1",
        "user_history": str(user_path),
        "ref_gh_history": str(ref_path),
        "common_rows": len(common_times),
        "initial_time": common_times[0],
        "final_time": common_times[-1],
        "native_maxima": maxima,
        "linf_differences": differences,
        "absolute_tolerance": tolerance,
        "pass": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("user_history", type=Path)
    parser.add_argument("ref_gh_history", type=Path)
    parser.add_argument("--absolute-tolerance", type=float, default=5.0e-15)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    result = compare(args.user_history, args.ref_gh_history,
                     args.absolute_tolerance)
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(output)
    print(output, end="")
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
