#!/usr/bin/env python3
"""Check bounded Ref-GH feedback-state continuity across history gaps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GENERATION_COLUMN = 18
XI_COLUMN = 19
XI_DOT_COLUMN = 20


def rows(path: Path) -> list[list[float]]:
    result = []
    for line in path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            result.append([float(value) for value in line.split()])
    if not result:
        raise ValueError(f"{path}: no history rows")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("histories", type=Path, nargs="+")
    parser.add_argument("--absolute-tolerance", type=float, default=1.0e-11)
    parser.add_argument("--maximum-gap", type=float, default=0.06)
    parser.add_argument("--v-max", type=float, default=0.25)
    parser.add_argument("--max-acceleration", type=float, default=0.5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    comparisons = []
    all_pass = len(args.histories) >= 2
    for left_path, right_path in zip(args.histories, args.histories[1:]):
        left = rows(left_path)[-1]
        right = rows(right_path)[0]
        gap = right[0] - left[0]
        xi_change = right[XI_COLUMN] - left[XI_COLUMN]
        xi_dot_change = right[XI_DOT_COLUMN] - left[XI_DOT_COLUMN]
        checks = {
            "positive_bounded_time_gap": (
                gap > 0.0 and gap <= args.maximum_gap + args.absolute_tolerance),
            "xi_monotone": xi_change >= -args.absolute_tolerance,
            "xi_rate_bounded": (
                xi_change <= args.v_max*gap + args.absolute_tolerance),
            "xi_dot_nonnegative": right[XI_DOT_COLUMN] >= -args.absolute_tolerance,
            "xi_dot_acceleration_bounded": (
                abs(xi_dot_change)
                <= args.max_acceleration*gap + args.absolute_tolerance),
            "generation_monotone": (
                right[GENERATION_COLUMN] >= left[GENERATION_COLUMN]),
        }
        boundary_pass = all(checks.values())
        all_pass = all_pass and boundary_pass
        comparisons.append({
            "left": str(left_path),
            "right": str(right_path),
            "left_time": left[0],
            "right_time": right[0],
            "time_gap": gap,
            "xi_change": xi_change,
            "xi_dot_change": xi_dot_change,
            "generation_change": (
                right[GENERATION_COLUMN] - left[GENERATION_COLUMN]),
            "checks": checks,
            "pass": boundary_pass,
        })
    result = {
        "schema": "ref-gh-feedback-segment-restart-history-v2",
        "histories": [str(path) for path in args.histories],
        "boundaries": comparisons,
        "absolute_tolerance": args.absolute_tolerance,
        "maximum_gap": args.maximum_gap,
        "v_max": args.v_max,
        "max_acceleration": args.max_acceleration,
        "interpretation": (
            "History output resumes after a cadence-sized gap; exact restart "
            "equivalence is qualified separately. This gate rejects a reset "
            "or a rate/acceleration-discontinuous replicated trajectory."),
        "pass": all_pass,
    }
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(output)
    print(output, end="")
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
