#!/usr/bin/env python3
"""Check measured fused-CPBC cost against the Z4c RHS kernel cost."""

import argparse
import math
import pathlib
import re


PATTERN = re.compile(
    r"Z4C_CHARACTERISTIC_CPBC .*?"
    r"performance_valid=(?P<valid>[01]) "
    r"kernel_seconds=(?P<kernel>\S+) "
    r"volume_rhs_seconds=(?P<rhs>\S+) "
    r"kernel_fraction=(?P<fraction>\S+)"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("log", type=pathlib.Path)
    parser.add_argument("--maximum-fraction", type=float, default=0.03)
    parser.add_argument("--warmup-samples", type=int, default=1)
    args = parser.parse_args()

    rows = []
    for match in PATTERN.finditer(args.log.read_text(encoding="utf-8")):
        if match.group("valid") != "1":
            continue
        row = tuple(
            float(match.group(name)) for name in ("kernel", "rhs", "fraction")
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in row):
            raise SystemExit("nonfinite or negative CPBC timing")
        if row[1] <= 0.0:
            raise SystemExit("valid CPBC timing has zero volume-RHS time")
        rows.append((row[0], row[1], row[0] / row[1], row[2]))
    if not rows:
        raise SystemExit("no CPBC timing diagnostics found")
    if args.warmup_samples < 0 or args.warmup_samples >= len(rows):
        raise SystemExit("invalid number of warmup samples")
    measured = rows[args.warmup_samples:]

    # The reported values bracket the preceding non-diagnostic cycle.
    # kernel_seconds and volume_rhs_seconds are independently reduced maxima
    # across ranks, so their ratio measures the synchronous critical-path cost.
    # The printed kernel_fraction is the maximum same-rank ratio and can be
    # dominated by a lightly loaded boundary rank; retain it as a diagnostic
    # but do not use it as the total-step performance gate.
    maximum = max(row[2] for row in measured)
    median = sorted(row[2] for row in measured)[len(measured) // 2]
    maximum_rank_fraction = max(row[3] for row in measured)
    print(
        "samples={} warmup_samples={} median_critical_path_fraction={:.8e} "
        "maximum_critical_path_fraction={:.8e} "
        "maximum_same_rank_fraction={:.8e}".format(
            len(rows), args.warmup_samples, median, maximum,
            maximum_rank_fraction))
    if maximum >= args.maximum_fraction:
        raise SystemExit(
            "CPBC fraction {:.6e} exceeds {:.6e}".format(
                maximum, args.maximum_fraction))


if __name__ == "__main__":
    main()
