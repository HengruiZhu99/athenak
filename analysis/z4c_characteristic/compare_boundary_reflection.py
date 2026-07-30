#!/usr/bin/env python3
"""Compare matched CPBC and extrapolating-Sommerfeld pulse reflections."""

import argparse
import math
import pathlib
import re


RATIO = re.compile(r"\bratio=([+\-0-9.eE]+)")


def load(root):
    values = {}
    for result in sorted(root.glob("axis*_side*_*/interior_reflection.txt")):
        match = RATIO.search(result.read_text())
        if match is None:
            raise SystemExit("{}: missing ratio".format(result))
        ratio = float(match.group(1))
        if not math.isfinite(ratio):
            raise SystemExit("{}: nonfinite ratio".format(result))
        values[result.parent.name] = ratio
    if not values:
        raise SystemExit("{}: no reflection results".format(root))
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cpbc", type=pathlib.Path)
    parser.add_argument("sommerfeld", type=pathlib.Path)
    parser.add_argument("--maximum-cpbc-ratio", type=float, default=0.02)
    parser.add_argument("--minimum-median-improvement", type=float, default=2.0)
    parser.add_argument("--minimum-worst-improvement", type=float, default=1.5)
    parser.add_argument("--maximum-case-regression", type=float, default=1.10)
    args = parser.parse_args()

    cpbc = load(args.cpbc)
    sommerfeld = load(args.sommerfeld)
    if set(cpbc) != set(sommerfeld):
        missing_cpbc = sorted(set(sommerfeld) - set(cpbc))
        missing_sommerfeld = sorted(set(cpbc) - set(sommerfeld))
        raise SystemExit(
            "unmatched cases: missing CPBC={} missing Sommerfeld={}".format(
                missing_cpbc, missing_sommerfeld))

    rows = []
    for name in sorted(cpbc):
        improvement = sommerfeld[name] / max(cpbc[name], 1.0e-300)
        rows.append((name, cpbc[name], sommerfeld[name], improvement))
        print(
            "{} cpbc={:.8e} sommerfeld={:.8e} improvement={:.6f}".format(
                name, cpbc[name], sommerfeld[name], improvement))

    cpbc_sorted = sorted(value for _, value, _, _ in rows)
    sommerfeld_sorted = sorted(value for _, _, value, _ in rows)
    middle = len(rows) // 2
    if len(rows) % 2:
        cpbc_median = cpbc_sorted[middle]
        sommerfeld_median = sommerfeld_sorted[middle]
    else:
        cpbc_median = 0.5 * (
            cpbc_sorted[middle - 1] + cpbc_sorted[middle])
        sommerfeld_median = 0.5 * (
            sommerfeld_sorted[middle - 1] + sommerfeld_sorted[middle])
    median_improvement = sommerfeld_median / max(cpbc_median, 1.0e-300)
    worst_improvement = max(sommerfeld.values()) / max(cpbc.values())
    worst_case_factor = max(
        cpbc[name] / max(sommerfeld[name], 1.0e-300) for name in cpbc)
    print(
        "SUMMARY cases={} cpbc_median={:.8e} sommerfeld_median={:.8e} "
        "median_improvement={:.6f} cpbc_worst={:.8e} "
        "sommerfeld_worst={:.8e} worst_improvement={:.6f} "
        "worst_case_regression={:.6f}".format(
            len(rows), cpbc_median, sommerfeld_median, median_improvement,
            max(cpbc.values()), max(sommerfeld.values()), worst_improvement,
            worst_case_factor))

    failures = []
    if max(cpbc.values()) >= args.maximum_cpbc_ratio:
        failures.append("CPBC maximum reflection exceeds its gate")
    if median_improvement < args.minimum_median_improvement:
        failures.append("median improvement is too small")
    if worst_improvement < args.minimum_worst_improvement:
        failures.append("worst-case improvement is too small")
    if worst_case_factor > args.maximum_case_regression:
        failures.append("at least one matched family regressed")
    if failures:
        raise SystemExit("; ".join(failures))


if __name__ == "__main__":
    main()
