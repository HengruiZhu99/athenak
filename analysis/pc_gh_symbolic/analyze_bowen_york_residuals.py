#!/usr/bin/env python3
"""Fail closed on the time-symmetric Bowen-York production-path audit ladder."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


EXACT_RMS = (
    "rms_state_primary",
    "rms_rhs_gradient",
    "rms_reduction_curl_algebraic",
)
CONVERGENT_RMS = (
    "rms_state_gradient",
    "rms_rhs_primary",
    "rms_GH_physical",
)


def read_table(path: Path) -> tuple[list[str], list[dict[str, float]]]:
    with path.open(encoding="utf-8") as stream:
        header_line = stream.readline()
        if not header_line.startswith("# "):
            raise AssertionError(f"{path}: missing named header")
        header = header_line[2:].split()
        rows = [dict(zip(header, map(float, line.split()), strict=True))
                for line in stream if line.strip() and not line.startswith("#")]
    rows.sort(key=lambda row: row["N"])
    if len(rows) < 4:
        raise AssertionError(
            f"{path}: need at least four resolutions to control hard-shell sampling, "
            f"found {len(rows)}")
    if len({row["N"] for row in rows}) != len(rows):
        raise AssertionError(f"{path}: duplicate resolutions")
    required = {"N", *EXACT_RMS, *CONVERGENT_RMS}
    missing = sorted(required - set(header))
    if missing:
        raise AssertionError(f"{path}: missing columns: {', '.join(missing)}")
    return header, rows


def fitted_order(rows: list[dict[str, float]], key: str) -> float:
    resolutions = np.asarray([row["N"] for row in rows])
    errors = np.asarray([row[key] for row in rows])
    if not np.all(np.isfinite(errors)) or np.any(errors <= 0.0):
        raise AssertionError(f"{key}: nonpositive or nonfinite error")
    return -float(np.polyfit(np.log(resolutions), np.log(errors), 1)[0])


def finest_order(rows: list[dict[str, float]], key: str) -> float:
    coarse, fine = rows[-2:]
    return math.log(coarse[key]/fine[key])/math.log(fine["N"]/coarse["N"])


def check_maxima(path: Path) -> None:
    with path.open(encoding="utf-8") as stream:
        if stream.readline().split() != ["#", "kind", "name", "max_abs", "x", "y", "z"]:
            raise AssertionError(f"{path}: malformed maximum-location header")
        rows = [line.split() for line in stream if line.strip()]
    if not rows:
        raise AssertionError(f"{path}: empty maximum-location table")
    if not all(len(row) == 6 and all(math.isfinite(float(value)) for value in row[2:])
               for row in rows):
        raise AssertionError(f"{path}: malformed or nonfinite maximum-location row")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("table", type=Path)
    parser.add_argument("--min-fit-order", type=float, default=1.8)
    parser.add_argument("--min-finest-order", type=float, default=1.85)
    parser.add_argument("--roundoff-tol", type=float, default=2.0e-12)
    args = parser.parse_args()
    _, rows = read_table(args.table)

    print("resolutions", " ".join(str(int(row["N"])) for row in rows))
    for key in EXACT_RMS:
        maximum = max(row[key] for row in rows)
        print(f"{key:36s} max={maximum:.6e}")
        if not math.isfinite(maximum) or maximum > args.roundoff_tol:
            raise AssertionError(
                f"{key}: {maximum:.6e} exceeds roundoff tolerance {args.roundoff_tol:.6e}")

    for key in CONVERGENT_RMS:
        values = [row[key] for row in rows]
        if not all(fine < coarse for coarse, fine in zip(values, values[1:])):
            raise AssertionError(f"{key}: residual does not decrease monotonically")
        fit = fitted_order(rows, key)
        finest = finest_order(rows, key)
        print(f"{key:36s} fit_order={fit:.6f} finest_order={finest:.6f}")
        if fit < args.min_fit_order:
            raise AssertionError(
                f"{key}: fitted order {fit:.6f} < {args.min_fit_order:.6f}")
        if finest < args.min_finest_order:
            raise AssertionError(
                f"{key}: finest order {finest:.6f} < {args.min_finest_order:.6f}")

    for row in rows:
        resolution = int(row["N"])
        check_maxima(args.table.parent/f"pc_gh_bowen_york-maxima-N{resolution}.dat")
    print("PASS: time-symmetric Bowen-York ADM-to-PC-GH pointwise ladder")


if __name__ == "__main__":
    main()
