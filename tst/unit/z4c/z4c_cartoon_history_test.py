#!/usr/bin/env python3
"""Validate the half-plane Cartoon constraint-history contract."""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path


FAMILIES = ("C", "H", "M", "Z")
LAYERS = range(5)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def close(a: float, b: float) -> bool:
    # Athena history output is intentionally printed with five digits after the
    # decimal in scientific notation.  Allow only the accumulated text-rounding
    # error from the five layer entries.
    return math.isclose(a, b, rel_tol=3.0e-5, abs_tol=5.0e-9)


def parse_history(path: Path) -> tuple[list[str], list[dict[str, float]]]:
    text = path.read_text(encoding="utf-8")
    header_lines = [line for line in text.splitlines() if line.startswith("#  [1]=")]
    require(len(header_lines) == 1, "history must contain exactly one indexed header")
    labels = re.findall(r"\[\d+\]=([^ ]+)", header_lines[0])
    require(len(labels) == len(set(labels)), "history labels must be unique")

    rows: list[dict[str, float]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        values = [float(token) for token in line.split()]
        require(len(values) == len(labels), "history row width differs from header")
        require(all(math.isfinite(value) for value in values),
                "history row contains a non-finite value")
        rows.append(dict(zip(labels, values, strict=True)))
    require(rows, "history contains no data rows")
    return labels, rows


def validate_row(row: dict[str, float]) -> None:
    require(row["Volume"] > 0.0, "physical diagnostic volume must be positive")
    require(row["off-Vol"] > 0.0, "off-axis physical volume must be positive")

    layer_counts = [row[f"L{layer}-N"] for layer in LAYERS]
    require(all(count > 0.0 and close(count, round(count)) for count in layer_counts),
            "radial-layer cell counts must be positive integers")
    require(close(row["ax-N"], sum(layer_counts)),
            "axis-tube count does not equal the five resolved radial layers")

    for family in FAMILIES:
        axis_sum = row[f"ax-{family}2"]
        off_sum = row[f"off-{family}2"]
        layer_sum = sum(row[f"L{layer}-{family}2"] for layer in LAYERS)
        maximum = row[f"{family}-Linf"]
        rho = row[f"{family}-rho"]
        z = row[f"{family}-z"]

        require(axis_sum >= 0.0 and off_sum >= 0.0 and maximum >= 0.0,
                f"{family} diagnostic must be nonnegative")
        require(close(axis_sum, layer_sum),
                f"{family} axis sum does not equal its radial-layer sums")
        require(-1.0e-14 <= rho <= 8.0 + 1.0e-14,
                f"{family} Linf rho is outside the active half-plane")
        require(-8.0 - 1.0e-14 <= z <= 8.0 + 1.0e-14,
                f"{family} Linf z is outside the active domain")

        # Every reported quadratic mean is bounded by the same active-cell
        # maximum.  This checks the squared-vs-unsquared family conventions as
        # well as the separation between unweighted axis and weighted off-axis
        # reductions.
        maximum2 = maximum * maximum
        require(axis_sum <= row["ax-N"] * maximum2 * (1.0 + 3.0e-5) + 5.0e-9,
                f"{family} axis RMS exceeds its global Linf value")
        require(off_sum <= row["off-Vol"] * maximum2 * (1.0 + 3.0e-5) + 5.0e-9,
                f"{family} off-axis RMS exceeds its global Linf value")
        require(row[f"{family}-norm2"] <=
                row["Volume"] * maximum2 * (1.0 + 3.0e-5) + 5.0e-9,
                f"{family} global physical RMS exceeds its Linf value")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", required=True, type=Path)
    args = parser.parse_args()
    labels, rows = parse_history(args.history)

    expected_suffix = []
    expected_suffix += [f"ax-{family}2" for family in FAMILIES] + ["ax-N"]
    expected_suffix += [f"off-{family}2" for family in FAMILIES] + ["off-Vol"]
    for layer in LAYERS:
        expected_suffix += [f"L{layer}-{family}2" for family in FAMILIES]
        expected_suffix += [f"L{layer}-N"]
    for family in FAMILIES:
        expected_suffix += [f"{family}-Linf", f"{family}-rho", f"{family}-z"]
    require(labels[-len(expected_suffix):] == expected_suffix,
            "Cartoon constraint-history inventory or ordering changed")

    for row in rows:
        validate_row(row)
    print("Cartoon axis/off-axis/layer/Linf history contract passed")


if __name__ == "__main__":
    main()
