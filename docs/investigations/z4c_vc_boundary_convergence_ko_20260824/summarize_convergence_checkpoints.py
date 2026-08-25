#!/usr/bin/env python3
"""Summarize pairwise/self convergence curves at required proper-time gates."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


METRICS = ("p_128_256", "p_256_512", "p_self")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with args.input.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(bool(rows), f"{args.input}: no rows")
    required = {"axisTau", "region", "field", *METRICS}
    require(not required - set(rows[0]), f"{args.input}: missing columns")

    output: list[dict[str, str | float]] = []
    regions = list(dict.fromkeys(row["region"] for row in rows))
    fields = list(dict.fromkeys(row["field"] for row in rows))
    for region in regions:
        for field in fields:
            selected = [row for row in rows
                        if row["region"] == region and row["field"] == field]
            selected.sort(key=lambda row: float(row["axisTau"]))
            require(bool(selected), f"missing {region} {field}")
            tau = np.asarray([float(row["axisTau"]) for row in selected])
            for metric in METRICS:
                values = np.asarray([float(row[metric]) for row in selected])
                early = values[(tau >= 0.5) & (tau <= 2.0) & np.isfinite(values)]
                late = values[(tau >= 3.0) & np.isfinite(values)]
                require(len(early) > 0 and len(late) > 0,
                        f"insufficient checkpoints for {region} {field} {metric}")
                output.append({
                    "region": region,
                    "field": field,
                    "metric": metric,
                    "early_median_tau_0p5_2": float(np.median(early)),
                    "at_tau_2": float(np.interp(2.0, tau, values)),
                    "at_tau_3": float(np.interp(3.0, tau, values)),
                    "terminal_tau": float(tau[-1]),
                    "terminal": float(values[-1]),
                    "minimum_tau_ge_3": float(np.min(late)),
                })

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(output)


if __name__ == "__main__":
    main()
