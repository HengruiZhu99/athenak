#!/usr/bin/env python3
"""Plot native-VC state and production-RHS coarse/fine defects."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def series(rows: list[dict[str, str]], transfer: str, category: str,
           variable: str) -> tuple[np.ndarray, np.ndarray]:
    selected = sorted(
        (row for row in rows if row["transfer_order"] == transfer and
         row["category"] == category and row["variable"] == variable),
        key=lambda row: int(row["resolution"]))
    return (np.asarray([int(row["resolution"]) for row in selected]),
            np.asarray([float(row["rms"]) for row in selected]))


def reference_slope(axis: plt.Axes, x: np.ndarray, y: np.ndarray,
                    order: int, label: str) -> None:
    positive = y[y > 0.0]
    if not len(positive):
        return
    anchor = positive[-1]
    reference = anchor * (x[-1] / x) ** order
    axis.loglog(x, reference, linestyle=":", color="0.5", linewidth=1.0,
                label=label)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    state = load(args.input_dir / "state_defect_metrics.csv")
    rhs = load(args.input_dir / "rhs_defect_metrics.csv")
    panels = [
        (state, "interface_layer_0", "Bz", "State at interface: Bz"),
        (rhs, "interface_layer_0", "Gamy", "RHS interface: Gamma_y"),
        (rhs, "interface_layer_1", "Gamx", "RHS first layer: Gamma_x"),
        (rhs, "all", "Gamy", "Global sampled RHS: Gamma_y"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), constrained_layout=True)
    for axis, (rows, category, variable, title) in zip(axes.flat, panels):
        plotted: list[tuple[np.ndarray, np.ndarray]] = []
        for transfer, color, marker in (("4", "#d95f02", "o"),
                                        ("6", "#1b9e77", "s")):
            x, y = series(rows, transfer, category, variable)
            axis.loglog(x, y, marker=marker, color=color, linewidth=1.8,
                        label=f"O4 FD + q{transfer}")
            plotted.append((x, y))
        x4, y4 = plotted[0]
        x6, y6 = plotted[1]
        if "State" in title:
            reference_slope(axis, x4, y4, 4, r"$N^{-4}$")
            reference_slope(axis, x6, y6, 6, r"$N^{-6}$")
        elif category == "all":
            reference_slope(axis, x4, y4, 2, r"$N^{-2}$ guide")
            reference_slope(axis, x6, y6, 4, r"$N^{-4}$ guide")
        else:
            reference_slope(axis, x4, y4, 2, r"$N^{-2}$")
            reference_slope(axis, x6, y6, 4, r"$N^{-4}$")
        axis.set_title(title)
        axis.set_xlabel("base resolution N")
        axis.set_ylabel("level-matched RMS defect")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(fontsize=8)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
