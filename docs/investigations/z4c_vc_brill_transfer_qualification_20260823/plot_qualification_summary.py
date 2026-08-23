#!/usr/bin/env python3
"""Make compact qualification plots from committed Phase-6 CSV evidence."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
HISTORY = (
    ROOT
    / "artifacts/perlmutter_phase6-fixed-brill-analysis/analysis"
    / "common_axis_time_history.csv"
)
ORDERS = (
    ROOT
    / "artifacts/perlmutter_phase6-fixed-brill-terminal-rhs-analysis/analysis"
    / "orders.csv"
)
OUT = ROOT / "artifacts/qualification_summary"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def plot_constraints() -> None:
    rows = read_csv(HISTORY)
    variables = ["C-norm2", "H-norm2", "M-norm2", "Z-norm2"]
    resolutions = ["N128", "N256", "N512"]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), sharex=True)
    for axis, variable in zip(axes.flat, variables):
        selected = sorted(
            (row for row in rows if row["variable"] == variable),
            key=lambda row: float(row["axis_tau"]),
        )
        tau = [float(row["axis_tau"]) for row in selected]
        for resolution in resolutions:
            values = [float(row[resolution]) for row in selected]
            axis.semilogy(tau, values, "o-", label=resolution)
        axis.set_title(variable)
        axis.grid(True, which="both", alpha=0.25)
        axis.set_ylabel("proper ring RMS")
    for axis in axes[-1, :]:
        axis.set_xlabel(r"central proper time $\tau_c/M$")
    axes[0, 0].legend(frameon=False)
    fig.suptitle("Fixed-grid VC/O4 Brill constraints (no AMR)")
    fig.tight_layout()
    fig.savefig(OUT / "phase6_fixed_grid_constraint_history.png", dpi=180)
    plt.close(fig)


def plot_regional_orders() -> None:
    rows = read_csv(ORDERS)
    fields = [
        ("state", "state_z4c_chi", r"$\chi$"),
        ("state", "state_z4c_alpha", r"$\alpha$"),
        ("state", "state_z4c_Theta", r"$\Theta$"),
        ("state", "state_z4c_Bx", r"$B^\rho$"),
        ("constraints", "con_H", "H"),
        ("constraints", "con_M", "M"),
        ("constraints", "con_C", "C"),
        ("constraints", "con_Z", "Z"),
    ]
    regions = [
        ("core_r8", "core r<=8", "ring_rms_order"),
        ("core_r8_mb_interior", "core MB interior", "ring_rms_order"),
        ("axis", "axis", "rms_order"),
        ("physical_boundary", "outer boundary", "ring_rms_order"),
    ]
    lookup = {
        (row["region"], row["family"], row["variable"]): row for row in rows
    }
    x = np.arange(len(fields))
    width = 0.19
    fig, axis = plt.subplots(figsize=(12.2, 5.8))
    for offset, (region, label, metric) in enumerate(regions):
        values = []
        for family, variable, _ in fields:
            raw = lookup[(region, family, variable)][metric]
            values.append(float(raw) if raw else np.nan)
        axis.bar(x + (offset - 1.5) * width, values, width, label=label)
    axis.axhline(4.0, color="black", linewidth=1.0, linestyle="--", label="O4")
    axis.axhline(0.0, color="black", linewidth=0.7)
    axis.set_xticks(x, [label for _, _, label in fields])
    axis.set_ylabel("observed N128/N256/N512 order")
    axis.set_title("Terminal fixed-grid regional convergence at tau_c=3.08 M")
    axis.grid(True, axis="y", alpha=0.25)
    axis.legend(ncol=3, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "phase6_fixed_grid_regional_orders.png", dpi=180)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    plot_constraints()
    plot_regional_orders()


if __name__ == "__main__":
    main()
