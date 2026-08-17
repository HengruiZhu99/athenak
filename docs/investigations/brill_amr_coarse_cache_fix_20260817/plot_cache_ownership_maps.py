#!/usr/bin/env python3
"""Plot physical maps from the exact coarse-cache ownership CSV evidence."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent


def plot_difference_map(source: Path, output: Path, title: str) -> None:
    grouped: dict[tuple[float, float], float] = {}
    with source.open(newline="", encoding="utf-8") as stream:
        for record in csv.DictReader(stream):
            key = (float(record["rho"]), float(record["z"]))
            grouped[key] = max(grouped.get(key, 0.0), abs(float(record["difference"])))
    if not grouped:
        raise RuntimeError(f"no nonzero records available for {title}")
    rho = np.asarray([key[0] for key in grouped])
    z = np.asarray([key[1] for key in grouped])
    magnitude = np.asarray([grouped[key] for key in grouped])
    fig, axis = plt.subplots(figsize=(8.0, 5.8), constrained_layout=True)
    image = axis.scatter(rho, z, c=np.log10(magnitude), s=14, cmap="magma",
                         linewidths=0)
    axis.scatter([5.109375], [-0.046875], marker="x", s=90, c="cyan",
                 linewidths=1.8, label="known seam hotspot")
    axis.set_xlabel(r"$\rho/M$")
    axis.set_ylabel(r"$z/M$")
    axis.set_title(title)
    axis.legend(loc="best")
    colorbar = fig.colorbar(image, ax=axis)
    colorbar.set_label(r"$\log_{10}\max_v|\Delta u_v|$")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> None:
    plot_difference_map(
        HERE / "changed_consumed_coarse_values.csv",
        HERE / "coarse_cache_overwrite_map.png",
        "Consumed coarse-cache values overwritten by local refresh")
    plot_difference_map(
        HERE / "preserve_received_fine_differences.csv",
        HERE / "preserve_received_fine_difference_map.png",
        "Downstream O6 fine ghosts: current minus preserve-received")


if __name__ == "__main__":
    main()
