#!/usr/bin/env python3
"""Plot the matched old/patched N256 continuation using AthenaK history data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


COL = {
    "time": 0, "dt": 1, "C": 2, "H": 3, "M": 4, "Z": 5,
    "K": 11, "nmb": 12, "Kret": 13, "level": 14, "cycle": 18,
}


def load(path: Path) -> np.ndarray:
    values = np.loadtxt(path)
    return values.reshape(1, -1) if values.ndim == 1 else values


def strict_dump(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def summary(data: np.ndarray) -> dict[str, object]:
    last = data[-1]
    return {
        "rows": int(data.shape[0]),
        "terminal_time": float(last[COL["time"]]),
        "terminal_cycle": int(last[COL["cycle"]]),
        "last_dt": float(last[COL["dt"]]),
        "last_constraints": {key: float(last[COL[key]]) for key in ("C", "H", "M", "Z")},
        "last_max_abs_K": float(last[COL["K"]]),
        "last_max_kretschmann": float(last[COL["Kret"]]),
        "last_nmb": int(last[COL["nmb"]]),
        "last_max_level": int(last[COL["level"]]),
        "max_level_seen": int(np.max(data[:, COL["level"]])),
        "max_nmb_seen": int(np.max(data[:, COL["nmb"]])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", type=Path, required=True)
    parser.add_argument("--patched", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    old, patched = load(args.old), load(args.patched)
    strict_dump(args.output / "short_pde_comparison.json", {
        "schema": "athenak_brill_coarse_cache_short_pde_comparison_v1",
        "old": summary(old), "patched": summary(patched),
    })

    colors = {"old": "#a33b20", "patched": "#1768ac"}
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True,
                             sharex=True)
    for label, data in (("old", old), ("patched", patched)):
        t = data[:, COL["time"]]
        for key, style in (("C", "-"), ("H", "--"), ("M", ":"), ("Z", "-.")):
            axes[0, 0].plot(t, data[:, COL[key]], style, color=colors[label],
                            alpha=0.9, label=f"{label} {key}")
        axes[0, 1].plot(t, data[:, COL["K"]], "-", color=colors[label],
                        label=f"{label} max|K|")
        axes[0, 1].plot(t, data[:, COL["Kret"]], "--", color=colors[label],
                        label=f"{label} max Kretschmann")
        axes[1, 0].plot(t, data[:, COL["dt"]], color=colors[label], label=label)
        axes[1, 1].step(t, data[:, COL["level"]], where="post",
                        color=colors[label], label=f"{label} level")
        axes[1, 1].step(t, data[:, COL["nmb"]] / 100.0, where="post", linestyle="--",
                        color=colors[label], label=f"{label} NMB/100")
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_ylabel("proper norm²")
    axes[0, 0].set_title("Constraints")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_ylabel("maximum")
    axes[0, 1].set_title("Curvature diagnostics")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_ylabel(r"$dt/M$")
    axes[1, 0].set_xlabel(r"$t/M$")
    axes[1, 1].set_ylabel("level and NMB/100")
    axes[1, 1].set_xlabel(r"$t/M$")
    for axis in axes.flat:
        axis.grid(alpha=0.22)
        axis.legend(fontsize=7, ncol=2)
    fig.savefig(args.output / "short_pde_history_comparison.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
