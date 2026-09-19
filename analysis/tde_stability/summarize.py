#!/usr/bin/env python3
"""Compare completed or running trumpet controls without modifying their outputs.

Example: python summarize.py --run 'baseline=/path/to/run' --output /tmp/report
History norms use their configured mask; only compare matching physical masks.
"""
import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def history(path):
    lines = path.read_text().splitlines()
    keys = re.findall(r"\[\d+\]=([^\s]+)", lines[1])
    rows = []
    for line in lines:
        if not line or line.startswith("#"):
            continue
        values = line.split()
        if len(values) != len(keys):  # A currently written row can be incomplete.
            continue
        rows.append([float(v) for v in values])
    data = np.asarray(rows)
    if not len(data):
        raise ValueError(f"No complete history rows: {path}")
    return dict(zip(keys, data.T))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="append", required=True, help="label=directory")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-time", type=float)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), layout="constrained",
                             facecolor="white")
    for spec in args.run:
        label, directory = spec.split("=", 1)
        directory = Path(directory)
        u = history(directory / "ks_background.user.hst")
        h = history(directory / "ks_background.z4c.user.hst")
        theta = np.sqrt(h.get("Theta-norm2", h.get("Theta-norm")))
        log = (directory / "run.log").read_text(errors="replace")
        invalid = re.findall(r"Z4C_INVALID_STATE time=([^\s]+).*", log)
        stop = ("invalid state" if invalid else
                "walltime" if "Terminating on wall clock limit" in log else
                "target time" if "Terminating on time limit" in log else
                "cycle limit" if "Terminating on cycle limit" in log else
                "running or unclassified; inspect scheduler/exit")
        row = {
            "directory": str(directory.resolve()), "last_history_time_M": float(u["time"][-1]),
            "stop": stop, "first_recorded_invalid_time_M": float(invalid[0]) if invalid else None,
            "finite_histories": bool(all(np.isfinite(v).all() for v in [*u.values(), *h.values()])),
            "max_recorded_bad_metric": float(u["bad-metric"].max()),
            "final_Theta_max": float(u["Theta-max"][-1]),
            "final_masked_Theta_L2": float(theta[-1]),
            "growth_windows": [],
        }
        for lo, hi in [(50, 100), (100, 200), (200, 300), (300, 500), (500, 750), (750, 1000)]:
            mask = (h["time"] >= lo) & (h["time"] <= hi) & (theta > 0) & np.isfinite(theta)
            if mask.sum() < 15:
                continue
            t, y = h["time"][mask], np.log(theta[mask])
            fit = np.polyfit(t, y, 1)
            variance = np.sum((y-y.mean())**2)
            row["growth_windows"].append({
                "range_M": [float(t[0]), float(t[-1])], "gamma_per_M": float(fit[0]),
                "R2": float(1-np.sum((y-np.polyval(fit, t))**2)/variance) if variance else None,
            })
        results[label] = row
        axes[0].semilogy(u["time"], np.where(u["Theta-max"] > 0, u["Theta-max"], np.nan), label=label)
        axes[1].semilogy(h["time"], np.where(theta > 0, theta, np.nan), label=label)
    axes[0].set_ylabel(r"Global $\max |\Theta|$")
    axes[1].set_ylabel(r"Masked proper-volume $\|\Theta\|_2$")
    for ax in axes:
        ax.set_xlabel(r"$t/M$")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
        if args.max_time:
            ax.set_xlim(0, args.max_time)
    fig.savefig(args.output / "comparison.png", dpi=180)
    plt.close(fig)
    (args.output / "results.json").write_text(json.dumps(results, indent=2)+"\n")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
