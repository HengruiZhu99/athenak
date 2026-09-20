#!/usr/bin/env python3
"""Summarize completed boundary controls, keeping ghost failures separate from histories."""
import argparse
import json
import re
from pathlib import Path

import numpy as np

LABELS = {
    "small-linear-long": "Baseline, linear ghosts",
    "small-quadratic-long": "Baseline, quadratic ghosts",
    "small-damped-long": "Constraint-radiation candidate",
    "outer-linear-long": "Baseline",
    "outer-damped-long": "Constraint radiation, linear ghosts",
    "outer-damped-quadratic": "Constraint radiation, quadratic ghosts",
    "outer-damped-cubic": "Constraint radiation, cubic ghosts",
    "outer-sponge-002-w128": "Sponge: 128M, 0.02/M",
    "outer-sponge-01-w128": "Sponge: 128M, 0.1/M",
    "outer-sponge-002-w256": "Sponge: 256M, 0.02/M",
    "outer-ghost-baseline": "8 blocks, original stencil",
    "outer-stencil-baseline": "8 blocks, repaired stencil",
    "outer-stencil-sponge": "8 blocks, repair + 256M sponge",
}

def history(path):
    lines = path.read_text().splitlines()
    names = re.findall(r"\[\d+\]=([^\s]+)", lines[1])
    values = np.atleast_2d(np.loadtxt(path))
    return {name: values[:, i] for i, name in enumerate(names)}


def summarize(path):
    log = (path / "run.log").read_text(errors="replace")
    h = history(path / "ks_background.user.hst")
    result = {"case": path.name, "last_history_M": float(h["time"][-1]),
              "last_theta_max": float(h["Theta-max"][-1]),
              "last_lapse_residual_max": float(h["alpha-res"][-1]),
              "last_shift_residual_max": float(h["beta-res"][-1]),
              "last_history_bad_metric": float(h["bad-metric"][-1]),
              "history_finite": all(np.isfinite(x).all() for x in h.values())}
    invalid = re.search(r"Z4C_INVALID_STATE ([^\n]+)", log)
    if invalid:
        result["fatal_invalid_state"] = {
            k: float(v) for k, v in re.findall(r"(\w+)=([-+.\deE]+)", invalid[1])}
    error = re.search(r"An error occurred during the primitive solve: ([^\n]+)", log)
    if error:
        times = re.findall(r"\btime=([-+.\deE]+)", log[:error.start()])
        section = log[error.end():error.end()+3000]
        coords = re.search(r"Location: \(([^)]+)\)\s*\(([^)]+)\)", section)
        det = re.search(r"detg = ([-+.\deE]+)", section)
        result["first_primitive_error"] = {
            "kind": error[1],
            "after_last_logged_time_M": float(times[-1]) if times else None,
            "local_mkji": [int(x) for x in coords[1].split(",")] if coords else None,
            "xyz_M": [float(x) for x in coords[2].split(",")] if coords else None,
            "detg": float(det[1]) if det else None,
            "note": "The preceding logged time is a lower bound, not the precise error time."}
    if invalid:
        result["outcome"] = "invalid_state_failure"
    elif "Terminating on time limit" in log:
        result["outcome"] = "time_target"
    elif "Terminating on cycle limit" in log:
        result["outcome"] = "cycle_limit"
    elif "walltime" in log.lower() or "wall time limit" in log.lower():
        result["outcome"] = "inspect_walltime_stop"
    else:
        result["outcome"] = "incomplete_or_unclassified"
    windows = ([(500, 1000), (1000, 1500), (1500, 2000), (2000, 3000), (3000, 4000), (4000, 5000)] if "outer" in path.name
               else [(200, 400), (400, 600), (600, 700)])
    result["theta_growth_windows"] = []
    for lo, hi in windows:
        mask = ((h["time"] >= lo) & (h["time"] <= hi) &
                np.isfinite(h["Theta-max"]) & (h["Theta-max"] > 0))
        if np.count_nonzero(mask) < 5:
            continue
        t, z = h["time"][mask], np.log(h["Theta-max"][mask])
        slope, intercept = np.polyfit(t, z, 1)
        rss = np.sum((z - slope*t - intercept)**2)
        tss = np.sum((z - np.mean(z))**2)
        result["theta_growth_windows"].append({
            "time_window_M": [float(t[0]), float(t[-1])],
            "gamma_per_M": float(slope), "log_fit_R2": float(1-rss/tss) if tss else None})
    return result, h


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    records = [summarize(p) for p in args.runs]
    (args.output / "results.json").write_text(
        json.dumps([r for r, _ in records], indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    for result, h in records:
        ax = axes[int("outer" in result["case"])]
        positive = (h["Theta-max"] > 0) & np.isfinite(h["Theta-max"])
        line, = ax.semilogy(h["time"][positive], h["Theta-max"][positive],
                           label=LABELS.get(result["case"], result["case"]))
        if "first_primitive_error" in result:
            ax.axvline(result["first_primitive_error"]["after_last_logged_time_M"],
                       color=line.get_color(), linestyle=":", alpha=.6)
    for ax, label in zip(axes, ["Centered trumpet", "Weak-field patch"]):
        ax.set(xlabel=r"$t/M$", ylabel=r"$\max |\Theta|$", title=label)
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.savefig(args.output / "constraint-comparison.png", dpi=180)
    # Constraint norms are separate from pointwise maxima and from gauge growth.
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), layout="constrained")
    for path, (result, h) in zip(args.runs, records):
        if "outer" not in result["case"]:
            continue
        con = history(path / "ks_background.z4c.user.hst")
        theta_key = next(k for k in con if k.startswith("Theta-norm"))
        series = [(con["time"], np.sqrt(np.maximum(con[theta_key], 0)/con["Volume"])),
                  (con["time"], np.sqrt(np.maximum(con["H-norm2"], 0)/con["Volume"])),
                  (h["time"], h["alpha-res"]), (h["time"], h["beta-res"])]
        for ax, (t, value) in zip(axes.flat, series):
            ok = (value > 0) & np.isfinite(value)
            ax.semilogy(t[ok], value[ok], label=LABELS.get(result["case"], result["case"]))
    for ax, label in zip(axes.flat, [r"$\sqrt{\langle\Theta^2\rangle}$",
            r"$\sqrt{\langle H^2\rangle}$", r"$\max|\delta\alpha|$",
            r"$\max|\delta\beta^i|$"]):
        ax.set(xlabel=r"$t/M$", ylabel=label)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=6)
    fig.savefig(args.output / "constraint-gauge-norms.png", dpi=180)
    print(json.dumps([{k: v for k, v in r.items() if k != "theta_growth_windows"}
                      for r, _ in records], indent=2))


if __name__ == "__main__":
    main()
