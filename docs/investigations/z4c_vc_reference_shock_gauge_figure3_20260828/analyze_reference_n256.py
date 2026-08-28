#!/usr/bin/env python3
"""Reduce the fresh N256 reference-gauge trajectory without curve alignment."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HEADER = re.compile(r"\[(\d+)\]=([^\s]+)")
CONSTRAINTS = ("C-norm2", "H-norm2", "M-norm2", "Z-norm2")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_history(paths: list[Path]) -> dict[str, np.ndarray]:
    labels: dict[str, int] = {}
    rows: dict[float, list[float]] = {}
    for path in paths:
        require(path.is_file(), f"missing history: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                labels.update({name: int(index) - 1 for index, name in HEADER.findall(line)})
            elif line.strip():
                row = [float(value) for value in line.split()]
                require("time" in labels, f"history data precedes header in {path}")
                rows[row[labels["time"]]] = row
    required = {
        "time", "dt", "cycle", "axisTau", "axisKret", "axisLapse",
        "max_abs_K", "maxAbsKret", "maxRefLev", "nmb_total",
        "C-Linf", "C-rho", "C-z", *CONSTRAINTS,
    }
    require(not required - labels.keys(), f"missing columns: {sorted(required-labels.keys())}")
    array = np.asarray([rows[time] for time in sorted(rows)], dtype=float)
    require(array.ndim == 2 and len(array) > 1, "history has too few rows")
    require(np.all(np.diff(array[:, labels["time"]]) > 0.0), "time is nonmonotone")
    require(np.all(np.diff(array[:, labels["axisTau"]]) >= 0.0), "proper time is nonmonotone")
    return {name: array[:, index] for name, index in labels.items()}


def read_reference(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    values: dict[str, tuple[list[float], list[float]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            xx, yy = values.setdefault(row["series"], ([], []))
            xx.append(float(row["tau"]))
            yy.append(float(row["log10_abs_I"]))
    require(set(values) == {"bamps", "prague", "sphGR"}, "reference series changed")
    return {name: (np.asarray(xx), np.asarray(yy)) for name, (xx, yy) in values.items()}


def zero_crossings(x: np.ndarray, y: np.ndarray) -> list[float]:
    crossings: list[float] = []
    for i in range(1, len(y)):
        if y[i] == 0.0:
            crossings.append(float(x[i]))
        elif y[i - 1] * y[i] < 0.0:
            fraction = -y[i - 1] / (y[i] - y[i - 1])
            crossings.append(float(x[i - 1] + fraction * (x[i] - x[i - 1])))
    return crossings


def milestone(x: np.ndarray, y: np.ndarray, lo: float, hi: float,
              choose: str) -> dict[str, float] | None:
    mask = (x >= lo) & (x <= hi) & np.isfinite(y)
    if not np.any(mask):
        return None
    indices = np.flatnonzero(mask)
    selected = indices[np.argmax(y[mask]) if choose == "max" else np.argmin(y[mask])]
    return {"tau": float(x[selected]), "log10_abs_I": float(y[selected])}


def direct_rmse(x: np.ndarray, y: np.ndarray, reference_x: np.ndarray,
                reference_y: np.ndarray) -> float | None:
    lo = max(float(x[0]), float(reference_x[0]))
    hi = min(float(x[-1]), float(reference_x[-1]))
    if hi <= lo:
        return None
    sample = reference_x[(reference_x >= lo) & (reference_x <= hi)]
    if len(sample) < 3:
        return None
    residual = np.interp(sample, x, y) - np.interp(sample, reference_x, reference_y)
    return float(np.sqrt(np.mean(residual * residual)))


def write_compact(path: Path, data: dict[str, np.ndarray]) -> None:
    columns = (
        "time", "axisTau", "dt", "cycle", "axisKret", "axisLapse",
        "max_abs_K", "maxAbsKret", "maxRefLev", "nmb_total", "C-Linf",
        "C-rho", "C-z", *CONSTRAINTS,
    )
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(zip(*(data[name] for name in columns)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, nargs="+", required=True)
    parser.add_argument("--reference", type=Path,
                        default=Path(__file__).with_name("figure3_published_curves.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    data = read_history(args.history)
    reference = read_reference(args.reference)
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    figures.mkdir(exist_ok=True)
    write_compact(args.output / "n256_history_compact.csv", data)

    valid_kret = np.isfinite(data["axisKret"]) & (np.abs(data["axisKret"]) > 0.0)
    tau = data["axisTau"][valid_kret]
    log_kret = np.log10(np.abs(data["axisKret"][valid_kret]))
    require(len(tau) > 1, "no finite nonzero origin Kretschmann history")

    milestones: dict[str, dict] = {}
    for name, (xx, yy) in reference.items():
        milestones[name] = {
            "first_peak": milestone(xx, yy, 9.0, 11.5, "max"),
            "deep_minimum": milestone(xx, yy, 11.5, 13.5, "min"),
            "rebound": milestone(xx, yy, 12.5, 14.5, "max"),
        }
    milestones["athenak_n256"] = {
        "first_peak": milestone(tau, log_kret, 9.0, 11.5, "max"),
        "deep_minimum": milestone(tau, log_kret, 11.5, 13.5, "min"),
        "rebound": milestone(tau, log_kret, 12.5, 14.5, "max"),
    }

    crossing_tau = zero_crossings(data["axisTau"], data["axisLapse"])
    summary = {
        "coordinate_time_final": float(data["time"][-1]),
        "axis_proper_time_final": float(data["axisTau"][-1]),
        "rows": int(len(data["time"])),
        "all_history_values_finite": bool(all(np.isfinite(values).all()
                                                  for values in data.values())),
        "axis_lapse_min": float(np.nanmin(data["axisLapse"])),
        "axis_lapse_max": float(np.nanmax(data["axisLapse"])),
        "axis_lapse_zero_crossings_tau": crossing_tau,
        "max_refinement_level": int(np.nanmax(data["maxRefLev"])),
        "max_meshblocks": int(np.nanmax(data["nmb_total"])),
        "max_abs_kretschmann": float(np.nanmax(data["maxAbsKret"])),
        "final_C_Linf": float(data["C-Linf"][-1]),
        "figure3_milestones": milestones,
        "direct_unshifted_log_curve_rmse": {
            name: direct_rmse(tau, log_kret, xx, yy)
            for name, (xx, yy) in reference.items()
        },
        "reached_reference_windows": {
            "first_peak": bool(tau[-1] >= 11.5),
            "deep_minimum": bool(tau[-1] >= 13.5),
            "rebound": bool(tau[-1] >= 14.5),
        },
        "curve_transform": "none",
    }
    (args.output / "n256_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    fig, axis = plt.subplots(figsize=(10.4, 6.2), constrained_layout=True)
    for name, (xx, yy) in reference.items():
        axis.plot(xx, yy, linewidth=1.3, alpha=0.72, label=f"published {name}")
    axis.plot(tau, log_kret, color="black", linewidth=1.7, label="AthenaK N256")
    axis.set(xlim=(0.0, 15.0), ylim=(-8.0, 8.0), xlabel="central proper time",
             ylabel=r"$\log_{10}|\mathrm{Kretschmann}(0)|$")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8, ncol=2)
    fig.savefig(figures / "figure3_reference_n256.png", dpi=240)
    fig.savefig(figures / "figure3_reference_n256.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10.0, 4.8), constrained_layout=True)
    axis.plot(data["axisTau"], data["axisLapse"], color="#984ea3")
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set(xlabel="central proper time", ylabel="lapse at origin")
    axis.grid(alpha=0.25)
    fig.savefig(figures / "axis_lapse_history.png", dpi=220)
    fig.savefig(figures / "axis_lapse_history.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True,
                             constrained_layout=True)
    for field in CONSTRAINTS:
        axes[0].semilogy(data["time"], np.maximum(data[field], 1.0e-300), label=field)
    radius = np.hypot(data["C-rho"], data["C-z"])
    axes[1].plot(data["time"], radius, color="#e41a1c", label="C-Linf radius")
    axes[1].axhline(5.0, color="black", linestyle="--", linewidth=0.8)
    axes[0].set_ylabel("squared inventory")
    axes[1].set(xlabel="coordinate time", ylabel="radius of C-Linf")
    axes[0].legend(fontsize=8, ncol=4)
    for axis in axes:
        axis.grid(alpha=0.23, which="both")
    fig.savefig(figures / "constraints_and_rho5_mode.png", dpi=220)
    fig.savefig(figures / "constraints_and_rho5_mode.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.0), sharex=True,
                             constrained_layout=True)
    axes[0].semilogy(data["time"], data["dt"])
    axes[1].step(data["time"], data["maxRefLev"], where="post")
    axes[2].step(data["time"], data["nmb_total"], where="post")
    axes[0].set_ylabel("dt")
    axes[1].set_ylabel("max level")
    axes[2].set(xlabel="coordinate time", ylabel="MeshBlocks")
    for axis in axes:
        axis.grid(alpha=0.23, which="both")
    fig.savefig(figures / "timestep_refinement_meshblocks.png", dpi=220)
    fig.savefig(figures / "timestep_refinement_meshblocks.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
