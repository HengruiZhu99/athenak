#!/usr/bin/env python3
"""Plot matched Brill amplitude histories with slice-global extrema."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


HEADER = re.compile(r"\[(\d+)\]=([^\s]+)")
REQUIRED = {"time", "axisTau", "axisKret", "axisLapse", "maxAbsKret", "minLapse"}


def read_history(path: Path) -> dict[str, np.ndarray]:
    labels: dict[str, int] = {}
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            labels.update({name: int(index) - 1 for index, name in HEADER.findall(line)})
        elif line.strip():
            row = [float(value) for value in line.split()]
            if not labels or len(row) <= max(labels.values()):
                raise RuntimeError(f"malformed history row in {path}")
            rows.append(row)
    missing = REQUIRED - labels.keys()
    if missing:
        raise RuntimeError(f"{path} is missing {sorted(missing)}")
    if not rows:
        raise RuntimeError(f"empty history: {path}")
    array = np.asarray(rows, dtype=float)
    if not np.all(np.isfinite(array)):
        raise RuntimeError(f"nonfinite history values in {path}")
    if not np.all(np.diff(array[:, labels["time"]]) > 0.0):
        raise RuntimeError(f"nonmonotone coordinate time in {path}")
    data = {name: array[:, index] for name, index in labels.items()}
    tolerance = 64.0 * np.finfo(float).eps
    if np.any(data["minLapse"] > data["axisLapse"] + tolerance):
        raise RuntimeError(f"slice minimum exceeds axis lapse in {path}")
    if np.any(data["maxAbsKret"] + tolerance < np.abs(data["axisKret"])):
        raise RuntimeError(f"slice maximum is below origin magnitude in {path}")
    return data


def read_reference(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    series: dict[str, tuple[list[float], list[float]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            xx, yy = series.setdefault(row["series"], ([], []))
            xx.append(float(row["tau"]))
            yy.append(float(row["log10_abs_I"]))
    return {name: (np.asarray(xx), np.asarray(yy))
            for name, (xx, yy) in series.items()}


def save(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", dpi=240)
    fig.savefig(output / f"{stem}.pdf")
    plt.close(fig)


def positive_log10(values: np.ndarray) -> np.ndarray:
    output = np.full_like(values, np.nan)
    valid = np.isfinite(values) & (np.abs(values) > 0.0)
    output[valid] = np.log10(np.abs(values[valid]))
    return output


def summarize(data: dict[str, np.ndarray]) -> dict[str, float | int]:
    axis_abs = np.abs(data["axisKret"])
    axis_peak = int(np.nanargmax(axis_abs))
    slice_peak = int(np.nanargmax(data["maxAbsKret"]))
    lapse_min = int(np.nanargmin(data["minLapse"]))
    return {
        "rows": int(len(data["time"])),
        "final_coordinate_time": float(data["time"][-1]),
        "final_central_proper_time": float(data["axisTau"][-1]),
        "peak_origin_abs_kretschmann": float(axis_abs[axis_peak]),
        "peak_origin_abs_kretschmann_coordinate_time": float(data["time"][axis_peak]),
        "peak_origin_abs_kretschmann_central_proper_time": float(data["axisTau"][axis_peak]),
        "peak_slice_abs_kretschmann": float(data["maxAbsKret"][slice_peak]),
        "peak_slice_abs_kretschmann_coordinate_time": float(data["time"][slice_peak]),
        "minimum_slice_lapse": float(data["minLapse"][lapse_min]),
        "minimum_slice_lapse_coordinate_time": float(data["time"][lapse_min]),
        "minimum_axis_lapse": float(np.min(data["axisLapse"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subcritical", type=Path, required=True)
    parser.add_argument("--supercritical", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    subcritical = read_history(args.subcritical)
    supercritical = read_history(args.supercritical)
    cases = {
        f"A=-0.047 native AMR (partial, t<={subcritical['time'][-1]:.2f}M)":
            subcritical,
        f"A=-0.050 native AMR (partial, t<={supercritical['time'][-1]:.2f}M)":
            supercritical,
    }
    reference = read_reference(args.reference)
    colors = {name: color for name, color in
              zip(cases, ("#0072B2", "#D55E00"), strict=True)}

    fig, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for name, (tau, value) in reference.items():
        axis.plot(tau, value, linewidth=1.1, alpha=0.58,
                  label=f"published {name}, A=-0.047")
    for name, data in cases.items():
        axis.plot(data["axisTau"], positive_log10(data["axisKret"]),
                  color=colors[name], linewidth=1.8, label=f"AthenaK {name}")
    axis.set(xlabel=r"central proper time $\tau_0/M$",
             ylabel=r"$\log_{10}|R_{abcd}R^{abcd}|_{\rho=z=0}$",
             title="Brill amplitudes: origin curvature (Figure-3 style)")
    axis.grid(alpha=0.22)
    axis.legend(fontsize=8, ncol=2)
    save(fig, args.output, "central_kretschmann_figure3_style")

    fig, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for name, data in cases.items():
        axis.plot(data["time"], positive_log10(data["maxAbsKret"]),
                  color=colors[name], linewidth=1.8, label=name)
    axis.set(xlabel=r"coordinate time $t/M$",
             ylabel=r"$\log_{10}\max_{\mathrm{slice}}|R_{abcd}R^{abcd}|$",
             title="Slice-global maximum Kretschmann scalar")
    axis.grid(alpha=0.22)
    axis.legend()
    save(fig, args.output, "slice_max_kretschmann_coordinate_time")

    fig, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for name, data in cases.items():
        axis.plot(data["time"], data["minLapse"], color=colors[name],
                  linewidth=1.8, label=name)
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.45)
    axis.set(xlabel=r"coordinate time $t/M$",
             ylabel=r"$\min_{\mathrm{slice}}\alpha$",
             title="Slice-global minimum lapse")
    axis.grid(alpha=0.22)
    axis.legend()
    save(fig, args.output, "slice_min_lapse_coordinate_time")

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 13.0), constrained_layout=True)
    for name, (tau, value) in reference.items():
        axes[0].plot(tau, value, linewidth=1.0, alpha=0.45,
                     label=f"published {name}, A=-0.047")
    for name, data in cases.items():
        axes[0].plot(data["axisTau"], positive_log10(data["axisKret"]),
                     color=colors[name], linewidth=1.7, label=name)
        axes[1].plot(data["time"], positive_log10(data["maxAbsKret"]),
                     color=colors[name], linewidth=1.7, label=name)
        axes[2].plot(data["time"], data["minLapse"], color=colors[name],
                     linewidth=1.7, label=name)
    axes[0].set(xlabel=r"central proper time $\tau_0/M$",
                ylabel=r"$\log_{10}|R^2|_0$")
    axes[1].set(xlabel=r"coordinate time $t/M$",
                ylabel=r"$\log_{10}\max_{\mathrm{slice}}|R^2|$")
    axes[2].set(xlabel=r"coordinate time $t/M$",
                ylabel=r"$\min_{\mathrm{slice}}\alpha$")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8, ncol=2)
    save(fig, args.output, "history_extrema_three_panel")

    summary = {
        "schema": "athenak.z4c.history_extrema_amplitude_comparison.v1",
        "cases": {name: summarize(data) for name, data in cases.items()},
        "diagnostic_definitions": {
            "axisKret": "origin Kretschmann scalar at rho=z=0",
            "maxAbsKret": "slice-global maximum absolute Kretschmann scalar",
            "minLapse": "slice-global minimum lapse over active evolution points",
        },
        "comparison_limitations": [
            "Both plotted AthenaK curves use native N512 dynamic AMR.",
            "The A=-0.047 segment is partial: it stopped at t=24.52459M on "
            "the fail-closed max_nmb_per_rank=256 gate, not a physical endpoint.",
            "The A=-0.047 plotted segment did not run the horizon finder; the "
            "matched retry produced only its t=0 history row before an I/O stall.",
            "The A=-0.050 segment is partial: its continuation ended at "
            "t=36.45853M on the internal wall-clock limit.",
            "Published Figure-3 reference curves correspond to A=-0.047 only.",
            "Curvature is plotted in raw code units and is not ADM-mass normalized.",
        ],
    }
    (args.output / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
