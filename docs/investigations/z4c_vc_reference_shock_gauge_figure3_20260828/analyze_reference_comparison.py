#!/usr/bin/env python3
"""Direct N256/N512 reference-gauge comparison with no curve transform."""

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
CONSTRAINTS = ("C-norm2", "H-norm2", "M-norm2", "Z-norm2")
REQUIRED = {
    "time", "dt", "cycle", "axisTau", "axisKret", "axisLapse",
    "max_abs_K", "maxAbsKret", "maxRefLev", "nmb_total",
    "C-Linf", "C-rho", "C-z", *CONSTRAINTS,
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def parse_history(path: Path) -> tuple[dict[str, int], list[list[float]]]:
    labels: dict[str, int] = {}
    rows: list[list[float]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            labels.update({name: int(index) - 1 for index, name in HEADER.findall(line)})
        elif line.strip():
            values = [float(value) for value in line.split()]
            require(labels and len(values) > max(labels.values()),
                    f"malformed history row in {path}")
            require(all(math.isfinite(value) for value in values),
                    f"nonfinite history row in {path}")
            rows.append(values)
    require(not REQUIRED - labels.keys(),
            f"{path} missing {sorted(REQUIRED - labels.keys())}")
    require(bool(rows), f"empty history: {path}")
    return labels, rows


def merge_histories(paths: list[Path]) -> dict[str, np.ndarray]:
    require(bool(paths), "at least one history is required")
    labels0: dict[str, int] | None = None
    by_cycle: dict[int, list[float]] = {}
    for path in paths:
        require(path.is_file(), f"missing history: {path}")
        labels, rows = parse_history(path)
        if labels0 is None:
            labels0 = labels
        require(labels == labels0, f"history schema changed in {path}")
        for row in rows:
            cycle = int(row[labels["cycle"]])
            if cycle in by_cycle:
                old = by_cycle[cycle]
                differing = [name for name, index in labels.items()
                             if row[index] != old[index]]
                require(set(differing) <= {"dt"},
                        f"restart overlap mismatch at cycle {cycle} in {path}: "
                        f"{differing}")
                by_cycle[cycle] = row
            else:
                by_cycle[cycle] = row
    labels = labels0 or {}
    array = np.asarray([by_cycle[key] for key in sorted(by_cycle)], dtype=float)
    require(len(array) > 1, "too few merged history rows")
    require(np.all(np.diff(array[:, labels["time"]]) > 0.0), "time is nonmonotone")
    require(np.all(np.diff(array[:, labels["axisTau"]]) >= 0.0),
            "central proper time is nonmonotone")
    return {name: array[:, index] for name, index in labels.items()}


def read_reference(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    values: dict[str, tuple[list[float], list[float]]] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            xx, yy = values.setdefault(row["series"], ([], []))
            xx.append(float(row["tau"]))
            yy.append(float(row["log10_abs_I"]))
    require(set(values) == {"bamps", "prague", "sphGR"}, "reference series changed")
    return {name: (np.asarray(xx), np.asarray(yy))
            for name, (xx, yy) in values.items()}


def milestone(x: np.ndarray, y: np.ndarray, lo: float, hi: float,
              mode: str) -> dict[str, float] | None:
    mask = (x >= lo) & (x <= hi) & np.isfinite(y)
    if not np.any(mask):
        return None
    indices = np.flatnonzero(mask)
    selected = indices[np.argmax(y[mask]) if mode == "max" else np.argmin(y[mask])]
    return {"tau": float(x[selected]), "log10_abs_I": float(y[selected])}


def first_threshold(data: dict[str, np.ndarray], field: str,
                    threshold: float) -> dict[str, float] | None:
    indices = np.flatnonzero(data[field] >= threshold)
    if not len(indices):
        return None
    i = int(indices[0])
    return {"coordinate_time": float(data["time"][i]),
            "central_proper_time": float(data["axisTau"][i]),
            "value": float(data[field][i])}


def curve(data: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(data["axisKret"]) & (np.abs(data["axisKret"]) > 0.0)
    return data["axisTau"][valid], np.log10(np.abs(data["axisKret"][valid]))


def direct_rmse(x: np.ndarray, y: np.ndarray, rx: np.ndarray,
                ry: np.ndarray) -> float | None:
    lo, hi = max(float(x[0]), float(rx[0])), min(float(x[-1]), float(rx[-1]))
    sample = rx[(rx >= lo) & (rx <= hi)]
    if len(sample) < 3:
        return None
    residual = np.interp(sample, x, y) - np.interp(sample, rx, ry)
    return float(np.sqrt(np.mean(np.square(residual))))


def summarize(data: dict[str, np.ndarray],
              reference: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    tau, log_kret = curve(data)
    summary: dict[str, object] = {
        "rows": int(len(data["time"])),
        "coordinate_time_final": float(data["time"][-1]),
        "central_proper_time_final": float(data["axisTau"][-1]),
        "axis_lapse_min": float(np.min(data["axisLapse"])),
        "max_refinement_level": int(np.max(data["maxRefLev"])),
        "max_meshblocks": int(np.max(data["nmb_total"])),
        "first_peak": milestone(tau, log_kret, 9.0, 11.5, "max"),
        "deep_minimum_through_13_3": milestone(tau, log_kret, 11.5, 13.3, "min"),
        "rebound_through_13_3": milestone(tau, log_kret, 12.5, 13.3, "max"),
        "direct_unshifted_rmse": {
            name: direct_rmse(tau, log_kret, xx, yy)
            for name, (xx, yy) in reference.items()
        },
        "constraint_thresholds": {
            field: {str(value): first_threshold(data, field, value)
                    for value in (1.0e-2, 1.0e-1, 1.0, 10.0)}
            for field in CONSTRAINTS
        },
        "constraints": {},
    }
    for field in CONSTRAINTS:
        i = int(np.argmax(data[field]))
        summary["constraints"][field] = {
            "initial": float(data[field][0]),
            "final": float(data[field][-1]),
            "maximum": float(data[field][i]),
            "maximum_coordinate_time": float(data["time"][i]),
            "maximum_central_proper_time": float(data["axisTau"][i]),
        }
    return summary


def matched_comparison(coarse: dict[str, np.ndarray],
                       fine: dict[str, np.ndarray]) -> dict:
    tau_max = min(float(coarse["axisTau"][-1]), float(fine["axisTau"][-1]))
    sample = coarse["axisTau"][coarse["axisTau"] <= tau_max]
    require(len(sample) > 2, "empty common proper-time interval")
    output: dict[str, object] = {"common_tau_max": tau_max, "constraints": {}}
    for field in CONSTRAINTS:
        low = np.interp(sample, coarse["axisTau"], coarse[field])
        high = np.interp(sample, fine["axisTau"], fine[field])
        ratio = high / np.maximum(low, np.finfo(float).tiny)
        windows: dict[str, object] = {}
        for lo, hi in ((0.0, 8.0), (8.0, 10.0), (10.0, 11.5), (11.5, 13.3)):
            mask = (sample >= lo) & (sample <= hi)
            windows[f"{lo:g}_{hi:g}"] = None if not np.any(mask) else {
                "median_n512_over_n256": float(np.median(ratio[mask])),
                "maximum_n256": float(np.max(low[mask])),
                "maximum_n512": float(np.max(high[mask])),
            }
        output["constraints"][field] = windows
    return output


def write_compact(path: Path, data: dict[str, np.ndarray]) -> None:
    columns = ("time", "axisTau", "dt", "cycle", "axisKret", "axisLapse",
               "max_abs_K", "maxAbsKret", "maxRefLev", "nmb_total",
               "C-Linf", "C-rho", "C-z", *CONSTRAINTS)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(zip(*(data[name] for name in columns)))


def save(fig: plt.Figure, root: Path, name: str) -> None:
    fig.savefig(root / f"{name}.png", dpi=230)
    fig.savefig(root / f"{name}.pdf")
    plt.close(fig)


def plots(data: dict[str, dict[str, np.ndarray]],
          reference: dict[str, tuple[np.ndarray, np.ndarray]], root: Path) -> None:
    colors = {"n256": "#377eb8", "n512": "#e41a1c"}
    fig, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for name, (xx, yy) in reference.items():
        axis.plot(xx, yy, linewidth=1.2, alpha=0.62, label=f"published {name}")
    for case, values in data.items():
        xx, yy = curve(values)
        axis.plot(xx, yy, color=colors[case], linewidth=1.7, label=case.upper())
    axis.set(xlim=(0.0, 13.5), ylim=(-8.0, 8.0), xlabel="central proper time",
             ylabel=r"$\log_{10}|\mathrm{Kretschmann}(0)|$")
    axis.grid(alpha=0.22); axis.legend(fontsize=8, ncol=2)
    save(fig, root, "figure3_n256_n512")

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True,
                             constrained_layout=True)
    for axis, field in zip(axes.flat, CONSTRAINTS):
        for case, values in data.items():
            axis.semilogy(values["axisTau"], np.maximum(values[field], 1.0e-14),
                          color=colors[case], label=case.upper())
        axis.set_title(field); axis.grid(alpha=0.22, which="both")
    axes[1, 0].set_xlabel("central proper time")
    axes[1, 1].set_xlabel("central proper time")
    axes[0, 0].legend()
    save(fig, root, "constraints_n256_n512")

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.5), sharex=True,
                             constrained_layout=True)
    for case, values in data.items():
        radius = np.hypot(values["C-rho"], values["C-z"])
        axes[0].semilogy(values["axisTau"], np.maximum(values["C-norm2"], 1.0e-14),
                         color=colors[case], label=case.upper())
        axes[1].plot(values["axisTau"], radius, color=colors[case], label=case.upper())
    axes[1].axhspan(4.0, 6.0, color="gray", alpha=0.15, label=r"$4<\rho<6$")
    axes[0].set_ylabel("C squared integral")
    axes[1].set(xlabel="central proper time", ylabel="radius of C-Linf")
    for axis in axes: axis.grid(alpha=0.22, which="both"); axis.legend()
    save(fig, root, "constraint_location_rho5")

    fig, axes = plt.subplots(3, 2, figsize=(12.0, 9.0), sharex=True,
                             constrained_layout=True)
    panels = (("axisLapse", "axis lapse", False), ("dt", "dt", True),
              ("max_abs_K", "max |K|", True),
              ("maxAbsKret", "max |Kretschmann|", True),
              ("maxRefLev", "max refinement level", False),
              ("nmb_total", "MeshBlocks", False))
    for axis, (field, label, logarithmic) in zip(axes.flat, panels):
        for case, values in data.items():
            plot = axis.semilogy if logarithmic else axis.plot
            plot(values["axisTau"], values[field], color=colors[case], label=case.upper())
        axis.set_ylabel(label); axis.grid(alpha=0.22, which="both")
    axes[0, 0].legend()
    axes[-1, 0].set_xlabel("central proper time")
    axes[-1, 1].set_xlabel("central proper time")
    save(fig, root, "lapse_timestep_curvature_topology")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n256-history", type=Path, nargs="+", required=True)
    parser.add_argument("--n512-history", type=Path, nargs="+", required=True)
    parser.add_argument("--reference", type=Path,
                        default=Path(__file__).with_name("figure3_published_curves.csv"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    reference = read_reference(args.reference)
    data = {"n256": merge_histories(args.n256_history),
            "n512": merge_histories(args.n512_history)}
    write_compact(args.output / "n256_history_compact.csv", data["n256"])
    write_compact(args.output / "n512_history_compact.csv", data["n512"])
    summary = {
        "schema": "athenak.z4c.reference_shock_n256_n512.v1",
        "curve_transform": "none",
        "cases": {case: summarize(values, reference) for case, values in data.items()},
        "matched_comparison": matched_comparison(data["n256"], data["n512"]),
        "claim_boundary": "two-resolution discriminator; no convergence order",
    }
    (args.output / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    plots(data, reference, args.output)


if __name__ == "__main__":
    main()
