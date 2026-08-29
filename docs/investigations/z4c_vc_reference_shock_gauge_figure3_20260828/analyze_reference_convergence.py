#!/usr/bin/env python3
"""Three-resolution, same-tree comparison for the reference-gauge campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from analyze_reference_comparison import merge_histories, read_reference, summarize


CONSTRAINTS = ("C-norm2", "H-norm2", "M-norm2", "Z-norm2")
WINDOWS = ((0.0, 8.0), (8.0, 10.0), (10.0, 11.5), (11.5, 13.3))


def finite_median(values: np.ndarray) -> float | None:
    selected = values[np.isfinite(values)]
    return None if not len(selected) else float(np.median(selected))


def window_summary(tau: np.ndarray, values: np.ndarray) -> dict[str, float | None]:
    output = {}
    for lo, hi in WINDOWS:
        mask = (tau >= lo) & (tau <= hi)
        output[f"{lo:g}_{hi:g}"] = finite_median(values[mask])
    return output


def save(fig: plt.Figure, root: Path, name: str) -> None:
    fig.savefig(root / f"{name}.png", dpi=230)
    fig.savefig(root / f"{name}.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    for case in ("n128", "n256", "n512"):
        parser.add_argument(f"--{case}-history", type=Path, nargs="+", required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = {case: merge_histories(getattr(args, f"{case}_history"))
            for case in ("n128", "n256", "n512")}
    reference = read_reference(args.reference)
    common_tau_max = min(float(values["axisTau"][-1]) for values in data.values())
    tau = data["n256"]["axisTau"]
    tau = tau[tau <= common_tau_max]
    aligned = {
        case: {field: np.interp(tau, values["axisTau"], values[field])
               for field in ("axisKret", "axisLapse", *CONSTRAINTS)}
        for case, values in data.items()
    }

    rows = []
    eps = np.finfo(float).tiny
    raw_diffs = {}
    for field in ("axisKret", "axisLapse"):
        d_low = np.abs(aligned["n128"][field] - aligned["n256"][field])
        d_high = np.abs(aligned["n256"][field] - aligned["n512"][field])
        order = np.log2(np.maximum(d_low, eps) / np.maximum(d_high, eps))
        raw_diffs[field] = {"n128_n256": d_low, "n256_n512": d_high, "order": order}
    constraint_orders = {}
    for field in CONSTRAINTS:
        amplitude = {case: np.sqrt(np.maximum(aligned[case][field], 0.0))
                     for case in data}
        p_low = np.log2(np.maximum(amplitude["n128"], eps) /
                        np.maximum(amplitude["n256"], eps))
        p_high = np.log2(np.maximum(amplitude["n256"], eps) /
                         np.maximum(amplitude["n512"], eps))
        constraint_orders[field] = {"n128_n256": p_low, "n256_n512": p_high}
    for index, value in enumerate(tau):
        row = {"axisTau": value}
        for case in data:
            row[f"{case}_axisKret"] = aligned[case]["axisKret"][index]
            row[f"{case}_axisLapse"] = aligned[case]["axisLapse"][index]
            for field in CONSTRAINTS:
                row[f"{case}_{field}_amplitude"] = math.sqrt(
                    max(aligned[case][field][index], 0.0))
        for field in raw_diffs:
            row[f"{field}_richardson_order"] = raw_diffs[field]["order"][index]
        for field in CONSTRAINTS:
            row[f"{field}_order_n128_n256"] = constraint_orders[field]["n128_n256"][index]
            row[f"{field}_order_n256_n512"] = constraint_orders[field]["n256_n512"][index]
        rows.append(row)
    with (args.output / "three_resolution_aligned.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)

    summary = {
        "schema": "athenak.z4c.reference_same_tree_convergence.v1",
        "common_tau_max": common_tau_max,
        "final_tau": {case: float(values["axisTau"][-1]) for case, values in data.items()},
        "curve_richardson_order_medians": {
            field: window_summary(tau, values["order"])
            for field, values in raw_diffs.items()
        },
        "constraint_amplitude_pair_order_medians": {
            field: {pair: window_summary(tau, order)
                    for pair, order in pairs.items()}
            for field, pairs in constraint_orders.items()
        },
        "constraint_amplitude_maxima": {
            case: {field: float(np.sqrt(np.max(np.maximum(values[field], 0.0))))
                   for field in CONSTRAINTS}
            for case, values in data.items()
        },
        "case_observations": {
            case: summarize(values, reference) for case, values in data.items()
        },
        "claim_boundary": (
            "same physical AMR tree and 2:1 cell-spacing sequence; observed orders are "
            "descriptive and support convergence only when consistent across both pairs and windows"
        ),
    }
    (args.output / "three_resolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    colors = {"n128": "#4daf4a", "n256": "#377eb8", "n512": "#e41a1c"}
    fig, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for name, (xx, yy) in reference.items():
        axis.plot(xx, yy, linewidth=1.15, alpha=0.55, label=f"published {name}")
    for case, values in data.items():
        valid = np.abs(values["axisKret"]) > 0.0
        axis.plot(values["axisTau"][valid], np.log10(np.abs(values["axisKret"][valid])),
                  color=colors[case], linewidth=1.55, label=case.upper())
    axis.set(xlim=(0.0, 13.5), ylim=(-8.0, 8.0), xlabel="central proper time",
             ylabel=r"$\log_{10}|\mathrm{Kretschmann}(0)|$")
    axis.grid(alpha=0.22); axis.legend(fontsize=8, ncol=2)
    save(fig, args.output, "figure3_n128_n256_n512")

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True,
                             constrained_layout=True)
    for axis, field in zip(axes.flat, CONSTRAINTS):
        for case, values in data.items():
            axis.semilogy(values["axisTau"], np.sqrt(np.maximum(values[field], 1.0e-30)),
                          color=colors[case], label=case.upper())
        axis.set_title(field.replace("-norm2", "")); axis.grid(alpha=0.22, which="both")
    axes[1, 0].set_xlabel("central proper time"); axes[1, 1].set_xlabel("central proper time")
    axes[0, 0].set_ylabel(r"$\sqrt{\int C^2 dV}$")
    axes[1, 0].set_ylabel(r"$\sqrt{\int C^2 dV}$")
    axes[0, 0].legend()
    save(fig, args.output, "constraint_amplitudes_n128_n256_n512")

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True,
                             constrained_layout=True)
    for axis, field in zip(axes.flat, CONSTRAINTS):
        axis.plot(tau, constraint_orders[field]["n128_n256"], label="N128/N256")
        axis.plot(tau, constraint_orders[field]["n256_n512"], label="N256/N512")
        axis.axhline(4.0, color="black", linestyle=":", linewidth=1.0)
        axis.set(title=field.replace("-norm2", ""), ylim=(-4.0, 8.0))
        axis.grid(alpha=0.22)
    axes[1, 0].set_xlabel("central proper time"); axes[1, 1].set_xlabel("central proper time")
    axes[0, 0].set_ylabel("pairwise amplitude order")
    axes[1, 0].set_ylabel("pairwise amplitude order")
    axes[0, 0].legend()
    save(fig, args.output, "constraint_pairwise_orders")


if __name__ == "__main__":
    main()
