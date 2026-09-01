#!/usr/bin/env python3
"""Same-tree N128--N1024 comparison for the reference-gauge campaign."""

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
BASE_CASES = ("n128", "n256", "n512")


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


def interpolate_on_support(tau: np.ndarray, source_tau: np.ndarray,
                           source_values: np.ndarray) -> np.ndarray:
    """Interpolate without silently extending a partial trajectory."""
    result = np.interp(tau, source_tau, source_values)
    result[(tau < source_tau[0]) | (tau > source_tau[-1])] = np.nan
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for case in BASE_CASES:
        parser.add_argument(f"--{case}-history", type=Path, nargs="+", required=True)
    parser.add_argument("--n1024-history", type=Path, nargs="+")
    parser.add_argument("--n1024-cfl040-ko010-history", type=Path, nargs="+")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--constraint-radius", type=float)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    resolution_cases = BASE_CASES + (("n1024",) if args.n1024_history else ())
    plot_cases = resolution_cases + (("n1024_cfl040_ko010",)
                                     if args.n1024_cfl040_ko010_history else ())
    histories = {case: getattr(args, f"{case}_history") for case in BASE_CASES}
    if args.n1024_history:
        histories["n1024"] = args.n1024_history
    if args.n1024_cfl040_ko010_history:
        histories["n1024_cfl040_ko010"] = args.n1024_cfl040_ko010_history
    data = {case: merge_histories(histories[case]) for case in plot_cases}
    tag = "_".join(plot_cases)
    resolution_word = "four" if len(resolution_cases) == 4 else "three"
    reference = read_reference(args.reference)
    common_tau_max = min(float(data[case]["axisTau"][-1])
                         for case in resolution_cases)
    tau = data["n256"]["axisTau"]
    tau = tau[tau <= common_tau_max]
    aligned = {
        case: {field: interpolate_on_support(tau, values["axisTau"], values[field])
               for field in ("axisKret", "axisLapse", *CONSTRAINTS)}
        for case, values in data.items()
    }

    rows = []
    eps = np.finfo(float).tiny
    pair_diffs = {}
    richardson_orders = {}
    for field in ("axisKret", "axisLapse"):
        pair_diffs[field] = {
            f"{low}_{high}": np.abs(aligned[low][field] - aligned[high][field])
            for low, high in zip(resolution_cases[:-1], resolution_cases[1:])
        }
        richardson_orders[field] = {}
        for low, middle, high in zip(resolution_cases[:-2],
                                     resolution_cases[1:-1],
                                     resolution_cases[2:]):
            d_low = pair_diffs[field][f"{low}_{middle}"]
            d_high = pair_diffs[field][f"{middle}_{high}"]
            richardson_orders[field][f"{low}_{middle}_{high}"] = (
                np.log2(np.maximum(d_low, eps)) -
                np.log2(np.maximum(d_high, eps)))
    constraint_orders = {}
    for field in CONSTRAINTS:
        amplitude = {case: np.sqrt(np.maximum(aligned[case][field], 0.0))
                     for case in data}
        constraint_orders[field] = {
            f"{low}_{high}": (np.log2(np.maximum(amplitude[low], eps)) -
                               np.log2(np.maximum(amplitude[high], eps)))
            for low, high in zip(resolution_cases[:-1], resolution_cases[1:])
        }
    ablation = None
    if "n1024_cfl040_ko010" in data:
        baseline = "n1024"
        variant = "n1024_cfl040_ko010"
        ablation = {
            "constraint_amplitude_variant_over_baseline_medians": {
                field: window_summary(
                    tau,
                    np.sqrt(np.maximum(aligned[variant][field], 0.0)) /
                    np.maximum(np.sqrt(np.maximum(aligned[baseline][field], 0.0)), eps))
                for field in CONSTRAINTS
            },
            "constraint_squared_integral_maximum_ratio": {
                field: float(np.max(data[variant][field]) /
                             np.maximum(np.max(data[baseline][field]), eps))
                for field in CONSTRAINTS
            },
        }
    for index, value in enumerate(tau):
        row = {"axisTau": value}
        for case in data:
            row[f"{case}_axisKret"] = aligned[case]["axisKret"][index]
            row[f"{case}_axisLapse"] = aligned[case]["axisLapse"][index]
            for field in CONSTRAINTS:
                row[f"{case}_{field}_amplitude"] = math.sqrt(
                    max(aligned[case][field][index], 0.0))
        for field, triples in richardson_orders.items():
            for triple, order in triples.items():
                row[f"{field}_richardson_order_{triple}"] = order[index]
        for field in CONSTRAINTS:
            for pair, order in constraint_orders[field].items():
                row[f"{field}_order_{pair}"] = order[index]
        rows.append(row)
    with (args.output / f"{resolution_word}_resolution_aligned.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)

    summary = {
        "schema": "athenak.z4c.reference_same_tree_convergence.v2",
        "common_tau_max": common_tau_max,
        "final_tau": {case: float(values["axisTau"][-1]) for case, values in data.items()},
        "curve_richardson_order_medians": {
            field: {triple: window_summary(tau, order)
                    for triple, order in triples.items()}
            for field, triples in richardson_orders.items()
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
        "same_resolution_ablation": ablation,
        "constraint_history_radius": args.constraint_radius,
        "claim_boundary": (
            "same physical AMR tree and 2:1 cell-spacing sequence; observed orders are "
            "descriptive and support convergence only when consistent across both pairs and windows"
        ),
    }
    (args.output / f"{resolution_word}_resolution_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    colors = {"n128": "#4daf4a", "n256": "#377eb8", "n512": "#e41a1c",
              "n1024": "#984ea3", "n1024_cfl040_ko010": "#ff7f00"}
    labels = {"n128": "N128", "n256": "N256", "n512": "N512",
              "n1024": "N1024 CFL 0.15, KO 0.50",
              "n1024_cfl040_ko010": "N1024 CFL 0.40, KO 0.10 (failed)"}
    linestyles = {case: ("--" if case == "n1024_cfl040_ko010" else "-")
                  for case in plot_cases}
    fig, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for name, (xx, yy) in reference.items():
        axis.plot(xx, yy, linewidth=1.15, alpha=0.55, label=f"published {name}")
    for case, values in data.items():
        valid = np.abs(values["axisKret"]) > 0.0
        axis.plot(values["axisTau"][valid], np.log10(np.abs(values["axisKret"][valid])),
                  color=colors[case], linestyle=linestyles[case], linewidth=1.55,
                  label=labels[case])
    axis.set(xlim=(0.0, 13.5), ylim=(-8.0, 8.0), xlabel="central proper time",
             ylabel=r"$\log_{10}|\mathrm{Kretschmann}(0)|$")
    axis.grid(alpha=0.22); axis.legend(fontsize=8, ncol=2)
    save(fig, args.output, f"figure3_{tag}")

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True,
                             constrained_layout=True)
    for axis, field in zip(axes.flat, CONSTRAINTS):
        for case, values in data.items():
            axis.semilogy(values["axisTau"], np.sqrt(np.maximum(values[field], 1.0e-30)),
                          color=colors[case], linestyle=linestyles[case],
                          label=labels[case])
        axis.set_title(field.replace("-norm2", "")); axis.grid(alpha=0.22, which="both")
    axes[1, 0].set_xlabel("central proper time"); axes[1, 1].set_xlabel("central proper time")
    domain = (r"_{r<" + f"{args.constraint_radius:g}" + "}"
              if args.constraint_radius is not None else "")
    axes[0, 0].set_ylabel(r"$\sqrt{\int" + domain + r" C^2 dV}$")
    axes[1, 0].set_ylabel(r"$\sqrt{\int" + domain + r" C^2 dV}$")
    axes[0, 0].legend()
    save(fig, args.output, f"constraint_amplitudes_{tag}")

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), sharex=True,
                             constrained_layout=True)
    for axis, field in zip(axes.flat, CONSTRAINTS):
        for pair, order in constraint_orders[field].items():
            axis.plot(tau, order, label=pair.upper().replace("_", "/"))
        axis.axhline(4.0, color="black", linestyle=":", linewidth=1.0)
        axis.set(title=field.replace("-norm2", ""), ylim=(-4.0, 8.0))
        axis.grid(alpha=0.22)
    axes[1, 0].set_xlabel("central proper time"); axes[1, 1].set_xlabel("central proper time")
    axes[0, 0].set_ylabel("pairwise amplitude order")
    axes[1, 0].set_ylabel("pairwise amplitude order")
    axes[0, 0].legend()
    resolution_tag = "_".join(resolution_cases)
    save(fig, args.output, f"constraint_pairwise_orders_{resolution_tag}")


if __name__ == "__main__":
    main()
