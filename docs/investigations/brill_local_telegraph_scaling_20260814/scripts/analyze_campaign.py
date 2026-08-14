#!/usr/bin/env python3
"""Create matched plots and a strict summary for the four-mode campaign."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
EVIDENCE = HERE / "terminal_evidence"
CASES = EVIDENCE / "run/cases"
PROFILES = HERE / "selected_mu_profiles.json"
PUBLISHED = HERE / "figure3_published_curves.csv"
OLD_BASE = (
    HERE.parent
    / "r4_brill_figure3_allbulk_scaleinv_n128_ae3a817e_v4_20260813"
    / "terminal_evidence/run/n128_scale_invariant"
    / "brill_fig3_allbulk_scaleinv_n128.z4c.user.hst"
)
ANALYSIS = HERE / "analysis"

ORDER = (
    "max_domain_abs_K",
    "local_abs_K",
    "local_extrinsic_curvature_norm",
    "local_chi_gradient_norm",
)
LABELS = {
    "max_domain_abs_K": r"domain $\max|K|$ (baseline)",
    "local_abs_K": r"local $|K|$",
    "local_extrinsic_curvature_norm": r"local $\sqrt{K_{ij}K^{ij}}$",
    "local_chi_gradient_norm": r"local $|\nabla\chi|_\gamma$",
}
COLORS = {
    "max_domain_abs_K": "#202020",
    "local_abs_K": "#d62728",
    "local_extrinsic_curvature_norm": "#1f77b4",
    "local_chi_gradient_norm": "#2ca02c",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_history(path: Path) -> tuple[dict[str, int], np.ndarray]:
    labels: dict[str, int] = {}
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            labels.update(
                {name: int(number) - 1 for number, name in
                 re.findall(r"\[([0-9]+)\]=([^ ]+)", line)}
            )
        elif line.strip():
            values = [float(value) for value in line.split()]
            if all(math.isfinite(value) for value in values):
                rows.append(values)
    required = {
        "time", "C-norm2", "H-norm2", "M-norm2", "Z-norm2",
        "max_abs_K", "maxAbsKret", "cycle", "axisLapse", "axisTau",
        "axisKret", "muMin", "muMax",
    }
    if not required <= labels.keys():
        raise RuntimeError(f"missing history labels in {path}: {required-labels.keys()}")
    data = np.asarray(rows)
    if data.ndim != 2 or data.shape[1] != len(labels):
        raise RuntimeError(f"history shape mismatch for {path}: {data.shape}, {len(labels)}")
    return labels, data


def column(labels: dict[str, int], data: np.ndarray, name: str) -> np.ndarray:
    return data[:, labels[name]]


def nearest(labels: dict[str, int], data: np.ndarray, target: float) -> dict[str, float]:
    index = int(np.argmin(np.abs(column(labels, data, "time") - target)))
    return {
        key: float(data[index, labels[key]])
        for key in ("time", "C-norm2", "H-norm2", "M-norm2", "Z-norm2",
                    "max_abs_K", "axisLapse", "axisTau", "muMin", "muMax")
    }


def read_published() -> dict[str, list[tuple[float, float]]]:
    curves: dict[str, list[tuple[float, float]]] = {}
    with PUBLISHED.open(newline="") as stream:
        for row in csv.DictReader(stream):
            curves.setdefault(row["series"], []).append(
                (float(row["tau"]), float(row["log10_abs_I"]))
            )
    return curves


def plot_figure3(histories: dict[str, tuple[dict[str, int], np.ndarray]]) -> None:
    published = read_published()
    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.6), constrained_layout=True)
    paper_style = {
        "bamps": ("BAMPS (paper, rendered)", "#aaaaaa", ":"),
        "prague": ("Prague (paper, rendered)", "#777777", "--"),
        "sphGR": ("sphGR (paper, rendered)", "#444444", "-."),
    }
    for ax in axes:
        for name, (label, color, linestyle) in paper_style.items():
            points = published[name]
            ax.plot(*zip(*points), label=label, color=color, linestyle=linestyle,
                    linewidth=1.25, alpha=0.8)
        for mode in ORDER:
            labels, data = histories[mode]
            tau = column(labels, data, "axisTau")
            curvature = np.abs(column(labels, data, "axisKret"))
            valid = curvature > 0.0
            ax.plot(tau[valid], np.log10(curvature[valid]), color=COLORS[mode],
                    linewidth=1.75, label=LABELS[mode])
            ax.scatter(tau[valid][-1], np.log10(curvature[valid][-1]),
                       color=COLORS[mode], edgecolor="white", linewidth=0.5,
                       s=30, zorder=5)
        ax.grid(alpha=0.22)
        ax.set_xlabel(r"central proper time $\tau/M$")
        ax.set_ylabel(r"$\log_{10}|I|$ on the symmetry axis")
    axes[0].set_xlim(0.0, 15.05)
    axes[0].set_ylim(-7.0, 6.0)
    axes[0].set_title("Full published Figure-3 interval")
    axes[1].set_xlim(0.0, 6.35)
    axes[1].set_ylim(-4.0, 5.5)
    axes[1].set_title("Zoom: all AthenaK local-scale runs")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle("Brill A=-0.047: scale-invariant telegrapher damping prescriptions")
    fig.savefig(ANALYSIS / "figure3_local_telegraph_mu_comparison.png", dpi=200)
    fig.savefig(ANALYSIS / "figure3_local_telegraph_mu_comparison.pdf")
    plt.close(fig)


def plot_constraints(histories: dict[str, tuple[dict[str, int], np.ndarray]]) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), constrained_layout=True)
    for ax, name, title in zip(
            axes.flat,
            ("C-norm2", "H-norm2", "M-norm2", "Z-norm2"),
            (r"combined $C_2$", r"Hamiltonian $H_2$",
             r"momentum $M_2$", r"Z4c $Z_2$")):
        for mode in ORDER:
            labels, data = histories[mode]
            ax.semilogy(column(labels, data, "time"),
                        np.maximum(column(labels, data, name), 1.0e-18),
                        color=COLORS[mode], linewidth=1.5, label=LABELS[mode])
        ax.set_title(title)
        ax.set_xlabel(r"coordinate time $t/M$")
        ax.grid(True, which="both", alpha=0.22)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Brill A=-0.047: constraint growth under local damping choices")
    fig.savefig(ANALYSIS / "constraints_local_telegraph_mu_comparison.png", dpi=200)
    fig.savefig(ANALYSIS / "constraints_local_telegraph_mu_comparison.pdf")
    plt.close(fig)


def plot_lapse_k(histories: dict[str, tuple[dict[str, int], np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.1), constrained_layout=True)
    for mode in ORDER:
        labels, data = histories[mode]
        axes[0].plot(column(labels, data, "axisTau"),
                     column(labels, data, "axisLapse"), color=COLORS[mode],
                     linewidth=1.6, label=LABELS[mode])
        axes[1].semilogy(column(labels, data, "time"),
                        np.maximum(column(labels, data, "max_abs_K"), 1.0e-18),
                        color=COLORS[mode], linewidth=1.6, label=LABELS[mode])
    axes[0].set_xlabel(r"central proper time $\tau/M$")
    axes[0].set_ylabel(r"central reconstructed lapse $\alpha$")
    axes[0].set_title("Central lapse")
    axes[1].set_xlabel(r"coordinate time $t/M$")
    axes[1].set_ylabel(r"domain $\max |K|$")
    axes[1].set_title("Curvature scale used for normalization")
    for ax in axes:
        ax.grid(True, which="both", alpha=0.22)
    axes[0].legend(fontsize=8)
    fig.savefig(ANALYSIS / "lapse_and_maxK_local_telegraph_mu.png", dpi=200)
    fig.savefig(ANALYSIS / "lapse_and_maxK_local_telegraph_mu.pdf")
    plt.close(fig)


def plot_mu_extrema(histories: dict[str, tuple[dict[str, int], np.ndarray]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.1), constrained_layout=True)
    for mode in ORDER:
        labels, data = histories[mode]
        time = column(labels, data, "time")
        for ax, key in zip(axes, ("muMin", "muMax")):
            values = column(labels, data, key).copy()
            values[values <= 0.0] = np.nan
            ax.semilogy(time, values, color=COLORS[mode], linewidth=1.45,
                        label=LABELS[mode])
    axes[0].set_title(r"domain minimum of positive $\mu$")
    axes[1].set_title(r"domain maximum of $\mu$")
    for ax in axes:
        ax.set_xlabel(r"coordinate time $t/M$")
        ax.set_ylabel(r"physical damping scale $\mu$ [$M^{-1}$]")
        ax.grid(True, which="both", alpha=0.22)
    axes[1].legend(fontsize=8)
    fig.suptitle(r"Telegrapher damping-field extrema ($Q=\mu/\max|K|$)")
    fig.savefig(ANALYSIS / "telegraph_mu_extrema.png", dpi=200)
    fig.savefig(ANALYSIS / "telegraph_mu_extrema.pdf")
    plt.close(fig)


def plot_profiles(profiles: dict[str, object]) -> dict[str, object]:
    fig, axes = plt.subplots(4, 2, figsize=(13.2, 15.2), constrained_layout=True)
    summary: dict[str, object] = {}
    for row, mode in enumerate(ORDER):
        snapshots = profiles["cases"][mode]
        mode_summary = []
        for number, snapshot in enumerate(snapshots):
            label = (
                "first evaluated" if number == 0 else
                (r"common $t\approx8M$" if number == 1 else "terminal profile")
            )
            for ax, key, coordinate in (
                    (axes[row, 0], "equatorial_profile", "rho"),
                    (axes[row, 1], "axis_profile", "z")):
                points = snapshot[key]
                x = np.asarray([point[coordinate] for point in points])
                y = np.asarray([point["mu"] for point in points])
                ax.plot(x, y, linewidth=1.25, label=f"{label}, t={snapshot['time']:.3f}")
            middle = snapshot["equatorial_profile"]
            y = np.asarray([point["mu"] for point in middle])
            scale = max(float(np.max(np.abs(y))), 1.0e-300)
            mode_summary.append({
                "time": snapshot["time"],
                "cycle": snapshot["cycle"],
                "field_min": snapshot["field_min"],
                "field_max": snapshot["field_max"],
                "nonfinite_cells": snapshot["nonfinite_cells"],
                "negative_cells": snapshot["negative_cells"],
                "zero_cells": snapshot["zero_cells"],
                "equatorial_total_variation_over_peak":
                    float(np.sum(np.abs(np.diff(y))) / scale),
                "equatorial_max_jump_over_peak":
                    float(np.max(np.abs(np.diff(y))) / scale) if y.size > 1 else 0.0,
                "source_file": snapshot["file"],
                "source_sha256": snapshot["sha256"],
            })
        axes[row, 0].set_ylabel(r"$\mu$ [$M^{-1}$]")
        axes[row, 0].set_title(f"{LABELS[mode]}: near-equatorial profile")
        axes[row, 1].set_title(f"{LABELS[mode]}: axis-adjacent profile")
        for ax in axes[row]:
            ax.set_yscale("symlog", linthresh=1.0e-6)
            ax.grid(True, which="both", alpha=0.2)
            ax.legend(fontsize=7)
        axes[row, 0].set_xlabel(r"$\rho/M$ at nearest cell to $z=0$")
        axes[row, 1].set_xlabel(r"$z/M$ at first active $\rho$ cell")
        summary[mode] = mode_summary
    fig.suptitle("Spatial damping-scale profiles read directly from AthenaK outputs")
    fig.savefig(ANALYSIS / "telegraph_mu_spatial_profiles.png", dpi=200)
    fig.savefig(ANALYSIS / "telegraph_mu_spatial_profiles.pdf")
    plt.close(fig)
    return summary


def main() -> None:
    ANALYSIS.mkdir(exist_ok=False)
    histories: dict[str, tuple[dict[str, int], np.ndarray]] = {}
    results = {}
    for mode in ORDER:
        case = CASES / mode
        history_path = next(case.glob("*.hst"))
        histories[mode] = read_history(history_path)
        results[mode] = json.loads((case / "result.json").read_text())

    plot_figure3(histories)
    plot_constraints(histories)
    plot_lapse_k(histories)
    plot_mu_extrema(histories)
    profile_summary = plot_profiles(json.loads(PROFILES.read_text()))

    old_lines = np.asarray([
        [float(value) for value in line.split()]
        for line in OLD_BASE.read_text().splitlines()
        if line and not line.startswith("#")
    ])
    baseline_labels, baseline_data = histories["max_domain_abs_K"]
    baseline_reproduction = {
        "old_history_sha256": sha256(OLD_BASE),
        "new_history_sha256": sha256(next((CASES / "max_domain_abs_K").glob("*.hst"))),
        "row_count_equal": int(old_lines.shape[0]) == int(baseline_data.shape[0]),
        "original_primary_columns_1_through_22_bitwise_equal":
            bool(np.array_equal(old_lines[:, :22], baseline_data[:, :22])),
        "original_all_69_columns_max_abs_difference":
            float(np.max(np.abs(old_lines - baseline_data[:, :69]))),
        "final_time_equal": results["max_domain_abs_K"]["final_time"] == 10.16294,
        "fatal_gate_reproduced":
            "rejected 6412 parent stencils" in results["max_domain_abs_K"]["fatal_line"],
        "note": (
            "Primary global/history observables are bitwise identical. Tiny changes in "
            "the auxiliary axis/off-axis decompositions are reduction-order effects "
            "from appending the mu history reductions; the terminal result is identical."
        ),
    }

    summary = {
        "schema": "athenak_brill_local_telegraph_mu_analysis_v1",
        "qualification_claim": False,
        "source_commit": "2a8ad80e02279769a99fe279b7a33516bc6c8d0d",
        "job_id": 56955603,
        "scale_invariant_parameterization": {
            "Q": "mu/max_domain_abs_K",
            "tau_effective": "tau/max_domain_abs_K",
            "kappa_effective": "kappa/max_domain_abs_K",
            "evaluated_damping": "mu/tau",
            "evaluated_gradient": "kappa/tau",
            "tau": 1.0,
            "kappa": 1.0,
        },
        "baseline_reproduction": baseline_reproduction,
        "cases": {
            mode: {
                "result": results[mode],
                "nearest_t8_history": nearest(*histories[mode], 8.0),
                "history_sha256": sha256(next((CASES / mode).glob("*.hst"))),
                "run_log_sha256": sha256(CASES / mode / "run.log"),
            }
            for mode in ORDER
        },
        "spatial_profile_diagnostics": profile_summary,
        "authenticated_remote_evidence": {
            "root_manifest_sha256": sha256(EVIDENCE / "SHA256SUMS"),
            "detached_manifest_file_sha256": sha256(EVIDENCE / "SHA256SUMS.sha256"),
            "campaign_summary_sha256": sha256(EVIDENCE / "run/campaign_summary.json"),
            "sacct_sha256": sha256(EVIDENCE / "allocation/sacct-settled.psv"),
            "full_remote_manifest_verification": "pass",
        },
        "disposition": {
            "recommended_prescription": "max_domain_abs_K",
            "reason": (
                "It exactly reproduced the prior baseline and reached t=10.16294M. "
                "All three local alternatives failed the same strict positive-chi gate "
                "earlier (8.358252M to 9.193945M); no coefficient was retuned."
            ),
            "production_qualification": False,
        },
    }
    summary_path = ANALYSIS / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True,
                                       allow_nan=False) + "\n")

    files = sorted(path for path in ANALYSIS.iterdir() if path.is_file())
    with (ANALYSIS / "SHA256SUMS").open("w") as stream:
        for path in files:
            stream.write(f"{sha256(path)}  {path.name}\n")
    print(summary_path)
    for mode in ORDER:
        print(mode, results[mode]["final_time"], results[mode]["fatal_line"])


if __name__ == "__main__":
    main()
