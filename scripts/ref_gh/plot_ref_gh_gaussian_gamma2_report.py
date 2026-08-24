#!/usr/bin/env python3
"""Generate deterministic figures for the 2026-08-24 Ref-GH report.

The T4/T5 plots are derived only from compact, committed histories.  The
Gaussian plot is an analytic design illustration and is labelled as such in
the report; it is not evolution evidence.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


ARTIFACT_ROOT = Path(
    "docs/fo_gh_artifacts/ref_gh_feedback_continuation_20260823"
)
T4_ROOT = ARTIFACT_ROOT / "aurora/job_8777607_t4_fail_closed"
T5_ROOT = ARTIFACT_ROOT / "aurora/job_8777824_t5_open_loop_fail"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 9.5,
            "axes.labelsize": 9.5,
            "axes.titlesize": 10.5,
            "legend.fontsize": 8.3,
            "figure.dpi": 160,
            "savefig.dpi": 200,
            "savefig.bbox": "tight",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "lines.linewidth": 1.65,
        }
    )


def save(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.savefig(
        output_dir / f"{stem}.pdf",
        metadata={
            "Creator": "plot_ref_gh_gaussian_gamma2_report.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    fig.savefig(output_dir / f"{stem}.png")
    plt.close(fig)


def load_history(path: Path) -> np.ndarray:
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.shape[1] < 31:
        raise ValueError(f"unexpected history schema in {path}: {data.shape}")
    return data


def plot_time_histories(output_dir: Path) -> None:
    t4 = load_history(T4_ROOT / "refgh_feedback_t4_outer24.combined.user.hst")
    t5 = load_history(T5_ROOT / "refgh_prescribed_t5_outer24.combined.user.hst")
    curves = (("T4 feedback", t4, "#1f77b4"), ("T5 prescribed", t5, "#d95f02"))

    fig, axes = plt.subplots(2, 2, figsize=(7.15, 5.25), sharex=True)
    for label, data, color in curves:
        time = data[:, 0]
        axes[0, 0].plot(time, data[:, 19], color=color, label=label)
        axes[0, 0].plot(time, data[:, 17], color=color, linestyle="--", alpha=0.75)
        axes[0, 1].plot(time, data[:, 8], color=color, label=label)
        axes[1, 0].plot(time, data[:, 11], color=color, label=label)
        for column, linestyle, name in (
            (28, "-", r"$\|C_{\mathrm{GH}}\|_2$"),
            (29, "--", r"$\|C_I\|_2$"),
            (30, ":", r"$\|C_{IJ}\|_2$"),
        ):
            axes[1, 1].semilogy(
                time,
                np.maximum(data[:, column], 1.0e-18),
                color=color,
                linestyle=linestyle,
                label=f"{label}: {name}",
            )

    axes[0, 0].set_ylabel("clock / activation")
    axes[0, 0].set_title(r"Continuation clock $\xi$ (solid) and $S(\xi)$ (dashed)")
    axes[0, 1].set_ylabel(r"$\kappa(G)$")
    axes[0, 1].set_title("Maximum relative-metric condition number")
    axes[1, 0].set_ylabel(r"$v_{\mathrm{rel}}^2$ maximum")
    axes[1, 0].set_title("Relative boost diagnostic")
    axes[1, 1].set_ylabel("global native norm")
    axes[1, 1].set_title("GH, reduction, and curl constraints")
    for axis in axes[1, :]:
        axis.set_xlabel(r"$t/M$")
    axes[0, 0].legend(loc="upper left")
    axes[0, 1].legend(loc="upper left")
    axes[1, 0].legend(loc="upper left")
    axes[1, 1].legend(loc="upper left", ncol=1, fontsize=7.1)
    fig.suptitle("Aurora medium-grid fixed-stitch discriminators", y=1.01)
    fig.tight_layout()
    save(fig, output_dir, "t4_t5_time_histories")


def plot_equal_activation(output_dir: Path) -> None:
    comparison = json.loads(
        (ARTIFACT_ROOT / "feedback_vs_open_loop_comparison.json").read_text()
    )
    samples = comparison["activation_samples"]
    xi = np.array([sample["xi"] for sample in samples])
    colors = {"controlled": "#1f77b4", "prescribed": "#d95f02"}
    labels = {"controlled": "T4 feedback", "prescribed": "T5 prescribed"}

    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.65))
    for key in ("controlled", "prescribed"):
        axes[0].plot(
            xi,
            [sample[key]["condition"] for sample in samples],
            color=colors[key],
            marker="o",
            markersize=3.2,
            label=labels[key],
        )
        axes[1].plot(
            xi,
            [sample[key]["v2"] for sample in samples],
            color=colors[key],
            marker="o",
            markersize=3.2,
            label=labels[key],
        )
        for field, linestyle, short in (
            ("gh_l2", "-", "GH"),
            ("reduction_l2", "--", "reduction"),
            ("curl_l2", ":", "curl"),
        ):
            axes[2].semilogy(
                xi,
                [sample[key][field] for sample in samples],
                color=colors[key],
                linestyle=linestyle,
                marker="o",
                markersize=2.7,
                label=f"{labels[key]} {short}",
            )

    axes[0].set_ylabel(r"$\kappa(G)$")
    axes[0].set_title("conditioning")
    axes[1].set_ylabel(r"$v_{\mathrm{rel}}^2$ maximum")
    axes[1].set_title("relative boost")
    axes[2].set_ylabel("global native norm")
    axes[2].set_title("constraints")
    for axis in axes:
        axis.set_xlabel(r"equal activation clock $\xi$")
    axes[0].legend(loc="upper left", fontsize=7.4)
    axes[1].legend(loc="upper left", fontsize=7.4)
    axes[2].legend(loc="upper left", fontsize=6.5, ncol=1)
    fig.tight_layout()
    save(fig, output_dir, "t4_t5_equal_activation")


def load_max_locations(root: Path) -> list[dict[str, str]]:
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for path in sorted(root.glob("to_*/*maxloc.tsv")):
        with path.open(newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                unique[(row["time"], row["diagnostic"])] = row
    return list(unique.values())


def plot_max_locations(output_dir: Path) -> None:
    diagnostics = (
        "GH_constraint",
        "reduction_constraint",
        "curl_constraint",
        "source_frame_correction",
    )
    markers = {
        "GH_constraint": "o",
        "reduction_constraint": "s",
        "curl_constraint": "^",
        "source_frame_correction": "D",
    }
    short = {
        "GH_constraint": "GH",
        "reduction_constraint": "reduction",
        "curl_constraint": "curl",
        "source_frame_correction": "frame source",
    }
    runs = (
        ("T4", load_max_locations(T4_ROOT), "#1f77b4"),
        ("T5", load_max_locations(T5_ROOT), "#d95f02"),
    )
    fig, axis = plt.subplots(figsize=(7.15, 3.05))
    axis.axhspan(0.30, 0.60, color="#d9d9d9", alpha=0.75, label="fixed stitch 0.30--0.60M")
    axis.axhline(4.0, color="black", linestyle="--", linewidth=1.0, label="nearest finest-box face")
    for run, rows, color in runs:
        for diagnostic in diagnostics:
            selected = sorted(
                (
                    row
                    for row in rows
                    if row["diagnostic"] == diagnostic
                    and float(row["maximum"]) > 1.0e-14
                ),
                key=lambda row: float(row["time"]),
            )
            if not selected:
                continue
            axis.plot(
                [float(row["time"]) for row in selected],
                [float(row["radius"]) for row in selected],
                color=color,
                marker=markers[diagnostic],
                markersize=3.4,
                linestyle="-" if diagnostic != "source_frame_correction" else ":",
                label=f"{run} {short[diagnostic]}",
            )
    axis.set_xlabel(r"$t/M$")
    axis.set_ylabel(r"radius of global maximum $r/M$")
    axis.set_ylim(0.0, 4.5)
    axis.set_title("Failure diagnostics localize inside the compact stitching shell")
    axis.legend(loc="upper left", ncol=3, fontsize=6.9)
    fig.tight_layout()
    save(fig, output_dir, "t4_t5_maximum_locations")


def plot_gaussian_profiles(output_dir: Path) -> None:
    radius = np.linspace(0.0, 12.0, 1201)
    fig, axes = plt.subplots(1, 3, figsize=(7.3, 2.55), sharex=True)
    colors = ("#1b9e77", "#7570b3", "#e7298a")
    for width, color in zip((2.0, 3.0, 4.0), colors):
        weight = np.exp(-(radius / width) ** 2)
        first = np.abs(-2.0 * radius * weight / width**2)
        second = np.abs((4.0 * radius**2 / width**4 - 2.0 / width**2) * weight)
        label = rf"$R_G/M={width:.0f}$"
        axes[0].plot(radius, weight, color=color, label=label)
        axes[1].plot(radius, first, color=color, label=label)
        axes[2].plot(radius, second, color=color, label=label)
    axes[0].set_ylabel(r"$W$")
    axes[1].set_ylabel(r"$M|dW/dr|$")
    axes[2].set_ylabel(r"$M^2|d^2W/dr^2|$")
    axes[0].set_title("attenuation")
    axes[1].set_title("first derivative")
    axes[2].set_title("second derivative")
    for axis in axes:
        axis.set_xlabel(r"$r/M$")
        axis.set_xlim(0.0, 12.0)
    axes[0].legend(loc="upper right", fontsize=7.6)
    fig.suptitle("Proposed broad Gaussian localization (analytic design, not run data)", y=1.01)
    fig.tight_layout()
    save(fig, output_dir, "gaussian_reference_profiles")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "docs/fo_gh_artifacts/ref_gh_gaussian_reference_gamma2_20260824/figures"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_time_histories(args.output_dir)
    plot_equal_activation(args.output_dir)
    plot_max_locations(args.output_dir)
    plot_gaussian_profiles(args.output_dir)


if __name__ == "__main__":
    main()
