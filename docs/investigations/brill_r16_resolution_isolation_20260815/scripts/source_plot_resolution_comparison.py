#!/usr/bin/env python3
"""Plot the authenticated R=16 N128/N256 resolution-isolation histories."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "comparison-plots"
REFERENCE = OUT / "reference" / "figure3_published_curves.csv"
N128 = (
    ROOT.parent
    / "r4_brill_figure3_five_run_domain64_comparison_20260815"
    / "evidence/domain16/run/cases/fixed_eta2_tau1_kappa1_l20_nocd_ko002"
    / "brill_fig3_fixed_eta2_tau1_kappa1_l20_nocd_ko002_n128.z4c.user.hst"
)
N256 = (
    ROOT / "terminal-remote-selected/run/case"
    / "brill_fig3_r16_n256_dchi001_tau1_nocd.z4c.user.hst"
)
HEADER = re.compile(r"\[(\d+)\]=([^\s]+)")
STYLES = {
    "N128": (r"N128, $d\chi_{\max}=0.02$", "#377eb8", "-"),
    "N256": (r"N256, $d\chi_{\max}=0.01$", "#e41a1c", "--"),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_history(path: Path) -> list[dict[str, float]]:
    labels: dict[str, int] = {}
    rows: list[dict[str, float]] = []
    for line in path.read_text(errors="strict").splitlines():
        if line.startswith("#"):
            for index, name in HEADER.findall(line):
                labels[name] = int(index) - 1
            continue
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if all(math.isfinite(value) for value in values):
            rows.append({name: values[index] for name, index in labels.items()})
    required = {
        "time", "dt", "C-norm2", "H-norm2", "M-norm2", "Z-norm2",
        "max_abs_K", "maxAbsKret", "nmb_total", "maxRefLev", "cycle",
        "axisLapse", "axisTau", "axisKret",
    }
    if not rows or not required <= labels.keys():
        raise RuntimeError(f"invalid history: {path}")
    return rows


def finite_xy(rows, x, y, positive=False):
    pairs = [
        (row[x], row[y]) for row in rows
        if math.isfinite(row[x]) and math.isfinite(row[y])
        and (not positive or row[y] > 0.0)
    ]
    return [item[0] for item in pairs], [item[1] for item in pairs]


def save(fig, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=220)
    fig.savefig(
        OUT / f"{stem}.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def plot_figure3(curves) -> None:
    with REFERENCE.open(newline="") as stream:
        paper = list(csv.DictReader(stream))
    paper_styles = {
        "bamps": ("BAMPS (paper rendering)", "#999999", ":"),
        "prague": ("Prague (paper rendering)", "#666666", "--"),
        "sphGR": ("sphGR (paper rendering)", "#222222", "-."),
    }
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.8), constrained_layout=True)
    for ax in axes:
        for key, (label, color, linestyle) in paper_styles.items():
            selected = [row for row in paper if row["series"] == key]
            ax.plot(
                [float(row["tau"]) for row in selected],
                [float(row["log10_abs_I"]) for row in selected],
                label=label, color=color, linestyle=linestyle, linewidth=1.15,
            )
        for key, rows in curves.items():
            label, color, linestyle = STYLES[key]
            points = [
                (row["axisTau"], math.log10(abs(row["axisKret"])))
                for row in rows if row["axisKret"] != 0.0
            ]
            ax.plot(
                [point[0] for point in points], [point[1] for point in points],
                label=label, color=color, linestyle=linestyle, linewidth=1.55,
            )
            ax.scatter(
                points[-1][0], points[-1][1], color=color, marker="x", s=38,
                zorder=5,
            )
        ax.set_xlim(0.0, 15.05)
        ax.set_xlabel(r"central proper time $\tau/M$")
        ax.set_ylabel(r"$\log_{10}|I|$ at the origin")
        ax.grid(alpha=0.23)
    axes[0].set_ylim(-7.0, 6.0)
    axes[0].set_title("Published Figure 3 range")
    axes[1].set_ylim(-7.0, 34.0)
    axes[1].set_title("Full finite AthenaK range")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle(
        "Brill A=-0.047, R=Z=16: resolution isolation over published Figure 3"
    )
    save(fig, "figure3_resolution_overlay")


def plot_constraints(curves) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.8), constrained_layout=True)
    specs = (
        ("C-norm2", r"$\|C\|_2$"),
        ("H-norm2", r"$\|H\|_2$"),
        ("M-norm2", r"$\|M\|_2$"),
        ("Z-norm2", r"$\|Z\|_2$"),
    )
    for ax, (field, title) in zip(axes.flat, specs):
        for key, rows in curves.items():
            label, color, linestyle = STYLES[key]
            xx, yy = finite_xy(rows, "time", field, positive=True)
            ax.semilogy(
                xx, yy, label=label, color=color, linestyle=linestyle,
                linewidth=1.45,
            )
        ax.axvline(curves["N128"][-1]["time"], color="#377eb8", alpha=0.35)
        ax.axvline(curves["N256"][-1]["time"], color="#e41a1c", alpha=0.35)
        ax.set_title(title)
        ax.set_xlabel(r"coordinate time $t/M$")
        ax.grid(alpha=0.23, which="both")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle(
        "Brill A=-0.047: global constraint histories (vertical lines mark failure)"
    )
    save(fig, "constraints_resolution_comparison")


def plot_gauge_amr(curves) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(15.0, 8.5), constrained_layout=True)
    specs = (
        ("axisLapse", "axis lapse", False),
        ("max_abs_K", r"$\max |K|$", True),
        ("maxAbsKret", r"$\max |I|$", True),
        ("maxRefLev", "maximum AMR level", False),
        ("nmb_total", "total MeshBlocks", True),
        ("dt", r"time step $\Delta t/M$", True),
    )
    for ax, (field, title, log_y) in zip(axes.flat, specs):
        for key, rows in curves.items():
            label, color, linestyle = STYLES[key]
            xx, yy = finite_xy(rows, "time", field, positive=log_y)
            (ax.semilogy if log_y else ax.plot)(
                xx, yy, label=label, color=color, linestyle=linestyle,
                linewidth=1.4,
            )
        ax.set_title(title)
        ax.set_xlabel(r"coordinate time $t/M$")
        ax.grid(alpha=0.23, which="both")
    axes[1, 0].axhline(20, color="#444444", linestyle=":", linewidth=1.0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle("Brill A=-0.047: gauge, curvature, refinement, and timestep")
    save(fig, "gauge_amr_resolution_comparison")


def write_summary(curves) -> None:
    summary = {
        "schema": "athenak_brill_r16_n128_n256_plot_bundle_v1",
        "qualification_claim": False,
        "inputs": {
            "N128": {"path": str(N128), "sha256": sha256(N128)},
            "N256": {"path": str(N256), "sha256": sha256(N256)},
            "published_rendered_curves": {
                "path": str(REFERENCE), "sha256": sha256(REFERENCE),
                "status": "rendered PDF polyline reconstruction, not paper raw data",
            },
        },
        "terminal": {
            key: {
                "time": rows[-1]["time"],
                "axis_proper_time": rows[-1]["axisTau"],
                "cycle": int(rows[-1]["cycle"]),
                "max_refinement_level": int(max(row["maxRefLev"] for row in rows)),
                "max_meshblocks": int(max(row["nmb_total"] for row in rows)),
            }
            for key, rows in curves.items()
        },
    }
    (OUT / "plot_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


def main() -> None:
    curves = {"N128": read_history(N128), "N256": read_history(N256)}
    plot_figure3(curves)
    plot_constraints(curves)
    plot_gauge_amr(curves)
    write_summary(curves)


if __name__ == "__main__":
    main()
