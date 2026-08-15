#!/usr/bin/env python3
"""Build the pre/post 2D-restriction Brill resolution comparison."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
PATHS = {
    "pre_n128": DATA / "pre_fix_n128_history.csv",
    "pre_n256": DATA / "pre_fix_n256_history.csv",
    "post_n128": DATA / "post_fix_n128_history.csv",
    "post_n256": DATA / "post_fix_n256_history.csv",
}
STYLES = {
    "pre_n128": ("N128 before fix", "#85add1", "--", 0.8),
    "pre_n256": ("N256 before fix", "#f29a96", "--", 0.8),
    "post_n128": ("N128 repaired", "#2166ac", "-", 1.0),
    "post_n256": ("N256 repaired", "#b2182b", "-", 1.0),
}
CONSTRAINTS = ("C-norm2", "H-norm2", "M-norm2", "Z-norm2")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, float]]:
    with path.open(newline="") as stream:
        rows = [{key: float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]
    if not rows or not all(math.isfinite(value) for row in rows for value in row.values()):
        raise RuntimeError(f"nonfinite or empty history: {path}")
    return rows


def l2(row: dict[str, float], field: str) -> float:
    # AthenaK's historical *-norm2 columns are volume integrals of the squared
    # diagnostic.  The plotted L2-like magnitude is their square root.
    return math.sqrt(max(row[field], 0.0))


def finite_xy(rows, x, y, *, sqrt_y=False, positive=False):
    pairs = []
    for row in rows:
        yy = l2(row, y) if sqrt_y else row[y]
        if math.isfinite(row[x]) and math.isfinite(yy) and (not positive or yy > 0):
            pairs.append((row[x], yy))
    return [item[0] for item in pairs], [item[1] for item in pairs]


def topology_events(rows):
    events = []
    for before, after in zip(rows, rows[1:]):
        if (after["nmb_total"] == before["nmb_total"] and
                after["maxRefLev"] == before["maxRefLev"]):
            continue
        factors = {}
        for field in ("C-norm2", "H-norm2", "M-norm2"):
            left, right = l2(before, field), l2(after, field)
            factors[field] = (right / left if left > 0 else None)
        events.append({
            "time": after["time"],
            "cycle": int(after["cycle"]),
            "nmb_before": int(before["nmb_total"]),
            "nmb_after": int(after["nmb_total"]),
            "level_before": int(before["maxRefLev"]),
            "level_after": int(after["maxRefLev"]),
            "l2_factors": factors,
            "max_abs_log10_l2_factor": max(
                (abs(math.log10(value)) for value in factors.values()
                 if value is not None and value > 0), default=0.0),
        })
    return events


def median_log_increment(rows, field, topology):
    values = []
    for before, after in zip(rows, rows[1:]):
        changed = (after["nmb_total"] != before["nmb_total"] or
                   after["maxRefLev"] != before["maxRefLev"])
        if changed != topology:
            continue
        left, right = l2(before, field), l2(after, field)
        if left > 0 and right > 0:
            values.append(abs(math.log10(right / left)))
    return statistics.median(values) if values else None


def save(fig, stem):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.png", dpi=220)
    fig.savefig(FIGURES / f"{stem}.pdf",
                metadata={"CreationDate": None, "ModDate": None})
    plt.close(fig)


def plot_constraints(curves):
    fig, axes = plt.subplots(2, 2, figsize=(13.6, 8.8), constrained_layout=True)
    labels = {
        "C-norm2": r"$\sqrt{\int C^2 dV}$",
        "H-norm2": r"$\sqrt{\int H^2 dV}$",
        "M-norm2": r"$\sqrt{\int M^2 dV}$",
        "Z-norm2": r"$\sqrt{\int Z^2 dV}$",
    }
    for ax, field in zip(axes.flat, CONSTRAINTS):
        for key, rows in curves.items():
            label, color, linestyle, alpha = STYLES[key]
            xx, yy = finite_xy(rows, "time", field, sqrt_y=True, positive=True)
            ax.semilogy(xx, yy, label=label, color=color, linestyle=linestyle,
                        alpha=alpha, linewidth=1.35)
        ax.set_title(labels[field])
        ax.set_xlabel(r"coordinate time $t/M$")
        ax.grid(alpha=0.22, which="both")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=4, fontsize=8)
    fig.suptitle("Brill A=-0.047: constraints before and after 2D restriction repair")
    save(fig, "constraints_pre_post_restriction")


def plot_topology(curves):
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 8.6), constrained_layout=True)
    for column, resolution in enumerate(("n128", "n256")):
        for phase, color, linestyle in (("pre", "#777777", "--"),
                                         ("post", "#2166ac" if resolution == "n128" else "#b2182b", "-")):
            key = f"{phase}_{resolution}"
            rows = curves[key]
            axes[0, column].step([row["time"] for row in rows],
                                 [row["nmb_total"] for row in rows], where="post",
                                 color=color, linestyle=linestyle,
                                 label=f"{phase}-fix MeshBlocks", linewidth=1.15)
            axes[1, column].semilogy([row["time"] for row in rows],
                                    [l2(row, "C-norm2") for row in rows],
                                    color=color, linestyle=linestyle,
                                    label=f"{phase}-fix C", linewidth=1.15)
            for event in topology_events(rows):
                axes[1, column].axvline(event["time"], color=color, alpha=0.055,
                                        linewidth=0.65)
        axes[0, column].set_title(f"{resolution.upper()}: AMR topology")
        axes[0, column].set_ylabel("MeshBlocks")
        axes[1, column].set_title(f"{resolution.upper()}: C diagnostic and topology events")
        axes[1, column].set_ylabel(r"$\sqrt{\int C^2 dV}$")
        for row in axes[:, column]:
            row.set_xlabel(r"coordinate time $t/M$")
            row.grid(alpha=0.2, which="both")
            row.legend(fontsize=8)
    fig.suptitle("Observed constraint discontinuities versus refinement topology changes")
    save(fig, "amr_topology_jump_comparison")


def plot_gauge_amr(curves):
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
        for key in ("post_n128", "post_n256"):
            label, color, linestyle, _ = STYLES[key]
            xx, yy = finite_xy(curves[key], "time", field, positive=log_y)
            (ax.semilogy if log_y else ax.plot)(
                xx, yy, label=label, color=color, linestyle=linestyle,
                linewidth=1.35)
        ax.set_title(title)
        ax.set_xlabel(r"coordinate time $t/M$")
        ax.grid(alpha=0.22, which="both")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=9)
    fig.suptitle("Repaired Brill runs: gauge, curvature, refinement, and timestep")
    save(fig, "repaired_gauge_amr_resolution_comparison")


def plot_figure3(curves):
    with (DATA / "figure3_published_curves.csv").open(newline="") as stream:
        paper = list(csv.DictReader(stream))
    paper_styles = {
        "bamps": ("BAMPS (paper rendering)", "#999999", ":"),
        "prague": ("Prague (paper rendering)", "#666666", "--"),
        "sphGR": ("sphGR (paper rendering)", "#222222", "-."),
    }
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.8), constrained_layout=True)
    for ax in axes:
        for key, (label, color, linestyle) in paper_styles.items():
            selected = [row for row in paper if row["series"] == key]
            ax.plot([float(row["tau"]) for row in selected],
                    [float(row["log10_abs_I"]) for row in selected],
                    label=label, color=color, linestyle=linestyle, linewidth=1.1)
        for key in ("post_n128", "post_n256"):
            label, color, linestyle, _ = STYLES[key]
            points = [(row["axisTau"], math.log10(abs(row["axisKret"])))
                      for row in curves[key] if row["axisKret"] != 0]
            ax.plot([point[0] for point in points], [point[1] for point in points],
                    label=label, color=color, linestyle=linestyle, linewidth=1.5)
            ax.scatter(*points[-1], color=color, marker="x", s=40, zorder=5)
        ax.set_xlim(0, 15.05)
        ax.set_xlabel(r"central proper time $\tau/M$")
        ax.set_ylabel(r"$\log_{10}|I|$ at the origin")
        ax.grid(alpha=0.23)
    axes[0].set_ylim(-7, 6)
    axes[0].set_title("Published Figure 3 range")
    axes[1].set_ylim(-7, 34)
    axes[1].set_title("Full finite AthenaK range")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle("Brill A=-0.047 after 2D restriction repair over paper Figure 3")
    save(fig, "figure3_repaired_resolution_overlay")


def write_summary(curves):
    cases = {}
    for key, rows in curves.items():
        events = topology_events(rows)
        cases[key] = {
            "history_sha256": sha256(PATHS[key]),
            "rows": len(rows),
            "terminal": {
                "time": rows[-1]["time"], "axisTau": rows[-1]["axisTau"],
                "cycle": int(rows[-1]["cycle"]),
                "maxRefLev": int(max(row["maxRefLev"] for row in rows)),
                "maxMeshBlocks": int(max(row["nmb_total"] for row in rows)),
            },
            "topology_event_count": len(events),
            "largest_topology_jump_events": sorted(
                events, key=lambda item: item["max_abs_log10_l2_factor"], reverse=True
            )[:12],
            "median_abs_log10_l2_increment": {
                field: {
                    "topology_change": median_log_increment(rows, field, True),
                    "stable_sample": median_log_increment(rows, field, False),
                } for field in ("C-norm2", "H-norm2", "M-norm2")
            },
        }
    paired = {}
    for resolution in ("n128", "n256"):
        pre_key, post_key = f"pre_{resolution}", f"post_{resolution}"
        reductions = {}
        for field in ("C-norm2", "H-norm2", "M-norm2"):
            pre = cases[pre_key]["median_abs_log10_l2_increment"][field]["topology_change"]
            post = cases[post_key]["median_abs_log10_l2_increment"][field]["topology_change"]
            reductions[field] = 1.0 - post / pre
        paired[resolution] = {
            "terminal_time_change_post_minus_pre": (
                cases[post_key]["terminal"]["time"] - cases[pre_key]["terminal"]["time"]
            ),
            "median_topology_log_increment_reduction_fraction": reductions,
        }
    summary = {
        "schema": "athenak_brill_r16_2d_restriction_fix_comparison_v1",
        "qualification_claim": False,
        "disposition": "restriction_inconsistency_fixed_but_late_instability_not_cured",
        "constraint_columns_are_squared_volume_integrals": True,
        "plotted_constraint_quantity": "sqrt(history column)",
        "source_commit": "345dd31d59cebd9c0c7231be43dcc6a72524bcc7",
        "cases": cases,
        "paired_comparison": paired,
    }
    (DATA / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n")


def main():
    curves = {key: read_csv(path) for key, path in PATHS.items()}
    plot_constraints(curves)
    plot_topology(curves)
    plot_gauge_amr(curves)
    plot_figure3(curves)
    write_summary(curves)


if __name__ == "__main__":
    main()
