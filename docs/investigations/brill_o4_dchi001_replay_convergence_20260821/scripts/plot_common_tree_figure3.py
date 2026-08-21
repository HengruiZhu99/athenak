#!/usr/bin/env python3
"""Overlay common-tree runs on the authenticated rendered Figure-3 curves."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


HEADER = re.compile(r"\[(\d+)\]=([^\s]+)")
COLORS = {"n128": "#377eb8", "n256": "#4daf4a", "n512": "#e41a1c"}
PAPER_STYLES = {
    "bamps": ("BAMPS (rendered paper curve)", "#aaaaaa", ":"),
    "prague": ("Prague (rendered paper curve)", "#777777", "--"),
    "sphGR": ("sphGR (rendered paper curve)", "#444444", "-."),
}


def read_history(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as stream:
        first = stream.readline()
    if "," in first and not first.startswith("#"):
        with path.open(newline="", encoding="utf-8") as stream:
            rows = [{name: float(value) for name, value in row.items()
                     if value not in (None, "")}
                    for row in csv.DictReader(stream)]
        if not rows or not {"axisTau", "axisKret"} <= rows[0].keys():
            raise RuntimeError(f"invalid history CSV: {path}")
        return rows
    labels: dict[str, int] = {}
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            labels.update({name: int(index) - 1 for index, name in HEADER.findall(line)})
        elif line.strip():
            values = [float(value) for value in line.split()]
            rows.append({name: values[index] for name, index in labels.items()})
    if not rows or not {"axisTau", "axisKret"} <= labels.keys():
        raise RuntimeError(f"invalid history: {path}")
    return rows


def read_reference(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or not {"series", "tau", "log10_abs_I"} <= rows[0].keys():
        raise RuntimeError(f"invalid Figure-3 reference: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    for case in ("n128", "n256", "n512"):
        parser.add_argument(f"--{case}", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--secondary", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    histories = {case: read_history(getattr(args, case)) for case in COLORS}
    secondary = read_history(args.secondary) if args.secondary else []
    reference = read_reference(args.reference)
    args.output.mkdir(parents=True, exist_ok=True)
    plotted = []
    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.8), constrained_layout=True)
    for axis in axes:
        for series, (label, color, linestyle) in PAPER_STYLES.items():
            selected = [row for row in reference if row["series"] == series]
            xx = [float(row["tau"]) for row in selected]
            yy = [float(row["log10_abs_I"]) for row in selected]
            axis.plot(xx, yy, label=label, color=color, linestyle=linestyle, linewidth=1.2)
        for case, rows in histories.items():
            points = [(row["axisTau"], math.log10(abs(row["axisKret"])))
                      for row in rows if row["axisKret"] != 0.0]
            axis.plot([point[0] for point in points], [point[1] for point in points],
                      color=COLORS[case], label=case.upper(), linewidth=1.45)
            axis.scatter(points[-1][0], points[-1][1], color=COLORS[case], marker="x", s=28)
        if secondary:
            points = [(row["axisTau"], math.log10(abs(row["axisKret"])))
                      for row in secondary if row["axisKret"] != 0.0]
            axis.plot([point[0] for point in points], [point[1] for point in points],
                      color="#984ea3", linestyle=":", linewidth=1.1,
                      label="prior N256 O4 dchi=0.02 (unmatched context)")
        axis.set_xlim(0.0, 15.05)
        axis.set_xlabel(r"central proper time $\tau_c/M$")
        axis.set_ylabel(r"$\log_{10}|I|$ on the symmetry axis")
        axis.grid(alpha=0.23)
    axes[0].set_ylim(-7.0, 16.5)
    axes[0].set_title("Published Figure-3 scale")
    axes[1].set_ylim(-7.0, 21.0)
    axes[1].set_title("Common-tree runs: complete finite range")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle("Brill A=-0.047: symmetric-O4 common-hierarchy comparison")
    figure_path = args.output / "fig3_o4_common_tree_n128_n256_n512.png"
    fig.savefig(figure_path, dpi=220)
    plt.close(fig)

    for row in reference:
        plotted.append({"source": "rendered_paper", "series": row["series"],
                        "tau": row["tau"], "log10_abs_I": row["log10_abs_I"]})
    for case, rows in histories.items():
        plotted.extend({"source": "athenak_history", "series": case,
                        "tau": row["axisTau"],
                        "log10_abs_I": math.log10(abs(row["axisKret"]))}
                       for row in rows if row["axisKret"] != 0.0)
    plotted.extend({"source": "secondary_unmatched_context", "series": "n256_dchi002_prior",
                    "tau": row["axisTau"],
                    "log10_abs_I": math.log10(abs(row["axisKret"]))}
                   for row in secondary if row["axisKret"] != 0.0)
    with (args.output / "figure3_plotted_data.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("source", "series", "tau", "log10_abs_I"),
                                lineterminator="\n")
        writer.writeheader(); writer.writerows(plotted)
    print("COMMON_TREE_FIGURE3_OVERLAY_PASS")


if __name__ == "__main__":
    main()
