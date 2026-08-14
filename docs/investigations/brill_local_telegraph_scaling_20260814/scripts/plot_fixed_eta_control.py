#!/usr/bin/env python3
"""Build the reduced fixed-eta history and the Figure-3 control overlay.

The default mode reads only committed reduced CSV files.  Passing
``--import-fixed-history`` refreshes the reduced fixed-eta CSV from an AthenaK
history file before plotting.  This keeps the Git artifact reproducible without
shipping the roughly one-megabyte native history table.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
FIGURES = ROOT / "figures"
REDUCED_FIXED = DATA / "fixed_shift_eta2_history.csv"

REDUCED_COLUMNS = (
    "case",
    "time",
    "dt",
    "C-norm2",
    "H-norm2",
    "M-norm2",
    "Z-norm2",
    "max_abs_K",
    "nmb_total",
    "maxRefLev",
    "cycle",
    "axisLapse",
    "axisTau",
    "axisKret",
    "maxAbsKret",
    "eta_shift",
    "S_beta",
)


def parse_history(path: Path) -> tuple[dict[str, int], list[list[float]]]:
    labels: dict[str, int] = {}
    rows: list[list[float]] = []
    pattern = re.compile(r"\[(\d+)\]=([^\s]+)")
    for line in path.read_text().splitlines():
        if line.startswith("#"):
            for index, name in pattern.findall(line):
                labels[name] = int(index) - 1
            continue
        if not line.strip():
            continue
        values = [float(value) for value in line.split()]
        if all(math.isfinite(value) for value in values):
            rows.append(values)
    required = set(REDUCED_COLUMNS) - {"case", "eta_shift", "S_beta"}
    missing = required - labels.keys()
    if missing:
        raise RuntimeError(f"missing history labels in {path}: {sorted(missing)}")
    if not rows:
        raise RuntimeError(f"no finite history rows in {path}")
    return labels, rows


def import_fixed_history(path: Path) -> None:
    labels, rows = parse_history(path)
    DATA.mkdir(parents=True, exist_ok=True)
    with REDUCED_FIXED.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=REDUCED_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            dt = row[labels["dt"]]
            record: dict[str, str | float] = {
                "case": "fixed_eta2",
                "eta_shift": 2.0,
                "S_beta": 2.0 * dt,
            }
            for name in REDUCED_COLUMNS:
                if name not in record:
                    record[name] = row[labels[name]]
            writer.writerow(record)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def finite_xy(
    rows: list[dict[str, str]], x_name: str, y_name: str
) -> tuple[list[float], list[float]]:
    x_values: list[float] = []
    y_values: list[float] = []
    for row in rows:
        x_value = float(row[x_name])
        y_value = float(row[y_name])
        if math.isfinite(x_value) and math.isfinite(y_value) and y_value != 0.0:
            x_values.append(x_value)
            y_values.append(math.log10(abs(y_value)))
    return x_values, y_values


def plot_overlay() -> None:
    published_rows = read_csv(DATA / "figure3_published_curves.csv")
    baseline_rows = [
        row
        for row in read_csv(DATA / "history_curves.csv")
        if row["prescription"] == "max_domain_abs_K"
    ]
    fixed_rows = read_csv(REDUCED_FIXED)
    if not baseline_rows or not fixed_rows:
        raise RuntimeError("baseline or fixed-eta reduced history is empty")

    paper_style = {
        "bamps": ("BAMPS (paper, rendered)", "#aaaaaa", ":"),
        "prague": ("Prague (paper, rendered)", "#777777", "--"),
        "sphGR": ("sphGR (paper, rendered)", "#444444", "-."),
    }
    published: dict[str, tuple[list[float], list[float]]] = {}
    for name in paper_style:
        selected = [row for row in published_rows if row["series"] == name]
        published[name] = (
            [float(row["tau"]) for row in selected],
            [float(row["log10_abs_I"]) for row in selected],
        )

    baseline_x, baseline_y = finite_xy(baseline_rows, "axisTau", "axisKret")
    fixed_x, fixed_y = finite_xy(fixed_rows, "axisTau", "axisKret")

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 5.65), constrained_layout=True)
    for ax in axes:
        for name, (label, color, linestyle) in paper_style.items():
            ax.plot(
                *published[name],
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=1.25,
                alpha=0.85,
            )
        ax.plot(
            baseline_x,
            baseline_y,
            color="#c94f49",
            linewidth=1.8,
            label=r"AthenaK: $\eta=2\max|K|$",
        )
        ax.plot(
            fixed_x,
            fixed_y,
            color="#436fad",
            linewidth=1.8,
            label=r"AthenaK control: fixed $\eta=2$",
        )
        ax.axvline(
            baseline_x[-1], color="#c94f49", linestyle=":", linewidth=1.3, alpha=0.8
        )
        ax.axvline(
            fixed_x[-1], color="#436fad", linestyle=":", linewidth=1.3, alpha=0.8
        )
        ax.set_xlabel(r"central proper time $\tau/M$")
        ax.set_ylabel(r"$\log_{10}|I|$ on the symmetry axis")
        ax.grid(alpha=0.24)
    axes[0].set_xlim(0.0, 15.05)
    axes[0].set_ylim(-7.0, 16.5)
    axes[0].set_title("Published Figure-3 interval")
    axes[1].set_xlim(0.0, 10.55)
    axes[1].set_ylim(-4.0, 16.5)
    axes[1].set_title("AthenaK diagnostic interval")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=8)
    fig.suptitle("Brill A=-0.047: fixed-shift control on the Figure-3 axes")
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / "figure3_fixed_shift_eta2_overlay.png", dpi=200)
    fig.savefig(
        FIGURES / "figure3_fixed_shift_eta2_overlay.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--import-fixed-history",
        type=Path,
        help="AthenaK fixed-eta history file to reduce before plotting",
    )
    args = parser.parse_args()
    if args.import_fixed_history is not None:
        import_fixed_history(args.import_fixed_history)
    plot_overlay()


if __name__ == "__main__":
    main()
