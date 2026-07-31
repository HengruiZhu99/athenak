#!/usr/bin/env python3
"""Plot the matched thin-z Sommerfeld and zero-rate CPBC TDE runs."""

import argparse
import math
import pathlib
import re
import types
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
from matplotlib.cm import ScalarMappable
import numpy as np

import compare_tde_boundaries as comparator


SOMMERFELD_COLOR = "#D55E00"
CPBC_COLOR = "#0072B2"
GRID_COLOR = "#B8B8B8"
HISTORY_SUFFIXES = {
    "problem": ".user.hst",
    "constraint": ".z4c.user.hst",
}


def history_path(run, kind):
    suffix = HISTORY_SUFFIXES[kind]
    candidates = [
        path for path in sorted(run.glob("*{}".format(suffix)))
        if kind != "problem" or not path.name.endswith(".z4c.user.hst")
    ]
    if len(candidates) != 1:
        raise SystemExit(
            "{}: expected one {} history, found {}".format(
                run, kind, len(candidates)))
    return candidates[0]


def read_history(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise SystemExit("{}: incomplete history".format(path))
    labels = {
        match.group(2): int(match.group(1)) - 1
        for match in re.finditer(r"\[(\d+)\]=(\S+)", lines[1])
    }
    rows = [
        [float(value) for value in line.split()]
        for line in lines[2:]
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rows or "time" not in labels:
        raise SystemExit("{}: missing history data or time".format(path))
    return labels, np.asarray(rows, dtype=np.float64)


def history_series(run, kind, label):
    labels, data = read_history(history_path(run, kind))
    if label not in labels:
        raise SystemExit("{}: missing history label {}".format(run, label))
    time = data[:, labels["time"]]
    value = data[:, labels[label]]
    finite_time = np.isfinite(time)
    time = time[finite_time]
    value = value[finite_time]
    order = np.argsort(time, kind="stable")
    time = time[order]
    value = value[order]
    if len(time) == 0:
        raise SystemExit("{}: no finite history times".format(run))

    # Restart boundaries repeat their final state in the next segment.
    unique_time = []
    unique_value = []
    for current_time, current_value in zip(time, value):
        if unique_time and abs(current_time - unique_time[-1]) <= 1.0e-12:
            unique_time[-1] = current_time
            unique_value[-1] = current_value
        else:
            unique_time.append(current_time)
            unique_value.append(current_value)
    return (
        np.asarray(unique_time, dtype=np.float64),
        np.asarray(unique_value, dtype=np.float64),
    )


def first_nonfinite_time(run):
    candidates = []
    for kind in HISTORY_SUFFIXES:
        labels, data = read_history(history_path(run, kind))
        time = data[:, labels["time"]]
        bad = np.isfinite(time) & ~np.all(np.isfinite(data), axis=1)
        candidates.extend(float(value) for value in time[bad])
    return min(candidates) if candidates else math.nan


def configure_style():
    # Matplotlib 3.3/NumPy 1.19 can probe masked symlog values outside the
    # represented color range while rendering.  The resulting overflow is
    # masked and does not affect plotted data or color limits.
    warnings.filterwarnings(
        "ignore", message="overflow encountered in power",
        category=RuntimeWarning)
    plt.rcParams.update({
        "axes.grid": True,
        "axes.linewidth": 0.8,
        "axes.titleweight": "bold",
        "figure.dpi": 120,
        "font.size": 10,
        "grid.alpha": 0.35,
        "grid.color": GRID_COLOR,
        "grid.linewidth": 0.6,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "savefig.bbox": "tight",
    })


def save_figure(figure, output_dir, stem, formats):
    outputs = []
    for file_format in formats:
        path = output_dir / "{}.{}".format(stem, file_format)
        metadata = {
            "Creator": "plot_tde_boundary_comparison.py",
            "Title": stem.replace("_", " "),
        }
        with np.errstate(over="ignore"):
            figure.savefig(
                path, format=file_format, dpi=220,
                metadata=metadata if file_format == "pdf" else None)
        outputs.append(path)
    plt.close(figure)
    return outputs


def positive_finite(time, value, maximum_time=30.0):
    mask = (
        np.isfinite(time)
        & np.isfinite(value)
        & (value > 0.0)
        & (time >= 0.0)
        & (time <= maximum_time + 1.0e-12)
    )
    return time[mask], value[mask]


def add_failure_marker(axis, failure_time):
    if math.isfinite(failure_time):
        axis.axvline(
            failure_time, color=SOMMERFELD_COLOR, linestyle=":", linewidth=1.2)


def plot_constraint_histories(sommerfeld, cpbc, output_dir, formats):
    failure_time = first_nonfinite_time(sommerfeld)
    panels = (
        ("H-norm2", "Hamiltonian constraint", r"$\|H\|_2$"),
        ("M-norm2", "Momentum constraint", r"$\|M\|_2$"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharex=True)
    for axis, (label, title, ylabel) in zip(axes, panels):
        for run, name, color in (
            (sommerfeld, "Sommerfeld", SOMMERFELD_COLOR),
            (cpbc, "zero-rate CPBC", CPBC_COLOR),
        ):
            time, value = history_series(run, "constraint", label)
            maximum_time = (
                failure_time if run == sommerfeld else 30.0)
            time, value = positive_finite(
                time, value, maximum_time=maximum_time)
            axis.semilogy(time, value, label=name, color=color)
        add_failure_marker(axis, failure_time)
        axis.set_title(title)
        axis.set_xlabel(r"$t/M$")
        axis.set_ylabel(ylabel)
        axis.set_xlim(0.0, 30.0)
    axes[0].legend(loc="upper left")
    if math.isfinite(failure_time):
        axes[1].annotate(
            "Sommerfeld first nonfinite\n$t/M={:.1f}$".format(failure_time),
            xy=(failure_time, 1.0), xycoords=("data", "axes fraction"),
            xytext=(8.2, 0.84), textcoords=("data", "axes fraction"),
            arrowprops={"arrowstyle": "->", "color": SOMMERFELD_COLOR},
            color=SOMMERFELD_COLOR, ha="left", va="top",
        )
    figure.suptitle("Thin-z no-sponge TDE constraint evolution")
    figure.tight_layout()
    return save_figure(
        figure, output_dir, "tde_boundary_constraint_histories", formats)


def plot_residual_histories(sommerfeld, cpbc, output_dir, formats):
    failure_time = first_nonfinite_time(sommerfeld)
    panels = (
        ("Theta-max", r"$\max|\Theta|$", (1.0e-7, 1.0e-1)),
        ("alpha-res", r"$\max|\delta\alpha|$", (1.0e-6, 1.0e-1)),
        ("beta-res", r"$\max|\delta\beta^i|$", (1.0e-7, 1.0e-2)),
        ("Gam-res", r"$\max|\delta\tilde{\Gamma}^i|$", (1.0e-7, 3.0e-2)),
    )
    figure, axes = plt.subplots(
        2, 2, figsize=(10.2, 7.1), sharex=True)
    for axis, (label, ylabel, limits) in zip(axes.ravel(), panels):
        for run, name, color in (
            (sommerfeld, "Sommerfeld", SOMMERFELD_COLOR),
            (cpbc, "zero-rate CPBC", CPBC_COLOR),
        ):
            time, value = history_series(run, "problem", label)
            maximum_time = (
                failure_time if run == sommerfeld else 30.0)
            time, value = positive_finite(
                time, value, maximum_time=maximum_time)
            axis.semilogy(time, value, label=name, color=color)
        add_failure_marker(axis, failure_time)
        axis.set_ylabel(ylabel)
        axis.set_xlim(0.0, 30.0)
        axis.set_ylim(*limits)
    axes[0, 0].legend(loc="upper left")
    axes[1, 0].set_xlabel(r"$t/M$")
    axes[1, 1].set_xlabel(r"$t/M$")
    figure.suptitle("Thin-z no-sponge TDE gauge and constraint residual maxima")
    figure.tight_layout()
    return save_figure(
        figure, output_dir, "tde_boundary_residual_histories", formats)


def closest_slice(run, target_time):
    arguments = types.SimpleNamespace(
        slice_id="xz_z4c",
        slice_plane="xz",
        slice_fixed_coordinate=0.0,
    )
    time, path = min(
        comparator.slice_series(run, arguments),
        key=lambda item: abs(item[0] - target_time),
    )
    if abs(time - target_time) > 5.0e-3:
        raise SystemExit(
            "{}: closest x-z slice is at {}, requested {}".format(
                run, time, target_time))
    measured_time, blocks = comparator.read_slice_binary(path, arguments)
    return measured_time, blocks


def plot_theta_slice(sommerfeld, cpbc, output_dir, formats, target_time):
    cases = []
    maximum = 0.0
    for run, label in (
        (sommerfeld, "Sommerfeld"),
        (cpbc, "zero-rate CPBC"),
    ):
        time, blocks = closest_slice(run, target_time)
        track_time, track_position = comparator.load_track(run)[:2]
        star_position = tuple(
            float(np.interp(time, track_time, track_position[:, component]))
            for component in range(3)
        )
        values = np.concatenate([
            np.asarray(block["fields"]["z4c_Theta"]).ravel()
            for block in blocks
        ])
        maximum = max(maximum, float(np.max(np.abs(values))))
        cases.append((label, time, blocks, star_position))
    if maximum <= 0.0 or not math.isfinite(maximum):
        raise SystemExit("invalid common Theta color scale")

    norm = SymLogNorm(
        linthresh=max(1.0e-8, maximum * 1.0e-4),
        linscale=0.8, vmin=-maximum, vmax=maximum, base=10)
    figure, axes = plt.subplots(
        1, 2, figsize=(11.0, 4.2), sharex=True, sharey=True)
    for axis, (label, time, blocks, star_position) in zip(axes, cases):
        for block in sorted(blocks, key=lambda item: item["level"]):
            axis.imshow(
                block["fields"]["z4c_Theta"],
                extent=(
                    block["column_min"], block["column_max"],
                    block["row_min"], block["row_max"],
                ),
                origin="lower", aspect="auto", interpolation="nearest",
                cmap="RdBu_r", norm=norm,
            )
        axis.scatter(
            [0.0], [0.0], marker="o", s=22, facecolors="none",
            edgecolors="black", linewidths=0.9, label="black hole")
        axis.scatter(
            [star_position[0]], [star_position[2]], marker="*", s=45,
            color="black", label="tracked star")
        axis.set_title("{} at $t/M={:.5f}$".format(label, time))
        axis.set_xlabel(r"$x/M$")
        axis.set_xlim(-48.0, 72.0)
        axis.set_ylim(-6.0, 42.0)
        axis.grid(False)
    axes[0].set_ylabel(r"$z/M$")
    axes[0].annotate(
        "close physical boundary",
        xy=(0.0, -6.0), xytext=(-35.0, 3.5),
        arrowprops={"arrowstyle": "->", "color": "black"},
        ha="left", va="bottom",
    )
    axes[1].legend(loc="upper right")
    # Matplotlib 3.3/NumPy 1.19 can probe masked symlog tick values outside
    # the represented range while constructing the colorbar.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message="overflow encountered in power",
            category=RuntimeWarning)
        colorbar = figure.colorbar(
            ScalarMappable(norm=norm, cmap="RdBu_r"),
            ax=axes, fraction=0.025, pad=0.025)
    colorbar.set_label(r"residual $\Theta$")
    figure.suptitle("Matched x-z residual slice on a common symmetric-log scale")
    figure.subplots_adjust(
        left=0.08, right=0.90, bottom=0.14, top=0.84, wspace=0.08)
    return save_figure(
        figure, output_dir, "tde_boundary_theta_slice_t7", formats)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sommerfeld", type=pathlib.Path, required=True)
    parser.add_argument("--cpbc", type=pathlib.Path, required=True)
    parser.add_argument("--output-dir", type=pathlib.Path, required=True)
    parser.add_argument("--slice-time", type=float, default=7.00313)
    parser.add_argument(
        "--formats", nargs="+", choices=("pdf", "png"), default=("pdf",))
    arguments = parser.parse_args()
    if (
        not arguments.sommerfeld.is_dir()
        or not arguments.cpbc.is_dir()
        or not math.isfinite(arguments.slice_time)
    ):
        raise SystemExit("invalid run directory or slice time")
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    outputs = []
    outputs.extend(plot_constraint_histories(
        arguments.sommerfeld, arguments.cpbc,
        arguments.output_dir, arguments.formats))
    outputs.extend(plot_residual_histories(
        arguments.sommerfeld, arguments.cpbc,
        arguments.output_dir, arguments.formats))
    outputs.extend(plot_theta_slice(
        arguments.sommerfeld, arguments.cpbc,
        arguments.output_dir, arguments.formats, arguments.slice_time))
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
