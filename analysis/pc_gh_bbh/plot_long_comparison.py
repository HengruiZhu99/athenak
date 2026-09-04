#!/usr/bin/env python3
"""Plot the long-domain CUDA Z4c/PC-GH head-on BBH comparison.

The script expects one AthenaK run directory per formulation.  It compares the
puncture tracks, common ADM constraints, and the (2,2) waveform at every shared
extraction radius.  Constraint panels mark mesh-change times parsed from all
segment logs so apparent spikes can be checked against AMR operations.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vis" / "python"))
import bin_convert  # noqa: E402


HISTORY_COLUMN = re.compile(r"\[(\d+)\]=(\S+)")
WAVE_COLUMN = re.compile(r"(\d+):([^\s]+)")
AMR_CHANGE = re.compile(
    r"AMR_CHANGE\s+cycle=(\d+)\s+time=([+\-\d.eE]+)\s+"
    r"created=(\d+)\s+deleted=(\d+)\s+blocks=(\d+)"
)
COLORS = {"Z4c": "#0072B2", "PC-GH": "#D55E00"}
RADII = (8, 12, 24, 32, 48, 56)


def merge_restart_segments(data: np.ndarray, time_column: int) -> np.ndarray:
    """Merge appended output segments, preferring data from the latest restart.

    AthenaK appends to text outputs after a restart.  The new segment starts at
    the checkpoint time, so it can overlap several samples from the preceding
    segment rather than merely repeating one timestamp.  Split at every
    non-increasing time and replace the overlapping tail with the newer data.
    """
    times = data[:, time_column]
    starts = np.flatnonzero(np.diff(times) <= 0.0) + 1
    segments = np.split(data, starts)
    merged = segments[0]
    for segment in segments[1:]:
        if segment.size == 0:
            continue
        restart_time = segment[0, time_column]
        merged = merged[merged[:, time_column] < restart_time]
        merged = np.vstack((merged, segment))
    if np.any(np.diff(merged[:, time_column]) <= 0.0):
        raise ValueError("restart-merged times are not strictly increasing")
    return merged


def load_table(path: Path, pattern: re.Pattern[str]) -> dict[str, np.ndarray]:
    """Read an AthenaK text table with numbered columns in the header."""
    header_lines = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.startswith("#"):
                break
            header_lines.append(line)
    columns = {
        name: int(number) - 1
        for number, name in pattern.findall(" ".join(header_lines))
    }
    if not columns:
        raise ValueError(f"no numbered columns found in {path}")
    data = np.atleast_2d(np.loadtxt(path))
    if not np.all(np.isfinite(data)):
        raise ValueError(f"non-finite value in {path}")
    if "time" in columns:
        try:
            data = merge_restart_segments(data, columns["time"])
        except ValueError as error:
            raise ValueError(f"invalid restart chronology in {path}") from error
    return {name: data[:, column] for name, column in columns.items()}


def only_path(run_dir: Path, pattern: str) -> Path:
    paths = sorted(run_dir.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"expected one {pattern!r} in {run_dir}, found {paths}")
    return paths[0]


def load_history(run_dir: Path, formulation: str) -> dict[str, np.ndarray]:
    suffix = "*.z4c.user.hst" if formulation == "Z4c" else "*.pcgh.hst"
    return load_table(only_path(run_dir, suffix), HISTORY_COLUMN)


def load_tracker(path: Path) -> dict[str, np.ndarray]:
    data = np.atleast_2d(np.loadtxt(path))
    if data.shape[1] != 8 or not np.all(np.isfinite(data)):
        raise ValueError(f"invalid compact-object tracker table {path}")
    try:
        data = merge_restart_segments(data, 1)
    except ValueError as error:
        raise ValueError(f"invalid restart chronology in {path}") from error
    names = ("cycle", "time", "x", "y", "z", "vx", "vy", "vz")
    return {name: data[:, column] for column, name in enumerate(names)}


def load_wave(run_dir: Path, radius: int) -> dict[str, np.ndarray]:
    wave_dir = run_dir / "waveforms"
    real_path = wave_dir / f"rpsi4_real_{radius:04d}.txt"
    imag_path = wave_dir / f"rpsi4_imag_{radius:04d}.txt"
    real = load_table(real_path, WAVE_COLUMN)
    imag = load_table(imag_path, WAVE_COLUMN)
    if not np.allclose(real["time"], imag["time"], rtol=0.0, atol=1.0e-12):
        raise ValueError(f"real/imaginary times differ at r={radius} in {run_dir}")
    return {
        "time": real["time"],
        "retarded_time": real["time"] - radius,
        "real_22": real["22"],
        "imag_22": imag["22"],
    }


def slice_lapse_minima(run_dir: Path, formulation: str
                       ) -> dict[int, dict[str, np.ndarray]]:
    """Locate each puncture from the lapse minimum in every saved xy slice."""
    stem = "z4c" if formulation == "Z4c" else "pcgh"
    paths = sorted((run_dir / "bin").glob(f"*.{stem}.[0-9]*.bin"))
    if not paths:
        raise ValueError(f"no {formulation} state slices found in {run_dir / 'bin'}")
    samples: dict[int, list[dict[str, float]]] = {0: [], 1: []}
    for path in paths:
        output = bin_convert.read_binary(path)
        if formulation == "Z4c":
            lapse = np.asarray(output["mb_data"]["z4c_alpha"])
        else:
            lapse = (np.asarray(output["mb_data"]["pcgh_w"])
                     * np.asarray(output["mb_data"]["pcgh_rho"]))
        if not np.all(np.isfinite(lapse)):
            raise ValueError(f"non-finite lapse in {path}")

        best: dict[int, dict[str, float] | None] = {0: None, 1: None}
        meshblock_cells = np.asarray([
            output["nx1_mb"], output["nx2_mb"], output["nx3_mb"]])
        for block, geometry in enumerate(output["mb_geometry"]):
            bounds = np.asarray(geometry).reshape(3, 2)
            spacing = (bounds[:, 1] - bounds[:, 0])/meshblock_cells
            offsets = np.asarray(output["mb_index"][block])[[0, 2, 4]]
            shape = lapse[block].shape[::-1]
            coordinates = [
                bounds[axis, 0] + (offsets[axis] + np.arange(shape[axis]) + 0.5)
                * spacing[axis]
                for axis in range(3)
            ]
            x, y, z = np.meshgrid(*coordinates, indexing="ij")
            block_lapse = np.transpose(lapse[block], (2, 1, 0))
            for puncture, sign in ((0, 1.0), (1, -1.0)):
                masked = np.where(sign*x > 0.0, block_lapse, np.inf)
                flat_index = int(np.argmin(masked))
                minimum = float(masked.flat[flat_index])
                if not np.isfinite(minimum):
                    continue
                index = np.unravel_index(flat_index, masked.shape)
                candidate = {
                    "time": float(output["time"]),
                    "x": float(x[index]), "y": float(y[index]), "z": float(z[index]),
                    "min_alpha": minimum, "dx": float(spacing[0]),
                }
                if best[puncture] is None or minimum < best[puncture]["min_alpha"]:
                    best[puncture] = candidate
        for puncture in (0, 1):
            if best[puncture] is None:
                raise ValueError(f"could not locate puncture {puncture} in {path}")
            samples[puncture].append(best[puncture])

    result = {}
    for puncture, rows in samples.items():
        rows.sort(key=lambda row: row["time"])
        result[puncture] = {
            key: np.asarray([row[key] for row in rows]) for key in rows[0]
        }
    return result


def rms(numerator: np.ndarray, volume: np.ndarray) -> np.ndarray:
    quotient = np.full_like(np.asarray(numerator, dtype=float), np.nan)
    np.divide(numerator, volume, out=quotient, where=volume > 0.0)
    return np.sqrt(np.maximum(quotient, 0.0))


def constraint_series(history: dict[str, np.ndarray], formulation: str
                      ) -> dict[str, np.ndarray]:
    if formulation == "Z4c":
        volume = history["Volume"]
        hamiltonian = rms(history["H-norm2"], volume)
        momentum = rms(history["M-norm2"], volume)
    else:
        volume = history["shared-Vol"]
        hamiltonian = rms(history["shared-H-n"], volume)
        momentum = rms(history["shared-M-n"], volume)
    return {"time": history["time"], "H": hamiltonian, "M": momentum}


def load_amr_changes(run_dir: Path) -> list[dict[str, float | int]]:
    changes: dict[tuple[int, float], dict[str, float | int]] = {}
    for log_path in sorted(run_dir.glob("*segment-*.log")):
        text = log_path.read_text(encoding="utf-8", errors="replace")
        for match in AMR_CHANGE.finditer(text):
            cycle, time, created, deleted, blocks = match.groups()
            event = {
                "cycle": int(cycle), "time": float(time),
                "created": int(created), "deleted": int(deleted),
                "blocks": int(blocks),
            }
            changes[(event["cycle"], event["time"])] = event
    return sorted(changes.values(), key=lambda event: float(event["time"]))


def draw_amr_times(axis: plt.Axes, changes: list[dict[str, float | int]]) -> None:
    for event in changes:
        axis.axvline(float(event["time"]), color="0.45", alpha=0.15,
                     linewidth=0.7, zorder=0)


def constraint_change_summary(
        series: dict[str, np.ndarray],
        changes: list[dict[str, float | int]],
        count: int = 12) -> dict[str, list[dict[str, object]]]:
    """Rank adjacent constraint changes and identify intervening AMR events."""
    times = series["time"]
    result: dict[str, list[dict[str, object]]] = {}
    for constraint in ("H", "M"):
        values = series[constraint]
        entries = []
        for index in range(1, len(times)):
            before = float(values[index - 1])
            after = float(values[index])
            if before <= 0.0 or after <= 0.0:
                continue
            ratio = after / before
            factor = max(ratio, 1.0 / ratio)
            lower = float(times[index - 1])
            upper = float(times[index])
            bracketed = [event for event in changes
                         if lower < float(event["time"]) <= upper]
            entries.append({
                "time_before": lower,
                "time_after": upper,
                "value_before": before,
                "value_after": after,
                "signed_ratio": ratio,
                "absolute_change_factor": factor,
                "amr_events_between_samples": bracketed,
            })
        entries.sort(key=lambda entry: float(entry["absolute_change_factor"]),
                     reverse=True)
        result[constraint] = entries[:count]
    return result


def constraint_peak_grid_phase(series: dict[str, np.ndarray],
                               tracker: dict[str, np.ndarray],
                               finest_dx: float = 1.0/16.0,
                               x_window: tuple[float, float] = (0.75, 1.75)) -> dict:
    """Relate local Hamiltonian peaks to puncture motion across finest cells."""
    values = series["H"]
    peak_indices = np.flatnonzero(
        (values[1:-1] > values[:-2]) & (values[1:-1] >= values[2:])) + 1
    entries = []
    for index in peak_indices:
        time = float(series["time"][index])
        puncture_x = abs(float(np.interp(
            time, tracker["time"], tracker["x"])))
        if not x_window[0] <= puncture_x <= x_window[1]:
            continue
        entries.append({
            "time": time,
            "H": float(values[index]),
            "abs_puncture_x": puncture_x,
        })
    cell_displacements = []
    for previous, current in zip(entries, entries[1:]):
        displacement = ((previous["abs_puncture_x"] - current["abs_puncture_x"])
                        / finest_dx)
        current["displacement_since_previous_peak_in_finest_cells"] = displacement
        cell_displacements.append(displacement)
    return {
        "finest_dx": finest_dx,
        "puncture_abs_x_window": list(x_window),
        "peak_count": len(entries),
        "median_peak_spacing_in_finest_cells": (
            float(np.median(cell_displacements)) if cell_displacements else float("nan")),
        "mean_peak_spacing_in_finest_cells": (
            float(np.mean(cell_displacements)) if cell_displacements else float("nan")),
        "entries": entries,
    }


def plot_constraints(series: dict[str, dict[str, np.ndarray]],
                     changes: dict[str, list[dict[str, float | int]]],
                     output_dir: Path) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    for row, constraint in enumerate(("H", "M")):
        axis = axes[row]
        for formulation in ("PC-GH", "Z4c"):
            values = np.where(series[formulation][constraint] > 0.0,
                              series[formulation][constraint], np.nan)
            width = 2.4 if formulation == "PC-GH" else 1.4
            zorder = 2 if formulation == "PC-GH" else 3
            axis.semilogy(series[formulation]["time"], values,
                          color=COLORS[formulation], linewidth=width,
                          label=formulation, zorder=zorder)
        all_changes = {
            (int(event["cycle"]), float(event["time"])): event
            for events in changes.values() for event in events
        }
        draw_amr_times(axis, list(all_changes.values()))
        axis.set_ylabel(f"{constraint} proper-volume RMS")
        axis.grid(alpha=0.25, which="both")
        axis.legend()
    axes[-1].set_xlabel(r"$t/M$")
    fig.suptitle("Head-on BBH constraints; gray lines are AMR changes")
    fig.tight_layout()
    path = output_dir / "constraint_overlay_with_amr.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_trajectories(run_dirs: dict[str, Path], output_dir: Path) -> tuple[Path, dict]:
    z4c_tracks = [
        load_tracker(only_path(run_dirs["Z4c"], f"*.co_{index}.txt"))
        for index in (0, 1)
    ]
    slices = {
        formulation: slice_lapse_minima(run_dir, formulation)
        for formulation, run_dir in run_dirs.items()
    }
    csv_path = output_dir / "puncture_trajectory_from_slice_minima.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("formulation", "puncture", "time", "x", "y", "z",
                         "min_alpha", "dx"))
        for formulation in ("PC-GH", "Z4c"):
            for puncture in (0, 1):
                track = slices[formulation][puncture]
                for row in zip(*(track[key] for key in (
                        "time", "x", "y", "z", "min_alpha", "dx"))):
                    writer.writerow((formulation, puncture, *row))

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    for index, linestyle in enumerate(("-", "--")):
        pcgh = slices["PC-GH"][index]
        axes[0].plot(pcgh["time"], pcgh["x"], color=COLORS["PC-GH"],
                     linestyle=linestyle, marker="o", markersize=3.5,
                     linewidth=2.8, zorder=2,
                     label=f"PC-GH slice minimum, puncture {index}")
        z4c = z4c_tracks[index]
        axes[0].plot(z4c["time"], z4c["x"], color=COLORS["Z4c"],
                     linestyle=linestyle, linewidth=1.4, zorder=3,
                     label=f"Z4c ODE tracker, puncture {index}")
        z4c_slice = slices["Z4c"][index]
        axes[0].plot(z4c_slice["time"], z4c_slice["x"], linestyle="none",
                     marker="x", markersize=4.0, color=COLORS["Z4c"], zorder=4)

        common_time = pcgh["time"]
        z4c_x = np.interp(common_time, z4c_slice["time"], z4c_slice["x"])
        axes[1].plot(common_time, np.abs(pcgh["x"] - z4c_x),
                     color="0.2", linestyle=linestyle, marker="o", markersize=3.0,
                     label=f"puncture {index}")
    axes[0].set_ylabel(r"$x/M$")
    axes[1].set_ylabel(r"$|x_{\rm PCGH,min}-x_{\rm Z4c,min}|/M$")
    axes[1].set_xlabel(r"$t/M$")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(ncol=2, fontsize=8)
    fig.suptitle("Puncture trajectories (Z4c drawn above PC-GH; x marks Z4c slice checks)")
    fig.tight_layout()
    path = output_dir / "puncture_trajectory_overlay.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    summary = {"slice_minima_csv": str(csv_path)}
    for formulation in ("Z4c", "PC-GH"):
        pair = slices[formulation]
        summary[formulation] = {
            "source": "lapse minima in saved xy slices",
            "final_time": float(min(pair[index]["time"][-1] for index in (0, 1))),
            "final_x": [float(pair[index]["x"][-1]) for index in (0, 1)],
            "finest_dx": float(min(np.min(pair[index]["dx"]) for index in (0, 1))),
        }
    z4c_errors = []
    for index in (0, 1):
        sample = slices["Z4c"][index]
        interpolated = np.interp(sample["time"], z4c_tracks[index]["time"],
                                 z4c_tracks[index]["x"])
        z4c_errors.extend(np.abs(sample["x"] - interpolated))
    summary["Z4c"]["max_slice_vs_ode_abs_x"] = float(np.max(z4c_errors))
    return path, summary


def plot_waveform_overlays(waves: dict[str, dict[int, dict[str, np.ndarray]]],
                           output_dir: Path) -> tuple[list[Path], dict]:
    fig, axes = plt.subplots(3, 2, figsize=(12.0, 10.0), sharex=True, sharey=True)
    for axis, radius in zip(axes.flat, RADII):
        for formulation in ("PC-GH", "Z4c"):
            wave = waves[formulation][radius]
            width = 2.5 if formulation == "PC-GH" else 1.3
            zorder = 2 if formulation == "PC-GH" else 3
            axis.plot(wave["retarded_time"], wave["real_22"],
                      color=COLORS[formulation], linewidth=width,
                      zorder=zorder, label=formulation)
        axis.axvline(20.0, color="0.45", linestyle=":", linewidth=0.8)
        axis.axvline(35.0, color="0.45", linestyle=":", linewidth=0.8)
        axis.set_title(f"r = {radius} M")
        axis.grid(alpha=0.25)
    for axis in axes[-1, :]:
        axis.set_xlabel(r"$(t-r)/M$")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$r\,\mathrm{Re}(\Psi_4^{22})$")
    axes[0, 0].legend()
    fig.suptitle("(2,2) waveform; dotted line separates early and late windows")
    fig.tight_layout()
    overlay_path = output_dir / "waveform_22_radii_overlay.png"
    fig.savefig(overlay_path, dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.5), sharex=True)
    radius_colors = plt.cm.viridis(np.linspace(0.05, 0.95, len(RADII)))
    for row, formulation in enumerate(("Z4c", "PC-GH")):
        for radius, color in zip(RADII, radius_colors):
            wave = waves[formulation][radius]
            axes[row, 0].plot(wave["retarded_time"], wave["real_22"],
                              color=color, label=f"r={radius}M")
            axes[row, 1].plot(wave["retarded_time"], wave["imag_22"],
                              color=color, label=f"r={radius}M")
        axes[row, 0].set_ylabel(f"{formulation}\n" + r"$r\,\mathrm{Re}(\Psi_4^{22})$")
        axes[row, 1].set_ylabel(f"{formulation}\n" + r"$r\,\mathrm{Im}(\Psi_4^{22})$")
    for axis in axes.flat:
        axis.axvline(20.0, color="0.45", linestyle=":", linewidth=0.8)
        axis.axvline(35.0, color="0.45", linestyle=":", linewidth=0.8)
        axis.grid(alpha=0.25)
        axis.set_xlabel(r"$(t-r)/M$")
    axes[0, 0].legend(ncol=2, fontsize=8)
    fig.suptitle("Finite-radius consistency and imaginary-mode symmetry leakage")
    fig.tight_layout()
    convergence_path = output_dir / "waveform_22_radius_consistency.png"
    fig.savefig(convergence_path, dpi=220)
    plt.close(fig)

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for formulation in ("Z4c", "PC-GH"):
        summary[formulation] = {}
        for radius in RADII:
            wave = waves[formulation][radius]
            real_norm = float(np.linalg.norm(wave["real_22"]))
            imag_norm = float(np.linalg.norm(wave["imag_22"]))
            absolute_real = np.abs(wave["real_22"])
            overall_index = int(np.argmax(absolute_real))
            pulse_windows = {}
            for name, mask in (
                    ("early_u_lt_20", wave["retarded_time"] < 20.0),
                    ("merger_u_20_to_35", (wave["retarded_time"] >= 20.0)
                                           & (wave["retarded_time"] < 35.0)),
                    ("late_u_ge_35", wave["retarded_time"] >= 35.0)):
                indices = np.flatnonzero(mask)
                if indices.size:
                    index = int(indices[np.argmax(absolute_real[indices])])
                    pulse_windows[name] = {
                        "peak_abs_real_22": float(absolute_real[index]),
                        "peak_retarded_time": float(wave["retarded_time"][index]),
                    }
            summary[formulation][str(radius)] = {
                "final_time": float(wave["time"][-1]),
                "peak_abs_real_22": float(absolute_real[overall_index]),
                "peak_retarded_time": float(wave["retarded_time"][overall_index]),
                "peak_abs_imag_22": float(np.max(np.abs(wave["imag_22"]))),
                "imag_to_real_l2": imag_norm / real_norm if real_norm > 0.0 else float("nan"),
                "pulse_windows": pulse_windows,
            }
            if "early_u_lt_20" in pulse_windows and "merger_u_20_to_35" in pulse_windows:
                early = pulse_windows["early_u_lt_20"]["peak_abs_real_22"]
                merger = pulse_windows["merger_u_20_to_35"]["peak_abs_real_22"]
                summary[formulation][str(radius)]["merger_to_early_peak_ratio"] = (
                    merger/early if early > 0.0 else float("nan"))
    return [overlay_path, convergence_path], summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z4c-dir", required=True, type=Path)
    parser.add_argument("--pcgh-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = {"Z4c": args.z4c_dir, "PC-GH": args.pcgh_dir}
    histories = {
        formulation: load_history(run_dir, formulation)
        for formulation, run_dir in run_dirs.items()
    }
    constraints = {
        formulation: constraint_series(histories[formulation], formulation)
        for formulation in run_dirs
    }
    changes = {
        formulation: load_amr_changes(run_dir)
        for formulation, run_dir in run_dirs.items()
    }
    waves = {
        formulation: {radius: load_wave(run_dir, radius) for radius in RADII}
        for formulation, run_dir in run_dirs.items()
    }

    products = [plot_constraints(constraints, changes, args.output_dir)]
    trajectory_path, trajectory_summary = plot_trajectories(run_dirs, args.output_dir)
    products.append(trajectory_path)
    waveform_paths, waveform_summary = plot_waveform_overlays(waves, args.output_dir)
    products.extend(waveform_paths)
    summary = {
        "run_directories": {key: str(value) for key, value in run_dirs.items()},
        "amr_changes": changes,
        "largest_adjacent_constraint_changes": {
            formulation: constraint_change_summary(
                constraints[formulation], changes[formulation])
            for formulation in run_dirs
        },
        "z4c_hamiltonian_peak_grid_phase": constraint_peak_grid_phase(
            constraints["Z4c"],
            load_tracker(only_path(run_dirs["Z4c"], "*.co_0.txt"))),
        "puncture_trajectory": trajectory_summary,
        "waveforms": waveform_summary,
    }
    summary_path = args.output_dir / "long_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    products.append(summary_path)
    print("\n".join(str(path) for path in products))


if __name__ == "__main__":
    main()
