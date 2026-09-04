#!/usr/bin/env python3
"""Plot the long-domain CUDA Z4c/PC-GH head-on BBH comparison.

The script expects one AthenaK run directory per formulation.  It compares the
puncture tracks, common ADM constraints, and the (2,2) waveform at every shared
extraction radius.  Constraint panels mark mesh-change times parsed from all
segment logs so apparent spikes can be checked against AMR operations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HISTORY_COLUMN = re.compile(r"\[(\d+)\]=(\S+)")
WAVE_COLUMN = re.compile(r"(\d+):([^\s]+)")
AMR_CHANGE = re.compile(
    r"AMR_CHANGE\s+cycle=(\d+)\s+time=([+\-\d.eE]+)\s+"
    r"created=(\d+)\s+deleted=(\d+)\s+blocks=(\d+)"
)
COLORS = {"Z4c": "#0072B2", "PC-GH": "#D55E00"}
RADII = (8, 12, 24, 32, 48, 56)


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
    # A restart can repeat its first saved sample.  Keep the last occurrence.
    _, reverse_indices = np.unique(data[::-1, 1], return_index=True)
    indices = np.sort(data.shape[0] - 1 - reverse_indices)
    data = data[indices]
    if np.any(np.diff(data[:, 1]) <= 0.0):
        raise ValueError(f"tracker times are not strictly increasing in {path}")
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
    for log_path in sorted(run_dir.glob("segment-*.log")):
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
    tracks = {
        formulation: [
            load_tracker(only_path(run_dir, f"*.co_{index}.txt"))
            for index in (0, 1)
        ]
        for formulation, run_dir in run_dirs.items()
    }
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    for formulation in ("PC-GH", "Z4c"):
        for index, (track, linestyle) in enumerate(zip(
                tracks[formulation], ("-", "--"))):
            width = 2.8 if formulation == "PC-GH" else 1.4
            zorder = 2 if formulation == "PC-GH" else 3
            axes[0].plot(track["time"], track["x"], color=COLORS[formulation],
                         linestyle=linestyle, linewidth=width, zorder=zorder,
                         label=f"{formulation}, puncture {index}")
            drift = np.hypot(track["y"], track["z"])
            axes[1].semilogy(track["time"], np.maximum(drift, 1.0e-30),
                            color=COLORS[formulation], linestyle=linestyle,
                            linewidth=width, zorder=zorder,
                            label=f"{formulation}, puncture {index}")
    axes[0].set_ylabel(r"$x/M$")
    axes[1].set_ylabel(r"$\sqrt{y^2+z^2}/M$")
    axes[1].set_xlabel(r"$t/M$")
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
        axis.legend(ncol=2, fontsize=8)
    fig.suptitle("Puncture trajectories (Z4c drawn above PC-GH)")
    fig.tight_layout()
    path = output_dir / "puncture_trajectory_overlay.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    summary = {
        formulation: {
            "final_time": float(min(track["time"][-1] for track in pair)),
            "max_transverse_drift": float(max(
                np.max(np.hypot(track["y"], track["z"])) for track in pair)),
            "final_x": [float(track["x"][-1]) for track in pair],
        }
        for formulation, pair in tracks.items()
    }
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
        axis.set_title(f"r = {radius} M")
        axis.grid(alpha=0.25)
    for axis in axes[-1, :]:
        axis.set_xlabel(r"$(t-r)/M$")
    for axis in axes[:, 0]:
        axis.set_ylabel(r"$r\,\mathrm{Re}(\Psi_4^{22})$")
    axes[0, 0].legend()
    fig.suptitle("(2,2) waveform at all extraction radii; Z4c drawn above PC-GH")
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
            summary[formulation][str(radius)] = {
                "final_time": float(wave["time"][-1]),
                "peak_abs_real_22": float(np.max(np.abs(wave["real_22"]))),
                "peak_abs_imag_22": float(np.max(np.abs(wave["imag_22"]))),
                "imag_to_real_l2": imag_norm / real_norm if real_norm > 0.0 else float("nan"),
            }
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
        "puncture_trajectory": trajectory_summary,
        "waveforms": waveform_summary,
    }
    summary_path = args.output_dir / "long_comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    products.append(summary_path)
    print("\n".join(str(path) for path in products))


if __name__ == "__main__":
    main()
