#!/usr/bin/env python3
"""Plot matched Z4c and PC-GH head-on binary-black-hole diagnostics.

The PC-GH history uses the shared ADM Hamiltonian and momentum diagnostics added
for a formulation-independent comparison.  The waveform plot shows both the
rotation-invariant ell=2 norm and representative real modes at r=8M.
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
ELL2_MODES = ("2-2", "2-1", "20", "21", "22")
COLORS = {"Z4c": "#0072B2", "PC-GH": "#D55E00"}


def load_history(path: Path) -> dict[str, np.ndarray]:
    """Load an AthenaK history file without depending on athena_read."""
    with path.open(encoding="utf-8") as stream:
        stream.readline()
        header = stream.readline()
    columns = {
        name: number - 1 for number, name in
        ((int(number), name) for number, name in HISTORY_COLUMN.findall(header))
    }
    data = np.atleast_2d(np.loadtxt(path))
    return {name: data[:, column] for name, column in columns.items()}


def load_wave_pair(directory: Path) -> dict[str, np.ndarray]:
    """Load the real and imaginary rPsi4 tables and require matching samples."""
    real_path = directory / "rpsi4_real_0008.txt"
    imag_path = directory / "rpsi4_imag_0008.txt"
    with real_path.open(encoding="utf-8") as stream:
        header = stream.readline()
    columns = {
        name: number - 1 for number, name in
        ((int(number), name) for number, name in WAVE_COLUMN.findall(header))
    }
    real = np.atleast_2d(np.loadtxt(real_path))
    imag = np.atleast_2d(np.loadtxt(imag_path))
    if real.shape != imag.shape or not np.allclose(real[:, 0], imag[:, 0], rtol=0.0,
                                                   atol=1.0e-12):
        raise ValueError(f"real and imaginary waveform samples do not match in {directory}")
    result = {"time": real[:, 0]}
    for mode in ELL2_MODES:
        column = columns[mode]
        result[f"real_{mode}"] = real[:, column]
        result[f"imag_{mode}"] = imag[:, column]
    result["ell2_norm"] = np.sqrt(sum(
        result[f"real_{mode}"]**2 + result[f"imag_{mode}"]**2
        for mode in ELL2_MODES
    ))
    return result


def rms(numerator: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Return a nonnegative volume RMS and preserve invalid samples as NaN."""
    quotient = np.full_like(np.asarray(numerator, dtype=float), np.nan)
    np.divide(numerator, volume, out=quotient, where=volume > 0.0)
    return np.sqrt(np.maximum(quotient, 0.0))


def finite_peak(time: np.ndarray, values: np.ndarray) -> dict[str, float]:
    valid = np.isfinite(time) & np.isfinite(values)
    if not np.any(valid):
        return {"time": float("nan"), "value": float("nan")}
    valid_indices = np.flatnonzero(valid)
    index = valid_indices[np.argmax(values[valid])]
    return {"time": float(time[index]), "value": float(values[index])}


def plot_constraints(z4c: dict[str, np.ndarray], pcgh: dict[str, np.ndarray],
                     output_dir: Path) -> tuple[Path, dict]:
    diagnostics = {
        "Z4c": {
            "time": z4c["time"],
            "H": rms(z4c["H-norm2"], z4c["Volume"]),
            "M": rms(z4c["M-norm2"], z4c["Volume"]),
        },
        "PC-GH": {
            "time": pcgh["time"],
            "H": rms(pcgh["shared-H-n"], pcgh["shared-Vol"]),
            "M": rms(pcgh["shared-M-n"], pcgh["shared-Vol"]),
        },
    }
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.0), sharex="col", sharey="row")
    for column, formulation in enumerate(("Z4c", "PC-GH")):
        data = diagnostics[formulation]
        for row, constraint in enumerate(("H", "M")):
            axis = axes[row, column]
            values = data[constraint]
            positive = np.where(values > 0.0, values, np.nan)
            axis.semilogy(data["time"], positive, color=COLORS[formulation], linewidth=1.6)
            axis.grid(alpha=0.25, which="both")
            axis.set_title(f"{formulation}: {constraint}")
            if column == 0:
                axis.set_ylabel("proper-volume RMS")
            if row == 1:
                axis.set_xlabel(r"$t/M$")
    fig.suptitle("Matched head-on BBH constraints (no constraint damping)")
    fig.tight_layout()
    path = output_dir / "constraint_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    summary = {}
    for formulation, data in diagnostics.items():
        summary[formulation] = {
            constraint: {
                "initial_rms": float(values[0]),
                "final_rms": float(values[-1]),
                "peak": finite_peak(data["time"], values),
            }
            for constraint, values in (("H", data["H"]), ("M", data["M"]))
        }
    summary["final_pcgh_to_z4c_ratio"] = {
        constraint: float(diagnostics["PC-GH"][constraint][-1]
                          / diagnostics["Z4c"][constraint][-1])
        for constraint in ("H", "M")
    }
    return path, summary


def plot_waveforms(z4c: dict[str, np.ndarray], pcgh: dict[str, np.ndarray],
                   output_dir: Path) -> tuple[Path, dict]:
    waves = {"Z4c": z4c, "PC-GH": pcgh}
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 7.0), sharex=True)
    for formulation, wave in waves.items():
        axes[0].plot(wave["time"], wave["ell2_norm"], color=COLORS[formulation],
                     linewidth=1.6, label=formulation)
        axes[1].plot(wave["time"], wave["real_20"], color=COLORS[formulation],
                     linewidth=1.6, label=f"{formulation} Re(2,0)")
        axes[1].plot(wave["time"], wave["real_22"], color=COLORS[formulation],
                     linewidth=1.1, linestyle="--", label=f"{formulation} Re(2,2)")
    axes[0].set_ylabel(r"$r\,|\Psi_4|_{\ell=2}$")
    axes[0].set_title(r"Rotation-invariant $\ell=2$ norm")
    axes[1].set_ylabel(r"$r\,\mathrm{Re}(\Psi_4^{\ell m})$")
    axes[1].set_xlabel(r"$t/M$ at extraction radius $r=8M$")
    axes[1].set_title("Representative modes")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    fig.suptitle("Matched head-on BBH waveform extraction")
    fig.tight_layout()
    path = output_dir / "waveform_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    summary = {
        formulation: {"ell2_peak": finite_peak(wave["time"], wave["ell2_norm"])}
        for formulation, wave in waves.items()
    }
    common_start = max(wave["time"][0] for wave in waves.values())
    common_end = min(wave["time"][-1] for wave in waves.values())
    common_time = np.linspace(common_start, common_end, 1001)
    z4c_norm = np.interp(common_time, z4c["time"], z4c["ell2_norm"])
    pcgh_norm = np.interp(common_time, pcgh["time"], pcgh["ell2_norm"])
    denominator = np.trapezoid(z4c_norm**2, common_time)
    summary["common_interval"] = [float(common_start), float(common_end)]
    summary["ell2_relative_l2_difference_to_z4c"] = (
        float(np.sqrt(np.trapezoid((pcgh_norm - z4c_norm)**2, common_time)
                      / denominator)) if denominator > 0.0 else float("nan"))
    return path, summary


def plot_pcgh_boundedness(path: Path, output_dir: Path) -> tuple[Path, dict]:
    names = path.read_text(encoding="utf-8").splitlines()[0].lstrip("# ").split()
    data = np.atleast_2d(np.loadtxt(path))
    columns = {name: data[:, index] for index, name in enumerate(names)}
    projection_names = (
        "dRw_reduction_project_max", "dRQ_reduction_project_max",
        "dRalpha_reduction_project_max", "dRB_reduction_project_max",
    )
    has_projection_monitor = all(name in columns for name in projection_names)
    nrows = 4 if has_projection_monitor else 3
    fig, axes = plt.subplots(nrows, 1, figsize=(10.5, 3.0*nrows + 1.0), sharex=True)
    for name, label in (("min_w", "min w"), ("min_alpha", "min lapse"),
                        ("min_eigenvalue", "min eig(gtilde)")):
        axes[0].plot(columns["time"], columns[name], label=label)
    for name, label in (("max_rho", "max rho"), ("max_K", "max |K|"),
                        ("max_Atilde", "max |Atilde|"),
                        ("max_beta", "max |beta|"),
                        ("max_abs_detg_minus_1", "max |det(gtilde)-1|")):
        axes[1].semilogy(columns["time"], np.maximum(columns[name], 1.0e-30), label=label)
    for name, label in (("max_p", "max |p|"), ("max_L", "max |L|"),
                        ("max_Q", "max |Q|"), ("max_B", "max |B|")):
        axes[2].semilogy(columns["time"], np.maximum(columns[name], 1.0e-30), label=label)
    axes[0].set_ylabel("positive lower bounds")
    axes[1].set_ylabel("primary upper bounds")
    axes[2].set_ylabel("gradient upper bounds")
    if has_projection_monitor:
        for name, label in zip(projection_names, ("p", "Q", "L", "B")):
            axes[3].semilogy(columns["time"], np.maximum(columns[name], 1.0e-30),
                            label=label)
        axes[3].set_ylabel("projection correction")
    axes[-1].set_xlabel(r"$t/M$")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, ncol=2)
    fig.suptitle("PC-GH boundedness monitor")
    fig.tight_layout()
    output = output_dir / "pcgh_boundedness.png"
    fig.savefig(output, dpi=200)
    plt.close(fig)
    summary = {
        "final_time": float(columns["time"][-1]),
        "all_finite": bool(np.all(np.isfinite(data))),
        "minimum_w": float(np.min(columns["min_w"])),
        "minimum_lapse": float(np.min(columns["min_alpha"])),
        "minimum_metric_eigenvalue": float(np.min(columns["min_eigenvalue"])),
        "maximum_abs_detg_minus_1": float(np.max(columns["max_abs_detg_minus_1"])),
        "maximum_constraint_projection_correction": float(
            np.max(columns["pcgh_projection_max"])),
        "maximum_fields": {
            name.removeprefix("max_"): float(np.max(columns[name]))
            for name in ("max_rho", "max_K", "max_Atilde", "max_beta", "max_p",
                         "max_L", "max_Q", "max_B")
        },
    }
    if has_projection_monitor:
        summary["maximum_reduction_projection_correction"] = {
            label: float(np.max(columns[name]))
            for name, label in zip(projection_names, ("p", "Q", "L", "B"))
        }
    return output, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--z4c-history", required=True, type=Path)
    parser.add_argument("--pcgh-history", required=True, type=Path)
    parser.add_argument("--z4c-wave-dir", required=True, type=Path)
    parser.add_argument("--pcgh-wave-dir", required=True, type=Path)
    parser.add_argument("--pcgh-boundedness", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    constraints_path, constraints = plot_constraints(
        load_history(args.z4c_history), load_history(args.pcgh_history), args.output_dir)
    waveform_path, waveform = plot_waveforms(
        load_wave_pair(args.z4c_wave_dir), load_wave_pair(args.pcgh_wave_dir),
        args.output_dir)
    products = [constraints_path, waveform_path]
    summary = {"constraints": constraints, "waveform": waveform}
    if args.pcgh_boundedness:
        boundedness_path, boundedness = plot_pcgh_boundedness(
            args.pcgh_boundedness, args.output_dir)
        products.append(boundedness_path)
        summary["pcgh_boundedness"] = boundedness
    summary_path = args.output_dir / "comparison_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    products.append(summary_path)
    print("\n".join(str(path) for path in products))


if __name__ == "__main__":
    main()
