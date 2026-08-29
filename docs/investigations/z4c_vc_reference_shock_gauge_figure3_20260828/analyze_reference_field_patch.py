#!/usr/bin/env python3
"""Measure stitched rho=4--6, z=-2--2 high-frequency content.

The finest replay patch is assembled before differencing or Fourier analysis,
so rho=5 and z={-1,0,1} remain interior same-level seams rather than artificial
FFT boundaries.  Metrics are descriptive diagnostics, not convergence proofs.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np


REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY / "vis" / "python"))
import bin_convert  # noqa: E402


PATCH = (4.0, 6.0, -2.0, 2.0)
REFERENCE_H = 1.0 / 32.0
HISTORY_HEADER = re.compile(r"\[(\d+)\]=([^\s]+)")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def normalized_norm(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def read_rank_set(rank0_path: Path) -> dict:
    """Read every per-rank state file without filtering by file size.

    ``bin_convert.read_all_ranks_binary`` discards files whose sizes differ
    from the largest file.  That is unsafe here because load-balanced ranks
    can legitimately own different numbers of MeshBlocks.
    """
    rank0_directory = str(rank0_path.parent)
    require("rank_00000000" in rank0_directory,
            "--all-ranks requires a rank_00000000 directory")
    paths = [Path(path) for path in sorted(glob.glob(
        rank0_directory.replace("rank_00000000", "rank_*") + "/" + rank0_path.name))]
    require(bool(paths), f"no per-rank files found for {rank0_path}")
    datasets = [bin_convert.read_binary(str(path)) for path in paths]
    reference = datasets[0]
    for path, data in zip(paths[1:], datasets[1:]):
        require(data["var_names"] == reference["var_names"],
                f"variable schema mismatch in {path}")
        require(data["time"] == reference["time"] and data["cycle"] == reference["cycle"],
                f"time/cycle mismatch in {path}")
        require(data["nx1_out_mb"] == reference["nx1_out_mb"] and
                data["nx2_out_mb"] == reference["nx2_out_mb"] and
                data["nx3_out_mb"] == reference["nx3_out_mb"],
                f"block output shape mismatch in {path}")
    combined = reference.copy()
    combined["mb_index"] = np.concatenate([data["mb_index"] for data in datasets])
    combined["mb_logical"] = np.concatenate([data["mb_logical"] for data in datasets])
    combined["mb_geometry"] = np.concatenate([data["mb_geometry"] for data in datasets])
    combined["mb_data"] = {
        variable: np.concatenate([data["mb_data"][variable] for data in datasets])
        for variable in reference["var_names"]
    }
    combined["n_mbs"] = len(combined["mb_index"])
    combined["rank_files"] = [str(path) for path in paths]
    return combined


def read_axis_tau(paths: list[Path]) -> tuple[np.ndarray, np.ndarray]:
    labels: dict[str, int] = {}
    by_cycle: dict[int, tuple[float, float]] = {}
    for path in paths:
        require(path.is_file(), f"missing history: {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                labels.update({name: int(index) - 1
                               for index, name in HISTORY_HEADER.findall(line)})
            elif line.strip():
                require({"time", "axisTau", "cycle"} <= labels.keys(),
                        "history row precedes required schema")
                values = [float(value) for value in line.split()]
                record = (values[labels["time"]], values[labels["axisTau"]])
                cycle = int(values[labels["cycle"]])
                require(cycle not in by_cycle or by_cycle[cycle] == record,
                        f"history overlap mismatch at cycle {cycle}")
                by_cycle[cycle] = record
    require(bool(by_cycle), "no history rows")
    records = [by_cycle[cycle] for cycle in sorted(by_cycle)]
    time = np.asarray([record[0] for record in records])
    tau = np.asarray([record[1] for record in records])
    require(np.all(np.diff(time) > 0.0), "merged history time is nonmonotone")
    return time, tau


def fraction_above(power: np.ndarray, wavenumber: np.ndarray,
                   threshold: float) -> float:
    total = float(np.sum(power))
    return math.nan if total <= 0.0 else float(np.sum(power[wavenumber >= threshold]) / total)


def power_quantile(power: np.ndarray, wavenumber: np.ndarray,
                   fraction: float) -> float:
    order = np.argsort(wavenumber.ravel())
    pp = power.ravel()[order]
    kk = wavenumber.ravel()[order]
    total = float(np.sum(pp))
    if total <= 0.0:
        return math.nan
    index = int(np.searchsorted(np.cumsum(pp), fraction * total, side="left"))
    return float(kk[min(index, len(kk) - 1)])


def detrend(values: np.ndarray, rho: np.ndarray, z: np.ndarray) -> np.ndarray:
    rr = 2.0 * (rho - rho[0]) / (rho[-1] - rho[0]) - 1.0
    zz = 2.0 * (z - z[0]) / (z[-1] - z[0]) - 1.0
    x, y = np.meshgrid(rr, zz)
    design = np.column_stack((np.ones(x.size), x.ravel(), y.ravel(),
                              np.square(x).ravel(), (x * y).ravel(),
                              np.square(y).ravel()))
    coefficients, *_ = np.linalg.lstsq(design, values.ravel(), rcond=None)
    return values - (design @ coefficients).reshape(values.shape)


def stitched_patch(data: dict, variable: str) -> tuple[np.ndarray, np.ndarray,
                                                        np.ndarray, dict]:
    xlo, xhi, ylo, yhi = PATCH
    logical = data["mb_logical"]
    geometry = data["mb_geometry"]
    levels = logical[:, 3]
    selected = [
        m for m, bounds in enumerate(geometry)
        if levels[m] == np.max(levels)
        and bounds[0] >= xlo - 1.0e-12 and bounds[1] <= xhi + 1.0e-12
        and bounds[2] >= ylo - 1.0e-12 and bounds[3] <= yhi + 1.0e-12
    ]
    require(len(selected) == 8, f"expected 8 finest blocks, found {len(selected)}")
    nx = data["nx1_out_mb"] - 1
    ny = data["nx2_out_mb"] - 1
    require(nx == ny and nx > 4, "unexpected vertex block shape")
    rho = np.linspace(xlo, xhi, 2 * nx + 1)
    z = np.linspace(ylo, yhi, 4 * ny + 1)
    values = np.zeros((len(z), len(rho)), dtype=float)
    counts = np.zeros_like(values, dtype=np.int16)
    duplicate_max_abs = 0.0
    duplicate_max_rel = 0.0
    locations: list[list[int]] = []
    for m in selected:
        bounds = geometry[m]
        i0 = int(round((bounds[0] - xlo) / (xhi - xlo) * (len(rho) - 1)))
        j0 = int(round((bounds[2] - ylo) / (yhi - ylo) * (len(z) - 1)))
        block = np.asarray(data["mb_data"][variable][m][0], dtype=float)
        require(block.shape == (ny + 1, nx + 1), "unexpected block payload shape")
        target = values[j0:j0 + ny + 1, i0:i0 + nx + 1]
        target_counts = counts[j0:j0 + ny + 1, i0:i0 + nx + 1]
        duplicate = target_counts > 0
        if np.any(duplicate):
            difference = np.abs(target[duplicate] / target_counts[duplicate] - block[duplicate])
            scale = np.maximum(np.abs(block[duplicate]), 1.0)
            duplicate_max_abs = max(duplicate_max_abs, float(np.max(difference)))
            duplicate_max_rel = max(duplicate_max_rel,
                                    float(np.max(difference / scale)))
        target += block
        target_counts += 1
        locations.append([int(value) for value in logical[m]])
    require(np.all(counts > 0), "stitched patch has holes")
    values /= counts
    require(duplicate_max_rel <= 2.0e-13,
            f"shared-node mismatch exceeds roundoff: {duplicate_max_rel}")
    return rho, z, values, {
        "logical_locations": sorted(locations),
        "duplicate_max_abs": duplicate_max_abs,
        "duplicate_max_relative": duplicate_max_rel,
        "spacing": float(rho[1] - rho[0]),
    }


def metrics(values: np.ndarray, rho: np.ndarray, z: np.ndarray) -> dict[str, float]:
    h = float(rho[1] - rho[0])
    require(math.isclose(h, float(z[1] - z[0]), rel_tol=0.0, abs_tol=1.0e-13),
            "anisotropic stitched spacing")
    d2_rho = values[:, :-2] - 2.0 * values[:, 1:-1] + values[:, 2:]
    d4_rho = (values[:, :-4] - 4.0 * values[:, 1:-3] +
              6.0 * values[:, 2:-2] - 4.0 * values[:, 3:-1] + values[:, 4:])
    d2_z = values[:-2, :] - 2.0 * values[1:-1, :] + values[2:, :]
    d4_z = (values[:-4, :] - 4.0 * values[1:-3, :] +
            6.0 * values[2:-2, :] - 4.0 * values[3:-1, :] + values[4:, :])
    eps = np.finfo(float).eps * max(normalized_norm(values), 1.0)
    eta_rho = normalized_norm(d4_rho) / (normalized_norm(d2_rho[:, 1:-1]) + eps)
    eta_z = normalized_norm(d4_z) / (normalized_norm(d2_z[1:-1, :]) + eps)

    residual = detrend(values, rho, z)
    window = np.outer(np.hanning(len(z)), np.hanning(len(rho)))
    transform = np.fft.rfft2(residual * window)
    power = np.square(np.abs(transform))
    kz = 2.0 * np.pi * np.fft.fftfreq(len(z), d=h)
    krho = 2.0 * np.pi * np.fft.rfftfreq(len(rho), d=h)
    kk_rho, kk_z = np.meshgrid(krho, kz)
    kk = np.hypot(kk_rho, kk_z)
    power[0, 0] = 0.0

    # Directional spectra are summed over transverse lines after the same
    # detrending.  They retain the physical seam inside the sampled interval.
    rho_power = np.sum(np.square(np.abs(np.fft.rfft(
        residual * np.hanning(len(rho))[None, :], axis=1))), axis=0)
    z_power = np.sum(np.square(np.abs(np.fft.rfft(
        residual * np.hanning(len(z))[:, None], axis=0))), axis=1)
    k_rho_1d = 2.0 * np.pi * np.fft.rfftfreq(len(rho), d=h)
    k_z_1d = 2.0 * np.pi * np.fft.rfftfreq(len(z), d=h)
    rho_power[0] = 0.0
    z_power[0] = 0.0

    output = {
        "spacing": h,
        "eta4_rho_l2": eta_rho,
        "eta4_z_l2": eta_z,
        "eta4_max_direction": max(eta_rho, eta_z),
        "eta4_dominant_direction": "rho" if eta_rho >= eta_z else "z",
        "fluctuation_rms": normalized_norm(residual),
        "k90": power_quantile(power, kk, 0.90),
        "k95": power_quantile(power, kk, 0.95),
        "k99": power_quantile(power, kk, 0.99),
    }
    for fraction in (0.50, 0.65, 0.80):
        tag = str(int(round(100.0 * fraction)))
        reference_threshold = fraction * np.pi / REFERENCE_H
        own_threshold = fraction * np.pi / h
        output[f"f2d_above_n256_nyquist_{tag}"] = fraction_above(
            power, kk, reference_threshold)
        output[f"f2d_above_own_nyquist_{tag}"] = fraction_above(
            power, kk, own_threshold)
        output[f"frho_above_n256_nyquist_{tag}"] = fraction_above(
            rho_power, k_rho_1d, reference_threshold)
        output[f"fz_above_n256_nyquist_{tag}"] = fraction_above(
            z_power, k_z_1d, reference_threshold)

    centers = rho[2:-2]
    seam = np.abs(centers - 5.0) <= 4.0 * h + 1.0e-14
    away = ((np.abs(centers - 5.0) >= 8.0 * h - 1.0e-14) &
            (centers >= rho[0] + 8.0 * h) & (centers <= rho[-1] - 8.0 * h))
    seam_rms = normalized_norm(d4_rho[:, seam])
    away_rms = normalized_norm(d4_rho[:, away]) if np.any(away) else math.nan
    output["d4_rho_seam_rms"] = seam_rms
    output["d4_rho_away_rms"] = away_rms
    output["d4_rho_seam_over_away"] = seam_rms / (away_rms + eps)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resolution", required=True)
    parser.add_argument("--state", type=Path, nargs="+", required=True)
    parser.add_argument("--history", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-ranks", action="store_true",
                        help="assemble all rank_* siblings of each rank-0 state")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    snapshots: list[dict[str, object]] = []
    history_time, history_tau = read_axis_tau(args.history)
    expected_variables: list[str] | None = None
    for path in args.state:
        require(path.is_file(), f"missing state file: {path}")
        data = read_rank_set(path) if args.all_ranks else bin_convert.read_binary(str(path))
        variables = list(data["var_names"])
        if expected_variables is None:
            expected_variables = variables
        require(variables == expected_variables, "state variable schema changed")
        require(history_time[0] <= data["time"] <= history_time[-1],
                f"snapshot outside history range: {path}")
        axis_tau = float(np.interp(data["time"], history_time, history_tau))
        snapshot = {"file": path.name, "time": float(data["time"]),
                    "axisTau": axis_tau,
                    "cycle": int(data["cycle"]), "variables": len(variables),
                    "rank_files": len(data.get("rank_files", [str(path)]))}
        for variable in variables:
            rho, z, values, stitch = stitched_patch(data, variable)
            row = {
                "resolution": args.resolution,
                "file": path.name,
                "time": float(data["time"]),
                "axisTau": axis_tau,
                "cycle": int(data["cycle"]),
                "variable": variable,
                **metrics(values, rho, z),
                "duplicate_max_abs": stitch["duplicate_max_abs"],
                "duplicate_max_relative": stitch["duplicate_max_relative"],
            }
            rows.append(row)
        snapshots.append(snapshot)

    require(bool(rows), "no field metrics produced")
    with (args.output / f"{args.resolution}_field_patch_metrics.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema": "athenak.z4c.reference_field_patch.v1",
        "resolution": args.resolution,
        "patch": {"rho": [PATCH[0], PATCH[1]], "z": [PATCH[2], PATCH[3]]},
        "reference_n256_spacing": REFERENCE_H,
        "snapshots": snapshots,
        "stitching": "finest active vertices, shared nodes verified then averaged",
        "trend": "two-dimensional quadratic least-squares subtraction",
        "window": "Hann only on four outer edges of the combined patch",
        "claim_boundary": "descriptive high-frequency diagnostic; not a convergence proof",
    }
    (args.output / f"{args.resolution}_field_patch_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
