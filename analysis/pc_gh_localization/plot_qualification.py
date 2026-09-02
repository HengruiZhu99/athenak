#!/usr/bin/env python3
"""Plot PC-GH one-puncture localization, convergence, horizon, and slice diagnostics.

History specifications have the form ``GAUGE:R:PATH``, where ``R`` is the finest
zones per M (16, 20, or 24).  Slice specifications point at the ``cart`` directory
of an SMR run.  AHFinderDirect tables are optional; when supplied, their commented
header must name at least time and area columns.  Separate Z4c inputs provide the
matched-control constraint ladder and exact-z=0 Cartesian slices.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import struct
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "vis" / "python"))
import athena_read  # noqa: E402


FAMILIES = ("GH", "ADM", "reduction", "curl", "algebraic")
COLORS = {
    "all": "#222222",
    "chi": "#0072B2",
    "r05": "#E69F00",
    "r1": "#009E73",
    "r2": "#CC79A7",
    "ah": "#D55E00",
}


def parse_spec(text: str) -> tuple[str, int, Path]:
    gauge, resolution, path = text.split(":", 2)
    return gauge, int(resolution), Path(path)


def safe_rms(numerator: np.ndarray, volume: np.ndarray) -> np.ndarray:
    result = np.full_like(np.asarray(numerator, dtype=float), np.nan)
    np.divide(numerator, volume, out=result, where=volume > 0.0)
    return np.sqrt(np.maximum(result, 0.0))


def regional_rms(history: dict[str, np.ndarray], region: str) -> dict[str, np.ndarray]:
    if region == "chi":
        names = {
            "cp": "Cperp-n2", "z": "Z-norm2", "h": "H-norm2",
            "m": "Mhat-norm2", "rx": "redX-norm2", "rq": "redQ-norm2",
            "ry": "redY-norm2", "rb": "redB-norm2", "cx": "curlX-n2",
            "cq": "curlQ-n2", "cy": "curlY-n2", "cb": "curlB-n2",
            "alg": ("detg-norm2", "trA-norm2", "trQ-norm2", "proj-norm2"),
            "vol": "Volume",
        }
    elif region == "all":
        names = {
            "cp": "all-Cp2", "z": "all-Z2", "h": "all-H2", "m": "all-M2",
            "rx": "all-rX2", "rq": "all-rQ2", "ry": "all-rY2", "rb": "all-rB2",
            "cx": "all-cX2", "cq": "all-cQ2", "cy": "all-cY2", "cb": "all-cB2",
            "alg": ("all-det2", "all-trA2", "all-trQ2", "all-prj2"),
            "vol": "all-Vol",
        }
    else:
        names = {
            key: f"{region}-{suffix}" for key, suffix in {
                "cp": "Cp2", "z": "Z2", "h": "H2", "m": "M2",
                "rx": "rX2", "rq": "rQ2", "ry": "rY2", "rb": "rB2",
                "cx": "cX2", "cq": "cQ2", "cy": "cY2", "cb": "cB2",
                "alg": "alg2", "vol": "Vol",
            }.items()
        }
    volume = history[names["vol"]]
    algebraic_names = names["alg"]
    if isinstance(algebraic_names, tuple):
        algebraic = sum(history[name] for name in algebraic_names)
    else:
        algebraic = history[algebraic_names]
    return {
        "GH": safe_rms(history[names["cp"]] + history[names["z"]], volume),
        "ADM": safe_rms(history[names["h"]] + history[names["m"]], volume),
        "reduction": safe_rms(sum(history[names[key]] for key in ("rx", "rq", "ry", "rb")), volume),
        "curl": safe_rms(sum(history[names[key]] for key in ("cx", "cq", "cy", "cb")), volume),
        "algebraic": safe_rms(algebraic, volume),
    }


def interpolate_at(time: np.ndarray, values: np.ndarray, target: float) -> float:
    valid = np.isfinite(time) & np.isfinite(values)
    if not np.any(valid):
        return float("nan")
    finite_time = time[valid]
    finite_values = values[valid]
    order = np.argsort(finite_time)
    return float(np.interp(target, finite_time[order], finite_values[order],
                           left=np.nan, right=np.nan))


def plot_constraint_evolution(histories, output_dir: Path, tmax: float) -> list[Path]:
    products = []
    regions = ("all", "chi", "r05", "r1", "r2", "ah")
    for gauge in sorted({key[0] for key in histories}):
        resolutions = sorted(n for g, n in histories if g == gauge)
        n = resolutions[-1]
        history = histories[gauge, n]
        time = history["time"]
        window = time <= tmax + 1.0e-12
        fig, axes = plt.subplots(2, 3, figsize=(13, 7.5), sharex=True)
        for axis, family in zip(axes.flat, FAMILIES):
            for region in regions:
                values = regional_rms(history, region)[family]
                if np.any(window & np.isfinite(values) & (values > 0.0)):
                    axis.semilogy(time[window], values[window],
                                  color=COLORS[region], label=region)
            axis.set_title(family)
            axis.grid(alpha=0.25)
            axis.set_xlim(0.0, tmax)
            axis.set_ylabel("coordinate/physical-volume RMS")
        axes[1, 2].axis("off")
        axes[0, 0].legend(ncol=2, fontsize=8)
        for axis in axes[1, :2]:
            axis.set_xlabel(r"$t/M$")
        fig.suptitle(
            f"PC-GH {gauge}, finest $\\Delta x=M/{n}$: constraint localization")
        fig.tight_layout()
        path = output_dir / f"constraints_evolution_{gauge}_N{n}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        products.append(path)
    return products


def plot_convergence(histories, output_dir: Path, tmax: float) -> tuple[Path, dict]:
    gauges = sorted({key[0] for key in histories})
    fig, axes = plt.subplots(len(gauges), len(FAMILIES), figsize=(17, 3.4*len(gauges)), squeeze=False)
    summary = {}
    for row, gauge in enumerate(gauges):
        resolutions = sorted(n for g, n in histories if g == gauge)
        spacings = np.asarray([1.0/n for n in resolutions])
        summary[gauge] = {}
        for col, family in enumerate(FAMILIES):
            axis = axes[row, col]
            summary[gauge][family] = {}
            for region, marker in (("chi", "o"), ("r1", "s"), ("r2", "^"), ("ah", "D"), ("all", "x")):
                endpoints = np.asarray([
                    interpolate_at(histories[gauge, n]["time"],
                                   regional_rms(histories[gauge, n], region)[family], tmax)
                    for n in resolutions
                ])
                valid = np.isfinite(endpoints) & (endpoints > 0.0)
                if np.any(valid):
                    axis.loglog(spacings[valid], endpoints[valid], marker=marker,
                                color=COLORS[region], label=region)
                orders = []
                for left, right in zip(range(len(resolutions) - 1), range(1, len(resolutions))):
                    if endpoints[left] > 0.0 and endpoints[right] > 0.0:
                        orders.append(float(np.log(endpoints[left]/endpoints[right])
                                            / np.log(spacings[left]/spacings[right])))
                    else:
                        orders.append(float("nan"))
                summary[gauge][family][region] = {
                    "N": resolutions,
                    "endpoint_rms": endpoints.tolist(),
                    "pair_orders": orders,
                }
            axis.set_title(f"{gauge}: {family}")
            axis.set_xlabel(r"$\Delta x/M$")
            axis.grid(alpha=0.25, which="both")
            if col == 0:
                axis.set_ylabel(f"RMS at t={tmax:g}M")
            if row == 0 and col == 0:
                axis.legend(fontsize=7)
    fig.tight_layout()
    path = output_dir / "constraint_convergence.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, summary


def plot_finiteness_failure(histories, output_dir: Path) -> tuple[Path, dict]:
    """Plot the first non-finite diagnostic time and the preceding finite sample."""
    summary = {}
    fig, axis = plt.subplots(figsize=(6.8, 4.5))
    for gauge in sorted({key[0] for key in histories}):
        resolutions = sorted(n for g, n in histories if g == gauge)
        last_finite = []
        first_bad = []
        summary[gauge] = {}
        for n in resolutions:
            history = histories[gauge, n]
            finite = np.ones_like(history["time"], dtype=bool)
            for name, values in history.items():
                if name not in ("time", "dt"):
                    finite &= np.isfinite(values)
            finite_indices = np.flatnonzero(finite)
            bad_indices = np.flatnonzero(~finite)
            last = float(history["time"][finite_indices[-1]]) if finite_indices.size else float("nan")
            first = float(history["time"][bad_indices[0]]) if bad_indices.size else float("nan")
            last_finite.append(last)
            first_bad.append(first)
            summary[gauge][str(n)] = {
                "last_all_finite_sample": last,
                "first_nonfinite_sample": first,
            }
        axis.plot(resolutions, first_bad, "o-", label=f"{gauge}: first non-finite")
        axis.plot(resolutions, last_finite, "x--", label=f"{gauge}: last finite")
    axis.set(xlabel=r"finest zones per $M$", ylabel=r"$t/M$",
             title="PC-GH one-puncture loss of finiteness")
    axis.set_xticks(sorted({n for _, n in histories}))
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    path = output_dir / "pcgh_finiteness_failure.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path, summary


def z4c_rms(history: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Return the native chi-excised, physical-volume Z4c RMS diagnostics."""
    volume = history["Volume"]
    return {
        "C": safe_rms(history["C-norm2"], volume),
        "H": safe_rms(history["H-norm2"], volume),
        "M": safe_rms(history["M-norm2"], volume),
        "Z": safe_rms(history["Z-norm2"], volume),
        "Theta": safe_rms(history["Theta-norm"], volume),
    }


def plot_z4c_controls(histories, output_dir: Path, tmax: float) -> tuple[list[Path], dict]:
    """Plot matched Z4c control evolution and its three-resolution endpoint ladder."""
    if not histories:
        return [], {}
    resolutions = sorted(histories)
    finest = resolutions[-1]
    finest_history = histories[finest]
    metrics = tuple(z4c_rms(finest_history))
    products = []

    fig, axis = plt.subplots(figsize=(7.5, 4.8))
    for metric, values in z4c_rms(finest_history).items():
        axis.semilogy(finest_history["time"], values, label=metric)
    axis.set(xlabel=r"$t/M$", ylabel="physical-volume RMS",
             xlim=(0.0, tmax), title=f"Matched Z4c control, finest $\\Delta x=M/{finest}$")
    axis.grid(alpha=0.25)
    axis.legend(ncol=3)
    fig.tight_layout()
    path = output_dir / f"z4c_constraint_evolution_N{finest}.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    products.append(path)

    spacings = np.asarray([1.0/n for n in resolutions])
    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 3.5), sharex=True)
    summary = {}
    for axis, metric in zip(axes, metrics):
        endpoints = np.asarray([
            interpolate_at(histories[n]["time"], z4c_rms(histories[n])[metric], tmax)
            for n in resolutions
        ])
        axis.loglog(spacings, endpoints, "o-")
        axis.set_title(metric)
        axis.set_xlabel(r"finest $\Delta x/M$")
        axis.grid(alpha=0.25, which="both")
        valid = np.isfinite(endpoints) & (endpoints > 0.0)
        order = float(np.polyfit(np.log(spacings[valid]), np.log(endpoints[valid]), 1)[0]) \
            if np.count_nonzero(valid) >= 2 else float("nan")
        summary[metric] = {
            "endpoint": {str(n): float(v) for n, v in zip(resolutions, endpoints)},
            "observed_order": order,
        }
    axes[0].set_ylabel("physical-volume RMS at target time")
    fig.suptitle(f"Matched Z4c constraint ladder at $t={tmax:g}M$")
    fig.tight_layout()
    path = output_dir / "z4c_constraint_convergence.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    products.append(path)
    return products, summary


_CART_METADATA = struct.Struct("=if3f3f3i?3xi")


def read_cartesian(path: Path) -> dict:
    """Read AthenaK's interpolated CartesianGridOutput binary format."""
    with path.open("rb") as stream:
        metadata = _CART_METADATA.unpack(stream.read(_CART_METADATA.size))
        cycle, time = metadata[:2]
        center = np.asarray(metadata[2:5])
        extent = np.asarray(metadata[5:8])
        numpoints = np.asarray(metadata[8:11], dtype=int)
        is_chebyshev, noutvars = metadata[11:13]
        (label_length,) = struct.unpack("=i", stream.read(4))
        labels = stream.read(label_length).decode().split()
        raw = np.frombuffer(stream.read(), dtype="=f4")
    expected = int(noutvars*np.prod(numpoints))
    if len(labels) != noutvars or raw.size != expected:
        raise ValueError(f"invalid Cartesian output dimensions in {path}")
    if is_chebyshev:
        raise ValueError("qualification slices require a uniform Cartesian output grid")
    data = raw.reshape((noutvars, numpoints[2], numpoints[1], numpoints[0]))
    coordinates = [
        np.linspace(center[d] - extent[d], center[d] + extent[d], numpoints[d])
        for d in range(3)
    ]
    return {
        "cycle": cycle,
        "time": time,
        "x": coordinates[0],
        "y": coordinates[1],
        "z": coordinates[2],
        "data": {label: data[index] for index, label in enumerate(labels)},
    }


def nearest_cart_files(directory: Path, stem: str, targets: tuple[float, ...]):
    candidates = []
    for path in sorted(directory.glob(f"*.{stem}.*.bin")):
        data = read_cartesian(path)
        candidates.append((float(data["time"]), path, data))
    if not candidates:
        return []
    return [min(candidates, key=lambda item: abs(item[0] - target)) for target in targets]


def plot_slices(slice_specs, output_dir: Path, tmax: float) -> list[Path]:
    products = []
    targets = (0.0, 0.5*tmax, tmax)
    for gauge, n, directory in slice_specs:
        state_files = nearest_cart_files(directory, "pcgh_slice", targets)
        con_files = nearest_cart_files(directory, "pcgh_con_slice", targets)
        if len(state_files) != len(targets) or len(con_files) != len(targets):
            continue
        panels = []
        for (_, _, state), (_, _, con) in zip(state_files, con_files):
            if not (np.array_equal(state["x"], con["x"])
                    and np.array_equal(state["y"], con["y"])):
                raise ValueError("state and constraint Cartesian grids differ")
            chi = state["data"]["pcgh_chi"][0]
            cp = con["data"]["pcgh_Cperp"][0]
            zx = con["data"]["pcgh_Zx"][0]
            zy = con["data"]["pcgh_Zy"][0]
            zz = con["data"]["pcgh_Zz"][0]
            ham = con["data"]["pcgh_H"][0]
            mx = con["data"]["pcgh_Mhatx"][0]
            my = con["data"]["pcgh_Mhaty"][0]
            mz = con["data"]["pcgh_Mhatz"][0]
            gh = np.sqrt(cp*cp + zx*zx + zy*zy + zz*zz)
            adm = np.sqrt(ham*ham + mx*mx + my*my + mz*mz)
            panels.append((state["x"], state["y"], chi, gh, adm))
        positive = np.concatenate([
            panel[index].ravel() for panel in panels for index in (3, 4)])
        positive = positive[np.isfinite(positive) & (positive > 0.0)]
        chi_values = np.concatenate([panel[2].ravel() for panel in panels])
        chi_values = chi_values[np.isfinite(chi_values)]
        chi_lo, chi_hi = (float(np.min(chi_values)), float(np.max(chi_values)))
        lo, hi = (np.percentile(np.log10(positive), [2, 99.5])
                  if positive.size else (-16.0, 0.0))
        fig, axes = plt.subplots(3, len(targets), figsize=(11.5, 10), sharex=True,
                                 sharey=True, constrained_layout=True)
        for col, ((actual_time, _, _), panel) in enumerate(zip(state_files, panels)):
            x, y, chi, gh, adm = panel
            images = (
                axes[0, col].pcolormesh(
                    x, y, chi, shading="nearest", cmap="viridis",
                    vmin=chi_lo, vmax=chi_hi),
                axes[1, col].pcolormesh(
                    x, y, np.log10(np.maximum(gh, 10.0**lo)),
                    shading="nearest", cmap="magma", vmin=lo, vmax=hi),
                axes[2, col].pcolormesh(
                    x, y, np.log10(np.maximum(adm, 10.0**lo)),
                    shading="nearest", cmap="magma", vmin=lo, vmax=hi),
            )
            digits = 4 if tmax < 0.1 else 2
            axes[0, col].set_title(f"t={actual_time:.{digits}f}M")
            for axis in axes[1:, col]:
                axis.contour(x, y, chi, levels=[0.0625], colors="cyan",
                             linewidths=0.7, alpha=0.9)
            axes[2, col].set_xlabel(r"$x/M$")
            for axis in axes[:, col]:
                axis.set_aspect("equal")
                axis.set_xlim(-2.0, 2.0)
                axis.set_ylim(-2.0, 2.0)
        axes[0, 0].set_ylabel("$y/M$\n$\\chi$")
        axes[1, 0].set_ylabel("$y/M$\n$\\log_{10}|C_{GH}|$")
        axes[2, 0].set_ylabel("$y/M$\n$\\log_{10}|C_{ADM}|$")
        fig.colorbar(images[0], ax=axes[0, :], shrink=0.7, label=r"$\chi$")
        fig.colorbar(images[1], ax=axes[1:, :], shrink=0.7, label="log10 constraint")
        fig.suptitle(f"PC-GH {gauge}, finest $\\Delta x=M/{n}$: equatorial evolution")
        path = output_dir / f"diagnostic_slices_{gauge}_N{n}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        products.append(path)
    return products


def plot_z4c_slices(slice_specs, output_dir: Path, tmax: float) -> list[Path]:
    """Plot exact-z=0 Cartesian slices for the matched Z4c controls."""
    products = []
    targets = (0.0, 0.5*tmax, tmax)
    for gauge, n, directory in slice_specs:
        state_files = nearest_cart_files(directory, "z4c_slice", targets)
        con_files = nearest_cart_files(directory, "z4c_con_slice", targets)
        if len(state_files) != len(targets) or len(con_files) != len(targets):
            continue
        panels = []
        for (_, _, state), (_, _, con) in zip(state_files, con_files):
            if not (np.array_equal(state["x"], con["x"])
                    and np.array_equal(state["y"], con["y"])):
                raise ValueError("Z4c state and constraint Cartesian grids differ")
            chi = state["data"]["z4c_chi"][0]
            aggregate = np.sqrt(np.maximum(con["data"]["con_C"][0], 0.0))
            ham = np.abs(con["data"]["con_H"][0])
            momentum = np.sqrt(np.maximum(con["data"]["con_M"][0], 0.0))
            adm = np.sqrt(ham*ham + momentum*momentum)
            panels.append((state["x"], state["y"], chi, aggregate, adm))
        positive = np.concatenate([
            panel[index].ravel() for panel in panels for index in (3, 4)])
        positive = positive[np.isfinite(positive) & (positive > 0.0)]
        chi_values = np.concatenate([panel[2].ravel() for panel in panels])
        chi_values = chi_values[np.isfinite(chi_values)]
        chi_lo, chi_hi = float(np.min(chi_values)), float(np.max(chi_values))
        lo, hi = (np.percentile(np.log10(positive), [2, 99.5])
                  if positive.size else (-16.0, 0.0))
        fig, axes = plt.subplots(3, len(targets), figsize=(11.5, 10), sharex=True,
                                 sharey=True, constrained_layout=True)
        for col, ((actual_time, _, _), panel) in enumerate(zip(state_files, panels)):
            x, y, chi, aggregate, adm = panel
            images = (
                axes[0, col].pcolormesh(x, y, chi, shading="nearest", cmap="viridis",
                                        vmin=chi_lo, vmax=chi_hi),
                axes[1, col].pcolormesh(
                    x, y, np.log10(np.maximum(aggregate, 10.0**lo)),
                    shading="nearest", cmap="magma", vmin=lo, vmax=hi),
                axes[2, col].pcolormesh(
                    x, y, np.log10(np.maximum(adm, 10.0**lo)),
                    shading="nearest", cmap="magma", vmin=lo, vmax=hi),
            )
            digits = 4 if tmax < 0.1 else 2
            axes[0, col].set_title(f"t={actual_time:.{digits}f}M")
            for axis in axes[1:, col]:
                axis.contour(x, y, chi, levels=[0.0625], colors="cyan",
                             linewidths=0.7, alpha=0.9)
            axes[2, col].set_xlabel(r"$x/M$")
            for axis in axes[:, col]:
                axis.set_aspect("equal")
                axis.set_xlim(-2.0, 2.0)
                axis.set_ylim(-2.0, 2.0)
        axes[0, 0].set_ylabel("$y/M$\n$\\chi$")
        axes[1, 0].set_ylabel("$y/M$\n$\\log_{10}|C_{Z4c}|$")
        axes[2, 0].set_ylabel("$y/M$\n$\\log_{10}|C_{ADM}|$")
        fig.colorbar(images[0], ax=axes[0, :], shrink=0.7, label=r"$\chi$")
        fig.colorbar(images[1], ax=axes[1:, :], shrink=0.7, label="log10 constraint")
        fig.suptitle(f"Matched Z4c, finest $\\Delta x=M/{n}$: equatorial evolution")
        path = output_dir / f"diagnostic_slices_z4c_N{n}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        products.append(path)
    return products


def read_horizon_table(path: Path) -> dict[str, np.ndarray]:
    comment_headers = []
    rows = []
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            tokens = re.findall(r"[A-Za-z][A-Za-z0-9_./()-]*", stripped[1:])
            if tokens:
                comment_headers.append(tokens)
            continue
        try:
            rows.append([float(value) for value in stripped.split()])
        except ValueError:
            continue
    if not rows:
        raise ValueError(f"no numeric horizon rows in {path}")
    width = len(rows[0])
    header = next((tokens for tokens in reversed(comment_headers) if len(tokens) == width), None)
    if header is None:
        raise ValueError(f"no {width}-column commented header in {path}")
    array = np.asarray(rows)
    return {name: array[:, index] for index, name in enumerate(header)}


def alias_column(table: dict[str, np.ndarray], patterns) -> np.ndarray | None:
    for name, values in table.items():
        normalized = name.lower().replace("-", "_")
        if any(re.search(pattern, normalized) for pattern in patterns):
            return values
    return None


def plot_horizons(horizon_specs, output_dir: Path, tmax: float) -> tuple[list[Path], dict]:
    histories = {}
    summary = {}
    for gauge, n, path in horizon_specs:
        table = read_horizon_table(path)
        time = alias_column(table, (r"^t$", r"time"))
        area = alias_column(table, (r"area",))
        if time is None or area is None:
            raise ValueError(f"horizon table {path} must name time and area columns")
        histories[gauge, n] = (time, area, table)
    products = []
    for gauge in sorted({key[0] for key in histories}):
        fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
        summary[gauge] = {}
        for _, n in sorted(key for key in histories if key[0] == gauge):
            time, area, table = histories[gauge, n]
            radius = np.sqrt(area/(4.0*np.pi))
            mirr = np.sqrt(area/(16.0*np.pi))
            axes[0, 0].plot(time, area/(16.0*np.pi) - 1.0, label=f"N={n}")
            axes[0, 1].plot(time, radius/2.0 - 1.0)
            axes[1, 0].plot(time, mirr - 1.0)
            coordinate_radius = alias_column(table, (r"coord.*radius", r"mean.*radius", r"radius"))
            if coordinate_radius is not None:
                axes[1, 1].plot(time, coordinate_radius)
            valid = time <= tmax + 1.0e-12
            summary[gauge][str(n)] = {
                "max_fractional_area_drift": float(np.nanmax(np.abs(area[valid]/(16*np.pi) - 1.0))),
                "max_fractional_areal_radius_drift": float(np.nanmax(np.abs(radius[valid]/2.0 - 1.0))),
                "max_fractional_mirr_drift": float(np.nanmax(np.abs(mirr[valid] - 1.0))),
            }
        titles = ("A/(16πM²)-1", "R_AH/(2M)-1", "M_irr/M-1", "coordinate radius")
        for axis, title in zip(axes.flat, titles):
            axis.set_title(title)
            axis.grid(alpha=0.25)
            axis.set_xlim(0.0, tmax)
        axes[0, 0].legend()
        axes[1, 0].set_xlabel(r"$t/M$")
        axes[1, 1].set_xlabel(r"$t/M$")
        fig.suptitle(f"{gauge}: apparent-horizon history")
        fig.tight_layout()
        path = output_dir / f"horizon_properties_{gauge}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        products.append(path)
    return products, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", action="append", default=[], metavar="GAUGE:N:PATH")
    parser.add_argument("--slice-dir", action="append", default=[], metavar="GAUGE:N:DIR")
    parser.add_argument("--z4c-history", action="append", default=[], metavar="Z4C:N:PATH")
    parser.add_argument("--z4c-slice-dir", action="append", default=[], metavar="Z4C:N:DIR")
    parser.add_argument("--horizon", action="append", default=[], metavar="GAUGE:N:PATH")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tmax", type=float, default=6.0)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    histories = {(gauge, n): athena_read.hst(path)
                 for gauge, n, path in map(parse_spec, args.history)}
    products = []
    convergence = {}
    finiteness = {}
    if histories:
        products += plot_constraint_evolution(histories, args.output_dir, args.tmax)
        path, convergence = plot_convergence(histories, args.output_dir, args.tmax)
        products.append(path)
        path, finiteness = plot_finiteness_failure(histories, args.output_dir)
        products.append(path)
    products += plot_slices(list(map(parse_spec, args.slice_dir)), args.output_dir, args.tmax)
    z4c_histories = {
        n: athena_read.hst(path)
        for _, n, path in map(parse_spec, args.z4c_history)
    }
    z4c_products, z4c_convergence = plot_z4c_controls(
        z4c_histories, args.output_dir, args.tmax)
    products += z4c_products
    products += plot_z4c_slices(
        list(map(parse_spec, args.z4c_slice_dir)), args.output_dir, args.tmax)
    horizon_products, horizon_summary = plot_horizons(
        list(map(parse_spec, args.horizon)), args.output_dir, args.tmax)
    products += horizon_products

    boundary = {
        "assumed_fastest_coordinate_speed": math.sqrt(2.0),
        "boundary_to_r2_time": (8.0 - 2.0)/math.sqrt(2.0),
        "boundary_to_initial_ah_time": (8.0 - 0.5)/math.sqrt(2.0),
        "flag_before_t6": True,
        "note": "Conservative 1+log lapse-speed estimate; the periodic mismatch exists at the boundary from t=0.",
    }
    report = {
        "tmax": args.tmax,
        "constraint_convergence": convergence,
        "finiteness": finiteness,
        "z4c_constraint_convergence": z4c_convergence,
        "horizon": horizon_summary,
        "boundary_contamination_estimate": boundary,
        "products": [str(path) for path in products],
    }
    summary_path = args.output_dir / "qualification_summary.json"
    summary_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    print(summary_path)
    for path in products:
        print(path)


if __name__ == "__main__":
    main()
