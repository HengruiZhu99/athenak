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


def history_column(history: dict[str, np.ndarray], *names: str) -> np.ndarray:
    """Return the first available name, accepting legacy labels in old artifacts."""
    for name in names:
        if name in history:
            return history[name]
    raise KeyError(f"none of the history columns {names!r} is present")


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
            "m": "Mhat-norm2", "rw": "redw-norm2", "rq": "redQ-norm2",
            "ra": "reda-norm2", "rb": "redB-norm2", "cpv": "curlp-n2",
            "cq": "curlQ-n2", "cl": "curlL-n2", "cb": "curlB-n2",
            "alg": ("detg-norm2", "trA-norm2", "trQ-norm2", "proj-norm2"),
            "vol": "Volume",
        }
    elif region == "all":
        names = {
            "cp": "all-Cp2", "z": "all-Z2", "h": "all-H2", "m": "all-M2",
            "rw": ("all-rw2", "all-rX2"), "rq": "all-rQ2",
            "ra": ("all-ra2", "all-rY2"), "rb": "all-rB2",
            "cpv": ("all-cp2", "all-cX2"), "cq": "all-cQ2",
            "cl": ("all-cL2", "all-cY2"), "cb": "all-cB2",
            "alg": ("all-det2", "all-trA2", "all-trQ2", "all-prj2"),
            "vol": "all-Vol",
        }
    else:
        names = {
            key: f"{region}-{suffix}" for key, suffix in {
                "cp": "Cp2", "z": "Z2", "h": "H2", "m": "M2",
                "rw": "rw2", "rq": "rQ2", "ra": "ra2", "rb": "rB2",
                "cpv": "cp2", "cq": "cQ2", "cl": "cL2", "cb": "cB2",
                "alg": "alg2", "vol": "Vol",
            }.items()
        }
    def column(key):
        value = names[key]
        return history_column(history, *value) if isinstance(value, tuple) else history[value]

    volume = column("vol")
    algebraic_names = names["alg"]
    if isinstance(algebraic_names, tuple):
        algebraic = sum(history[name] for name in algebraic_names)
    else:
        algebraic = history[algebraic_names]
    return {
        "GH": safe_rms(history[names["cp"]] + history[names["z"]], volume),
        "ADM": safe_rms(history[names["h"]] + history[names["m"]], volume),
        "reduction": safe_rms(sum(column(key) for key in ("rw", "rq", "ra", "rb")), volume),
        "curl": safe_rms(sum(column(key) for key in ("cpv", "cq", "cl", "cb")), volume),
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
    previous_last = {16: 0.0216506, 20: 0.0129904, 24: 0.0108253}
    previous_bad = {16: 0.0270633, 20: 0.0173205, 24: 0.0180422}
    old_n = sorted(previous_bad)
    axis.plot(old_n, [previous_bad[n] for n in old_n], "o:", color="#999999",
              label="old PC-GH: first bad")
    axis.plot(old_n, [previous_last[n] for n in old_n], "x:", color="#555555",
              label="old PC-GH: last finite")
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
    summary["previous_pcgh"] = {
        str(n): {"last_finite": previous_last[n], "first_bad": previous_bad[n]}
        for n in old_n
    }
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
            chi = state["data"]["pcgh_w"][0]**2
            cp = con["data"]["pcgh_Cperp"][0]
            zx = con["data"]["pcgh_Zx"][0]
            zy = con["data"]["pcgh_Zy"][0]
            zz = con["data"]["pcgh_Zz"][0]
            ham = con["data"]["pcgh_H"][0]
            mx = con["data"]["pcgh_alphaMx"][0]
            my = con["data"]["pcgh_alphaMy"][0]
            mz = con["data"]["pcgh_alphaMz"][0]
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


def read_named_table(path: Path) -> dict[str, np.ndarray]:
    table = np.genfromtxt(path, names=True)
    if table.dtype.names is None:
        raise ValueError(f"no named columns in {path}")
    return {name: np.atleast_1d(table[name]).astype(float)
            for name in table.dtype.names}


def maximum_columns(table: dict[str, np.ndarray], names) -> np.ndarray:
    return np.maximum.reduce([np.abs(table[name]) for name in names])


def plot_boundedness(boundedness, output_dir: Path) -> tuple[list[Path], dict]:
    """Plot all required regular-field, geometry, constraint, and RHS bounds."""
    products = []
    summary = {}
    for gauge in sorted({key[0] for key in boundedness}):
        resolutions = sorted(n for g, n in boundedness if g == gauge)
        n = resolutions[-1]
        table = boundedness[gauge, n]
        time = table["time"]
        panels = (
            ("w and rho", (("min_w", "min w"), ("max_w", "max w"),
                            ("min_rho", "min rho"), ("max_rho", "max rho"))),
            ("regular gradients", (("max_p", "|p|"), ("max_L", "|L|"))),
            ("GH fields", (("max_Cperp", "|Cperp|"), ("max_Z", "|Z|"))),
            ("curvature", (("max_K", "|K|"), ("max_Atilde", "|Atilde|"))),
            ("gauge and reductions", (("max_beta", "|beta|"), ("max_Q", "|Q|"),
                                      ("max_B", "|B|"))),
            ("conformal SPD", (("min_detg", "det g"), ("min_minor1", "minor 1"),
                               ("min_minor2", "minor 2"),
                               ("min_eigenvalue", "min eigenvalue"))),
        )
        fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True)
        for axis, (title, series) in zip(axes.flat, panels):
            for name, label in series:
                axis.plot(time, table[name], label=label)
            axis.set_title(title)
            axis.grid(alpha=0.25)
            axis.legend(fontsize=7, ncol=2)
        diagnostic_groups = {
            "GH": ("pcgh_Cperp_max", "pcgh_Zx_max", "pcgh_Zy_max", "pcgh_Zz_max"),
            "ADM": ("pcgh_H_max", "pcgh_alphaMx_max", "pcgh_alphaMy_max",
                    "pcgh_alphaMz_max"),
            "reduction": ("pcgh_red_w_max", "pcgh_red_Q_max",
                          "pcgh_red_alpha_max", "pcgh_red_B_max"),
            "curl": ("pcgh_curl_p_max", "pcgh_curl_Q_max",
                     "pcgh_curl_L_max", "pcgh_curl_B_max"),
            "algebraic": ("pcgh_detg_max", "pcgh_trA_max", "pcgh_trQ_max",
                          "pcgh_projection_max"),
        }
        axis = axes[3, 0]
        for label, names in diagnostic_groups.items():
            values = maximum_columns(table, names)
            axis.semilogy(time, np.maximum(values, np.finfo(float).tiny), label=label)
        axis.set_title("all constraint maxima")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7, ncol=2)
        axis = axes[3, 1]
        axis.semilogy(time, np.maximum(table["pcgh_rhs_primary_max"],
                                      np.finfo(float).tiny), label="primary RHS")
        axis.semilogy(time, np.maximum(table["pcgh_rhs_gradient_max"],
                                      np.finfo(float).tiny), label="gradient RHS")
        axis.set_title("RHS maxima")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7)
        for axis in axes[-1, :]:
            axis.set_xlabel(r"$t/M$")
        fig.suptitle(f"PC-GH {gauge}, finest $\\Delta x=M/{n}$: strict bounds")
        fig.tight_layout()
        path = output_dir / f"field_boundedness_{gauge}_N{n}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        products.append(path)

        summary[gauge] = {}
        for resolution in resolutions:
            candidate = boundedness[gauge, resolution]
            field_summary = {}
            for name in ("w", "rho", "p", "L", "Cperp", "Z", "K", "Atilde",
                         "beta", "Q", "B"):
                column = f"max_{name}"
                field_summary[name] = float(np.nanmax(candidate[column]))
            field_summary["minimum_w"] = float(np.nanmin(candidate["min_w"]))
            field_summary["minimum_rho"] = float(np.nanmin(candidate["min_rho"]))
            field_summary["minimum_metric_eigenvalue"] = float(
                np.nanmin(candidate["min_eigenvalue"]))
            field_summary["minimum_metric_determinant"] = float(
                np.nanmin(candidate["min_detg"]))
            field_summary["minimum_metric_minor1"] = float(
                np.nanmin(candidate["min_minor1"]))
            field_summary["minimum_metric_minor2"] = float(
                np.nanmin(candidate["min_minor2"]))
            field_summary["maximum_abs_detg_minus_one"] = float(
                np.nanmax(candidate["max_abs_detg_minus_1"]))
            field_summary["maximum_primary_rhs"] = float(
                np.nanmax(candidate["pcgh_rhs_primary_max"]))
            field_summary["maximum_gradient_rhs"] = float(
                np.nanmax(candidate["pcgh_rhs_gradient_max"]))
            summary[gauge][str(resolution)] = field_summary

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for name in ("w", "rho", "p", "L", "Cperp", "Z", "K", "Atilde",
                     "beta", "Q", "B"):
            axes[0].semilogy(resolutions,
                            [summary[gauge][str(r)][name] for r in resolutions],
                            "o-", label=name)
        axes[0].set(xlabel=r"finest zones per $M$", ylabel="maximum over run",
                    title="regular-field refinement trend")
        axes[0].legend(fontsize=7, ncol=3)
        for name, label in (("minimum_w", "min w"), ("minimum_rho", "min rho"),
                            ("minimum_metric_eigenvalue", "min metric eigenvalue")):
            axes[1].semilogy(resolutions,
                            [summary[gauge][str(r)][name] for r in resolutions],
                            "o-", label=label)
        axes[1].set(xlabel=r"finest zones per $M$", ylabel="minimum over run",
                    title="positivity refinement trend")
        axes[1].legend(fontsize=8)
        for axis in axes:
            axis.set_xticks(resolutions)
            axis.grid(alpha=0.25, which="both")
        fig.suptitle(f"PC-GH {gauge}: boundedness across resolution")
        fig.tight_layout()
        path = output_dir / f"field_boundedness_refinement_{gauge}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        products.append(path)
    return products, summary


def state_field_magnitudes(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    def vector(names):
        return np.sqrt(sum(data[name][0]**2 for name in names))

    return {
        "w": data["pcgh_w"][0],
        "rho": data["pcgh_rho"][0],
        "p": vector(("pcgh_p1", "pcgh_p2", "pcgh_p3")),
        "L": vector(("pcgh_L1", "pcgh_L2", "pcgh_L3")),
        "Cperp": np.abs(data["pcgh_Cperp"][0]),
        "Z": vector(("pcgh_Zx", "pcgh_Zy", "pcgh_Zz")),
        "K": np.abs(data["pcgh_K"][0]),
        "Atilde": vector(("pcgh_Atxx", "pcgh_Atxy", "pcgh_Atxz",
                          "pcgh_Atyy", "pcgh_Atyz", "pcgh_Atzz")),
        "beta": vector(("pcgh_betax", "pcgh_betay", "pcgh_betaz")),
        "Q": vector(tuple(f"pcgh_Q{d}{ij}" for d in (1, 2, 3)
                          for ij in ("xx", "xy", "xz", "yy", "yz", "zz"))),
        "B": vector(tuple(f"pcgh_B{d}{j}" for d in (1, 2, 3)
                          for j in (1, 2, 3))),
    }


def radial_envelope(state: dict, values: np.ndarray, outer_radius: float = 0.75):
    x, y = np.meshgrid(state["x"], state["y"], indexing="xy")
    radius = np.sqrt(x*x + y*y)
    positive_radius = radius[radius > 0.0]
    inner = float(np.min(positive_radius))
    edges = np.geomspace(0.8*inner, outer_radius, 15)
    centers = np.sqrt(edges[:-1]*edges[1:])
    envelope = np.full_like(centers, np.nan)
    for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        selected = (radius >= left) & (radius < right) & np.isfinite(values)
        if np.any(selected):
            envelope[index] = np.max(np.abs(values[selected]))
    return centers, envelope, radius


def plot_puncture_profiles(slice_specs, output_dir: Path, tmax: float) -> tuple[list[Path], dict]:
    products = []
    summary = {}
    fields = ("w", "rho", "p", "L", "Cperp", "Z", "K", "Atilde", "beta", "Q", "B")
    by_gauge = {}
    for gauge, n, directory in slice_specs:
        files = nearest_cart_files(directory, "pcgh_slice", (tmax,))
        if files:
            by_gauge[gauge, n] = files[0][2]
    for gauge in sorted({key[0] for key in by_gauge}):
        resolutions = sorted(n for g, n in by_gauge if g == gauge)
        fig, axes = plt.subplots(3, 4, figsize=(14, 10), sharex=True)
        summary[gauge] = {}
        for field, axis in zip(fields, axes.flat):
            summary[gauge][field] = {}
            inner_maxima = []
            for n in resolutions:
                state = by_gauge[gauge, n]
                values = state_field_magnitudes(state["data"])[field]
                radius_axis, envelope, radius = radial_envelope(state, values)
                valid = np.isfinite(envelope) & (envelope > 0.0) & (radius_axis <= 0.5)
                slope = (float(np.polyfit(np.log(radius_axis[valid]),
                                          np.log(envelope[valid]), 1)[0])
                         if np.count_nonzero(valid) >= 4 else float("nan"))
                selected = (radius <= 0.5) & np.isfinite(values)
                inner_maximum = float(np.max(np.abs(values[selected])))
                inner_maxima.append(inner_maximum)
                summary[gauge][field][str(n)] = {
                    "fitted_inner_power": slope,
                    "inner_r_le_0p5_max": inner_maximum,
                    "sample_time": float(state["time"]),
                }
                axis.loglog(radius_axis, envelope, "o-", ms=3, label=f"N={n}")
            valid_refinement = np.isfinite(inner_maxima) & (np.asarray(inner_maxima) > 0.0)
            exponent = (float(np.polyfit(np.log(np.asarray(resolutions)[valid_refinement]),
                                         np.log(np.asarray(inner_maxima)[valid_refinement]), 1)[0])
                        if np.count_nonzero(valid_refinement) >= 2 else float("nan"))
            summary[gauge][field]["refinement_exponent"] = exponent
            axis.set_title(field)
            axis.grid(alpha=0.25, which="both")
        axes.flat[-1].axis("off")
        axes[0, 0].legend(fontsize=7)
        for axis in axes[-1, :3]:
            axis.set_xlabel(r"coordinate $r/M$")
        for axis in axes[:, 0]:
            axis.set_ylabel("radial max envelope")
        fig.suptitle(f"PC-GH {gauge}: puncture profiles near $t={tmax:g}M$")
        fig.tight_layout()
        path = output_dir / f"puncture_power_{gauge}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        products.append(path)
    return products, summary


def plot_initial_data_comparison(pcgh_specs, z4c_specs, output_dir: Path) -> tuple[list[Path], dict]:
    """Compare chi=w^2 after the changed PC-GH initial-data conversion."""
    pcgh = {}
    for gauge, n, directory in pcgh_specs:
        files = nearest_cart_files(directory, "pcgh_slice", (0.0,))
        if files and (n not in pcgh or gauge == "direct"):
            pcgh[n] = files[0][2]
    z4c = {}
    for _, n, directory in z4c_specs:
        files = nearest_cart_files(directory, "z4c_slice", (0.0,))
        if files:
            z4c[n] = files[0][2]
    common = sorted(set(pcgh) & set(z4c))
    if not common:
        return [], {}
    summary = {}
    errors = []
    for n in common:
        if not (np.array_equal(pcgh[n]["x"], z4c[n]["x"])
                and np.array_equal(pcgh[n]["y"], z4c[n]["y"])):
            raise ValueError(f"PC-GH and Z4c initial grids differ at N={n}")
        difference = pcgh[n]["data"]["pcgh_w"][0]**2 - z4c[n]["data"]["z4c_chi"][0]
        errors.append(float(np.max(np.abs(difference))))
        summary[str(n)] = {
            "max_abs_chi_difference": errors[-1],
            "rms_chi_difference": float(np.sqrt(np.mean(difference*difference))),
        }
    n = common[-1]
    difference = pcgh[n]["data"]["pcgh_w"][0]**2 - z4c[n]["data"]["z4c_chi"][0]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].semilogy(common, np.maximum(errors, np.finfo(float).tiny), "o-")
    axes[0].set(xlabel=r"finest zones per $M$", ylabel=r"$\|w^2-\chi_{Z4c}\|_\infty$",
                title="initial-data conversion consistency")
    axes[0].grid(alpha=0.25)
    image = axes[1].pcolormesh(pcgh[n]["x"], pcgh[n]["y"], difference,
                               shading="nearest", cmap="coolwarm")
    axes[1].set(xlabel=r"$x/M$", ylabel=r"$y/M$", title=f"N={n} signed difference")
    axes[1].set_aspect("equal")
    fig.colorbar(image, ax=axes[1])
    fig.tight_layout()
    path = output_dir / "initial_data_pcgh_z4c.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [path], summary


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
    parser.add_argument("--boundedness", action="append", default=[], metavar="GAUGE:N:PATH")
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
    slice_specs = list(map(parse_spec, args.slice_dir))
    products += plot_slices(slice_specs, args.output_dir, args.tmax)
    boundedness = {(gauge, n): read_named_table(path)
                   for gauge, n, path in map(parse_spec, args.boundedness)}
    boundedness_products, boundedness_summary = plot_boundedness(
        boundedness, args.output_dir) if boundedness else ([], {})
    products += boundedness_products
    puncture_products, puncture_summary = plot_puncture_profiles(
        slice_specs, args.output_dir, args.tmax)
    products += puncture_products
    z4c_histories = {
        n: athena_read.hst(path)
        for _, n, path in map(parse_spec, args.z4c_history)
    }
    z4c_products, z4c_convergence = plot_z4c_controls(
        z4c_histories, args.output_dir, args.tmax)
    products += z4c_products
    z4c_slice_specs = list(map(parse_spec, args.z4c_slice_dir))
    products += plot_z4c_slices(z4c_slice_specs, args.output_dir, args.tmax)
    initial_products, initial_summary = plot_initial_data_comparison(
        slice_specs, z4c_slice_specs, args.output_dir)
    products += initial_products
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
        "boundedness": boundedness_summary,
        "puncture_power": puncture_summary,
        "initial_data_comparison": initial_summary,
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
