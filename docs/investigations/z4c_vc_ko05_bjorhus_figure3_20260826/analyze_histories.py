#!/usr/bin/env python3
"""Reduce the KO=0.5 native-VC common-tree history campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASES = ("n128", "n256", "n512")
CONSTRAINTS = ("C-norm2", "H-norm2", "M-norm2", "Z-norm2")
HEADER = re.compile(r"\[(\d+)\]=([^\s]+)")
COLORS = {"n128": "#377eb8", "n256": "#4daf4a", "n512": "#e41a1c"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_history(paths: list[Path]) -> dict[str, np.ndarray]:
    """Concatenate restart segments, retaining the last row at duplicate times."""
    labels: dict[str, int] = {}
    rows: dict[float, list[float]] = {}
    for path in paths:
        require(path.is_file(), f"missing history {path}")
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                labels.update({name: int(index) - 1 for index, name in HEADER.findall(line)})
            elif line.strip():
                row = [float(value) for value in line.split()]
                rows[row[labels["time"]]] = row
    required = {
        "time", "dt", "cycle", "axisTau", "axisKret", "axisLapse", "max_abs_K",
        "maxAbsKret", "maxRefLev", "nmb_total", "C-Linf", "C-rho", "C-z",
        *CONSTRAINTS,
    }
    require(not required - labels.keys(), f"history missing {sorted(required - labels.keys())}")
    array = np.asarray([rows[key] for key in sorted(rows)], dtype=float)
    require(array.ndim == 2 and len(array) > 1, "history has too few rows")
    require(np.isfinite(array).all(), "history contains nonfinite values")
    require(np.all(np.diff(array[:, labels["time"]]) > 0.0), "history time is nonmonotone")
    return {name: array[:, index] for name, index in labels.items()}


def read_jsonl(path: Path) -> list[dict]:
    require(path.is_file(), f"missing JSONL {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def authority_events(path: Path) -> list[dict]:
    events = [row for row in read_jsonl(path) if row.get("type") == "event"]
    require(events and [row["event"] for row in events] == list(range(len(events))),
            "authority event sequence is incomplete")
    return events


def verify_replay(path: Path, authority: list[dict]) -> list[dict]:
    rows = read_jsonl(path)
    require(rows and all(row.get("exact_match") is True for row in rows),
            f"{path}: non-exact replay row")
    expected = authority[1:len(rows) + 1]
    require([row["event"] for row in rows] == [row["event"] for row in expected],
            f"{path}: event indices differ")
    require([row["tree_checksum"] for row in rows] ==
            [row["tree_checksum"] for row in expected], f"{path}: tree checksums differ")
    return rows


def interpolate(table: dict[str, np.ndarray], field: str, time: np.ndarray) -> np.ndarray:
    return np.interp(time, table["time"], table[field])


def write_csv(path: Path, rows: list[dict]) -> None:
    require(rows, f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def pair_inventory_order(coarse: float, fine: float) -> float:
    """Amplitude order for squared inventories, as required by the campaign."""
    return 0.5 * math.log2(coarse / fine) if coarse > 0.0 and fine > 0.0 else math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    for case in CASES:
        parser.add_argument(f"--{case}", type=Path, nargs="+", required=True)
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--n128-replay", type=Path, required=True)
    parser.add_argument("--n512-replay", type=Path, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    histories = {case: read_history(getattr(args, case)) for case in CASES}
    authority = authority_events(args.authority)
    replays = {
        "n128": verify_replay(args.n128_replay, authority),
        "n512": verify_replay(args.n512_replay, authority),
    }
    require(all(len(rows) == len(authority) - 1 for rows in replays.values()),
            "a replay did not consume the full authority")

    common_end = min(float(table["time"][-1]) for table in histories.values())
    time = np.linspace(0.0, common_end, 1001)
    inventory_rows: list[dict] = []
    for field in CONSTRAINTS:
        values = {case: interpolate(table, field, time) for case, table in histories.items()}
        for index, coordinate_time in enumerate(time):
            inventory_rows.append({
                "time": coordinate_time,
                "field": field,
                "n128_inventory": values["n128"][index],
                "n256_inventory": values["n256"][index],
                "n512_inventory": values["n512"][index],
                "q128_256_amplitude": pair_inventory_order(
                    values["n128"][index], values["n256"][index]),
                "q256_512_amplitude": pair_inventory_order(
                    values["n256"][index], values["n512"][index]),
            })

    tau_start = max(float(table["axisTau"][0]) for table in histories.values())
    tau_end = min(float(table["axisTau"][-1]) for table in histories.values())
    require(tau_end > tau_start, "empty common proper-time interval")
    tau = np.linspace(tau_start, tau_end, 1001)
    central_rows: list[dict] = []
    for index, proper_time in enumerate(tau):
        value = {
            case: float(np.interp(proper_time, table["axisTau"], table["axisKret"]))
            for case, table in histories.items()
        }
        e_coarse = abs(value["n128"] - value["n256"])
        e_fine = abs(value["n256"] - value["n512"])
        central_rows.append({
            "axisTau": proper_time,
            "n128_axisKret": value["n128"],
            "n256_axisKret": value["n256"],
            "n512_axisKret": value["n512"],
            "E128_256": e_coarse,
            "E256_512": e_fine,
            "observed_order": math.log2(e_coarse / e_fine)
            if e_coarse > 0.0 and e_fine > 0.0 else math.nan,
        })

    history_rows: list[dict] = []
    columns = ("time", "axisTau", "dt", "cycle", *CONSTRAINTS, "axisKret",
               "axisLapse", "max_abs_K", "maxAbsKret", "maxRefLev", "nmb_total",
               "C-Linf", "C-rho", "C-z", "H-Linf", "H-rho", "H-z",
               "M-Linf", "M-rho", "M-z", "Z-Linf", "Z-rho", "Z-z")
    for case, table in histories.items():
        for index in range(len(table["time"])):
            history_rows.append({"resolution": case, **{
                field: float(table[field][index]) for field in columns
            }})

    output = args.output
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(output / "history_compact.csv", history_rows)
    write_csv(output / "constraint_inventory_orders.csv", inventory_rows)
    write_csv(output / "central_observable_convergence.csv", central_rows)

    reference: dict[str, tuple[list[float], list[float]]] = {}
    with args.reference.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            xx, yy = reference.setdefault(row["series"], ([], []))
            xx.append(float(row["tau"]))
            yy.append(float(row["log10_abs_I"]))

    fig, axis = plt.subplots(figsize=(10.4, 6.2), constrained_layout=True)
    for name, (xx, yy) in reference.items():
        if name in ("bamps", "prague"):
            axis.plot(xx, yy, linewidth=1.35, alpha=0.70, label=f"paper {name}")
    for case in CASES:
        table = histories[case]
        mask = np.abs(table["axisKret"]) > 0.0
        axis.plot(table["axisTau"][mask], np.log10(np.abs(table["axisKret"][mask])),
                  color=COLORS[case], linewidth=1.5, label=f"AthenaK {case.upper()}")
    axis.set_xlim(0.0, 15.0)
    axis.set_ylim(-8.0, 8.0)
    axis.set_xlabel(r"central proper time $\tau$")
    axis.set_ylabel(r"$\log_{10}|I(0)|$")
    axis.grid(alpha=0.22)
    axis.legend(ncol=2, fontsize=8)
    fig.savefig(figures / "figure3_three_resolution.png", dpi=240)
    fig.savefig(figures / "figure3_three_resolution.pdf")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10.4, 5.6), constrained_layout=True)
    for name, (xx, yy) in reference.items():
        if name in ("bamps", "prague"):
            axis.plot(xx, yy, linewidth=1.35, alpha=0.70, label=f"paper {name}")
    for case in CASES:
        table = histories[case]
        mask = np.abs(table["axisKret"]) > 0.0
        axis.plot(table["axisTau"][mask], np.log10(np.abs(table["axisKret"][mask])),
                  color=COLORS[case], linewidth=1.5, label=f"AthenaK {case.upper()}")
    axis.set_xlim(8.0, 14.5)
    axis.set_ylim(-8.0, 7.0)
    axis.set_xlabel(r"central proper time $\tau$")
    axis.set_ylabel(r"$\log_{10}|I(0)|$")
    axis.set_title("Published first peak, minimum, and rebound region")
    axis.grid(alpha=0.22)
    axis.legend(ncol=2, fontsize=8)
    fig.savefig(figures / "figure3_peak_minimum_rebound_zoom.png", dpi=240)
    fig.savefig(figures / "figure3_peak_minimum_rebound_zoom.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    for axis, field in zip(axes.flat, CONSTRAINTS):
        for case in CASES:
            table = histories[case]
            axis.semilogy(table["time"], table[field], color=COLORS[case],
                          label=case.upper())
        axis.set_title(field)
        axis.set_xlabel("coordinate time")
        axis.grid(alpha=0.25, which="both")
    axes[0, 0].legend()
    fig.savefig(figures / "constraint_inventories.png", dpi=220)
    fig.savefig(figures / "constraint_inventories.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0), constrained_layout=True)
    for axis, field in zip(axes.flat, CONSTRAINTS):
        selected = [row for row in inventory_rows if row["field"] == field]
        axis.plot([row["time"] for row in selected],
                  [row["q128_256_amplitude"] for row in selected],
                  color=COLORS["n128"], label="q128-256")
        axis.plot([row["time"] for row in selected],
                  [row["q256_512_amplitude"] for row in selected],
                  color=COLORS["n512"], label="q256-512")
        axis.axhline(4.0, color="black", linestyle="--", linewidth=0.8)
        axis.axhline(0.0, color="gray", linewidth=0.7)
        axis.set_title(field)
        axis.set_xlabel("coordinate time")
        axis.set_ylabel("constraint-amplitude order q")
        axis.grid(alpha=0.25)
    axes[0, 0].legend()
    fig.savefig(figures / "constraint_amplitude_orders.png", dpi=220)
    fig.savefig(figures / "constraint_amplitude_orders.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(4, 1, figsize=(11.0, 10.0), sharex=True,
                             constrained_layout=True)
    for case in CASES:
        table = histories[case]
        axes[0].semilogy(table["time"], table["dt"], color=COLORS[case],
                         label=case.upper())
        axes[1].step(table["time"], table["maxRefLev"], where="post",
                     color=COLORS[case])
        axes[2].step(table["time"], table["nmb_total"], where="post",
                     color=COLORS[case])
        axes[3].semilogy(table["time"], np.maximum(table["C-Linf"], 1e-300),
                         color=COLORS[case])
    axes[0].set_ylabel("dt")
    axes[0].legend()
    axes[1].set_ylabel("max level")
    axes[2].set_ylabel("MeshBlocks")
    axes[3].set_ylabel("C Linf")
    axes[3].set_xlabel("coordinate time")
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
    fig.savefig(figures / "timestep_level_meshblocks_mode.png", dpi=220)
    fig.savefig(figures / "timestep_level_meshblocks_mode.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 8.2), sharex=True,
                             constrained_layout=True)
    for case in CASES:
        table = histories[case]
        axes[0].semilogy(table["time"], np.maximum(table["C-Linf"], 1e-300),
                         color=COLORS[case], label=case.upper())
        axes[1].plot(table["time"], table["C-rho"], color=COLORS[case])
        axes[2].plot(table["time"], table["C-z"], color=COLORS[case])
    axes[0].set_ylabel("C Linf")
    axes[0].legend()
    axes[1].set_ylabel("rho of C max")
    axes[1].axhline(5.0, color="black", linestyle=":", linewidth=0.8)
    axes[2].set_ylabel("z of C max")
    axes[2].set_xlabel("coordinate time")
    for axis in axes:
        axis.grid(alpha=0.25, which="both")
    fig.savefig(figures / "rho5_constraint_mode.png", dpi=220)
    fig.savefig(figures / "rho5_constraint_mode.pdf")
    plt.close(fig)

    terminal = {case: {field: float(table[field][-1]) for field in columns}
                for case, table in histories.items()}
    summary = {
        "schema": "z4c_vc_ko05_history_analysis_v1",
        "authority_events_including_initial": len(authority),
        "authority_last": {key: authority[-1].get(key) for key in
                           ("event", "time", "cycle", "leaf_count", "max_level",
                            "tree_checksum")},
        "replay": {case: {"events": len(rows), "all_exact": True,
                           "last_event": rows[-1]["event"]}
                   for case, rows in replays.items()},
        "common_coordinate_time": [0.0, common_end],
        "common_axis_proper_time": [tau_start, tau_end],
        "terminal": terminal,
        "claim_limits": [
            "pairwise constraint q is computed from squared inventories as 0.5*log2(Ih/Ih2)",
            "central-observable order is a three-level difference diagnostic and is singular near crossings",
            "no stability or convergence claim follows from exact tree replay alone",
        ],
    }
    (output / "history_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
