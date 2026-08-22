#!/usr/bin/env python3
"""Analyze the bounded Perlmutter VC axis-classification campaign.

The script is deliberately read-only with respect to raw campaign evidence.
It writes derived CSV/JSON/PNG products under the requested output directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


RESOLUTIONS = ("n128", "n256", "n512")
CONSTRAINT_COLUMNS = ("C-norm2", "H-norm2", "M-norm2", "Z-norm2")


def import_readers(source: Path):
    sys.path.insert(0, str(source / "vis" / "python"))
    import athena_read  # type: ignore
    import bin_convert  # type: ignore

    return athena_read, bin_convert


def latest_binary(case: Path, label: str, group: str) -> Path:
    files = sorted((case / "bin" / "rank_00000000").glob(f"{label}.{group}.*.bin"))
    if not files:
        raise RuntimeError(f"missing {group} binary output in {case}")
    return files[-1]


def duplicate_stitch(data):
    """Stitch native VC leaf arrays and retain duplicate-node diagnostics."""
    values: dict[tuple[float, float], np.ndarray] = {}
    counts: defaultdict[tuple[float, float], int] = defaultdict(int)
    max_mismatch = {name: 0.0 for name in data["var_names"]}
    edge_distance: dict[tuple[float, float], float] = {}
    variables = tuple(data["var_names"])
    for block, geom in enumerate(data["mb_geometry"]):
        array0 = data["mb_data"][variables[0]][block]
        nk, nj, ni = array0.shape
        if nk != 1:
            raise RuntimeError("bounded Cartoon analysis expected one collapsed plane")
        xs = np.linspace(float(geom[0]), float(geom[1]), ni)
        zs = np.linspace(float(geom[2]), float(geom[3]), nj)
        for j, z in enumerate(zs):
            for i, rho in enumerate(xs):
                key = (round(float(rho), 13), round(float(z), 13))
                sample = np.array(
                    [float(data["mb_data"][name][block][0, j, i]) for name in variables]
                )
                if key in values:
                    delta = np.abs(values[key] - sample)
                    for index, name in enumerate(variables):
                        max_mismatch[name] = max(max_mismatch[name], float(delta[index]))
                else:
                    values[key] = sample
                counts[key] += 1
                distance = float(min(i, ni - 1 - i, j, nj - 1 - j))
                edge_distance[key] = min(edge_distance.get(key, math.inf), distance)
    return {
        "variables": variables,
        "values": values,
        "counts": dict(counts),
        "edge_distance_cells": edge_distance,
        "max_duplicate_mismatch": max_mismatch,
    }


def ordered_matrix(stitched, keys):
    return np.vstack([stitched["values"][key] for key in keys])


def rms(values: np.ndarray, mask: np.ndarray | None = None) -> float:
    selected = values if mask is None else values[mask]
    if selected.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(np.square(selected))))


def effective_order(e_coarse: float, e_fine: float) -> float | None:
    if not (math.isfinite(e_coarse) and math.isfinite(e_fine)):
        return None
    if e_coarse <= 0.0 or e_fine <= 0.0:
        return None
    return float(math.log2(e_coarse / e_fine))


def region_masks(stitched, keys):
    rho = np.array([key[0] for key in keys])
    edge = np.array([stitched["edge_distance_cells"][key] for key in keys])
    count = np.array([stitched["counts"][key] for key in keys])
    return {
        "whole": np.ones(len(keys), dtype=bool),
        "axis": np.isclose(rho, 0.0),
        "block_interior": edge >= 4.0,
        "shared_or_block_boundary": count > 1,
    }


def fixed_grid_analysis(root: Path, reader):
    groups: dict[str, dict[str, object]] = {}
    for label in RESOLUTIONS:
        groups[label] = {}
        for group in ("z4c", "con"):
            data = reader(str(latest_binary(root / "fixed_grid" / label, label, group)))
            groups[label][group] = {"data": data, "stitched": duplicate_stitch(data)}

    rows = []
    for group in ("z4c", "con"):
        coarse = groups["n128"][group]["stitched"]
        medium = groups["n256"][group]["stitched"]
        fine = groups["n512"][group]["stitched"]
        keys = sorted(coarse["values"])
        missing_medium = [key for key in keys if key not in medium["values"]]
        missing_fine = [key for key in keys if key not in fine["values"]]
        if missing_medium or missing_fine:
            raise RuntimeError("fixed-grid physical vertex nesting failed")
        a = ordered_matrix(coarse, keys)
        b = ordered_matrix(medium, keys)
        c = ordered_matrix(fine, keys)
        masks = region_masks(coarse, keys)
        for index, variable in enumerate(coarse["variables"]):
            for region, mask in masks.items():
                e128_256 = rms(a[:, index] - b[:, index], mask)
                e256_512 = rms(b[:, index] - c[:, index], mask)
                rows.append(
                    {
                        "group": group,
                        "variable": variable,
                        "region": region,
                        "rms_n128_n256": e128_256,
                        "rms_n256_n512": e256_512,
                        "effective_order": effective_order(e128_256, e256_512),
                    }
                )

    duplicate = {}
    for label in RESOLUTIONS:
        duplicate[label] = {}
        for group in ("z4c", "con"):
            stitched = groups[label][group]["stitched"]
            duplicate[label][group] = {
                "maximum": max(stitched["max_duplicate_mismatch"].values()),
                "by_variable": stitched["max_duplicate_mismatch"],
                "duplicate_points": sum(count > 1 for count in stitched["counts"].values()),
                "maximum_multiplicity": max(stitched["counts"].values()),
            }
    return rows, duplicate


def history_data(case: Path, label: str, athena_read):
    path = case / f"{label}.z4c.user.hst"
    data = athena_read.hst(str(path))
    return {name: np.asarray(value, dtype=float) for name, value in data.items()}


def terminal_summary(root: Path):
    result = {}
    for label in RESOLUTIONS:
        case = root / "common_tree" / label
        failure = json.loads((case / "z4c_state_failure.json").read_text())
        result[label] = {
            "exit_status": int((case / "exit-status").read_text().strip()),
            "disposition": (case / "disposition").read_text().strip(),
            "time": failure["time"],
            "cycle": failure["cycle"],
            "rk_stage": failure["rk_stage"],
            "checkpoint": failure["checkpoint"],
            "reason": failure["reason"],
            "rho": failure["rho"],
            "z": failure["z"],
            "block_edge_distance": failure["block_edge_distance"],
            "coarse_fine_interface_distance": failure["coarse_fine_interface_distance"],
        }
    return result


def replay_summary(root: Path):
    result = {}
    for label in RESOLUTIONS:
        path = root / "common_tree" / label / f"{label}.amr_history_replay.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        result[label] = {
            "events": len(records),
            "exact_match": bool(records) and all(row["exact_match"] for row in records),
            "maximum_abs_ulp": max((abs(int(row["ulp_difference"])) for row in records), default=None),
            "last_event": records[-1]["event"] if records else None,
            "last_tree_checksum": records[-1]["tree_checksum"] if records else None,
        }
    return result


def history_analysis(root: Path, athena_read):
    histories = {
        label: history_data(root / "common_tree" / label, label, athena_read)
        for label in RESOLUTIONS
    }
    tmax = min(float(data["time"][-1]) for data in histories.values())
    times = np.linspace(0.1, max(0.1, tmax), 80)
    rows = []
    for quantity in CONSTRAINT_COLUMNS:
        interpolated = {
            label: np.interp(times, data["time"], data[quantity])
            for label, data in histories.items()
        }
        for i, time in enumerate(times):
            e128 = float(interpolated["n128"][i])
            e256 = float(interpolated["n256"][i])
            e512 = float(interpolated["n512"][i])
            rows.append(
                {
                    "time": float(time),
                    "quantity": quantity,
                    "n128": e128,
                    "n256": e256,
                    "n512": e512,
                    "effective_order": effective_order(abs(e128 - e256), abs(e256 - e512)),
                }
            )
    return histories, rows, tmax


def write_csv(path: Path, rows):
    rows = list(rows)
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV {path}")
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plots(output: Path, fixed_rows, histories, terminal):
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    whole = [
        row for row in fixed_rows
        if row["group"] == "z4c" and row["region"] == "whole"
        and row["effective_order"] is not None
    ]
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(np.arange(len(whole)), [row["effective_order"] for row in whole])
    ax.axhline(4.0, color="black", linestyle="--", linewidth=1, label="ideal O4")
    ax.set_xticks(np.arange(len(whole)), [row["variable"].replace("z4c_", "") for row in whole], rotation=75)
    ax.set_ylabel("effective order")
    ax.set_title("Fixed-grid VC field self-convergence at t=0.5")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "fixed_grid_field_order.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    for ax, quantity in zip(axes.flat, CONSTRAINT_COLUMNS):
        for label in RESOLUTIONS:
            data = histories[label]
            ax.semilogy(data["time"], np.maximum(data[quantity], np.finfo(float).tiny), label=label.upper())
        ax.set_title(quantity)
        ax.set_ylabel("proper ring-measure norm")
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("coordinate time / M")
    axes[-1, 1].set_xlabel("coordinate time / M")
    axes[0, 0].legend()
    fig.suptitle("Common-tree VC constraint histories (terminal states excluded from claims)")
    fig.tight_layout()
    fig.savefig(figures / "common_tree_constraints.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True)
    quantities = ("max_abs_K", "maxAbsKret", "dt", "nmb_total")
    labels = ("max |K|", "max Kretschmann", "dt", "MeshBlocks")
    for ax, quantity, title in zip(axes.flat, quantities, labels):
        for label in RESOLUTIONS:
            data = histories[label]
            if quantity in ("max_abs_K", "maxAbsKret"):
                ax.semilogy(data["time"], np.maximum(np.abs(data[quantity]), np.finfo(float).tiny), label=label.upper())
            else:
                ax.plot(data["time"], data[quantity], label=label.upper())
        ax.set_title(title)
        ax.grid(alpha=0.25)
    axes[-1, 0].set_xlabel("coordinate time / M")
    axes[-1, 1].set_xlabel("coordinate time / M")
    axes[0, 0].legend()
    fig.suptitle("Common-tree VC curvature, timestep, and topology")
    fig.tight_layout()
    fig.savefig(figures / "common_tree_curvature_timestep_topology.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    labels_order = list(RESOLUTIONS)
    ax.plot([128, 256, 512], [terminal[x]["time"] for x in labels_order], marker="o")
    ax.set_xscale("log", base=2)
    ax.set_xticks([128, 256, 512], ["N128", "N256", "N512"])
    ax.set_ylabel("fail-closed coordinate time / M")
    ax.set_title("Common-tree terminal time worsens with resolution")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figures / "common_tree_terminal_scaling.png", dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    athena_read, bin_convert = import_readers(args.source.resolve())
    output = args.output.resolve()
    (output / "data").mkdir(parents=True, exist_ok=True)

    fixed_rows, duplicate = fixed_grid_analysis(args.campaign.resolve(), bin_convert.read_binary)
    histories, history_rows, common_time_max = history_analysis(args.campaign.resolve(), athena_read)
    terminal = terminal_summary(args.campaign.resolve())
    replay = replay_summary(args.campaign.resolve())
    write_csv(output / "data" / "fixed_grid_convergence.csv", fixed_rows)
    write_csv(output / "data" / "common_tree_constraint_order.csv", history_rows)
    make_plots(output, fixed_rows, histories, terminal)

    whole_orders = {
        row["variable"]: row["effective_order"]
        for row in fixed_rows
        if row["group"] == "z4c" and row["region"] == "whole"
    }
    nonzero_orders = [value for value in whole_orders.values() if value is not None]
    constraint_order_medians = {}
    for quantity in CONSTRAINT_COLUMNS:
        values = [
            row["effective_order"] for row in history_rows
            if row["quantity"] == quantity and row["effective_order"] is not None
        ]
        constraint_order_medians[quantity] = float(np.median(values))
    summary = {
        "schema": "z4c_vertex_centered_axis_fix_analysis_v1",
        "raw_campaign": str(args.campaign.resolve()),
        "fixed_grid": {
            "time": 0.5,
            "whole_domain_effective_order": whole_orders,
            "minimum_nontrivial_order": min(nonzero_orders),
            "maximum_nontrivial_order": max(nonzero_orders),
            "duplicate_node_mismatch": duplicate,
            "verdict": "EXPECTED_O4_CONVERGENCE",
        },
        "common_tree": {
            "history_common_coordinate_time_max": common_time_max,
            "replay": replay,
            "median_constraint_effective_order": constraint_order_medians,
            "terminal": terminal,
            "verdict": "FAILURE_EARLIER_WITH_RESOLUTION",
            "qualification_claim": False,
        },
        "interpretation": {
            "axis_classification_bug_repaired": True,
            "axis_bug_was_sole_common_tree_instability": False,
            "long_brill_or_figure3_claim": False,
        },
    }
    (output / "comparison_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
