#!/usr/bin/env python3
"""Reanalyze authenticated Rout=16 histories and causal-speed contracts.

This script intentionally consumes the committed, authenticated reductions from
the native-authority investigation.  It does not reconstruct spatially
restricted norms; that is handled by ``analyze_radial_constraints.py`` from the
binary leaf outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


CASES = ("n128", "n256", "n512")
FIELDS = ("C-norm2", "H-norm2", "M-norm2", "Z-norm2")
COLORS = {"n128": "#377eb8", "n256": "#4daf4a", "n512": "#e41a1c"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    require(bool(rows), f"empty CSV: {path}")
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    require(bool(rows), f"no rows for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def interpolated_crossing(time: np.ndarray, value: np.ndarray,
                          threshold: float) -> float | None:
    indices = np.flatnonzero(value >= threshold)
    if len(indices) == 0:
        return None
    index = int(indices[0])
    if index == 0:
        return float(time[0])
    x0, x1 = float(time[index - 1]), float(time[index])
    y0, y1 = float(value[index - 1]), float(value[index])
    if y1 == y0:
        return x1
    return x0 + (threshold - y0) * (x1 - x0) / (y1 - y0)


def causal_trace(path: Path, boundary: float) -> tuple[list[dict], dict]:
    raw = read_csv(path)
    by_time: dict[float, float] = {}
    for row in raw:
        time = float(row["time"])
        speed = float(row["max_coordinate_speed"])
        by_time[time] = max(speed, by_time.get(time, -math.inf))
    time = np.asarray(sorted(by_time), dtype=float)
    speed = np.asarray([by_time[value] for value in time], dtype=float)
    require(len(time) > 1 and np.all(np.diff(time) > 0.0),
            f"invalid timestep contract: {path}")
    require(np.isfinite(speed).all() and np.all(speed >= 0.0),
            f"invalid coordinate speed: {path}")
    distance = np.zeros_like(time)
    distance[1:] = np.cumsum(
        0.5 * (speed[1:] + speed[:-1]) * np.diff(time)
    )
    rows = [
        {
            "time": float(t),
            "max_coordinate_speed": float(v),
            "integrated_coordinate_reach": float(d),
            "protected_radius": float(max(0.0, boundary - d)),
        }
        for t, v, d in zip(time, speed, distance)
    ]
    reach = {
        str(radius): interpolated_crossing(time, distance, boundary - radius)
        for radius in (4.0, 8.0, 12.0)
    }
    return rows, {
        "terminal_time": float(time[-1]),
        "maximum_coordinate_speed": float(np.max(speed)),
        "integrated_coordinate_reach": float(distance[-1]),
        "terminal_protected_radius": float(max(0.0, boundary - distance[-1])),
        "first_possible_reach_time_by_radius": reach,
    }


def at_tau(rows: list[dict], field: str, target: float) -> dict:
    selected = [row for row in rows if row["field"] == field]
    tau = np.asarray([float(row["axisTau"]) for row in selected])
    nearest = int(np.argmin(np.abs(tau - target)))
    return selected[nearest]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--convergence", type=Path, required=True)
    for case in CASES:
        parser.add_argument(f"--{case}-timestep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_rows = read_csv(args.convergence)
    rows: list[dict] = []
    for source in source_rows:
        field = source["field"]
        require(field in FIELDS, f"unexpected constraint field {field}")
        values = {case: float(source[case]) for case in CASES}
        direct12 = (math.log2(values["n128"] / values["n256"])
                    if values["n128"] > 0.0 and values["n256"] > 0.0
                    else math.nan)
        direct23 = (math.log2(values["n256"] / values["n512"])
                    if values["n256"] > 0.0 and values["n512"] > 0.0
                    else math.nan)
        rows.append({
            "axisTau": float(source["axisTau"]),
            "field": field,
            **values,
            "p_128_256": direct12,
            "p_256_512": direct23,
            "p_self": float(source["p"]),
            "E_128_256": float(source["E_128_256"]),
            "E_256_512": float(source["E_256_512"]),
        })

    output = args.output
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    write_csv(output / "small_domain_global_convergence.csv", rows)

    causal_summary: dict[str, dict] = {}
    for case in CASES:
        trace, summary = causal_trace(getattr(args, f"{case}_timestep"), 16.0)
        write_csv(output / f"small_domain_causality_{case}.csv", trace)
        causal_summary[case] = summary

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), constrained_layout=True)
    for axis, field in zip(axes.flat, FIELDS):
        selected = [row for row in rows if row["field"] == field]
        for case in CASES:
            axis.semilogy([row["axisTau"] for row in selected],
                          [row[case] for row in selected],
                          color=COLORS[case], label=case.upper())
        axis.set_title(field)
        axis.set_xlabel(r"central proper time $\tau_c/M$")
        axis.grid(alpha=0.24, which="both")
    axes[0, 0].legend()
    fig.suptitle(r"Authenticated $R_{out}=16M$ global constraints")
    fig.savefig(figures / "original_rout16_global_figure2.png", dpi=240)
    fig.savefig(figures / "original_rout16_global_figure2.pdf")
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12.4, 8.2), constrained_layout=True)
    for axis, field in zip(axes.flat, FIELDS):
        selected = [row for row in rows if row["field"] == field]
        axis.plot([row["axisTau"] for row in selected],
                  [row["p_128_256"] for row in selected],
                  label="direct N128-N256", color="#377eb8")
        axis.plot([row["axisTau"] for row in selected],
                  [row["p_256_512"] for row in selected],
                  label="direct N256-N512", color="#e41a1c")
        axis.plot([row["axisTau"] for row in selected],
                  [row["p_self"] for row in selected],
                  label="self difference", color="#666666", alpha=0.75)
        axis.axhline(4.0, color="black", linestyle="--", linewidth=0.8)
        axis.axhline(0.0, color="black", linewidth=0.6)
        axis.set_ylim(-2.0, 10.0)
        axis.set_title(field)
        axis.set_xlabel(r"central proper time $\tau_c/M$")
        axis.set_ylabel("observed order")
        axis.grid(alpha=0.24)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Direct zero-solution orders versus self-difference order")
    fig.savefig(figures / "small_domain_global_pairwise_orders.png", dpi=240)
    fig.savefig(figures / "small_domain_global_pairwise_orders.pdf")
    plt.close(fig)

    terminal_tau = min(max(float(row["axisTau"]) for row in rows
                           if row["field"] == field) for field in FIELDS)
    samples: dict[str, dict] = {}
    for field in FIELDS:
        selected = [row for row in rows if row["field"] == field]
        tau = np.asarray([row["axisTau"] for row in selected], dtype=float)
        samples[field] = {
            "tau_2": at_tau(rows, field, 2.0),
            "tau_3": at_tau(rows, field, 3.0),
            "terminal": at_tau(rows, field, terminal_tau),
            "early_median_tau_0p5_to_2": {
                key: float(np.nanmedian([row[key] for row in selected
                                         if 0.5 <= row["axisTau"] <= 2.0]))
                for key in ("p_128_256", "p_256_512", "p_self")
            },
            "late_minimum_tau_ge_3": {
                key: float(np.nanmin([row[key] for row in selected
                                      if row["axisTau"] >= 3.0]))
                for key in ("p_128_256", "p_256_512", "p_self")
            },
        }
    summary = {
        "schema": "z4c_vc_boundary_ko_small_domain_global_v1",
        "source_convergence_csv": str(args.convergence.resolve()),
        "terminal_common_axis_tau": terminal_tau,
        "constraint_samples": samples,
        "causality": causal_summary,
    }
    (output / "small_domain_global_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
