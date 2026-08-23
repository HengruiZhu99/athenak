#!/usr/bin/env python3
"""Role- and region-resolved fixed-grid Brill t=5 RHS/constraint audit."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


STATE_PREFIX = "state_z4c_"
RHS_PREFIX = "rhs_z4c_"
CONSTRAINTS = ("con_C", "con_H", "con_M", "con_Z", "con_Mx", "con_My", "con_Mz")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load(run: Path) -> list[dict[str, str]]:
    rows = []
    paths = sorted(run.glob("rhs.rank*.csv")) + sorted(run.glob("rhs.rank*.csv.gz"))
    for path in paths:
        context = (gzip.open(path, "rt", newline="", encoding="utf-8")
                   if path.suffix == ".gz"
                   else path.open(newline="", encoding="utf-8"))
        with context as stream:
            rows.extend(csv.DictReader(stream))
    require(rows and all(row["schema"] == "z4c_vc_rhs_field_v2" for row in rows),
            f"invalid terminal RHS diagnostic in {run}")
    require(all(abs(float(row["time"]) - 5.0) < 1.0e-12 and
                int(row["rk_stage"]) == 1 for row in rows),
            f"terminal diagnostic in {run} is not the t=5 first restart stage")
    return rows


def physical_key(row: dict[str, str]) -> tuple[float, float]:
    return round(float(row["rho"]), 13), round(float(row["x2"]), 13)


def columns(row: dict[str, str], prefix: str) -> list[str]:
    return [name for name in row if name.startswith(prefix)]


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(values * values)))


def ring_rms(values: np.ndarray, keys: list[tuple[float, float]]) -> float:
    weights = np.asarray([key[0] for key in keys])
    require(float(np.sum(weights)) > 0.0, "ring measure has no positive radius")
    return float(np.sqrt(np.sum(weights * values * values) / np.sum(weights)))


def observed_order(coarse: float, fine: float) -> float | None:
    return math.log(coarse / fine, 2.0) if coarse > 0.0 and fine > 0.0 else None


def region(row: dict[str, str], name: str) -> bool:
    rho, zed = float(row["rho"]), float(row["x2"])
    radius = math.hypot(rho, zed)
    dx = (float(row["x1max"]) - float(row["x1min"])) / int(row["nx1_intervals"])
    edge_distance = int(row["local_edge_distance"]) * dx
    if name == "full":
        return True
    if name == "core_r8":
        return radius <= 8.0
    if name == "core_r12":
        return radius <= 12.0
    if name == "core_r8_mb_interior":
        return radius <= 8.0 and edge_distance > 0.5
    if name == "core_r12_mb_interior":
        return radius <= 12.0 and edge_distance > 0.5
    if name == "axis":
        return row["role"] == "axis"
    if name == "same_level":
        return row["role"] == "shared_same_level"
    if name == "physical_boundary":
        return row["role"] == "physical_boundary"
    if name == "independent_interior":
        return row["role"] == "independent_interior"
    raise ValueError(name)


def prepare(resolution: int, rows: list[dict[str, str]]) -> dict[str, object]:
    state_columns = columns(rows[0], STATE_PREFIX)
    rhs_columns = columns(rows[0], RHS_PREFIX)
    require(len(state_columns) == 25 and len(rhs_columns) == 25,
            "terminal diagnostic does not contain all evolved state/RHS fields")
    groups: dict[tuple[float, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[physical_key(row)].append(row)
    owners = {physical_key(row): row for row in rows
              if int(row["canonical_owner"]) == 1}
    require(len(owners) == (129 * 257),
            f"N{resolution} stride did not preserve the common N128 vertex lattice")
    shared_spread = {"state": 0.0, "rhs": 0.0, "constraints": 0.0}
    family_columns = {"state": state_columns, "rhs": rhs_columns,
                      "constraints": list(CONSTRAINTS)}
    for group in groups.values():
        if len(group) <= 1:
            continue
        for family, names in family_columns.items():
            for name in names:
                values = [float(row[name]) for row in group]
                shared_spread[family] = max(
                    shared_spread[family], max(values) - min(values))
    return {"resolution": resolution, "owners": owners,
            "columns": family_columns, "shared_spread": shared_spread,
            "cycle": int(rows[0]["cycle"]), "rows": len(rows)}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", nargs=2, action="append", metavar=("N", "PATH"),
                        required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    cases = sorted((prepare(int(n), load(Path(path).resolve()))
                    for n, path in args.run), key=lambda item: item["resolution"])
    require([case["resolution"] for case in cases] == [128, 256, 512],
            "terminal RHS analyzer requires N128/N256/N512")
    require(cases[0]["columns"] == cases[1]["columns"] == cases[2]["columns"],
            "terminal diagnostic inventory changed with resolution")

    regions = ("full", "core_r8", "core_r12", "core_r8_mb_interior",
               "core_r12_mb_interior", "axis", "same_level",
               "physical_boundary", "independent_interior")
    difference_rows: list[dict[str, object]] = []
    pair_metrics = []
    for coarse, fine in zip(cases, cases[1:]):
        shared_keys = coarse["owners"].keys() & fine["owners"].keys()
        require(len(shared_keys) == len(coarse["owners"]),
                "fine diagnostic omitted common physical vertices")
        pair = {}
        for region_name in regions:
            keys = sorted(key for key in shared_keys
                          if region(coarse["owners"][key], region_name))
            if not keys:
                continue
            pair[region_name] = {}
            for family, names in coarse["columns"].items():
                pair[region_name][family] = {}
                for name in names:
                    values = np.asarray([
                        float(fine["owners"][key][name]) -
                        float(coarse["owners"][key][name]) for key in keys])
                    maximum = int(np.argmax(np.abs(values)))
                    metrics = {"rms": rms(values),
                               "ring_rms": (ring_rms(values, keys)
                                            if any(key[0] > 0.0 for key in keys)
                                            else None),
                               "linf": float(np.max(np.abs(values))),
                               "linf_rho": keys[maximum][0],
                               "linf_z": keys[maximum][1]}
                    pair[region_name][family][name] = metrics
                    difference_rows.append({
                        "region": region_name, "family": family,
                        "variable": name,
                        "coarse_resolution": coarse["resolution"],
                        "fine_resolution": fine["resolution"], **metrics})
        pair_metrics.append(pair)

    order_rows: list[dict[str, object]] = []
    for region_name in pair_metrics[0]:
        for family in pair_metrics[0][region_name]:
            for name, coarse_metrics in pair_metrics[0][region_name][family].items():
                fine_metrics = pair_metrics[1][region_name][family][name]
                order_rows.append({
                    "region": region_name, "family": family, "variable": name,
                    "rms_order": observed_order(coarse_metrics["rms"],
                                                fine_metrics["rms"]),
                    "ring_rms_order": observed_order(coarse_metrics["ring_rms"],
                                                     fine_metrics["ring_rms"])
                    if coarse_metrics["ring_rms"] is not None else None,
                    "linf_order": observed_order(coarse_metrics["linf"],
                                                 fine_metrics["linf"]),
                })
    write_csv(output / "differences.csv", difference_rows)
    write_csv(output / "orders.csv", order_rows)
    summary = {
        "schema": "z4c_vc_fixed_brill_terminal_rhs_v1",
        "runs": [{key: case[key] for key in
                  ("resolution", "cycle", "rows", "shared_spread")}
                 for case in cases],
        "orders": order_rows,
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
