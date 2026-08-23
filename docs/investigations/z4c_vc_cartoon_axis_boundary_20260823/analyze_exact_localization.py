#!/usr/bin/env python3
"""Analyze exact-time VC Brill state, RHS, constraints, and RHS term censuses."""

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
TARGETS = {"0000": 0.0, "0125": 0.125, "0250": 0.25, "0500": 0.5,
           "0750": 0.75, "1000": 1.0, "1250": 1.25}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def load_rows(path: Path, target_time: float) -> list[dict[str, str]]:
    with gzip.open(path, "rt", newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream)
                if int(row["rk_stage"]) == 1 and
                abs(float(row["time"]) - target_time) < 2.0e-14]
    require(rows and all(row["schema"] == "z4c_vc_rhs_field_v2" for row in rows),
            f"missing exact stage-1 rows in {path}")
    return rows


def key(row: dict[str, str]) -> tuple[float, float]:
    return round(float(row["rho"]), 13), round(float(row["x2"]), 13)


def columns(row: dict[str, str], prefix: str) -> list[str]:
    return [name for name in row if name.startswith(prefix)]


def prepare(resolution: int, rows: list[dict[str, str]]) -> dict:
    families = {"state": columns(rows[0], STATE_PREFIX),
                "rhs": columns(rows[0], RHS_PREFIX),
                "constraints": list(CONSTRAINTS)}
    require(len(families["state"]) == len(families["rhs"]) == 25,
            "state/RHS inventory is incomplete")
    owners = {key(row): row for row in rows if int(row["canonical_owner"]) == 1}
    require(len(owners) == 129 * 257,
            f"N{resolution} common-vertex inventory is incomplete: {len(owners)}")
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    spread = {family: 0.0 for family in families}
    for group in groups.values():
        if len(group) < 2:
            continue
        for family, names in families.items():
            for name in names:
                values = [float(row[name]) for row in group]
                spread[family] = max(spread[family], max(values) - min(values))
    return {"resolution": resolution, "owners": owners,
            "families": families, "shared_spread": spread,
            "cycle": int(rows[0]["cycle"]), "time": float(rows[0]["time"])}


def in_region(row: dict[str, str], name: str) -> bool:
    rho, zed = float(row["rho"]), float(row["x2"])
    radius = math.hypot(rho, zed)
    h = 0.125
    outer_distance = min(16.0 - rho, 16.0 - abs(zed))
    seam_r = abs(math.remainder(rho, 4.0)) < 1.0e-12
    seam_z = abs(math.remainder(zed, 4.0)) < 1.0e-12
    if name == "full":
        return True
    if name == "core_r8":
        return radius <= 8.0 + 1.0e-12
    if name == "core_r8_mb_interior":
        return radius <= 8.0 + 1.0e-12 and not seam_r and not seam_z and rho > 4*h
    if name == "axis":
        return abs(rho) < 1.0e-12
    if name == "axis_core_r8":
        return abs(rho) < 1.0e-12 and abs(zed) <= 8.0 + 1.0e-12
    if name == "axis_center":
        return abs(rho) < 1.0e-12 and abs(zed) < 1.0e-12
    if name == "axis_outer_corner":
        return abs(rho) < 1.0e-12 and abs(abs(zed) - 16.0) < 1.0e-12
    if name.startswith("radial_layer_"):
        fields = name.split("_")
        layer = int(fields[2])
        on_layer = abs(rho - layer * h) < 1.0e-12
        return on_layer and ("core" not in fields or abs(zed) <= 8.0 + 1.0e-12)
    if name == "z0_seam":
        return abs(zed) < 1.0e-12
    if name == "z0_seam_core_r8":
        return abs(zed) < 1.0e-12 and rho <= 8.0 + 1.0e-12
    if name == "same_level":
        return row["role"] == "shared_same_level"
    if name == "physical_boundary":
        return row["role"] == "physical_boundary"
    if name == "independent_interior":
        return row["role"] == "independent_interior"
    if name.startswith("outer_layer_"):
        return abs(outer_distance - int(name.rsplit("_", 1)[1]) * h) < 1.0e-12
    raise ValueError(name)


def metrics(values: np.ndarray, keys: list[tuple[float, float]]) -> dict:
    rho = np.asarray([item[0] for item in keys])
    result = {"rms": float(np.sqrt(np.mean(values**2))),
              "linf": float(np.max(np.abs(values)))}
    result["ring_rms"] = (float(np.sqrt(np.sum(rho * values**2) / np.sum(rho)))
                          if np.sum(rho) > 0.0 else None)
    return result


def observed_order(first: float | None, second: float | None) -> float | None:
    if first is None or second is None or first <= 0.0 or second <= 0.0:
        return None
    return math.log(first / second, 2.0)


def parse_tokens(line: str) -> dict[str, str]:
    result = {"record": line.split()[0]}
    for token in line.split()[1:]:
        if "=" in token:
            key_name, value = token.split("=", 1)
            result[key_name] = value
    return result


def load_term_log(path: Path, target_time: float) -> list[dict[str, str]]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        records = [parse_tokens(line) for line in stream if line.strip()]
    return [record for record in records
            if int(record.get("stage", -1)) == 1 and
            abs(float(record.get("time", "nan")) - target_time) < 2.0e-14]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    regions = ("full", "core_r8", "core_r8_mb_interior", "axis",
               "axis_core_r8", "axis_center", "axis_outer_corner",
               "radial_layer_1", "radial_layer_1_core_r8",
               "radial_layer_2", "radial_layer_2_core_r8",
               "radial_layer_3", "radial_layer_3_core_r8",
               "radial_layer_4", "radial_layer_4_core_r8",
               "z0_seam", "z0_seam_core_r8", "same_level", "physical_boundary",
               "independent_interior", "outer_layer_0", "outer_layer_1",
               "outer_layer_2", "outer_layer_3", "outer_layer_4")
    difference_rows, order_rows, term_rows = [], [], []
    run_records = []
    earliest = None
    for tag, tau in TARGETS.items():
        target_time = None
        cases = []
        term_cases = []
        for resolution in (128, 256, 512):
            directory = args.root / f"N{resolution}" / f"tau{tag}" / "diagnostic"
            rhs_path = directory / "rhs.rank000000.csv.gz"
            # Read the exact target from the first CSV record before filtering.
            with gzip.open(rhs_path, "rt", newline="", encoding="utf-8") as stream:
                first = next(csv.DictReader(stream))
            this_time = float(first["time"])
            target_time = this_time if target_time is None else target_time
            require(abs(this_time - target_time) < 2.0e-14,
                    f"coordinate-time mismatch at tau={tau}")
            cases.append(prepare(resolution, load_rows(rhs_path, target_time)))
            term_cases.append(load_term_log(directory / "z4c_rhs_stage_rank0.log.gz",
                                            target_time))
        require(cases[0]["families"] == cases[1]["families"] == cases[2]["families"],
                "diagnostic inventory changed with resolution")
        run_records.append({"axis_tau": tau, "coordinate_time": target_time,
                            "runs": [{key_name: case[key_name] for key_name in
                                      ("resolution", "cycle", "time", "shared_spread")}
                                     for case in cases]})
        pair_data = []
        for coarse, fine in zip(cases, cases[1:]):
            shared = coarse["owners"].keys() & fine["owners"].keys()
            require(len(shared) == len(coarse["owners"]), "common vertices missing")
            this_pair = {}
            for region_name in regions:
                selected = sorted(point for point in shared
                                  if in_region(coarse["owners"][point], region_name))
                if not selected:
                    continue
                this_pair[region_name] = {}
                for family, names in coarse["families"].items():
                    this_pair[region_name][family] = {}
                    for name in names:
                        delta = np.asarray([float(fine["owners"][point][name]) -
                                            float(coarse["owners"][point][name])
                                            for point in selected])
                        item = metrics(delta, selected)
                        this_pair[region_name][family][name] = item
                        difference_rows.append({"axis_tau": tau,
                                                "coordinate_time": target_time,
                                                "region": region_name,
                                                "family": family, "variable": name,
                                                "coarse_resolution": coarse["resolution"],
                                                "fine_resolution": fine["resolution"],
                                                **item})
            pair_data.append(this_pair)
        for region_name in pair_data[0]:
            for family in pair_data[0][region_name]:
                for name, first in pair_data[0][region_name][family].items():
                    second = pair_data[1][region_name][family][name]
                    item = {"axis_tau": tau, "coordinate_time": target_time,
                            "region": region_name, "family": family,
                            "variable": name}
                    for kind in ("rms", "ring_rms", "linf"):
                        item[f"difference_128_256_{kind}"] = first[kind]
                        item[f"difference_256_512_{kind}"] = second[kind]
                        item[f"observed_order_{kind}"] = observed_order(first[kind],
                                                                         second[kind])
                    order_rows.append(item)
                    p = item["observed_order_rms"]
                    if (tau > 0.0 and p is not None and p < 0.0 and
                            item["difference_256_512_rms"] > 1.0e-12):
                        candidate = {"axis_tau": tau, "coordinate_time": target_time,
                                     "region": region_name, "family": family,
                                     "variable": name, "observed_order_rms": p,
                                     "difference_256_512_rms":
                                         item["difference_256_512_rms"]}
                        if earliest is None or tau < earliest["axis_tau"]:
                            earliest = candidate
        for resolution, records in zip((128, 256, 512), term_cases):
            for record in records:
                row = {"axis_tau": tau, "coordinate_time": target_time,
                       "resolution": resolution, "record": record["record"]}
                for name in ("variable", "term", "abs_max", "value", "rho", "z",
                             "rhs_difference", "geometric_difference",
                             "trace_difference", "nonlinear_difference",
                             "lie_difference", "hessian_difference",
                             "ricci_tensor_difference", "trace_lapse_difference",
                             "trace_ricci_difference"):
                    row[name] = record.get(name)
                term_rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "differences.csv", difference_rows)
    write_csv(args.output_dir / "orders.csv", order_rows)
    write_csv(args.output_dir / "rhs_term_maxima.csv", term_rows)
    summary = {"schema": "z4c_vc_exact_localization_v1", "runs": run_records,
               "earliest_meaningful_negative_rms_order": earliest,
               "negative_meaningful_order_count": sum(
                   row["observed_order_rms"] is not None and
                   row["observed_order_rms"] < 0.0 and
                   row["difference_256_512_rms"] > 1.0e-12 for row in order_rows)}
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
