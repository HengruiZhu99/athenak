#!/usr/bin/env python3
"""Compare production C++ O2/O4/O6 constraints on the exact T3_06 state.

The provenance masks and ring-volume weights come from analyze_existing_event.py.
This script does not implement derivatives or ADM constraints; it consumes the
three constraint arrays written by Z4c::EvaluateDiagnosticConstraints.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np

import analyze_existing_event as base


SCHEMA = "athenak_z4c_amr_derivative_order_comparison_v1"
PHASE = "t3_06_PHYSICAL_OR_AXIS_BC"
ORDERS = ("o2", "o4", "o6")
METRICS = ("C", "H2", "M2", "Z2")


class OrderError(RuntimeError):
    pass


def strict_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_constraint(path: Path, metadata: dict[str, Any]) -> np.ndarray:
    shape = metadata.get("constraint_shape")
    if not isinstance(shape, list) or len(shape) != 5 or not all(
            isinstance(item, int) and item > 0 for item in shape):
        raise OrderError("invalid constraint_shape")
    if path.stat().st_size != math.prod(shape) * 8:
        raise OrderError(f"binary size mismatch: {path}")
    values = np.fromfile(path, dtype="<f8").reshape(shape)
    if not np.all(np.isfinite(values)):
        raise OrderError(f"non-finite constraint data: {path}")
    return values


def pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    if left.size != right.size or left.size == 0:
        raise OrderError("invalid correlation input")
    left_centered = left - np.mean(left)
    right_centered = right - np.mean(right)
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator == 0.0:
        return None
    return float(np.dot(left_centered, right_centered) / denominator)


def analyze(args: argparse.Namespace) -> None:
    existing = args.existing_raw.resolve() / PHASE
    audit = args.audit_event.resolve() / PHASE
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)

    existing_meta = base.strict_load(existing / "phase.json")
    audit_meta = base.strict_load(audit / "phase.json")
    existing_topology = base.topology(existing)
    audit_topology = base.topology(audit)
    if existing_topology != audit_topology:
        raise OrderError("audit and established topology differ")
    if existing_meta["active_bounds"] != audit_meta["active_bounds"]:
        raise OrderError("audit and established active bounds differ")

    constraints = {
        order: load_constraint(audit / f"constraints_{order}.bin", audit_meta)
        for order in ORDERS
    }
    captured_o6 = base.load_view(existing, "constraints", existing_meta)
    if not np.array_equal(constraints["o6"], captured_o6):
        raise OrderError("fresh production O6 audit does not reproduce established T3_06")

    masks = read_csv(args.masks.resolve())
    expected_cells = len(existing_topology) * 32 * 32
    if len(masks) != expected_cells:
        raise OrderError(f"mask inventory {len(masks)} != expected {expected_cells}")

    totals: dict[tuple[str, str, str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}
    magnitudes: dict[tuple[str, str], list[float]] = {}
    maxima: dict[tuple[str, str, str, str], tuple[float, dict[str, str]]] = {}
    for row in masks:
        gid, i, j = int(row["gid"]), int(row["i"]), int(row["j"])
        local_m = int(existing_topology[gid]["local_m"])
        bounds = audit_meta["active_bounds"]
        stored_i, stored_j, stored_k = bounds["is"] + i, bounds["js"] + j, bounds["ks"]
        provenance = row["stencil_class"]
        origin = row["hierarchy_origin"]
        counts[(provenance, origin)] = counts.get((provenance, origin), 0) + 1
        for order in ORDERS:
            cell = constraints[order][local_m, :, stored_k, stored_j, stored_i]
            for metric in METRICS:
                squared = float(base.metric_field(cell, metric))
                magnitude = math.sqrt(max(squared, 0.0))
                magnitudes.setdefault((order, metric), []).append(magnitude)
                for measure in ("coordinate", "proper"):
                    weight = float(row[f"{measure}_weight"])
                    for mask_name in ("ALL", provenance):
                        key = (order, metric, measure, mask_name)
                        totals[key] = totals.get(key, 0.0) + squared * weight
                max_key = (order, metric, provenance, origin)
                prior = maxima.get(max_key)
                if prior is None or magnitude > prior[0]:
                    maxima[max_key] = (magnitude, row)

    rows: list[dict[str, Any]] = []
    masks_present = ["ALL"] + sorted({row["stencil_class"] for row in masks})
    for mask_name in masks_present:
        cell_count = len(masks) if mask_name == "ALL" else sum(
            count for (provenance, _), count in counts.items() if provenance == mask_name)
        for metric in METRICS:
            for measure in ("coordinate", "proper"):
                o6 = totals[("o6", metric, measure, mask_name)]
                for order in ORDERS:
                    value = totals[(order, metric, measure, mask_name)]
                    rows.append({
                        "mask": mask_name,
                        "cells": cell_count,
                        "metric": metric,
                        "measure": measure,
                        "order": order,
                        "integral": value,
                        "integral_over_o6": value / o6 if o6 != 0.0 else None,
                    })
    write_csv(output / "derivative_order_integrals_by_provenance.csv", rows,
              ["mask", "cells", "metric", "measure", "order", "integral",
               "integral_over_o6"])

    correlation_rows: list[dict[str, Any]] = []
    for metric in METRICS:
        o6 = np.asarray(magnitudes[("o6", metric)])
        for order in ("o2", "o4"):
            other = np.asarray(magnitudes[(order, metric)])
            correlation_rows.append({
                "metric": metric,
                "comparison": f"{order}_vs_o6",
                "pearson_pointwise_magnitude": pearson(other, o6),
                "global_argmax_same_cell": int(np.argmax(other)) == int(np.argmax(o6)),
                "other_argmax_flat_cell": int(np.argmax(other)),
                "o6_argmax_flat_cell": int(np.argmax(o6)),
            })
    write_csv(output / "derivative_order_pointwise_correlations.csv", correlation_rows,
              ["metric", "comparison", "pearson_pointwise_magnitude",
               "global_argmax_same_cell", "other_argmax_flat_cell",
               "o6_argmax_flat_cell"])

    maximum_rows: list[dict[str, Any]] = []
    for (order, metric, provenance, origin), (value, row) in sorted(maxima.items()):
        maximum_rows.append({
            "order": order, "metric": metric, "stencil_class": provenance,
            "hierarchy_origin": origin, "magnitude": value,
            "gid": int(row["gid"]), "i": int(row["i"]), "j": int(row["j"]),
            "rho": float(row["rho"]), "z": float(row["z"]),
        })
    write_csv(output / "derivative_order_maxima.csv", maximum_rows,
              ["order", "metric", "stencil_class", "hierarchy_origin",
               "magnitude", "gid", "i", "j", "rho", "z"])

    summary = {
        "schema": SCHEMA,
        "qualification_claim": False,
        "exact_state": {
            "phase": PHASE,
            "fresh_o6_byte_equal_established_o6": True,
            "cells": len(masks),
        },
        "proper_global_integrals": {
            metric: {
                order: totals[(order, metric, "proper", "ALL")]
                for order in ORDERS
            } for metric in METRICS
        },
        "same_level_proper_integrals": {
            metric: {
                order: totals[(order, metric, "proper", "SAME_LEVEL_SEAM")]
                for order in ORDERS
            } for metric in METRICS
        },
        "pointwise_correlations": correlation_rows,
    }
    strict_dump(output / "derivative_order_summary.json", summary)


def self_test() -> None:
    assert math.isclose(
        pearson(np.asarray([1.0, 2.0]), np.asarray([2.0, 4.0])), 1.0,
        rel_tol=0.0, abs_tol=4.0 * np.finfo(float).eps)
    assert pearson(np.asarray([1.0, 1.0]), np.asarray([2.0, 4.0])) is None


def parse() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--existing-raw", type=Path)
    parser.add_argument("--audit-event", type=Path)
    parser.add_argument("--masks", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse()
    if arguments.self_test:
        self_test()
    else:
        if None in (arguments.existing_raw, arguments.audit_event,
                    arguments.masks, arguments.output):
            raise SystemExit("analysis arguments are required")
        analyze(arguments)
