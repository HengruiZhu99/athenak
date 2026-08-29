#!/usr/bin/env python3
"""Classify constraint maxima relative to the replayed physical AMR tree."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np


HEADER = re.compile(r"\[(\d+)\]=([^\s]+)")
FIELDS = ("C", "H", "M", "Z")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def read_history(paths: list[Path]) -> dict[str, np.ndarray]:
    labels0: dict[str, int] | None = None
    by_cycle: dict[int, list[float]] = {}
    for path in paths:
        labels: dict[str, int] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("#"):
                labels.update({name: int(index) - 1
                               for index, name in HEADER.findall(line)})
            elif line.strip():
                values = [float(value) for value in line.split()]
                require(labels and "cycle" in labels, "history data precedes schema")
                cycle = int(values[labels["cycle"]])
                if cycle in by_cycle:
                    differing = [name for name, index in labels.items()
                                 if values[index] != by_cycle[cycle][index]]
                    require(set(differing) <= {"dt"},
                            f"history overlap mismatch at cycle {cycle}: {differing}")
                by_cycle[cycle] = values
        if labels0 is None:
            labels0 = labels
        require(labels == labels0, f"history schema changed in {path}")
    labels = labels0 or {}
    required = {"time", "axisTau", "cycle", *(f"{field}-Linf" for field in FIELDS),
                *(f"{field}-rho" for field in FIELDS),
                *(f"{field}-z" for field in FIELDS),
                *(f"{field}-norm2" for field in FIELDS)}
    require(not required - labels.keys(), f"missing columns {sorted(required-labels.keys())}")
    array = np.asarray([by_cycle[cycle] for cycle in sorted(by_cycle)], dtype=float)
    return {name: array[:, index] for name, index in labels.items()}


def blocks(authority: list[dict]) -> tuple[list[dict], dict]:
    header = next(row for row in authority if row.get("type") == "header")
    event = max((row for row in authority if row.get("type") == "event"),
                key=lambda row: int(row["event"]))
    root_level = int(header["root_level"])
    root_blocks = [int(value) for value in header["root_blocks"]]
    domain = [float.fromhex(value) for value in header["domain_hex"]]
    output = []
    for level, lx1, lx2, lx3 in event["leaves"]:
        scale = 2 ** (int(level) - root_level)
        dx = (domain[1] - domain[0]) / (root_blocks[0] * scale)
        dz = (domain[3] - domain[2]) / (root_blocks[1] * scale)
        output.append({
            "level": int(level), "lx1": int(lx1), "lx2": int(lx2),
            "rho0": domain[0] + int(lx1) * dx,
            "rho1": domain[0] + (int(lx1) + 1) * dx,
            "z0": domain[2] + int(lx2) * dz,
            "z1": domain[2] + (int(lx2) + 1) * dz,
        })
    require(len(output) == int(event["leaf_count"]), "authority leaf count mismatch")
    return output, {"header": header, "event": event, "domain": domain}


def interfaces(leaves: list[dict]) -> dict[str, list[tuple[str, float, float, float]]]:
    output: dict[str, list[tuple[str, float, float, float]]] = {
        "same_level_seam": [], "coarse_fine_interface": []}
    tolerance = 1.0e-12
    for i, left in enumerate(leaves):
        for right in leaves[i + 1:]:
            category = ("same_level_seam" if left["level"] == right["level"]
                        else "coarse_fine_interface")
            if abs(left["rho1"] - right["rho0"]) <= tolerance:
                lo, hi = max(left["z0"], right["z0"]), min(left["z1"], right["z1"])
                if hi > lo + tolerance:
                    output[category].append(("rho", left["rho1"], lo, hi))
            if abs(right["rho1"] - left["rho0"]) <= tolerance:
                lo, hi = max(left["z0"], right["z0"]), min(left["z1"], right["z1"])
                if hi > lo + tolerance:
                    output[category].append(("rho", right["rho1"], lo, hi))
            if abs(left["z1"] - right["z0"]) <= tolerance:
                lo, hi = max(left["rho0"], right["rho0"]), min(left["rho1"], right["rho1"])
                if hi > lo + tolerance:
                    output[category].append(("z", left["z1"], lo, hi))
            if abs(right["z1"] - left["z0"]) <= tolerance:
                lo, hi = max(left["rho0"], right["rho0"]), min(left["rho1"], right["rho1"])
                if hi > lo + tolerance:
                    output[category].append(("z", right["z1"], lo, hi))
    # Repeated subsegments from one-to-many coarse/fine neighbors do not alter
    # the distance, but deterministic deduplication simplifies the manifest.
    for category in output:
        output[category] = sorted(set(output[category]))
    return output


def segment_distance(rho: float, z: float,
                     segment: tuple[str, float, float, float]) -> float:
    orientation, coordinate, lo, hi = segment
    if orientation == "rho":
        along = 0.0 if lo <= z <= hi else min(abs(z - lo), abs(z - hi))
        return math.hypot(rho - coordinate, along)
    along = 0.0 if lo <= rho <= hi else min(abs(rho - lo), abs(rho - hi))
    return math.hypot(z - coordinate, along)


def classify(rho: float, z: float, leaves: list[dict], faces: dict,
             domain: list[float], cells_per_block: int) -> dict[str, object]:
    tolerance = 1.0e-11
    owners = [block for block in leaves
              if block["rho0"] - tolerance <= rho <= block["rho1"] + tolerance
              and block["z0"] - tolerance <= z <= block["z1"] + tolerance]
    require(bool(owners), f"location ({rho},{z}) lies outside leaf tree")
    h = min((block["rho1"] - block["rho0"]) / cells_per_block for block in owners)
    distances = {
        category: min((segment_distance(rho, z, segment) for segment in segments),
                      default=math.inf)
        for category, segments in faces.items()
    }
    axis_distance = abs(rho - domain[0])
    outer_distance = min(abs(rho - domain[1]), abs(z - domain[2]), abs(z - domain[3]))
    stencil_radius = 3.0
    if axis_distance <= stencil_radius * h + tolerance:
        category = "axis"
    elif outer_distance <= stencil_radius * h + tolerance:
        category = "outer_boundary"
    elif distances["coarse_fine_interface"] <= stencil_radius * h + tolerance:
        category = "coarse_fine_interface"
    elif distances["same_level_seam"] <= stencil_radius * h + tolerance:
        category = "same_level_seam"
    else:
        category = "clean_interior"
    return {
        "classification": category,
        "local_h": h,
        "owner_levels": ";".join(str(level) for level in sorted({b["level"] for b in owners})),
        "distance_axis_over_h": axis_distance / h,
        "distance_outer_over_h": outer_distance / h,
        "distance_same_level_over_h": distances["same_level_seam"] / h,
        "distance_coarse_fine_over_h": distances["coarse_fine_interface"] / h,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--authority", type=Path, required=True)
    parser.add_argument("--n256-history", type=Path, nargs="+", required=True)
    parser.add_argument("--n512-history", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    authority = read_jsonl(args.authority)
    leaves, metadata = blocks(authority)
    faces = interfaces(leaves)
    histories = {"n256": read_history(args.n256_history),
                 "n512": read_history(args.n512_history)}
    rows: list[dict[str, object]] = []
    for case, data in histories.items():
        cells = 32 if case == "n256" else 64
        for index in range(len(data["time"])):
            for field in FIELDS:
                rho, z = float(data[f"{field}-rho"][index]), float(data[f"{field}-z"][index])
                rows.append({
                    "resolution": case, "time": float(data["time"][index]),
                    "axisTau": float(data["axisTau"][index]),
                    "cycle": int(data["cycle"][index]), "field": field,
                    "norm2": float(data[f"{field}-norm2"][index]),
                    "Linf": float(data[f"{field}-Linf"][index]), "rho": rho, "z": z,
                    **classify(rho, z, leaves, faces, metadata["domain"], cells),
                })
    with (args.output / "constraint_location_classification.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)

    summary: dict[str, object] = {
        "schema": "athenak.z4c.constraint_location_classification.v1",
        "authority_event": metadata["event"]["event"],
        "tree_checksum": metadata["event"]["tree_checksum"],
        "leaf_count": metadata["event"]["leaf_count"],
        "same_level_face_segments": len(faces["same_level_seam"]),
        "coarse_fine_face_segments": len(faces["coarse_fine_interface"]),
        "stencil_near_threshold_local_h": 3.0,
        "cases": {},
        "claim_boundary": "geometric location classification; not a causal attribution",
    }
    for case in histories:
        summary["cases"][case] = {}
        for field in FIELDS:
            selected = [row for row in rows if row["resolution"] == case and
                        row["field"] == field]
            high = [row for row in selected if float(row["norm2"]) >= 1.0e-2]
            summary["cases"][case][field] = {
                "all_rows": dict(Counter(str(row["classification"]) for row in selected)),
                "norm2_at_least_1e-2": dict(Counter(
                    str(row["classification"]) for row in high)),
                "rho_4_to_6_fraction_when_high": (
                    None if not high else sum(4.0 < float(row["rho"]) < 6.0 for row in high) /
                    len(high)),
            }
    (args.output / "constraint_location_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
