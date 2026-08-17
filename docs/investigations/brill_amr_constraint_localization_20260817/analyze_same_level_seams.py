#!/usr/bin/env python3
"""Conditional CASE-B audit of the worst cycle-1722 same-level seams.

The production ADM constraints differentiate the 13 stored ADM fields and use
Gamma^i/Theta point values.  This script dumps the exact O6 ADM read union for
the worst seam cells, compares receiver ghost bytes with sender active bytes,
and compares local-block derivatives with derivatives stitched directly from
same-level active cells.  It never recomputes an ADM constraint.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

import analyze_existing_event as base


SCHEMA = "athenak_z4c_amr_same_level_seam_audit_v1"
PHASE = "t3_06_PHYSICAL_OR_AXIS_BC"
FD1 = np.asarray([-1 / 60, 3 / 20, -3 / 4, 0.0,
                  3 / 4, -3 / 20, 1 / 60], dtype=np.float64)
FD2 = np.asarray([1 / 90, -3 / 20, 3 / 2, -49 / 18,
                  3 / 2, -3 / 20, 1 / 90], dtype=np.float64)
OFFSETS = tuple(range(-3, 4))
NONZERO = tuple(value for value in OFFSETS if value != 0)


class SeamError(RuntimeError):
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


def key(level: int, rho: Fraction, z: Fraction) -> tuple[int, Fraction, Fraction]:
    return level, rho, z


def cell_center(row: dict[str, str], i: int, j: int, nx: int, ny: int
                ) -> tuple[Fraction, Fraction, Fraction, Fraction]:
    dx1 = (Fraction(row["x1max"]) - Fraction(row["x1min"])) / nx
    dx2 = (Fraction(row["x2max"]) - Fraction(row["x2min"])) / ny
    rho = Fraction(row["x1min"]) + Fraction(2 * i + 1, 2) * dx1
    z = Fraction(row["x2min"]) + Fraction(2 * j + 1, 2) * dx2
    return rho, z, dx1, dx2


def build_active_map(view: np.ndarray, metadata: dict[str, Any],
                     topo: dict[int, dict[str, str]]) -> dict[
                         tuple[int, Fraction, Fraction], tuple[int, int, int, np.ndarray]]:
    result: dict[tuple[int, Fraction, Fraction], tuple[int, int, int, np.ndarray]] = {}
    for gid, row in topo.items():
        local_m = int(row["local_m"])
        values = base.active(view, metadata, local_m)
        ny, nx = values.shape[-2:]
        for j in range(ny):
            for i in range(nx):
                rho, z, _, _ = cell_center(row, i, j, nx, ny)
                item_key = key(int(row["level"]), rho, z)
                if item_key in result:
                    raise SeamError(f"duplicate active center {item_key}")
                result[item_key] = (gid, i, j, values[:, j, i])
    return result


def select_worst(cells: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    seams = [row for row in cells if row["stencil_class"] == "SAME_LEVEL_SEAM"]
    selected: dict[tuple[int, int, int], dict[str, str]] = {}
    for metric in ("C", "H2", "M2"):
        ordered = sorted(seams, key=lambda row: float(
            row[f"{metric}_proper_contribution"]), reverse=True)
        for row in ordered[:count]:
            selected[(int(row["gid"]), int(row["i"]), int(row["j"]))] = row
    return sorted(selected.values(), key=lambda row: (
        -max(float(row[f"{metric}_proper_contribution"]) for metric in
             ("C", "H2", "M2")), int(row["gid"]), int(row["j"]), int(row["i"])))


def paired_coordinate(face: str, rho: Fraction, z: Fraction,
                      dx1: Fraction, dx2: Fraction) -> tuple[Fraction, Fraction]:
    if face == "inner_x1":
        return rho - dx1, z
    if face == "outer_x1":
        return rho + dx1, z
    if face == "inner_x2":
        return rho, z - dx2
    if face == "outer_x2":
        return rho, z + dx2
    raise SeamError(f"unknown face {face}")


def active_value(active_map: dict[tuple[int, Fraction, Fraction], tuple[
        int, int, int, np.ndarray]], level: int, rho: Fraction,
                 z: Fraction) -> tuple[int, int, int, np.ndarray]:
    result = active_map.get(key(level, rho, z))
    if result is None:
        raise SeamError(f"missing same-level active source at level={level}, rho={rho}, z={z}")
    return result


def stored_value(view: np.ndarray, metadata: dict[str, Any], row: dict[str, str],
                 component: int, i: int, j: int) -> float:
    bounds = metadata["active_bounds"]
    return float(view[int(row["local_m"]), component, bounds["ks"],
                      bounds["js"] + j, bounds["is"] + i])


def bits(value: float) -> int:
    return int(np.asarray(value, dtype="<f8").view("<u8"))


def local_derivatives(view: np.ndarray, metadata: dict[str, Any],
                      row: dict[str, str], component: int, i: int, j: int,
                      dx1: float, dx2: float) -> dict[str, float]:
    def value(di: int, dj: int) -> float:
        return stored_value(view, metadata, row, component, i + di, j + dj)
    radial = np.asarray([value(offset, 0) for offset in OFFSETS])
    axial = np.asarray([value(0, offset) for offset in OFFSETS])
    mixed = np.asarray([[value(di, dj) for di in OFFSETS] for dj in OFFSETS])
    return {
        "D1_rho": float(np.dot(FD1, radial) / dx1),
        "D1_z": float(np.dot(FD1, axial) / dx2),
        "D2_rho": float(np.dot(FD2, radial) / (dx1 * dx1)),
        "D2_z": float(np.dot(FD2, axial) / (dx2 * dx2)),
        "Dmix_rho_z": float(np.einsum("j,i,ji->", FD1, FD1, mixed) /
                             (dx1 * dx2)),
    }


def stitched_derivatives(active_map: dict[tuple[int, Fraction, Fraction], tuple[
        int, int, int, np.ndarray]], level: int, rho: Fraction, z: Fraction,
                         component: int, dx1: Fraction, dx2: Fraction
                         ) -> dict[str, float]:
    def value(di: int, dj: int) -> float:
        return float(active_value(active_map, level, rho + di * dx1,
                                  z + dj * dx2)[3][component])
    radial = np.asarray([value(offset, 0) for offset in OFFSETS])
    axial = np.asarray([value(0, offset) for offset in OFFSETS])
    mixed = np.asarray([[value(di, dj) for di in OFFSETS] for dj in OFFSETS])
    h1, h2 = float(dx1), float(dx2)
    return {
        "D1_rho": float(np.dot(FD1, radial) / h1),
        "D1_z": float(np.dot(FD1, axial) / h2),
        "D2_rho": float(np.dot(FD2, radial) / (h1 * h1)),
        "D2_z": float(np.dot(FD2, axial) / (h2 * h2)),
        "Dmix_rho_z": float(np.einsum("j,i,ji->", FD1, FD1, mixed) / (h1 * h2)),
    }


def stencil_offsets() -> list[tuple[int, int, str]]:
    result: dict[tuple[int, int], set[str]] = {(0, 0): {"center"}}
    for offset in OFFSETS:
        result.setdefault((offset, 0), set()).add("pure_rho")
        result.setdefault((0, offset), set()).add("pure_z")
    for dj in OFFSETS:
        for di in OFFSETS:
            if FD1[di + 3] != 0.0 and FD1[dj + 3] != 0.0:
                result.setdefault((di, dj), set()).add("mixed_rho_z")
    return [(di, dj, ";".join(sorted(operators)))
            for (di, dj), operators in sorted(result.items(), key=lambda item:
                                               (item[0][1], item[0][0]))]


def analyze(args: argparse.Namespace) -> None:
    raw, analysis, output = args.raw.resolve(), args.analysis.resolve(), args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    phase = raw / PHASE
    metadata, topo = base.strict_load(phase / "phase.json"), base.topology(phase)
    adm_all = base.load_view(phase, "adm", metadata)
    u0_all = base.load_view(phase, "u0", metadata)
    adm_active = build_active_map(adm_all, metadata, topo)
    cells = read_csv(analysis / "stencil_provenance_cells.csv")
    selected = select_worst(cells, args.top_per_metric)
    if not selected:
        raise SeamError("no same-level seam cells selected")

    top_rows: list[dict[str, Any]] = []
    stencil_rows: list[dict[str, Any]] = []
    derivative_rows: list[dict[str, Any]] = []
    center_rows: list[dict[str, Any]] = []
    max_ghost_mismatch = 0.0
    bit_mismatches = 0
    max_local_stitched = 0.0
    for selection_rank, cell in enumerate(selected, start=1):
        gid, i, j = int(cell["gid"]), int(cell["i"]), int(cell["j"])
        row, level = topo[gid], int(topo[gid]["level"])
        faces = cell["crossed_faces"].split(";")
        if len(faces) != 1:
            raise SeamError("same-level selection did not cross exactly one face")
        face = faces[0]
        rho, z, dx1, dx2 = cell_center(row, i, j, 32, 32)
        paired_rho, paired_z = paired_coordinate(face, rho, z, dx1, dx2)
        paired_gid, paired_i, paired_j, _ = active_value(
            adm_active, level, paired_rho, paired_z)
        top_rows.append({
            "selection_rank": selection_rank, "receiver_gid": gid,
            "receiver_i": i, "receiver_j": j, "receiver_rho": float(rho),
            "receiver_z": float(z), "face": face, "paired_gid": paired_gid,
            "paired_i": paired_i, "paired_j": paired_j,
            "hierarchy_origin": cell["hierarchy_origin"],
            "C_proper_contribution": float(cell["C_proper_contribution"]),
            "H2_proper_contribution": float(cell["H2_proper_contribution"]),
            "M2_proper_contribution": float(cell["M2_proper_contribution"]),
        })

        # Gamma^i and Theta are pointwise inputs to Z and C, not derivative inputs.
        bounds = metadata["active_bounds"]
        local_m = int(row["local_m"])
        for component in range(14, 18):
            center_rows.append({
                "selection_rank": selection_rank, "gid": gid, "i": i, "j": j,
                "rho": float(rho), "z": float(z), "variable": base.Z4C_NAMES[component],
                "value": float(u0_all[local_m, component, bounds["ks"],
                                      bounds["js"] + j, bounds["is"] + i]),
            })

        for component, name in enumerate(base.ADM_NAMES):
            for di, dj, operators in stencil_offsets():
                point_rho, point_z = rho + di * dx1, z + dj * dx2
                source_gid, source_i, source_j, source_values = active_value(
                    adm_active, level, point_rho, point_z)
                receiver_value = stored_value(adm_all, metadata, row, component,
                                              i + di, j + dj)
                expected = float(source_values[component])
                byte_equal = bits(receiver_value) == bits(expected)
                mismatch = abs(receiver_value - expected)
                max_ghost_mismatch = max(max_ghost_mismatch, mismatch)
                bit_mismatches += int(not byte_equal)
                status = "ACTIVE" if source_gid == gid else "SAME_LEVEL_GHOST"
                stencil_rows.append({
                    "selection_rank": selection_rank, "receiver_gid": gid,
                    "receiver_i": i, "receiver_j": j, "variable": name,
                    "offset_i": di, "offset_j": dj, "operators": operators,
                    "source_gid": source_gid, "source_i": source_i,
                    "source_j": source_j, "source_status": status,
                    "source_rho": float(point_rho), "source_z": float(point_z),
                    "writer": ("SAME_RANK_SAME_LEVEL_COPY" if status.endswith("GHOST")
                               else "ACTIVE_STATE"),
                    "receiver_stored_value": receiver_value,
                    "expected_sender_active_value": expected,
                    "absolute_mismatch": mismatch, "exact_byte_equal": byte_equal,
                })

            sides = (("receiver", gid, i, j, rho, z),
                     ("paired", paired_gid, paired_i, paired_j, paired_rho, paired_z))
            side_values: dict[str, dict[str, float]] = {}
            for side, side_gid, side_i, side_j, side_rho, side_z in sides:
                local = local_derivatives(adm_all, metadata, topo[side_gid], component,
                                          side_i, side_j, float(dx1), float(dx2))
                stitched = stitched_derivatives(adm_active, level, side_rho, side_z,
                                                 component, dx1, dx2)
                side_values[side] = local
                for operator in local:
                    residual = local[operator] - stitched[operator]
                    max_local_stitched = max(max_local_stitched, abs(residual))
                    derivative_rows.append({
                        "selection_rank": selection_rank, "face": face,
                        "side": side, "gid": side_gid, "i": side_i, "j": side_j,
                        "rho": float(side_rho), "z": float(side_z),
                        "variable": name, "operator": operator,
                        "local_block_value": local[operator],
                        "stitched_active_value": stitched[operator],
                        "local_minus_stitched": residual,
                        "opposite_side_local_value": None,
                        "across_seam_local_jump": None,
                    })
            for record in derivative_rows[-10:]:
                opposite = "paired" if record["side"] == "receiver" else "receiver"
                record["opposite_side_local_value"] = side_values[opposite][record["operator"]]
                record["across_seam_local_jump"] = (record["local_block_value"] -
                                                     record["opposite_side_local_value"])

    write_csv(output / "same_level_top_cells.csv", top_rows, list(top_rows[0]))
    write_csv(output / "same_level_exact_adm_stencils.csv", stencil_rows,
              list(stencil_rows[0]))
    write_csv(output / "same_level_derivative_comparison.csv", derivative_rows,
              list(derivative_rows[0]))
    write_csv(output / "same_level_center_z4c_inputs.csv", center_rows,
              list(center_rows[0]))
    summary = {
        "schema": SCHEMA,
        "phase": PHASE,
        "selected_cell_count": len(top_rows),
        "selection": f"union of top {args.top_per_metric} proper contributions for C/H2/M2",
        "adm_stencil_value_count": len(stencil_rows),
        "same_level_ghost_stencil_value_count": sum(
            row["source_status"] == "SAME_LEVEL_GHOST" for row in stencil_rows),
        "sender_receiver_bit_mismatch_count": bit_mismatches,
        "maximum_sender_receiver_absolute_mismatch": max_ghost_mismatch,
        "maximum_local_minus_stitched_derivative": max_local_stitched,
        "all_sender_receiver_values_byte_exact": bit_mismatches == 0,
        "all_local_and_stitched_derivatives_byte_equivalent": max_local_stitched == 0.0,
        "qualification_claim": False,
    }
    strict_dump(output / "same_level_seam_summary.json", summary)
    print(json.dumps(summary, sort_keys=True, indent=2))


def self_test() -> None:
    if len(stencil_offsets()) != 49:
        raise SeamError(f"O6 stencil union should contain 49 points, got {len(stencil_offsets())}")
    polynomial = np.asarray([float(offset) ** 2 for offset in OFFSETS])
    if not math.isclose(float(np.dot(FD1, polynomial)), 0.0, abs_tol=1.0e-12):
        raise SeamError("O6 first derivative moment self-test failed")
    if not math.isclose(float(np.dot(FD2, polynomial)), 2.0,
                        rel_tol=0.0, abs_tol=1.0e-11):
        raise SeamError("O6 second derivative moment self-test failed")
    print("analyze_same_level_seams self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path)
    parser.add_argument("--analysis", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-per-metric", type=int, default=8)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.raw is None or args.analysis is None or args.output is None:
        parser.error("--raw, --analysis, and --output are required")
    analyze(args)


if __name__ == "__main__":
    main()
