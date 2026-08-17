#!/usr/bin/env python3
"""Strict two-arm, zero-PDE Cartoon Z4c AMR transfer comparison.

This analyzer consumes the raw T0--T5 diagnostic bytes from matched high-order
and limited-O2 target transactions.  It uses one common T0 parent-constraint
representation on the accepted child lattice, forms disjoint spatial budgets,
and applies the prospective causal gate.  It does not recompute constraints or
alter any production state.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import cartoon_amr_jump_analyze as base


SCHEMA = "athenak_z4c_amr_transfer_comparison_v1"
REGIONS = [
    "AXIS_OR_PHYSICAL_BOUNDARY",
    "COARSE_FINE_INTERFACE",
    "MESHBLOCK_EDGE_OR_CORNER",
    "INTERIOR",
]
FAMILIES = {"C": 0, "H": 1, "M": 2}
MEASURES = ("proper_ring", "coordinate_ring")


class ComparisonError(RuntimeError):
    pass


def strict_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def event_roots(root: Path, ranks: int, before: int, after: int,
                cycle: int) -> tuple[list[Path], dict[str, Any], list[dict[str, str]]]:
    roots: list[Path] = []
    event_name = f"event_c{cycle:08d}_l{before}_to_l{after}"
    schemas: list[dict[str, Any]] = []
    proposal: list[dict[str, str]] | None = None
    for rank in range(ranks):
        rank_root = root / f"rank{rank:04d}"
        schema = base.strict_load(rank_root / "schema.json")
        if schema.get("schema") != base.SCHEMA or schema.get("real_bytes") != 8:
            raise ComparisonError(f"bad diagnostic schema in {rank_root}")
        event = rank_root / event_name
        if not event.is_dir():
            raise ComparisonError(f"missing exact target event {event}")
        rows = base.read_csv(event / "t1_topology_proposal.csv")
        if proposal is None:
            proposal = rows
        elif rows != proposal:
            raise ComparisonError("rank topology proposals differ")
        phase = base.strict_load(event / "t1_phase.json")
        if int(phase.get("cycle", -1)) != cycle:
            raise ComparisonError("target cycle does not match event identity")
        with (rank_root / "post_event_cycles.jsonl").open(encoding="utf-8") as stream:
            post = [json.loads(line) for line in stream if line.strip()]
        if [int(row["cycle"]) for row in post] != [cycle]:
            raise ComparisonError("arm did not stop at T5 of the target cycle")
        exposure = rank_root / "rk_stage_exposure.jsonl"
        if exposure.exists() and exposure.read_text(encoding="utf-8").strip():
            raise ComparisonError("arm contains a post-event RK-stage exposure")
        roots.append(event)
        schemas.append(schema)
    if proposal is None:
        raise ComparisonError("empty target topology proposal")
    if any(schema != schemas[0] for schema in schemas[1:]):
        raise ComparisonError("rank schemas differ")
    return roots, schemas[0], proposal


def validate_diagnostic_manifest(case_root: Path, diagnostic_root: Path,
                                 expected_sha256: str) -> None:
    manifest = case_root / "diagnostic.SHA256SUMS"
    if base.sha256(manifest) != expected_sha256:
        raise ComparisonError(f"diagnostic manifest hash mismatch in {case_root}")
    listed: set[Path] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if not separator or len(digest) != 64:
            raise ComparisonError(f"malformed diagnostic manifest line in {manifest}")
        candidate = (case_root / relative).resolve()
        try:
            candidate.relative_to(case_root.resolve())
            candidate.relative_to(diagnostic_root.resolve())
        except ValueError as error:
            raise ComparisonError(f"diagnostic manifest path escaped root: {relative}") from error
        if not candidate.is_file() or base.sha256(candidate) != digest:
            raise ComparisonError(f"diagnostic manifest entry mismatch: {relative}")
        listed.add(candidate)
    actual = {path.resolve() for path in diagnostic_root.rglob("*") if path.is_file()}
    if listed != actual:
        raise ComparisonError("diagnostic manifest inventory is incomplete or excessive")


def validate_provenance(path: Path, diagnostic_root: Path,
                        expected_transfer: str, ranks: int) -> dict[str, Any]:
    record = base.strict_load(path)
    required_strings = [
        "source_commit", "source_tree", "executable_sha256", "input_sha256",
        "restart_sha256", "node", "gpu_model", "hardware_binding_sha256",
        "command_sha256", "diagnostic_manifest_sha256",
    ]
    if record.get("schema") != "athenak_z4c_amr_zero_pde_provenance_v1":
        raise ComparisonError(f"wrong provenance schema in {path}")
    if record.get("amr_transfer") != expected_transfer:
        raise ComparisonError(f"wrong transfer provenance in {path}")
    if record.get("rank_count") != ranks or record.get("post_cycles") != 0:
        raise ComparisonError(f"wrong rank/stop contract in {path}")
    if record.get("qualification_claim") is not False:
        raise ComparisonError(f"provenance promoted an unsupported claim in {path}")
    for key in required_strings:
        value = record.get(key)
        if not isinstance(value, str) or not value:
            raise ComparisonError(f"missing provenance key {key} in {path}")
    validate_diagnostic_manifest(path.parent.resolve(), diagnostic_root.resolve(),
                                 record["diagnostic_manifest_sha256"])
    return record


def comparable_provenance(high: dict[str, Any], limited: dict[str, Any]) -> None:
    identical = [
        "source_commit", "source_tree", "executable_sha256", "input_sha256",
        "restart_sha256", "rank_count", "node", "gpu_model",
        "hardware_binding_sha256",
    ]
    for key in identical:
        if high[key] != limited[key]:
            raise ComparisonError(f"arms differ in required provenance field {key}")


def selected_children(proposal: list[dict[str, str]]) -> set[int]:
    result = {
        int(row["new_gid"]) for row in proposal
        if int(row["new_level"]) == int(row["old_level"]) + 1
    }
    if not result:
        raise ComparisonError("target proposal contains no refined child")
    return result


def common_before(event: list[Path], children: set[int],
                  proposal: list[dict[str, str]]) -> tuple[
                      dict[int, np.ndarray], dict[int, np.ndarray],
                      dict[int, dict[str, str]], dict[str, int]]:
    t2 = base.PHASES[0]
    topology = base.phase_topology(event, t2)
    old_constraints = base.phase_gid_views(
        event, "t0_00_ACCEPTED_OLD_STATE", "constraints")
    old_adm = base.phase_gid_views(event, "t0_00_ACCEPTED_OLD_STATE", "adm")
    t2_coarse = base.phase_gid_views(event, t2, "coarse_u0")
    metadata = base.strict_load(event[0] / t2 / "phase.json")
    active = metadata["active_bounds"]
    coarse = metadata["coarse_active_bounds"]
    geometry = {**active, **coarse}
    nx1 = active["ie"] - active["is"] + 1
    nx2 = active["je"] - active["js"] + 1
    coarse_shape = next(iter(t2_coarse.values())).shape
    by_gid = {int(row["new_gid"]): row for row in proposal}
    constraints: dict[int, np.ndarray] = {}
    adm: dict[int, np.ndarray] = {}
    for gid in sorted(children):
        row = by_gid[gid]
        old_gid = int(row["old_gid"])
        ox1, ox2 = int(row["new_lx1"]) & 1, int(row["new_lx2"]) & 1
        con_shape = (old_constraints[old_gid].shape[0],) + coarse_shape[1:]
        adm_shape = (old_adm[old_gid].shape[0],) + coarse_shape[1:]
        con_source = base.child_coarse_from_parent(
            old_constraints[old_gid], con_shape, ox1, ox2, nx1, nx2)
        adm_source = base.child_coarse_from_parent(
            old_adm[old_gid], adm_shape, ox1, ox2, nx1, nx2)
        constraints[gid] = base.prolong_active(
            con_source, geometry, None, "high_order")
        adm[gid] = base.prolong_active(adm_source, geometry, None, "high_order")
    return constraints, adm, topology, active


def active_views(event: list[Path], phase: str, name: str,
                 children: set[int]) -> dict[int, np.ndarray]:
    raw = base.phase_gid_views(event, phase, name)
    result: dict[int, np.ndarray] = {}
    metadata_by_gid: dict[int, dict[str, Any]] = {}
    for root in event:
        metadata = base.strict_load(root / phase / "phase.json")
        for row in base.read_csv(root / phase / "topology.csv"):
            metadata_by_gid[int(row["gid"])] = metadata
    for gid in sorted(children):
        result[gid] = base.active_slice(raw[gid], metadata_by_gid[gid])
    return result


def region_for_cell(gid: int, i: int, j: int, nx: int, ny: int,
                    topology: dict[int, dict[str, str]]) -> str:
    row = topology[gid]
    x1min, x1max = float(row["x1min"]), float(row["x1max"])
    x2min, x2max = float(row["x2min"]), float(row["x2max"])
    dx1, dx2 = (x1max - x1min) / nx, (x2max - x2min) / ny
    rho = x1min + (i + 0.5) * dx1
    z = x2min + (j + 0.5) * dx2
    global_x1min = min(float(other["x1min"]) for other in topology.values())
    global_x1max = max(float(other["x1max"]) for other in topology.values())
    global_x2min = min(float(other["x2min"]) for other in topology.values())
    global_x2max = max(float(other["x2max"]) for other in topology.values())
    physical_cells = min(
        (rho - global_x1min) / dx1, (global_x1max - rho) / dx1,
        (z - global_x2min) / dx2, (global_x2max - z) / dx2)
    if physical_cells <= 4.0:
        return REGIONS[0]
    faces = base.coarse_fine_faces(gid, topology)
    distances: list[float] = []
    if "inner_x1" in faces: distances.append(i + 0.5)
    if "outer_x1" in faces: distances.append(nx - i - 0.5)
    if "inner_x2" in faces: distances.append(j + 0.5)
    if "outer_x2" in faces: distances.append(ny - j - 0.5)
    if distances and min(distances) <= 4.0:
        return REGIONS[1]
    if min(i + 0.5, nx - i - 0.5, j + 0.5, ny - j - 0.5) <= 4.0:
        return REGIONS[2]
    return REGIONS[3]


def family_squared(constraints: np.ndarray, family: str) -> np.ndarray:
    values = constraints[FAMILIES[family]]
    if family == "H":
        return values * values
    return values


def budgets(constraints: dict[int, np.ndarray], adm: dict[int, np.ndarray],
            topology: dict[int, dict[str, str]], children: set[int],
            arm: str, state: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    accum: dict[tuple[str, str, str], dict[str, Any]] = {}
    cells: list[dict[str, Any]] = []
    for measure in MEASURES:
        for family in FAMILIES:
            for region in REGIONS + ["GLOBAL"]:
                accum[(measure, family, region)] = {
                    "integral": 0.0, "volume": 0.0, "cell_count": 0,
                    "max_magnitude": -math.inf, "max_rho": None, "max_z": None,
                    "nonfinite_count": 0, "negative_squared_count": 0,
                }
    for gid in sorted(children):
        con, metric, row = constraints[gid], adm[gid], topology[gid]
        ny, nx = con.shape[-2:]
        dx1 = (float(row["x1max"]) - float(row["x1min"])) / nx
        dx2 = (float(row["x2max"]) - float(row["x2min"])) / ny
        rho_values = float(row["x1min"]) + (np.arange(nx) + 0.5) * dx1
        z_values = float(row["x2min"]) + (np.arange(ny) + 0.5) * dx2
        detg = base.determinant(metric)
        if not np.all(np.isfinite(detg)) or not np.all(detg > 0.0):
            raise ComparisonError(f"invalid determinant in {arm}/{state}/gid={gid}")
        for j, z in enumerate(z_values):
            for i, rho in enumerate(rho_values):
                region = region_for_cell(gid, i, j, nx, ny, topology)
                coordinate = 2.0 * math.pi * float(rho) * dx1 * dx2
                weights = {"coordinate_ring": coordinate,
                           "proper_ring": coordinate * math.sqrt(float(detg[j, i]))}
                cell = {"arm": arm, "state": state, "gid": gid, "i": i, "j": j,
                        "rho": float(rho), "z": float(z), "region": region}
                for family in FAMILIES:
                    squared = float(family_squared(con, family)[j, i])
                    magnitude = math.sqrt(max(0.0, squared))
                    cell[f"{family}_squared"] = squared
                    finite = math.isfinite(squared)
                    for measure, weight in weights.items():
                        for target_region in (region, "GLOBAL"):
                            target = accum[(measure, family, target_region)]
                            target["volume"] += weight
                            target["cell_count"] += 1
                            if finite:
                                target["integral"] += squared * weight
                                if squared < 0.0:
                                    target["negative_squared_count"] += 1
                                if magnitude > target["max_magnitude"]:
                                    target["max_magnitude"] = magnitude
                                    target["max_rho"], target["max_z"] = float(rho), float(z)
                            else:
                                target["nonfinite_count"] += 1
                cells.append(cell)
    rows: list[dict[str, Any]] = []
    for (measure, family, region), values in accum.items():
        if values["nonfinite_count"]:
            raise ComparisonError(
                f"nonfinite constraint magnitude in {arm}/{state}/{family}/{region}")
        values["rms"] = (math.sqrt(values["integral"] / values["volume"])
                         if values["integral"] >= 0.0 else None)
        if values["max_magnitude"] == -math.inf:
            values["max_magnitude"] = None
        rows.append({"arm": arm, "state": state, "measure": measure,
                     "family": family, "region": region, **values})
    for measure in MEASURES:
        for family in FAMILIES:
            regional = sum(row["integral"] for row in rows
                           if row["measure"] == measure and row["family"] == family
                           and row["region"] in REGIONS)
            global_value = next(row["integral"] for row in rows
                                if row["measure"] == measure and row["family"] == family
                                and row["region"] == "GLOBAL")
            tolerance = 4096.0 * np.finfo(np.float64).eps * max(1.0, abs(global_value))
            if abs(regional - global_value) > tolerance:
                raise ComparisonError(
                    f"regional budget does not close for {arm}/{state}/{measure}/{family}")
    return rows, cells


def index_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str, str, str, str], dict[str, Any]]:
    return {(row["arm"], row["state"], row["measure"], row["family"], row["region"]): row
            for row in rows}


def classify(rows: list[dict[str, Any]]) -> dict[str, Any]:
    index = index_rows(rows)
    families: dict[str, Any] = {}
    improved: list[str] = []
    localization_consistent = True
    for family in FAMILIES:
        item: dict[str, Any] = {}
        dominant: dict[str, str] = {}
        for measure in MEASURES:
            high_before = index[("high_order", "T0", measure, family, "GLOBAL")]["integral"]
            high_after = index[("high_order", "T5", measure, family, "GLOBAL")]["integral"]
            limited_before = index[("limited_o2", "T0", measure, family, "GLOBAL")]["integral"]
            limited_after = index[("limited_o2", "T5", measure, family, "GLOBAL")]["integral"]
            if not math.isclose(high_before, limited_before, rel_tol=0.0,
                                abs_tol=4096.0 * np.finfo(float).eps *
                                max(1.0, abs(high_before))):
                raise ComparisonError(f"arms do not share the same T0 {measure}/{family}")
            if high_before <= 0.0 or limited_before <= 0.0:
                raise ComparisonError(f"zero common-lattice before norm for {family}")
            high_ratio, limited_ratio = high_after / high_before, limited_after / limited_before
            excess = {
                region: index[("high_order", "T5", measure, family, region)]["integral"] -
                        index[("high_order", "T0", measure, family, region)]["integral"]
                for region in REGIONS
            }
            dominant[measure] = max(REGIONS, key=lambda region: excess[region])
            region = dominant[measure]
            limited_excess = (
                index[("limited_o2", "T5", measure, family, region)]["integral"] -
                index[("limited_o2", "T0", measure, family, region)]["integral"])
            item[measure] = {
                "high_order_jump_ratio": high_ratio,
                "limited_o2_jump_ratio": limited_ratio,
                "improvement_factor": high_ratio / limited_ratio if limited_ratio > 0 else None,
                "no_more_than_25_percent_worse": limited_ratio <= 1.25 * high_ratio,
                "dominant_high_order_excess_region": region,
                "dominant_high_order_excess": excess[region],
                "limited_o2_excess_in_same_region": limited_excess,
                "same_region_reduced": limited_excess < excess[region],
            }
        localization_consistent = localization_consistent and (
            dominant["proper_ring"] == dominant["coordinate_ring"])
        proper = item["proper_ring"]
        if (proper["improvement_factor"] is not None and
                proper["improvement_factor"] >= 2.0):
            improved.append(family)
        families[family] = item
    no_worse = all(families[family]["proper_ring"]["no_more_than_25_percent_worse"]
                   for family in FAMILIES)
    same_region = all(families[family]["proper_ring"]["same_region_reduced"]
                      for family in improved)
    causal = len(improved) >= 2 and no_worse and same_region and localization_consistent
    edge_dominant = causal and sum(
        families[family]["proper_ring"]["dominant_high_order_excess_region"] ==
        "MESHBLOCK_EDGE_OR_CORNER" for family in improved) >= 2
    if causal and edge_dominant:
        disposition = "causal_gate_pass_meshblock_edge_dominant"
    elif causal:
        disposition = "causal_gate_pass_spatially_distributed"
    else:
        disposition = "causal_gate_fail_or_mixed"
    return {"families": families, "factor_two_families": improved,
            "no_family_worse_over_25_percent": no_worse,
            "same_dominant_region_reduced": same_region,
            "proper_coordinate_localization_consistent": localization_consistent,
            "causal_gate_pass": causal, "meshblock_edge_dominant": edge_dominant,
            "disposition": disposition}


def plot_maps(cells: list[dict[str, Any]], output: Path) -> None:
    for family in FAMILIES:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        for column, arm in enumerate(("high_order", "limited_o2")):
            before = {(row["gid"], row["i"], row["j"]): row
                      for row in cells if row["arm"] == arm and row["state"] == "T0"}
            after = [row for row in cells if row["arm"] == arm and row["state"] == "T5"]
            delta = np.asarray([row[f"{family}_squared"] -
                                before[(row["gid"], row["i"], row["j"])][f"{family}_squared"]
                                for row in after])
            scale = max(float(np.max(np.abs(delta))), np.finfo(float).tiny)
            scatter = axes[0, column].scatter(
                [row["rho"] for row in after], [row["z"] for row in after],
                c=delta, s=6, cmap="coolwarm", vmin=-scale, vmax=scale)
            axes[0, column].set_title(f"{arm}: delta {family} squared")
            fig.colorbar(scatter, ax=axes[0, column])
            difference = []
            if arm == "limited_o2":
                high_after = {(row["gid"], row["i"], row["j"]): row
                              for row in cells if row["arm"] == "high_order" and
                              row["state"] == "T5"}
                difference = [row[f"{family}_squared"] -
                              high_after[(row["gid"], row["i"], row["j"])][f"{family}_squared"]
                              for row in after]
                diff_scale = max(max(abs(value) for value in difference), np.finfo(float).tiny)
                diff_scatter = axes[1, column].scatter(
                    [row["rho"] for row in after], [row["z"] for row in after],
                    c=difference, s=6, cmap="coolwarm", vmin=-diff_scale,
                    vmax=diff_scale)
                axes[1, column].set_title(f"limited_o2 - high_order T5 {family} squared")
                fig.colorbar(diff_scatter, ax=axes[1, column])
            else:
                region_code = {region: index for index, region in enumerate(REGIONS)}
                region_scatter = axes[1, column].scatter(
                    [row["rho"] for row in after], [row["z"] for row in after],
                    c=[region_code[row["region"]] for row in after], s=6,
                    cmap="tab10", vmin=0, vmax=len(REGIONS) - 1)
                axes[1, column].set_title("disjoint spatial regions")
                fig.colorbar(region_scatter, ax=axes[1, column], ticks=range(len(REGIONS)))
            for axis in axes[:, column]:
                axis.set_xlabel(r"$\rho/M$")
                axis.set_ylabel(r"$z/M$")
                axis.set_aspect("equal")
        fig.savefig(output / f"target_event_{family}_regional_maps.png", dpi=180)
        plt.close(fig)


def analyze(args: argparse.Namespace) -> None:
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    high_event, high_schema, high_proposal = event_roots(
        args.high_root.resolve(), args.ranks, args.level_before,
        args.level_after, args.expected_cycle)
    limited_event, limited_schema, limited_proposal = event_roots(
        args.limited_root.resolve(), args.ranks, args.level_before,
        args.level_after, args.expected_cycle)
    if high_schema.get("amr_transfer") != "high_order" or \
            limited_schema.get("amr_transfer") != "limited_o2":
        raise ComparisonError("diagnostic roots do not identify the requested arms")
    if high_proposal != limited_proposal:
        raise ComparisonError("transfer arms accepted different topology proposals")
    high_prov = validate_provenance(
        args.high_provenance, args.high_root, "high_order", args.ranks)
    limited_prov = validate_provenance(
        args.limited_provenance, args.limited_root, "limited_o2", args.ranks)
    comparable_provenance(high_prov, limited_prov)
    children = selected_children(high_proposal)
    high_before, high_before_adm, high_topology, active = common_before(
        high_event, children, high_proposal)
    limited_before, limited_before_adm, limited_topology, limited_active = common_before(
        limited_event, children, limited_proposal)
    if high_topology != limited_topology or active != limited_active:
        raise ComparisonError("arms differ in accepted child lattice")
    for gid in children:
        if not np.array_equal(high_before[gid], limited_before[gid]) or \
                not np.array_equal(high_before_adm[gid], limited_before_adm[gid]):
            raise ComparisonError("arms do not share byte-identical reconstructed T0 state")
    phase = "t5_00_ADM_OR_CONSTRAINT_RECOMPUTATION"
    high_after = active_views(high_event, phase, "constraints", children)
    high_after_adm = active_views(high_event, phase, "adm", children)
    limited_after = active_views(limited_event, phase, "constraints", children)
    limited_after_adm = active_views(limited_event, phase, "adm", children)
    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for arm, before, before_adm, after, after_adm in [
            ("high_order", high_before, high_before_adm, high_after, high_after_adm),
            ("limited_o2", limited_before, limited_before_adm,
             limited_after, limited_after_adm)]:
        before_rows, before_cells = budgets(
            before, before_adm, high_topology, children, arm, "T0")
        after_rows, after_cells = budgets(
            after, after_adm, high_topology, children, arm, "T5")
        rows.extend(before_rows + after_rows)
        cells.extend(before_cells + after_cells)
    verdict = classify(rows)
    verdict.update({
        "schema": SCHEMA, "qualification_claim": False,
        "event_cycle": args.expected_cycle, "level_before": args.level_before,
        "level_after": args.level_after, "refined_child_count": len(children),
        "rank_count": args.ranks, "high_order_provenance": high_prov,
        "limited_o2_provenance": limited_prov,
        "limitations": [
            "This is a zero-PDE target-event transfer comparison, not convergence evidence.",
            "The common T0 constraints are authoritative parent constraint bytes mapped "
            "with one arm-independent O6 point representation; constraints are not "
            "recomputed from a synthetic fine evolved state.",
            "No Figure-3 reproduction or production-default transfer is claimed.",
        ],
    })
    with (output / "regional_budgets.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output / "cell_regions.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cells[0]))
        writer.writeheader()
        writer.writerows(cells)
    strict_dump(output / "verdict.json", verdict)
    plot_maps(cells, output)
    report = "# Zero-PDE Cartoon Z4c AMR transfer comparison\n\n"
    report += f"Disposition: `{verdict['disposition']}`. " \
              "This target-event artifact has `qualification_claim=false`.\n\n"
    report += "The proper-ring measure is authoritative; the coordinate-ring measure " \
              "is the densitization cross-check. Every refined child cell belongs to " \
              "exactly one region and all regional budgets close to roundoff.\n\n"
    report += "## Prospective gate\n\n"
    report += f"Factor-two families: `{verdict['factor_two_families']}`; " \
              f"no-worse gate: `{verdict['no_family_worse_over_25_percent']}`; " \
              f"localization agreement: " \
              f"`{verdict['proper_coordinate_localization_consistent']}`.\n"
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    products = sorted(path for path in output.iterdir() if path.is_file())
    with (output / "SHA256SUMS").open("w", encoding="utf-8") as stream:
        for path in products:
            stream.write(f"{base.sha256(path)}  {path.name}\n")


def self_test() -> None:
    topology = {
        0: {"level": "1", "lx1": "0", "lx2": "0", "x1min": "0",
            "x1max": "1", "x2min": "-1", "x2max": "0"},
        1: {"level": "1", "lx1": "1", "lx2": "0", "x1min": "1",
            "x1max": "2", "x2min": "-1", "x2max": "0"},
        2: {"level": "0", "lx1": "0", "lx2": "1", "x1min": "0",
            "x1max": "2", "x2min": "0", "x2max": "1"},
    }
    if region_for_cell(0, 0, 8, 32, 32, topology) != REGIONS[0]:
        raise ComparisonError("axis-region precedence failed")
    if region_for_cell(0, 16, 31, 32, 32, topology) != REGIONS[1]:
        raise ComparisonError("coarse-fine region precedence failed")
    if region_for_cell(0, 31, 16, 32, 32, topology) != REGIONS[2]:
        raise ComparisonError("MeshBlock-edge classification failed")
    if region_for_cell(0, 16, 16, 32, 32, topology) != REGIONS[3]:
        raise ComparisonError("interior classification failed")
    print("cartoon_amr_transfer_compare self-test: PASS")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--high-root", type=Path, required=True)
    analyze_parser.add_argument("--limited-root", type=Path, required=True)
    analyze_parser.add_argument("--high-provenance", type=Path, required=True)
    analyze_parser.add_argument("--limited-provenance", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--ranks", type=int, required=True)
    analyze_parser.add_argument("--level-before", type=int, required=True)
    analyze_parser.add_argument("--level-after", type=int, required=True)
    analyze_parser.add_argument("--expected-cycle", type=int, required=True)
    sub.add_parser("self-test")
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command == "self-test":
        self_test()
    else:
        analyze(args)


if __name__ == "__main__":
    main()
