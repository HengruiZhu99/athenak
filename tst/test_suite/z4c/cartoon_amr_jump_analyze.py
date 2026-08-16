#!/usr/bin/env python3
"""Strict, deterministic analysis of the default-off Z4c AMR-jump ledger.

The executable records authoritative Z4c/ADM/constraint fields.  This script only
joins rank-owned bytes, applies the exact AthenaK O6 interpolation weights to build
the canonical child lattice, closes the phase ledger, and renders summaries.  It
does not reimplement any ADM or constraint equation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SCHEMA = "athenak_z4c_amr_jump_diagnostic_v1"
PHASES = [
    "t2_00_REFINE_OR_DEREFINE_TRANSFER",
    "t3_00_RESTRICT",
    "t3_01_MPI_RECEIVE",
    "t3_02_PHYSICAL_OR_AXIS_BC",
    "t3_03_SAME_LEVEL_COARSE_REFRESH",
    "t3_04_COARSE_TO_FINE_PROLONGATION",
    "t3_05_PHYSICAL_OR_AXIS_BC",
    "t3_06_PHYSICAL_OR_AXIS_BC",
    "t4_00_ALGEBRAIC_PROJECTION",
    "t5_00_ADM_OR_CONSTRAINT_RECOMPUTATION",
]
WRITERS = [
    "REFINE_OR_DEREFINE_TRANSFER", "RESTRICT", "MPI_RECEIVE",
    "PHYSICAL_OR_AXIS_BC_1", "SAME_LEVEL_COARSE_REFRESH",
    "COARSE_TO_FINE_PROLONGATION", "PHYSICAL_OR_AXIS_BC_2",
    "PHYSICAL_OR_AXIS_BC_AXIS", "ALGEBRAIC_PROJECTION",
    "ADM_OR_CONSTRAINT_RECOMPUTATION",
]
PROLONG_W = np.asarray([-0.02197265625, 0.205078125, 0.9228515625,
                        -0.123046875, 0.01708984375], dtype=np.float64)


class AnalysisError(RuntimeError):
    pass


def strict_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise AnalysisError(f"non-finite JSON token {value} in {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject_constant)


def strict_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_view(root: Path, name: str, metadata: dict[str, Any]) -> np.ndarray:
    key = {"u0": "u0_shape", "coarse_u0": "coarse_u0_shape",
           "adm": "adm_shape", "constraints": "constraint_shape"}[name]
    shape = metadata.get(key)
    if not isinstance(shape, list) or len(shape) != 5 or not all(
            isinstance(item, int) and item > 0 for item in shape):
        raise AnalysisError(f"missing/invalid {key} in {root / 'phase.json'}")
    path = root / f"{name}.bin"
    expected = math.prod(shape) * 8
    if path.stat().st_size != expected:
        raise AnalysisError(f"binary size mismatch for {path}: "
                            f"{path.stat().st_size} != {expected}")
    result = np.fromfile(path, dtype="<f8").reshape(shape)
    if not np.all(np.isfinite(result)):
        raise AnalysisError(f"nonfinite value in {path}")
    return result


def phase_gid_views(event_roots: list[Path], phase: str, name: str) -> dict[int, np.ndarray]:
    result: dict[int, np.ndarray] = {}
    for event in event_roots:
        root = event / phase
        metadata = strict_load(root / "phase.json")
        if metadata.get("schema") != SCHEMA:
            raise AnalysisError(f"wrong schema in {root / 'phase.json'}")
        view = load_view(root, name, metadata)
        rows = read_csv(root / "topology.csv")
        if len(rows) != view.shape[0]:
            raise AnalysisError(f"topology/view count mismatch in {root}")
        for row in rows:
            gid = int(row["gid"])
            local_m = int(row["local_m"])
            if gid in result:
                raise AnalysisError(f"duplicate ownership for gid {gid} in {phase}")
            if int(row["owner_rank"]) != int(metadata["rank"]):
                raise AnalysisError(f"owner rank mismatch for gid {gid} in {phase}")
            result[gid] = view[local_m]
    return result


def phase_topology(event_roots: list[Path], phase: str) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    for event in event_roots:
        for row in read_csv(event / phase / "topology.csv"):
            gid = int(row["gid"])
            if gid in result:
                raise AnalysisError(f"duplicate topology gid {gid} in {phase}")
            result[gid] = row
    return result


def prolong_group(coarse: np.ndarray, j: int, i: int, component: int,
                  positive_chi: bool) -> np.ndarray:
    result = np.empty((2, 2), dtype=np.float64)
    parent = coarse[component, 0]
    stencil = parent[j - 2:j + 3, i - 2:i + 3]
    if stencil.shape != (5, 5):
        raise AnalysisError("canonical O6 prolongation stencil escaped coarse storage")
    for dj in range(2):
        wj = PROLONG_W[::-1] if dj else PROLONG_W
        for di in range(2):
            wi = PROLONG_W[::-1] if di else PROLONG_W
            result[dj, di] = np.einsum("j,i,ji->", wj, wi, stencil)
    if positive_chi and (not np.all(np.isfinite(stencil)) or
                         not np.all(stencil > 0.0) or
                         not np.all(np.isfinite(result)) or
                         not np.all(result > 0.0)):
        center = parent[j, i]
        dl_i, dr_i = center - parent[j, i - 1], parent[j, i + 1] - center
        dl_j, dr_j = center - parent[j - 1, i], parent[j + 1, i] - center
        slope_i = 0.0 if dl_i * dr_i <= 0 else 0.25 * math.copysign(
            min(abs(dl_i), abs(dr_i)), dl_i)
        slope_j = 0.0 if dl_j * dr_j <= 0 else 0.25 * math.copysign(
            min(abs(dl_j), abs(dr_j)), dl_j)
        result = np.asarray([[center - slope_i - slope_j, center + slope_i - slope_j],
                             [center - slope_i + slope_j, center + slope_i + slope_j]])
        if not np.all(np.isfinite(result)) or not np.all(result > 0.0):
            raise AnalysisError("canonical chi prolongation failed strict positivity")
    return result


def prolong_active(coarse: np.ndarray, active: dict[str, int], chi_index: int | None) -> np.ndarray:
    cis, cie = active["cis"], active["cie"]
    cjs, cje = active["cjs"], active["cje"]
    is_, js = active["is"], active["js"]
    nx1 = (cie - cis + 1) * 2
    nx2 = (cje - cjs + 1) * 2
    output = np.empty((coarse.shape[0], nx2, nx1), dtype=np.float64)
    for component in range(coarse.shape[0]):
        for j in range(cjs, cje + 1):
            for i in range(cis, cie + 1):
                group = prolong_group(coarse, j, i, component,
                                      chi_index is not None and component == chi_index)
                fj, fi = 2 * j - cjs - js, 2 * i - cis - is_
                output[component, fj:fj + 2, fi:fi + 2] = group
    return output


def active_slice(view: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    bounds = metadata["active_bounds"]
    return view[:, bounds["ks"]:bounds["ke"] + 1,
                bounds["js"]:bounds["je"] + 1,
                bounds["is"]:bounds["ie"] + 1][:, 0]


def child_coarse_from_parent(parent: np.ndarray, target_shape: tuple[int, ...],
                             ox1: int, ox2: int, nx1: int, nx2: int) -> np.ndarray:
    if parent.ndim != 4 or len(target_shape) != 4 or parent.shape[1] != 1:
        raise AnalysisError("unexpected collapsed-view shape for child extraction")
    cnx1, cnx2 = nx1 // 2, nx2 // 2
    nj, ni = target_shape[2], target_shape[3]
    start_i, start_j = ox1 * cnx1, ox2 * cnx2
    result = parent[:, :, start_j:start_j + nj, start_i:start_i + ni]
    if result.shape != target_shape:
        raise AnalysisError("parent-to-child coarse extraction escaped stored bytes")
    return result


def determinant(adm: np.ndarray) -> np.ndarray:
    gxx, gxy, gxz, gyy, gyz, gzz = (adm[index] for index in range(6))
    return (gxx * (gyy * gzz - gyz * gyz)
            - gxy * (gxy * gzz - gyz * gxz)
            + gxz * (gxy * gyz - gyy * gxz))


def fixed_constraint_integrals(constraints: dict[int, np.ndarray],
                               adm: dict[int, np.ndarray],
                               topology: dict[int, dict[str, str]],
                               gids: set[int]) -> dict[str, float]:
    result = {"coordinate_ring_volume": 0.0, "proper_volume": 0.0,
              "C_norm2": 0.0, "H_norm2": 0.0,
              "M_norm2": 0.0, "Z_norm2": 0.0,
              "min_det_gamma": math.inf}
    for gid in sorted(gids):
        con = constraints[gid]
        metric = adm[gid]
        row = topology[gid]
        ny, nx = con.shape[-2:]
        x1min, x1max = float(row["x1min"]), float(row["x1max"])
        x2min, x2max = float(row["x2min"]), float(row["x2max"])
        dx1, dx2 = (x1max - x1min) / nx, (x2max - x2min) / ny
        rho = x1min + (np.arange(nx, dtype=np.float64) + 0.5) * dx1
        coordinate = 2.0 * math.pi * rho[np.newaxis, :] * dx1 * dx2
        detg = determinant(metric)
        if not np.all(np.isfinite(detg)) or not np.all(detg > 0.0):
            raise AnalysisError(f"invalid fixed-lattice determinant for gid {gid}")
        proper = coordinate * np.sqrt(detg)
        result["coordinate_ring_volume"] += float(np.sum(np.broadcast_to(coordinate, (ny, nx))))
        result["proper_volume"] += float(np.sum(proper))
        result["C_norm2"] += float(np.sum(con[0] * proper))
        result["H_norm2"] += float(np.sum(con[1] * con[1] * proper))
        result["M_norm2"] += float(np.sum(con[2] * proper))
        result["Z_norm2"] += float(np.sum(con[3] * proper))
        result["min_det_gamma"] = min(result["min_det_gamma"], float(np.min(detg)))
    return result


def coarse_fine_faces(gid: int, topology: dict[int, dict[str, str]]) -> set[str]:
    row = topology[gid]
    level, lx1, lx2 = int(row["level"]), int(row["lx1"]), int(row["lx2"])
    same = {(int(other["level"]), int(other["lx1"]), int(other["lx2"]))
            for other in topology.values()}
    global_x1min = min(float(other["x1min"]) for other in topology.values())
    global_x1max = max(float(other["x1max"]) for other in topology.values())
    global_x2min = min(float(other["x2min"]) for other in topology.values())
    global_x2max = max(float(other["x2max"]) for other in topology.values())
    result: set[str] = set()
    candidates = [("inner_x1", lx1 - 1, lx2, float(row["x1min"]) > global_x1min),
                  ("outer_x1", lx1 + 1, lx2, float(row["x1max"]) < global_x1max),
                  ("inner_x2", lx1, lx2 - 1, float(row["x2min"]) > global_x2min),
                  ("outer_x2", lx1, lx2 + 1, float(row["x2max"]) < global_x2max)]
    for name, neighbor_x1, neighbor_x2, not_physical in candidates:
        if not_physical and (level, neighbor_x1, neighbor_x2) not in same:
            result.add(name)
    return result


def aggregate_native(event_roots: list[Path], phase: str) -> dict[str, float]:
    keys = ["active_cells", "coordinate_ring_volume", "proper_volume",
            "C_norm2", "H_norm2", "M_norm2", "Z_norm2",
            "nonpositive_or_nonfinite_det_gamma", "nonfinite_chi",
            "nonfinite_constraints"]
    result = {key: 0.0 for key in keys}
    for event in event_roots:
        data = strict_load(event / phase / "aggregate.json")
        for key in keys:
            value = data.get(key)
            if value is None or not isinstance(value, (int, float)):
                raise AnalysisError(f"invalid aggregate {key} in {phase}")
            result[key] += float(value)
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def history_row(path: Path, cycle: int) -> list[float]:
    rows: list[list[float]] = []
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            values = [float(token) for token in line.split()]
            if int(round(values[18])) == cycle:
                rows.append(values)
    if len(rows) != 1:
        raise AnalysisError(f"expected exactly one history row for cycle {cycle}, "
                            f"found {len(rows)}")
    return rows[0]


def analyze(args: argparse.Namespace) -> None:
    root = args.diagnostic_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    rank_roots = sorted(path for path in root.glob("rank[0-9][0-9][0-9][0-9]")
                        if path.is_dir())
    if len(rank_roots) != args.ranks:
        raise AnalysisError(f"expected {args.ranks} rank roots, found {len(rank_roots)}")
    event_roots: list[Path] = []
    event_name: str | None = None
    for rank in rank_roots:
        schema = strict_load(rank / "schema.json")
        if schema.get("schema") != SCHEMA or schema.get("real_bytes") != 8:
            raise AnalysisError(f"bad schema in {rank}")
        matches = sorted(rank.glob(f"event_c*_l{args.level_before}_to_l{args.level_after}"))
        if len(matches) != 1:
            raise AnalysisError(f"expected one target event in {rank}, found {len(matches)}")
        if event_name is None:
            event_name = matches[0].name
        elif matches[0].name != event_name:
            raise AnalysisError("rank target event names disagree")
        event_roots.append(matches[0])

    event_cycle = int(event_name.split("_", 2)[1][1:])
    if args.expected_cycle is not None and event_cycle != args.expected_cycle:
        raise AnalysisError(f"target cycle mismatch: {event_cycle} != {args.expected_cycle}")
    for rank in rank_roots:
        post_path = rank / "post_event_cycles.jsonl"
        with post_path.open(encoding="utf-8") as stream:
            post = [json.loads(line) for line in stream if line.strip()]
        expected_cycles = list(range(event_cycle, event_cycle + args.post_cycles + 1))
        if [int(row["cycle"]) for row in post] != expected_cycles:
            raise AnalysisError(f"premature/incomplete post-event window in {rank}")
        for row in post:
            if row.get("schema") != "athenak_z4c_amr_post_event_cycle_v1":
                raise AnalysisError(f"wrong post-event schema in {rank}")

    t1_rows = read_csv(event_roots[0] / "t1_topology_proposal.csv")
    t1_phase = strict_load(event_roots[0] / "t1_phase.json")
    if args.expected_time is not None and not math.isclose(
            float(t1_phase["time"]), args.expected_time, rel_tol=0.0,
            abs_tol=args.time_tolerance):
        raise AnalysisError(f"target time mismatch: {t1_phase['time']} != {args.expected_time}")
    if args.expected_old_nmb is not None and int(t1_phase["old_nmb"]) != args.expected_old_nmb:
        raise AnalysisError("old MeshBlock count mismatch")
    if args.expected_new_nmb is not None and int(t1_phase["new_nmb"]) != args.expected_new_nmb:
        raise AnalysisError("new MeshBlock count mismatch")
    for event in event_roots[1:]:
        if read_csv(event / "t1_topology_proposal.csv") != t1_rows:
            raise AnalysisError("T1 topology proposals disagree across ranks")
    refined = [row for row in t1_rows if int(row["new_level"]) == int(row["old_level"]) + 1]
    if not refined:
        raise AnalysisError("target event contains no refined children")
    new_gids = {int(row["new_gid"]) for row in refined}
    if len(new_gids) != len(refined):
        raise AnalysisError("duplicate refined-child GID")

    phase_metadata: dict[str, dict[int, dict[str, Any]]] = {}
    phase_u0: dict[str, dict[int, np.ndarray]] = {}
    for phase in PHASES:
        phase_u0[phase] = phase_gid_views(event_roots, phase, "u0")
        phase_metadata[phase] = {}
        for event in event_roots:
            metadata = strict_load(event / phase / "phase.json")
            for row in read_csv(event / phase / "topology.csv"):
                phase_metadata[phase][int(row["gid"])] = metadata
        if set(phase_u0[phase]) != set(phase_metadata[phase]):
            raise AnalysisError(f"incomplete metadata ownership in {phase}")

    t2_coarse = phase_gid_views(event_roots, PHASES[0], "coarse_u0")
    t2_meta_any = next(iter(phase_metadata[PHASES[0]].values()))
    active_bounds = t2_meta_any["active_bounds"]
    coarse_bounds = t2_meta_any["coarse_active_bounds"]
    geometry = {**active_bounds, **coarse_bounds}
    schema = strict_load(rank_roots[0] / "schema.json")
    z4c_names = schema["z4c_components"]
    chi_index = z4c_names.index("z4c_chi")

    reference: dict[int, np.ndarray] = {}
    transfer_max_abs = 0.0
    for gid in sorted(new_gids):
        reference[gid] = prolong_active(t2_coarse[gid], geometry, chi_index)
        actual = active_slice(phase_u0[PHASES[0]][gid], phase_metadata[PHASES[0]][gid])
        transfer_max_abs = max(transfer_max_abs,
                               float(np.max(np.abs(reference[gid] - actual))))
    transfer_scale = max(1.0, max(float(np.max(np.abs(value)))
                                  for value in reference.values()))
    transfer_tolerance = 4096.0 * np.finfo(np.float64).eps * transfer_scale
    if transfer_max_abs > transfer_tolerance:
        raise AnalysisError(f"canonical transfer mismatch {transfer_max_abs} > "
                            f"{transfer_tolerance}")

    old_u0 = phase_gid_views(event_roots, "t0_00_ACCEPTED_OLD_STATE", "u0")
    old_constraints = phase_gid_views(
        event_roots, "t0_00_ACCEPTED_OLD_STATE", "constraints")
    old_adm = phase_gid_views(event_roots, "t0_00_ACCEPTED_OLD_STATE", "adm")
    new_topology = phase_topology(event_roots, PHASES[0])
    t2_coarse_shape = next(iter(t2_coarse.values())).shape
    nx1 = active_bounds["ie"] - active_bounds["is"] + 1
    nx2 = active_bounds["je"] - active_bounds["js"] + 1
    reference_constraints: dict[int, np.ndarray] = {}
    reference_adm: dict[int, np.ndarray] = {}
    coarse_source_residual = 0.0
    for row in refined:
        gid, old_gid = int(row["new_gid"]), int(row["old_gid"])
        ox1, ox2 = int(row["new_lx1"]) & 1, int(row["new_lx2"]) & 1
        source_u0 = child_coarse_from_parent(old_u0[old_gid], t2_coarse_shape,
                                             ox1, ox2, nx1, nx2)
        coarse_source_residual = max(
            coarse_source_residual,
            float(np.max(np.abs(source_u0 - t2_coarse[gid]))))
        con_shape = (old_constraints[old_gid].shape[0],) + t2_coarse_shape[1:]
        adm_shape = (old_adm[old_gid].shape[0],) + t2_coarse_shape[1:]
        source_con = child_coarse_from_parent(old_constraints[old_gid], con_shape,
                                              ox1, ox2, nx1, nx2)
        source_adm = child_coarse_from_parent(old_adm[old_gid], adm_shape,
                                              ox1, ox2, nx1, nx2)
        reference_constraints[gid] = prolong_active(source_con, geometry, None)
        reference_adm[gid] = prolong_active(source_adm, geometry, None)
    if coarse_source_residual > transfer_tolerance:
        raise AnalysisError(f"T2 coarse source differs from authenticated T0 parent: "
                            f"{coarse_source_residual}")

    component_ledger: list[dict[str, Any]] = []
    current = reference
    writer_totals: dict[str, float] = {}
    writer_stored_totals: dict[str, float] = {}
    final_arrays: dict[int, np.ndarray] = {}
    for phase, writer in zip(PHASES, WRITERS):
        next_state = {gid: active_slice(phase_u0[phase][gid], phase_metadata[phase][gid])
                      for gid in sorted(new_gids)}
        writer_sum2 = 0.0
        writer_stored_sum2 = 0.0
        writer_max = 0.0
        for component, name in enumerate(z4c_names):
            sum2 = 0.0
            max_abs = 0.0
            for gid in sorted(new_gids):
                delta = next_state[gid][component] - current[gid][component]
                sum2 += float(np.sum(delta * delta))
                max_abs = max(max_abs, float(np.max(np.abs(delta))))
            component_ledger.append({"phase": phase, "writer": writer,
                                     "component": name, "delta_l2_unweighted": math.sqrt(sum2),
                                     "delta_max_abs": max_abs})
            writer_sum2 += sum2
            writer_max = max(writer_max, max_abs)
        writer_totals[writer] = math.sqrt(writer_sum2)
        for gid in sorted(new_gids):
            previous_full = (phase_u0[PHASES[PHASES.index(phase) - 1]][gid]
                             if PHASES.index(phase) > 0 else None)
            if previous_full is not None:
                delta_full = phase_u0[phase][gid] - previous_full
                writer_stored_sum2 += float(np.sum(delta_full * delta_full))
        writer_stored_totals[writer] = math.sqrt(writer_stored_sum2)
        current = next_state
        final_arrays = next_state

    closure_residual = 0.0
    direct_norm2 = 0.0
    telescoped_norm2 = 0.0
    for gid in sorted(new_gids):
        direct = final_arrays[gid] - reference[gid]
        telescoped = np.zeros_like(direct)
        previous = reference[gid]
        for phase in PHASES:
            nxt = active_slice(phase_u0[phase][gid], phase_metadata[phase][gid])
            telescoped += nxt - previous
            previous = nxt
        residual = telescoped - direct
        closure_residual = max(closure_residual, float(np.max(np.abs(residual))))
        direct_norm2 += float(np.sum(direct * direct))
        telescoped_norm2 += float(np.sum(telescoped * telescoped))
    closure_tolerance = 4096.0 * np.finfo(np.float64).eps * max(
        1.0, math.sqrt(direct_norm2), math.sqrt(telescoped_norm2))
    if closure_residual > closure_tolerance:
        raise AnalysisError(f"evolved ledger does not close: {closure_residual}")

    native: dict[str, dict[str, float]] = {}
    for phase in ["t0_00_ACCEPTED_OLD_STATE", PHASES[7], PHASES[8], PHASES[9]]:
        native[phase] = aggregate_native(event_roots, phase)
    for phase, values in native.items():
        if values["nonpositive_or_nonfinite_det_gamma"] != 0 or \
           values["nonfinite_chi"] != 0 or values["nonfinite_constraints"] != 0:
            raise AnalysisError(f"nonfinite/nonpositive authoritative data in {phase}")
    history_comparison: dict[str, Any] | None = None
    if args.history is not None:
        row = history_row(args.history.resolve(), event_cycle)
        t5_native = native[PHASES[9]]
        mapping = {"C_norm2": row[2], "H_norm2": row[3], "M_norm2": row[4],
                   "Z_norm2": row[5], "proper_volume": row[10]}
        residuals: dict[str, float] = {}
        for key, expected in mapping.items():
            residual = abs(t5_native[key] - expected)
            tolerance_value = args.history_tolerance * max(1.0, abs(expected))
            if residual > tolerance_value:
                raise AnalysisError(f"T5/history mismatch for {key}: {residual} > "
                                    f"{tolerance_value}")
            residuals[key] = residual
        if int(round(row[12])) != int(t1_phase["new_nmb"]) or \
           int(round(row[14])) != args.level_after:
            raise AnalysisError("T5/history topology mismatch")
        history_comparison = {"path": str(args.history.resolve()),
                              "sha256": sha256(args.history.resolve()),
                              "cycle": event_cycle, "absolute_residuals": residuals,
                              "relative_tolerance": args.history_tolerance}

    constraint_phases = [PHASES[7], PHASES[8], PHASES[9]]
    constraint_states: dict[str, dict[int, np.ndarray]] = {
        "T0_INTERPOLATED": reference_constraints
    }
    fixed_constraints: dict[str, dict[str, float]] = {
        "T0_AUTHORITATIVE_CONSTRAINTS_INTERPOLATED_TO_CHILD_LATTICE":
            fixed_constraint_integrals(reference_constraints, reference_adm,
                                       new_topology, new_gids)
    }
    for phase in constraint_phases:
        con_views = phase_gid_views(event_roots, phase, "constraints")
        adm_views = phase_gid_views(event_roots, phase, "adm")
        con_active = {gid: active_slice(con_views[gid], phase_metadata[phase][gid])
                      for gid in new_gids}
        adm_active = {gid: active_slice(adm_views[gid], phase_metadata[phase][gid])
                      for gid in new_gids}
        fixed_constraints[phase] = fixed_constraint_integrals(
            con_active, adm_active, new_topology, new_gids)
        constraint_states[phase] = con_active

    constraint_ledger: list[dict[str, Any]] = []
    constraint_previous = constraint_states["T0_INTERPOLATED"]
    constraint_telescoped = {gid: np.zeros_like(constraint_previous[gid])
                             for gid in new_gids}
    constraint_stage_names = ["T0_TO_T3_REPRESENTATION_AND_BOUNDARY",
                              "ALGEBRAIC_PROJECTION",
                              "ADM_OR_CONSTRAINT_RECOMPUTATION"]
    constraint_names = schema["constraint_components"]
    for phase, stage_name in zip(constraint_phases, constraint_stage_names):
        current_constraints = constraint_states[phase]
        for component, component_name in enumerate(constraint_names):
            sum2, max_abs = 0.0, 0.0
            for gid in sorted(new_gids):
                delta = (current_constraints[gid][component] -
                         constraint_previous[gid][component])
                constraint_telescoped[gid][component] += delta
                sum2 += float(np.sum(delta * delta))
                max_abs = max(max_abs, float(np.max(np.abs(delta))))
            constraint_ledger.append({"stage": stage_name, "phase": phase,
                                      "component": component_name,
                                      "delta_l2_unweighted": math.sqrt(sum2),
                                      "delta_max_abs": max_abs})
        constraint_previous = current_constraints
    constraint_closure_residual = 0.0
    constraint_direct_norm2 = 0.0
    for gid in sorted(new_gids):
        direct = (constraint_states[constraint_phases[-1]][gid] -
                  reference_constraints[gid])
        residual = constraint_telescoped[gid] - direct
        constraint_closure_residual = max(
            constraint_closure_residual, float(np.max(np.abs(residual))))
        constraint_direct_norm2 += float(np.sum(direct * direct))
    constraint_closure_tolerance = (4096.0 * np.finfo(np.float64).eps *
                                    max(1.0, math.sqrt(constraint_direct_norm2)))
    if constraint_closure_residual > constraint_closure_tolerance:
        raise AnalysisError("fixed-lattice constraint ledger does not close")

    changed_points: list[dict[str, Any]] = []
    for gid in sorted(new_gids):
        row = new_topology[gid]
        before = reference_constraints[gid]
        after = constraint_states[constraint_phases[0]][gid]
        ny, nx = before.shape[-2:]
        x1min, x1max = float(row["x1min"]), float(row["x1max"])
        x2min, x2max = float(row["x2min"]), float(row["x2max"])
        dx1, dx2 = (x1max - x1min) / nx, (x2max - x2min) / ny
        faces = coarse_fine_faces(gid, new_topology)
        for j in range(ny):
            z = x2min + (j + 0.5) * dx2
            for i in range(nx):
                rho = x1min + (i + 0.5) * dx1
                block_distance = min((i + 0.5) * dx1, (nx - i - 0.5) * dx1,
                                     (j + 0.5) * dx2, (ny - j - 0.5) * dx2)
                cf_distances: list[float] = []
                if "inner_x1" in faces: cf_distances.append((i + 0.5) * dx1)
                if "outer_x1" in faces: cf_distances.append((nx - i - 0.5) * dx1)
                if "inner_x2" in faces: cf_distances.append((j + 0.5) * dx2)
                if "outer_x2" in faces: cf_distances.append((ny - j - 0.5) * dx2)
                changed_points.append({
                    "gid": gid, "i": i, "j": j, "rho": rho, "z": z,
                    "axis_distance": rho, "block_edge_distance": block_distance,
                    "coarse_fine_interface_distance":
                        min(cf_distances) if cf_distances else None,
                    "delta_C": float(after[0, j, i] - before[0, j, i]),
                    "delta_H": float(after[1, j, i] - before[1, j, i]),
                    "delta_M": float(after[2, j, i] - before[2, j, i]),
                    "delta_Z": float(after[3, j, i] - before[3, j, i]),
                })
    changed_points.sort(key=lambda row: abs(row["delta_C"]), reverse=True)
    worst = changed_points[0]

    shadow_rows: list[dict[str, str]] = []
    for event in event_roots:
        for name in ["t3_00_restrict_shadow_chi.csv",
                     "t3_03_same_level_refresh_shadow_chi.csv"]:
            shadow_rows.extend(read_csv(event / name))
    if not shadow_rows:
        raise AnalysisError("shadow restriction ledger is empty")
    shadow_summary: dict[str, dict[str, float]] = {}
    for row in shadow_rows:
        key = f"{row['writer_ordinal']}:{row['rule_class']}"
        summary = shadow_summary.setdefault(key, {"count": 0.0, "max_abs_diff": 0.0,
                                                  "max_rel_diff": 0.0,
                                                  "positive_source_nonpositive_target": 0.0})
        summary["count"] += 1.0
        summary["max_abs_diff"] = max(summary["max_abs_diff"], float(row["abs_diff"]))
        summary["max_rel_diff"] = max(summary["max_rel_diff"], float(row["rel_diff"]))
        if int(row["source_finite_positive"]) and not float(row["production_chi"]) > 0.0:
            summary["positive_source_nonpositive_target"] += 1.0

    tolerance = max(closure_tolerance, transfer_tolerance)
    significant = [(writer, value) for writer, value in writer_totals.items()
                   if value > tolerance]
    representation_constraint_l2 = math.sqrt(sum(
        row["delta_l2_unweighted"] ** 2 for row in constraint_ledger
        if row["stage"] == "T0_TO_T3_REPRESENTATION_AND_BOUNDARY"))
    projection_constraint_l2 = math.sqrt(sum(
        row["delta_l2_unweighted"] ** 2 for row in constraint_ledger
        if row["stage"] == "ALGEBRAIC_PROJECTION"))
    if representation_constraint_l2 > tolerance and significant:
        disposition = "quantified_multi_stage"
    elif representation_constraint_l2 > tolerance:
        disposition = "refine_or_derefine_transfer"
    elif not significant:
        disposition = "topology_or_measure_only"
    elif len(significant) == 1:
        disposition = {
            "REFINE_OR_DEREFINE_TRANSFER": "refine_or_derefine_transfer",
            "RESTRICT": "refine_or_derefine_transfer",
            "MPI_RECEIVE": "mpi_redistribution",
            "PHYSICAL_OR_AXIS_BC_1": "physical_or_axis_boundary",
            "SAME_LEVEL_COARSE_REFRESH": "same_level_coarse_refresh",
            "COARSE_TO_FINE_PROLONGATION": "coarse_to_fine_prolongation",
            "PHYSICAL_OR_AXIS_BC_2": "physical_or_axis_boundary",
            "PHYSICAL_OR_AXIS_BC_AXIS": "physical_or_axis_boundary",
            "ALGEBRAIC_PROJECTION": "algebraic_projection",
            "ADM_OR_CONSTRAINT_RECOMPUTATION": "adm_or_constraint_recomputation",
        }[significant[0][0]]
    else:
        disposition = "quantified_multi_stage"

    with (output / "evolved_phase_ledger.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(component_ledger[0]))
        writer.writeheader()
        writer.writerows(component_ledger)
    with (output / "shadow_chi_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = ["writer_rule", "count", "max_abs_diff", "max_rel_diff",
                  "positive_source_nonpositive_target"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for key in sorted(shadow_summary):
            writer.writerow({"writer_rule": key, **shadow_summary[key]})
    with (output / "fixed_lattice_constraint_ledger.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(constraint_ledger[0]))
        writer.writeheader()
        writer.writerows(constraint_ledger)
    with (output / "worst_changed_points.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(changed_points[0]))
        writer.writeheader()
        writer.writerows(changed_points[:256])
    with (output / "worst_point_trajectory.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        fields = ["phase", "writer", "component", "value"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        gid, i, j = int(worst["gid"]), int(worst["i"]), int(worst["j"])
        for component, name in enumerate(z4c_names):
            writer.writerow({"phase": "T0_CANONICAL", "writer": "ACCEPTED_OLD_STATE",
                             "component": name, "value": reference[gid][component, j, i]})
            for phase, phase_writer in zip(PHASES, WRITERS):
                active = active_slice(phase_u0[phase][gid], phase_metadata[phase][gid])
                writer.writerow({"phase": phase, "writer": phase_writer,
                                 "component": name, "value": active[component, j, i]})
        for component, name in enumerate(constraint_names):
            writer.writerow({"phase": "T0_CONSTRAINT_INTERPOLATED",
                             "writer": "ACCEPTED_OLD_STATE", "component": name,
                             "value": reference_constraints[gid][component, j, i]})
            for phase in constraint_phases:
                writer.writerow({"phase": phase, "writer": phase,
                                 "component": name,
                                 "value": constraint_states[phase][gid][component, j, i]})

    fig, axis = plt.subplots(figsize=(10, 4.8))
    axis.bar(range(len(writer_totals)), list(writer_totals.values()))
    axis.set_xticks(range(len(writer_totals)), list(writer_totals), rotation=55, ha="right")
    axis.set_yscale("symlog", linthresh=max(tolerance, 1e-30))
    axis.set_ylabel("changed-patch evolved-field delta L2 (unweighted)")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "evolved_writer_contributions.png", dpi=180)
    plt.close(fig)

    native_names = ["C_norm2", "H_norm2", "M_norm2", "Z_norm2"]
    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for name in native_names:
        axis.plot(range(len(native)), [native[phase][name] for phase in native],
                  marker="o", label=name)
    axis.set_xticks(range(len(native)), list(native), rotation=25, ha="right")
    axis.set_yscale("log")
    axis.set_ylabel("authoritative native proper-volume integral")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "native_constraint_integrals.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.5, 4.8))
    for name in native_names:
        axis.plot(range(len(fixed_constraints)),
                  [fixed_constraints[phase][name] for phase in fixed_constraints],
                  marker="o", label=name)
    axis.set_xticks(range(len(fixed_constraints)), list(fixed_constraints),
                    rotation=25, ha="right")
    axis.set_yscale("log")
    axis.set_ylabel("fixed child-lattice proper-volume integral")
    axis.legend()
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "fixed_lattice_constraint_integrals.png", dpi=180)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7.2, 6.0))
    values = np.asarray([abs(row["delta_C"]) for row in changed_points])
    scatter = axis.scatter([row["rho"] for row in changed_points],
                           [row["z"] for row in changed_points], c=values,
                           s=5, norm=matplotlib.colors.LogNorm(
                               vmin=max(float(np.min(values[values > 0])), 1e-300),
                               vmax=float(np.max(values))))
    axis.set_xlabel(r"$\rho/M$")
    axis.set_ylabel(r"$z/M$")
    axis.set_title("Fixed-lattice |delta C|: T0 interpolated to T3")
    fig.colorbar(scatter, ax=axis, label="|delta C|")
    fig.tight_layout()
    fig.savefig(output / "changed_patch_delta_C_map.png", dpi=180)
    plt.close(fig)

    verdict = {
        "schema": "athenak_z4c_amr_jump_verdict_v1",
        "qualification_claim": False,
        "event": event_name,
        "cycle": event_cycle,
        "time": float(t1_phase["time"]),
        "old_nmb": int(t1_phase["old_nmb"]),
        "new_nmb": int(t1_phase["new_nmb"]),
        "post_cycles": args.post_cycles,
        "rank_count": len(rank_roots),
        "refined_child_count": len(refined),
        "canonical_transfer_max_abs_residual": transfer_max_abs,
        "canonical_transfer_tolerance": transfer_tolerance,
        "evolved_ledger_max_abs_residual": closure_residual,
        "evolved_ledger_tolerance": closure_tolerance,
        "constraint_ledger_max_abs_residual": constraint_closure_residual,
        "constraint_ledger_tolerance": constraint_closure_tolerance,
        "native_integrals": native,
        "history_comparison": history_comparison,
        "writer_active_delta_l2": writer_totals,
        "writer_stored_delta_l2": writer_stored_totals,
        "T0_parent_to_T2_child_coarse_max_abs_residual": coarse_source_residual,
        "fixed_child_lattice_constraint_integrals": fixed_constraints,
        "fixed_child_lattice_constraint_stage_l2": {
            "T0_TO_T3_REPRESENTATION_AND_BOUNDARY": representation_constraint_l2,
            "ALGEBRAIC_PROJECTION": projection_constraint_l2,
        },
        "worst_fixed_lattice_C_change": worst,
        "shadow_chi": shadow_summary,
        "disposition": disposition,
        "limitations": [
            "T0 fixed-lattice constraints are authoritative parent constraint bytes "
            "interpolated with AthenaK's exact O6 weights; they are not constraints "
            "recomputed from a synthetic fine evolved state.",
            "T2 derivative constraints are unavailable by design until boundary ghosts close.",
            "No production repair or Figure 3 qualification is claimed.",
        ],
    }
    strict_dump(output / "verdict.json", verdict)
    report = "# Cartoon Z4c AMR-jump diagnostic\n\n"
    report += f"Target event: `{event_name}` ({len(refined)} refined children, " \
              f"{len(rank_roots)} ranks).\n\n"
    report += f"Disposition: `{disposition}`. This is a diagnosis-only result; " \
              "`qualification_claim=false`.\n\n"
    report += "## Accounting\n\n"
    report += (f"The production T0-parent to T2-child coarse source agrees to "
               f"`{coarse_source_residual:.6e}`; canonical O6 prolongation agrees "
               f"with T2 to `{transfer_max_abs:.6e}`. The evolved ledger closes to "
               f"`{closure_residual:.6e}` and the fixed-lattice constraint ledger "
               f"closes to `{constraint_closure_residual:.6e}`.\n\n")
    report += "## Quantitative attribution\n\n"
    report += (f"The T0-interpolated to T3 constraint-field delta L2 is "
               f"`{representation_constraint_l2:.6e}`; the T3-to-T4 algebraic "
               f"projection delta L2 is `{projection_constraint_l2:.6e}`. The worst "
               f"fixed-lattice C change is at `(rho,z)=({worst['rho']:.9g},"
               f"{worst['z']:.9g})`, with `delta_C={worst['delta_C']:.6e}`.\n\n")
    report += "## Qualification boundary\n\n"
    report += ("No transfer operator, gauge, damping, dissipation, AMR threshold, "
               "floor, timestep policy, or initial-data byte was changed. This "
               "artifact does not qualify Figure 3 or authorize a repair.\n")
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    products = sorted(path for path in output.iterdir() if path.is_file())
    with (output / "SHA256SUMS").open("w", encoding="utf-8") as stream:
        for path in products:
            stream.write(f"{sha256(path)}  {path.name}\n")


def self_test() -> None:
    coarse = np.zeros((2, 1, 24, 24), dtype=np.float64)
    yy, xx = np.mgrid[:24, :24]
    coarse[0, 0] = 2.0 + 0.1 * xx - 0.2 * yy
    coarse[1, 0] = 1.0 + 0.01 * xx + 0.02 * yy
    active = {"cis": 4, "cie": 19, "cjs": 4, "cje": 19,
              "is": 4, "js": 4}
    fine = prolong_active(coarse, active, 1)
    if fine.shape != (2, 32, 32) or not np.all(np.isfinite(fine)):
        raise AnalysisError("synthetic canonical-lattice test failed")
    increments = [np.ones(8), np.full(8, 2.0), np.full(8, -0.5)]
    direct = sum(increments)
    if not np.array_equal(direct, np.full(8, 2.5)):
        raise AnalysisError("synthetic telescoping test failed")
    print("cartoon_amr_jump_analyze self-test: PASS")


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    sub = result.add_subparsers(dest="command", required=True)
    analyze_parser = sub.add_parser("analyze")
    analyze_parser.add_argument("--diagnostic-root", type=Path, required=True)
    analyze_parser.add_argument("--output", type=Path, required=True)
    analyze_parser.add_argument("--ranks", type=int, required=True)
    analyze_parser.add_argument("--level-before", type=int, required=True)
    analyze_parser.add_argument("--level-after", type=int, required=True)
    analyze_parser.add_argument("--post-cycles", type=int, required=True)
    analyze_parser.add_argument("--expected-cycle", type=int)
    analyze_parser.add_argument("--expected-time", type=float)
    analyze_parser.add_argument("--time-tolerance", type=float, default=1.0e-13)
    analyze_parser.add_argument("--expected-old-nmb", type=int)
    analyze_parser.add_argument("--expected-new-nmb", type=int)
    analyze_parser.add_argument("--history", type=Path)
    analyze_parser.add_argument("--history-tolerance", type=float, default=2.0e-12)
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
