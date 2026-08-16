#!/usr/bin/env python3
"""Audit the accepted parent state immediately before a target AMR event.

This is offline analysis of AthenaK-authored field bytes.  It evaluates the
production O6-configured R/P self-shadow, O6-versus-O4 derivative disagreement,
and block-local odd/even content on only the parent MeshBlocks selected for
refinement.  It does not alter or advance a numerical state.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROLONG = np.asarray([-0.02197265625, 0.205078125, 0.9228515625,
                      -0.123046875, 0.01708984375], dtype=np.float64)
RESTRICT = np.asarray([-0.0390625, 0.46875, 0.703125,
                       -0.15625, 0.0234375], dtype=np.float64)
RESTRICT_EDGE = np.asarray([0.2734375, 1.09375, -0.546875,
                            0.21875, -0.0390625], dtype=np.float64)
FAMILIES = {
    "chi": ["z4c_chi"],
    "K": ["z4c_Khat"],
    "Atilde": ["z4c_Axx", "z4c_Axy", "z4c_Axz", "z4c_Ayy",
               "z4c_Ayz", "z4c_Azz"],
    "Gammatilde": ["z4c_Gamx", "z4c_Gamy", "z4c_Gamz"],
    "gammatilde": ["z4c_gxx", "z4c_gxy", "z4c_gxz", "z4c_gyy",
                   "z4c_gyz", "z4c_gzz"],
}


class AuditError(RuntimeError):
    pass


def strict_load(path: Path) -> Any:
    def reject(value: str) -> None:
        raise AuditError(f"nonfinite JSON token {value} in {path}")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream, parse_constant=reject)


def strict_dump(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def restrict_one(field: np.ndarray, fi: int, fj: int, nx: int = 32,
                 ng: int = 4) -> float:
    offset_i = fi < nx // 2 + ng
    offset_j = fj < nx // 2 + ng
    ref_i = fi - 1 if offset_i else fi - 2
    ref_j = fj - 1 if offset_j else fj - 2
    outer = nx + 2 * ng - 2
    edge_i = fi in (0, ng, ng + nx - 2, outer)
    edge_j = fj in (0, ng, ng + nx - 2, outer)
    if fi == ng:
        ref_i += 1
    if fj == ng:
        ref_j += 1
    if fi == ng + nx - 2:
        ref_i -= 1
    if fj == ng + nx - 2:
        ref_j -= 1
    if fi == 0:
        ref_i = 0
    if fj == 0:
        ref_j = 0
    if fi == outer:
        ref_i = nx + ng - 1
    if fj == outer:
        ref_j = nx + ng - 1
    wi = RESTRICT_EDGE if edge_i else RESTRICT
    wj = RESTRICT_EDGE if edge_j else RESTRICT
    if not offset_i:
        wi = wi[::-1]
    if not offset_j:
        wj = wj[::-1]
    stencil = field[ref_j:ref_j + 5, ref_i:ref_i + 5]
    if stencil.shape != (5, 5):
        raise AuditError(f"restriction stencil escaped storage at ({fi},{fj})")
    return float(np.einsum("j,i,ji->", wj, wi, stencil))


def restrict_stored(field: np.ndarray, nx: int = 32, ng: int = 4) -> np.ndarray:
    coarse = np.full((nx // 2 + 2 * ng, nx // 2 + 2 * ng), np.nan)
    cis = ng
    for cj in range(cis - 2, cis + nx // 2 + 2):
        for ci in range(cis - 2, cis + nx // 2 + 2):
            coarse[cj, ci] = restrict_one(
                field, 2 * ci - cis, 2 * cj - cis, nx, ng)
    return coarse


def prolong_active(coarse: np.ndarray, positive: bool, nx: int = 32,
                   ng: int = 4) -> tuple[np.ndarray, int]:
    result = np.empty((nx, nx), dtype=np.float64)
    fallbacks = 0
    for cj in range(ng, ng + nx // 2):
        for ci in range(ng, ng + nx // 2):
            parent = coarse[cj - 2:cj + 3, ci - 2:ci + 3]
            if parent.shape != (5, 5) or not np.all(np.isfinite(parent)):
                raise AuditError("prolongation parent stencil is incomplete")
            group = np.empty((2, 2), dtype=np.float64)
            for dj in range(2):
                wj = PROLONG[::-1] if dj else PROLONG
                for di in range(2):
                    wi = PROLONG[::-1] if di else PROLONG
                    group[dj, di] = np.einsum("j,i,ji->", wj, wi, parent)
            if positive and (not np.all(parent > 0.0) or not np.all(group > 0.0)):
                center = coarse[cj, ci]
                dl_i, dr_i = center - coarse[cj, ci - 1], coarse[cj, ci + 1] - center
                dl_j, dr_j = center - coarse[cj - 1, ci], coarse[cj + 1, ci] - center
                slope_i = 0.0 if dl_i * dr_i <= 0.0 else 0.25 * math.copysign(
                    min(abs(dl_i), abs(dr_i)), dl_i)
                slope_j = 0.0 if dl_j * dr_j <= 0.0 else 0.25 * math.copysign(
                    min(abs(dl_j), abs(dr_j)), dl_j)
                group = np.asarray([[center - slope_i - slope_j,
                                     center + slope_i - slope_j],
                                    [center - slope_i + slope_j,
                                     center + slope_i + slope_j]])
                fallbacks += 1
            if positive and (not np.all(np.isfinite(group)) or not np.all(group > 0.0)):
                raise AuditError("self-shadow chi fallback is not strictly positive")
            fj, fi = 2 * (cj - ng), 2 * (ci - ng)
            result[fj:fj + 2, fi:fi + 2] = group
    return result, fallbacks


D1_O6 = np.asarray([-1/60, 3/20, -3/4, 0, 3/4, -3/20, 1/60])
D1_O4 = np.asarray([1/12, -2/3, 0, 2/3, -1/12])
D2_O6 = np.asarray([1/90, -3/20, 3/2, -49/18, 3/2, -3/20, 1/90])
D2_O4 = np.asarray([-1/12, 4/3, -5/2, 4/3, -1/12])


def centered(field: np.ndarray, weights: np.ndarray, axis: int,
             spacing: float, derivative: int, nx: int = 32,
             ng: int = 4) -> np.ndarray:
    radius = len(weights) // 2
    output = np.zeros((nx, nx), dtype=np.float64)
    for offset, weight in enumerate(weights, start=-radius):
        js = slice(ng + offset, ng + nx + offset) if axis == 0 else slice(ng, ng + nx)
        is_ = slice(ng + offset, ng + nx + offset) if axis == 1 else slice(ng, ng + nx)
        output += weight * field[js, is_]
    return output / spacing ** derivative


def parity_metrics(active: np.ndarray) -> dict[str, float]:
    centered_field = active - float(np.mean(active))
    norm = float(np.linalg.norm(centered_field))
    if norm == 0.0:
        return {"i": 0.0, "j": 0.0, "ij": 0.0}
    ny, nx = active.shape
    pi = np.broadcast_to((-1.0) ** np.arange(nx), (ny, nx))
    pj = np.broadcast_to(((-1.0) ** np.arange(ny))[:, None], (ny, nx))
    return {
        "i": abs(float(np.sum(centered_field * pi))) / (norm * math.sqrt(nx * ny)),
        "j": abs(float(np.sum(centered_field * pj))) / (norm * math.sqrt(nx * ny)),
        "ij": abs(float(np.sum(centered_field * pi * pj))) /
              (norm * math.sqrt(nx * ny)),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / max(denominator, np.finfo(np.float64).tiny)


def analyze(args: argparse.Namespace) -> None:
    root = args.diagnostic_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    ranks = sorted(path for path in root.glob("rank[0-9][0-9][0-9][0-9]")
                   if path.is_dir())
    if len(ranks) != args.ranks:
        raise AuditError(f"expected {args.ranks} rank roots, found {len(ranks)}")
    events = []
    for rank in ranks:
        matches = list(rank.glob(f"event_c{args.cycle:08d}_l2_to_l3"))
        if len(matches) != 1:
            raise AuditError(f"target event inventory mismatch in {rank}")
        events.append(matches[0])
    schemas = [strict_load(rank / "schema.json") for rank in ranks]
    names = schemas[0]["z4c_components"]
    if any(schema["z4c_components"] != names for schema in schemas[1:]):
        raise AuditError("component schemas disagree")
    name_index = {name: index for index, name in enumerate(names)}
    for family_names in FAMILIES.values():
        if any(name not in name_index for name in family_names):
            raise AuditError("required Z4c component is missing")

    parent_gids: set[int] = set()
    for event in events:
        proposal = read_csv(event / "t1_topology_proposal.csv")
        parent_gids.update(int(row["old_gid"]) for row in proposal
                           if int(row["new_level"]) == int(row["old_level"]) + 1)
    if not parent_gids:
        raise AuditError("target event has no refined parents")

    blocks: dict[int, tuple[np.ndarray, dict[str, str]]] = {}
    t0_hashes = []
    for event in events:
        phase = event / "t0_00_ACCEPTED_OLD_STATE"
        metadata = strict_load(phase / "phase.json")
        shape = metadata["u0_shape"]
        raw = np.fromfile(phase / "u0.bin", dtype="<f8")
        if raw.size != math.prod(shape):
            raise AuditError("T0 field byte count mismatch")
        view = raw.reshape(shape)
        if not np.all(np.isfinite(view)):
            raise AuditError("T0 contains nonfinite evolved data")
        topology = read_csv(phase / "topology.csv")
        if len(topology) != shape[0]:
            raise AuditError("T0 topology/view mismatch")
        for row in topology:
            gid = int(row["gid"])
            if gid in parent_gids:
                if gid in blocks:
                    raise AuditError(f"duplicate parent ownership for gid {gid}")
                blocks[gid] = (view[int(row["local_m"])], row)
        t0_hashes.append({"path": str((phase / "u0.bin").resolve()),
                          "sha256": sha256(phase / "u0.bin")})
    if set(blocks) != parent_gids:
        raise AuditError("not all target parents were found in T0")

    accum: dict[str, dict[str, list[float]]] = {}
    for family in FAMILIES:
        accum[family] = {key: [] for key in (
            "shadow_residual", "shadow_state", "shadow_max",
            "shadow_edge_residual", "shadow_edge_state",
            "shadow_interior_residual", "shadow_interior_state",
            "d1_o6_o4", "d1_o6", "d1_max", "d2_o6_o4", "d2_o6", "d2_max",
            "nyquist_i", "nyquist_j", "nyquist_ij")}
    max_locations: dict[str, dict[str, Any]] = {}
    edge_mask = np.ones((32, 32), dtype=bool)
    edge_mask[4:-4, 4:-4] = False
    chi_fallbacks = 0
    for gid in sorted(blocks):
        state, topology = blocks[gid]
        if state.shape[1:] != (1, 40, 40):
            raise AuditError(f"unexpected stored block shape {state.shape} for gid {gid}")
        spacings = (float(topology["dx2"]), float(topology["dx1"]))
        for family, family_names in FAMILIES.items():
            for component_name in family_names:
                component = name_index[component_name]
                field = state[component, 0]
                active = field[4:36, 4:36]
                coarse = restrict_stored(field)
                shadow, fallbacks = prolong_active(
                    coarse, positive=(component_name == "z4c_chi"))
                chi_fallbacks += fallbacks
                residual = shadow - active
                accum[family]["shadow_residual"].append(float(np.sum(residual ** 2)))
                accum[family]["shadow_state"].append(float(np.sum(active ** 2)))
                accum[family]["shadow_max"].append(float(np.max(np.abs(residual))))
                accum[family]["shadow_edge_residual"].append(
                    float(np.sum(residual[edge_mask] ** 2)))
                accum[family]["shadow_edge_state"].append(
                    float(np.sum(active[edge_mask] ** 2)))
                accum[family]["shadow_interior_residual"].append(
                    float(np.sum(residual[~edge_mask] ** 2)))
                accum[family]["shadow_interior_state"].append(
                    float(np.sum(active[~edge_mask] ** 2)))
                local_index = np.unravel_index(np.argmax(np.abs(residual)), residual.shape)
                local_max = float(abs(residual[local_index]))
                if family not in max_locations or local_max > max_locations[family]["value"]:
                    rho = (float(topology["x1min"]) +
                           (int(local_index[1]) + 0.5) * float(topology["dx1"]))
                    z = (float(topology["x2min"]) +
                         (int(local_index[0]) + 0.5) * float(topology["dx2"]))
                    max_locations[family] = {
                        "value": local_max, "gid": gid, "component": component_name,
                        "active_i": int(local_index[1]), "active_j": int(local_index[0]),
                        "rho": rho, "z": z,
                        "within_four_cells_of_block_edge": bool(edge_mask[local_index])}
                for axis, spacing in enumerate(spacings):
                    d1_6 = centered(field, D1_O6, axis, spacing, 1)
                    d1_4 = centered(field, D1_O4, axis, spacing, 1)
                    d2_6 = centered(field, D2_O6, axis, spacing, 2)
                    d2_4 = centered(field, D2_O4, axis, spacing, 2)
                    accum[family]["d1_o6_o4"].append(float(np.sum((d1_6-d1_4)**2)))
                    accum[family]["d1_o6"].append(float(np.sum(d1_6**2)))
                    accum[family]["d1_max"].append(float(np.max(np.abs(d1_6-d1_4))))
                    accum[family]["d2_o6_o4"].append(float(np.sum((d2_6-d2_4)**2)))
                    accum[family]["d2_o6"].append(float(np.sum(d2_6**2)))
                    accum[family]["d2_max"].append(float(np.max(np.abs(d2_6-d2_4))))
                parity = parity_metrics(active)
                accum[family]["nyquist_i"].append(parity["i"])
                accum[family]["nyquist_j"].append(parity["j"])
                accum[family]["nyquist_ij"].append(parity["ij"])

    rows = []
    for family in FAMILIES:
        values = accum[family]
        rows.append({
            "family": family,
            "component_count": len(FAMILIES[family]),
            "parent_block_count": len(parent_gids),
            "PR_relative_l2": safe_ratio(math.sqrt(sum(values["shadow_residual"])),
                                          math.sqrt(sum(values["shadow_state"]))),
            "PR_max_abs": max(values["shadow_max"]),
            "PR_edge_band_relative_l2": safe_ratio(
                math.sqrt(sum(values["shadow_edge_residual"])),
                math.sqrt(sum(values["shadow_edge_state"]))),
            "PR_interior_relative_l2": safe_ratio(
                math.sqrt(sum(values["shadow_interior_residual"])),
                math.sqrt(sum(values["shadow_interior_state"]))),
            "PR_max_location": max_locations[family],
            "D1_O6_O4_relative_l2": safe_ratio(math.sqrt(sum(values["d1_o6_o4"])),
                                                math.sqrt(sum(values["d1_o6"]))),
            "D1_O6_O4_max_abs": max(values["d1_max"]),
            "D2_O6_O4_relative_l2": safe_ratio(math.sqrt(sum(values["d2_o6_o4"])),
                                                math.sqrt(sum(values["d2_o6"]))),
            "D2_O6_O4_max_abs": max(values["d2_max"]),
            "block_nyquist_i_rms": math.sqrt(float(np.mean(np.square(values["nyquist_i"])))),
            "block_nyquist_j_rms": math.sqrt(float(np.mean(np.square(values["nyquist_j"])))),
            "block_nyquist_ij_rms": math.sqrt(float(np.mean(np.square(values["nyquist_ij"])))),
            "block_nyquist_max": max(values["nyquist_i"] + values["nyquist_j"] +
                                      values["nyquist_ij"]),
        })
    with (output / "parent_state_audit.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    labels = [row["family"] for row in rows]
    positions = np.arange(len(labels))
    axes[0].bar(positions-0.19, [row["PR_edge_band_relative_l2"] for row in rows],
                width=0.38, label="four-cell edge band")
    axes[0].bar(positions+0.19, [row["PR_interior_relative_l2"] for row in rows],
                width=0.38, label="interior")
    axes[0].set_xticks(positions, labels)
    axes[0].set_title(r"self-shadow $\|u-PRu\|_2/\|u\|_2$")
    axes[0].legend(fontsize=8)
    axes[1].bar(positions-0.19, [row["D1_O6_O4_relative_l2"] for row in rows],
                width=0.38, label="D1")
    axes[1].bar(positions+0.19, [row["D2_O6_O4_relative_l2"] for row in rows],
                width=0.38, label="D2")
    axes[1].set_xticks(positions, labels)
    axes[1].set_title("O6-O4 derivative disagreement")
    axes[1].legend()
    axes[2].bar(labels, [row["block_nyquist_max"] for row in rows])
    axes[2].set_title("max block-local odd/even projection")
    for axis in axes:
        axis.set_yscale("log")
        axis.tick_params(axis="x", rotation=35)
        axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "parent_state_audit.png", dpi=180)
    plt.close(fig)

    verdict = {
        "schema": "athenak_brill_parent_state_audit_v1",
        "qualification_claim": False,
        "cycle": args.cycle,
        "source_commit": args.source_commit,
        "restart_sha256": args.restart_sha256,
        "parent_gids": sorted(parent_gids),
        "parent_block_count": len(parent_gids),
        "chi_PR_fallback_groups": chi_fallbacks,
        "T0_field_files": t0_hashes,
        "families": rows,
        "definitions": {
            "PR": "production O6-configured 2D point restriction followed by production high-order prolongation; chi uses the production strict-positive sibling fallback",
            "PR_edge_split": "edge band means active cells within four cells of any MeshBlock edge; interior excludes that band",
            "derivative_disagreement": "relative L2 of centered O6 minus centered O4 first/second derivatives over target-parent active cells",
            "nyquist": "RMS and maximum block-local normalized projections onto i, j, and ij alternating patterns after removing each block-component mean"
        },
        "limitations": [
            "This is a local target-parent resolution indicator, not a convergence proof.",
            "Block-local alternating projections can identify unresolved grid-scale content but do not establish its causal origin."
        ]
    }
    strict_dump(output / "parent_state_audit.json", verdict)
    products = sorted(path for path in output.iterdir() if path.is_file())
    with (output / "SHA256SUMS").open("w", encoding="utf-8") as stream:
        for path in products:
            stream.write(f"{sha256(path)}  {path.name}\n")


def self_test() -> None:
    yy, xx = np.mgrid[:40, :40]
    for field in (np.ones((40, 40)), 10.0 + 0.1 * xx - 0.07 * yy):
        coarse = restrict_stored(field)
        shadow, fallback = prolong_active(coarse, positive=True)
        if fallback != 0 or not np.allclose(shadow, field[4:36, 4:36],
                                             rtol=0.0, atol=3.0e-13):
            raise AuditError("R/P polynomial self-test failed")
    for axis in (0, 1):
        spacing = 0.125
        d1_6 = centered(2.0 + xx * spacing + 3.0 * yy * spacing,
                        D1_O6, axis, spacing, 1)
        d1_4 = centered(2.0 + xx * spacing + 3.0 * yy * spacing,
                        D1_O4, axis, spacing, 1)
        expected = 3.0 if axis == 0 else 1.0
        if not np.allclose(d1_6, expected) or not np.allclose(d1_4, expected):
            raise AuditError("derivative self-test failed")
    print("parent-state audit self-test: PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--diagnostic-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--ranks", type=int, default=1)
    parser.add_argument("--cycle", type=int, default=1722)
    parser.add_argument("--source-commit")
    parser.add_argument("--restart-sha256")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if any(value is None for value in (
            args.diagnostic_root, args.output, args.source_commit,
            args.restart_sha256)):
        parser.error("diagnostic root, output, source commit, and restart hash are required")
    analyze(args)


if __name__ == "__main__":
    main()
