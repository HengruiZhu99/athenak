#!/usr/bin/env python3
"""Audit Z4c coarse-cache ownership and replay one captured O6 boundary transfer.

This tool is deliberately limited to the captured two-dimensional Cartoon topology.
It derives all writer and consumer ranges from the production index formulae in
``buffs_cc.cpp`` and replays the collapsed-x3 O6 prolongation literally.  It never
reimplements the ADM constraints and therefore cannot promote an independent
constraint result.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from fractions import Fraction
import json
import math
import os
from pathlib import Path
from typing import Any, Iterable

import numpy as np

import cartoon_amr_jump_analyze as base


SCHEMA = "athenak_z4c_coarse_cache_ownership_audit_v1"
VARS = [
    "chi", "gxx", "gxy", "gxz", "gyy", "gyz", "gzz", "Khat",
    "Axx", "Axy", "Axz", "Ayy", "Ayz", "Azz", "Gamx", "Gamy",
    "Gamz", "Theta", "alpha", "betax", "betay", "betaz", "Bx", "By", "Bz",
]
P5 = np.asarray([-45, 420, 1890, -252, 35], dtype=np.float64) / 2048.0
ROUND_GATE = 128.0 * np.finfo(np.float64).eps


class AuditError(RuntimeError):
    pass


@dataclass(frozen=True)
class Block:
    m: int
    gid: int
    rank: int
    level: int
    lx1: int
    lx2: int
    x1lo: Fraction
    x1hi: Fraction
    x2lo: Fraction
    x2hi: Fraction
    dx: Fraction
    bcs: tuple[int, int, int, int]


@dataclass(frozen=True)
class Relation:
    receiver: int
    neighbor: int
    ox1: int
    ox2: int
    f1: int


def strict_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


def write_csv(path: Path, records: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def load_blocks(path: Path) -> list[Block]:
    result: list[Block] = []
    for row in read_rows(path):
        dx1 = Fraction(row["dx1"])
        dx2 = Fraction(row["dx2"])
        if dx1 != dx2:
            raise AuditError("only square two-dimensional blocks are supported")
        result.append(Block(
            m=int(row["local_m"]), gid=int(row["gid"]), rank=int(row["owner_rank"]),
            level=int(row["level"]), lx1=int(row["lx1"]), lx2=int(row["lx2"]),
            x1lo=Fraction(row["x1min"]), x1hi=Fraction(row["x1max"]),
            x2lo=Fraction(row["x2min"]), x2hi=Fraction(row["x2max"]), dx=dx1,
            bcs=(int(row["inner_x1_bc"]), int(row["outer_x1_bc"]),
                 int(row["inner_x2_bc"]), int(row["outer_x2_bc"])),
        ))
    if [block.m for block in result] != list(range(len(result))):
        raise AuditError("topology local indices are not dense and ordered")
    return result


def positive_overlap(a0: Fraction, a1: Fraction,
                     b0: Fraction, b1: Fraction) -> bool:
    return min(a1, b1) > max(a0, b0)


def relations(blocks: list[Block]) -> list[Relation]:
    output: list[Relation] = []
    for receiver in blocks:
        for neighbor in blocks:
            if receiver.gid == neighbor.gid:
                continue
            if neighbor.x1hi == receiver.x1lo:
                ox1 = -1
            elif neighbor.x1lo == receiver.x1hi:
                ox1 = 1
            elif positive_overlap(receiver.x1lo, receiver.x1hi,
                                  neighbor.x1lo, neighbor.x1hi):
                ox1 = 0
            else:
                continue
            if neighbor.x2hi == receiver.x2lo:
                ox2 = -1
            elif neighbor.x2lo == receiver.x2hi:
                ox2 = 1
            elif positive_overlap(receiver.x2lo, receiver.x2hi,
                                  neighbor.x2lo, neighbor.x2hi):
                ox2 = 0
            else:
                continue
            if ox1 == 0 and ox2 == 0:
                continue
            if abs(receiver.level - neighbor.level) > 1:
                raise AuditError("topology violates 2:1 balance")
            # In two dimensions f1 selects the receiver child in the direction
            # tangential to a coarse face.  Edge buffers have no subface selector.
            if neighbor.level < receiver.level and (ox1 == 0) != (ox2 == 0):
                f1 = (receiver.lx2 & 1) if ox1 != 0 else (receiver.lx1 & 1)
            else:
                f1 = 0
            output.append(Relation(receiver.m, neighbor.m, ox1, ox2, f1))
    return output


def axis_range(offset: int, start: int, end: int, width: int,
               tangential_extension: int = 0, child: int = 0) -> range:
    if offset < 0:
        return range(start - width, start)
    if offset > 0:
        return range(end + 1, end + width + 1)
    lower, upper = start, end
    if tangential_extension:
        if child == 0:
            upper += tangential_extension
        else:
            lower -= tangential_extension
    return range(lower, upper + 1)


def recv_range(kind: str, relation: Relation, ng: int = 4,
               cis: int = 4, cie: int = 19) -> tuple[range, range]:
    if kind == "same":
        return (axis_range(relation.ox1, cis, cie, ng),
                axis_range(relation.ox2, cis, cie, ng))
    if kind not in ("coarser", "prolong"):
        raise AuditError(f"unknown receive range kind {kind}")
    width = ng if kind == "coarser" else ng // 2
    i = axis_range(relation.ox1, cis, cie, width,
                   width if relation.ox1 == 0 else 0, relation.f1)
    j = axis_range(relation.ox2, cis, cie, width,
                   width if relation.ox2 == 0 else 0, relation.f1)
    return i, j


def refresh_range(relation: Relation, ng: int = 4, nx: int = 32,
                  cis: int = 4, fine_start: int = 4) -> tuple[range, range]:
    fine_end = fine_start + nx - 1
    extent = nx + 2 * ng

    def one(offset: int) -> range:
        fine = axis_range(offset, fine_start, fine_end, ng)
        lower = (fine.start + cis) // 2
        upper = (fine.stop - 1 + cis) // 2
        while lower <= upper and (lower - cis) * 2 + fine_start < 0:
            lower += 1
        while lower <= upper and (upper - cis) * 2 + fine_start + 1 >= extent:
            upper -= 1
        return range(lower, upper + 1)

    return one(relation.ox1), one(relation.ox2)


def cells(irange: range, jrange: range) -> Iterable[tuple[int, int]]:
    for j in jrange:
        for i in irange:
            yield j, i


def phase_view(event: Path, phase: str, name: str) -> np.ndarray:
    metadata = base.strict_load(event / phase / "phase.json")
    return base.load_view(event / phase, name, metadata)


def physical_coordinate(block: Block, j: int, i: int) -> tuple[float, float]:
    rho, z = exact_coarse_coordinate(block, j, i)
    return float(rho), float(z)


def exact_coarse_coordinate(block: Block, j: int, i: int
                            ) -> tuple[Fraction, Fraction]:
    spacing = 2 * block.dx
    rho = block.x1lo + (Fraction(i - 4) + Fraction(1, 2)) * spacing
    z = block.x2lo + (Fraction(j - 4) + Fraction(1, 2)) * spacing
    return rho, z


def global_coarse_owner(blocks: list[Block]) -> dict[tuple[int, Fraction, Fraction], int]:
    owner: dict[tuple[int, Fraction, Fraction], int] = {}
    for block in blocks:
        spacing = 2 * block.dx
        for j in range(4, 20):
            z = block.x2lo + (Fraction(j - 4) + Fraction(1, 2)) * spacing
            for i in range(4, 20):
                rho = block.x1lo + (Fraction(i - 4) + Fraction(1, 2)) * spacing
                key = (block.level, rho, z)
                if key in owner:
                    raise AuditError(f"duplicate active coarse owner at {key}")
                owner[key] = block.gid
    return owner


def prolong_value(coarse: np.ndarray, m: int, v: int, j: int, i: int,
                  high_j: bool, high_i: bool) -> float:
    wi = P5[::-1] if high_i else P5
    wj = P5[::-1] if high_j else P5
    stencil = coarse[m, v, 0, j - 2:j + 3, i - 2:i + 3]
    if stencil.shape != (5, 5):
        raise AuditError("O6 prolongation stencil left coarse allocation")
    return float(np.einsum("j,i,ji->", wj, wi, stencil))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        dummy = Relation(0, 1, -1, 0, 0)
        assert list(recv_range("same", dummy)[0]) == [0, 1, 2, 3]
        assert list(refresh_range(dummy)[0]) == [2, 3]
        assert list(recv_range("prolong", dummy)[0]) == [2, 3]
        assert abs(float(P5.sum()) - 1.0) < 1.0e-15
        print("coarse-cache ownership audit self-test passed")
        return
    if args.event is None or args.output is None:
        parser.error("--event and --output are required unless --self-test is used")

    event = args.event.resolve()
    output = args.output.resolve()
    blocks = load_blocks(event / "t3_01_MPI_RECEIVE" / "topology.csv")
    by_m = {block.m: block for block in blocks}
    links = relations(blocks)
    owner_map = global_coarse_owner(blocks)

    writers: dict[tuple[int, int, int], list[str]] = {}
    refresh_cells: set[tuple[int, int, int]] = set()
    same_cells: set[tuple[int, int, int]] = set()
    same_sender: dict[tuple[int, int, int], int] = {}

    def add_writer(key: tuple[int, int, int], label: str) -> None:
        writers.setdefault(key, []).append(label)

    for block in blocks:
        for j, i in cells(range(4, 20), range(4, 20)):
            add_writer((block.m, j, i), "local_active_restriction")
        for face, flag in enumerate(block.bcs):
            if flag == 0:
                continue
            if face == 0:
                region = cells(range(0, 4), range(0, 24))
            elif face == 1:
                region = cells(range(20, 24), range(0, 24))
            elif face == 2:
                region = cells(range(0, 24), range(0, 4))
            else:
                region = cells(range(0, 24), range(20, 24))
            for j, i in region:
                add_writer((block.m, j, i), "physical_boundary")

    for link in links:
        receiver = by_m[link.receiver]
        neighbor = by_m[link.neighbor]
        if neighbor.level == receiver.level:
            irange, jrange = recv_range("same", link)
            for j, i in cells(irange, jrange):
                key = (receiver.m, j, i)
                add_writer(key, f"same_level_receive:gid={neighbor.gid}")
                same_cells.add(key)
                rho, z = exact_coarse_coordinate(receiver, j, i)
                owner_gid = owner_map.get((receiver.level, rho, z))
                if owner_gid != neighbor.gid:
                    raise AuditError(
                        "same-level coarse receive does not map to the geometric owner: "
                        f"receiver={receiver.gid} neighbor={neighbor.gid} "
                        f"coordinate=({rho},{z}) owner={owner_gid}")
                same_sender[key] = neighbor.gid
            irange, jrange = refresh_range(link)
            for j, i in cells(irange, jrange):
                key = (receiver.m, j, i)
                add_writer(key, f"same_level_local_refresh:gid={neighbor.gid}")
                refresh_cells.add(key)
        elif neighbor.level < receiver.level:
            irange, jrange = recv_range("coarser", link)
            for j, i in cells(irange, jrange):
                add_writer((receiver.m, j, i), f"coarser_receive:gid={neighbor.gid}")

    consumers: dict[tuple[int, int, int], list[dict[str, int]]] = {}
    target_rows: list[dict[str, Any]] = []
    coarser_relation_count = 0
    for link in links:
        receiver = by_m[link.receiver]
        neighbor = by_m[link.neighbor]
        if neighbor.level >= receiver.level:
            continue
        coarser_relation_count += 1
        irange, jrange = recv_range("prolong", link)
        for j, i in cells(irange, jrange):
            target = {"receiver_gid": receiver.gid, "neighbor_gid": neighbor.gid,
                      "coarse_j": j, "coarse_i": i, "ox1": link.ox1,
                      "ox2": link.ox2, "f1": link.f1}
            target_rows.append(target)
            for sj in range(j - 2, j + 3):
                for si in range(i - 2, i + 3):
                    consumers.setdefault((receiver.m, sj, si), []).append(target)

    need = set(consumers)
    ownership_rows: list[dict[str, Any]] = []
    source_counts: dict[str, int] = {}
    duplicate = missing = overwritten = overwritten_needed = 0
    for key in sorted(need):
        m, j, i = key
        labels = writers.get(key, [])
        authoritative = [label for label in labels
                         if not label.startswith("same_level_local_refresh")]
        refresh = [label for label in labels
                   if label.startswith("same_level_local_refresh")]
        if not authoritative:
            missing += 1
        if len(authoritative) > 1:
            duplicate += 1
        if refresh:
            overwritten_needed += 1
        for label in set(label.split(":", 1)[0] for label in authoritative):
            source_counts[label] = source_counts.get(label, 0) + 1
        block = by_m[m]
        rho, z = physical_coordinate(block, j, i)
        ownership_rows.append({
            "receiver_gid": block.gid, "coarse_j": j, "coarse_i": i,
            "rho": f"{rho:.17g}", "z": f"{z:.17g}",
            "authoritative_writers": "|".join(authoritative),
            "refresh_writers": "|".join(refresh),
            "authoritative_writer_count": len(authoritative),
            "consumer_count": len(consumers[key]),
        })
    overwritten = len(refresh_cells & same_cells)

    coarse_received = phase_view(event, "t3_01_MPI_RECEIVE", "coarse_u0")
    coarse_current = phase_view(event, "t3_03_SAME_LEVEL_COARSE_REFRESH", "coarse_u0")
    fine_after = phase_view(event, "t3_04_COARSE_TO_FINE_PROLONGATION", "u0")
    coarse_preserved = coarse_current.copy()
    for key in refresh_cells & same_cells:
        m, j, i = key
        coarse_preserved[m, :, 0, j, i] = coarse_received[m, :, 0, j, i]

    changed_consumed_rows: list[dict[str, Any]] = []
    for key in sorted((refresh_cells & same_cells) & need):
        m, j, i = key
        receiver = by_m[m]
        delta = coarse_current[m, :, 0, j, i] - coarse_received[m, :, 0, j, i]
        rho, z = physical_coordinate(receiver, j, i)
        for v, difference in enumerate(delta):
            scale = max(1.0, abs(float(coarse_received[m, v, 0, j, i])))
            if abs(float(difference)) <= ROUND_GATE * scale:
                continue
            for target in consumers[key]:
                changed_consumed_rows.append({
                    "receiver_gid": receiver.gid,
                    "sender_gid": same_sender.get(key, -1),
                    "coarse_j": j, "coarse_i": i,
                    "rho": f"{rho:.17g}", "z": f"{z:.17g}",
                    "variable": VARS[v],
                    "received_value": f"{coarse_received[m, v, 0, j, i]:.17g}",
                    "refreshed_value": f"{coarse_current[m, v, 0, j, i]:.17g}",
                    "difference": f"{difference:.17g}",
                    "consumer_neighbor_gid": target["neighbor_gid"],
                    "consumer_coarse_j": target["coarse_j"],
                    "consumer_coarse_i": target["coarse_i"],
                    "consumer_ox1": target["ox1"], "consumer_ox2": target["ox2"],
                })

    fine_rows: list[dict[str, Any]] = []
    production_max = 0.0
    production_count = 0
    difference_count = 0
    seen_fine: set[tuple[int, int, int, int]] = set()
    seam_rho, seam_z = 5.109375, -0.046875
    min_seam_distance = math.inf
    for target in target_rows:
        receiver = next(block for block in blocks if block.gid == target["receiver_gid"])
        m = receiver.m
        j, i = target["coarse_j"], target["coarse_i"]
        fi = (i - 4) * 2 + 4
        fj = (j - 4) * 2 + 4
        for dj in (0, 1):
            for di in (0, 1):
                for v in range(25):
                    key = (m, v, fj + dj, fi + di)
                    current = prolong_value(coarse_current, m, v, j, i,
                                            bool(dj), bool(di))
                    preserved = prolong_value(coarse_preserved, m, v, j, i,
                                              bool(dj), bool(di))
                    captured = float(fine_after[m, v, 0, fj + dj, fi + di])
                    production_max = max(production_max, abs(current - captured))
                    production_count += 1
                    scale = max(1.0, abs(current), abs(preserved))
                    if abs(current - preserved) <= ROUND_GATE * scale or key in seen_fine:
                        continue
                    seen_fine.add(key)
                    difference_count += 1
                    rho = float(receiver.x1lo +
                                (Fraction(fi + di - 4) + Fraction(1, 2)) * receiver.dx)
                    z = float(receiver.x2lo +
                              (Fraction(fj + dj - 4) + Fraction(1, 2)) * receiver.dx)
                    distance = math.hypot(rho - seam_rho, z - seam_z)
                    min_seam_distance = min(min_seam_distance, distance)
                    fine_rows.append({
                        "receiver_gid": receiver.gid, "fine_j": fj + dj,
                        "fine_i": fi + di, "rho": f"{rho:.17g}", "z": f"{z:.17g}",
                        "variable": VARS[v], "current": f"{current:.17g}",
                        "preserved": f"{preserved:.17g}",
                        "difference": f"{current - preserved:.17g}",
                        "captured": f"{captured:.17g}",
                        "current_minus_captured": f"{current - captured:.17g}",
                        "distance_to_known_seam": f"{distance:.17g}",
                    })

    if production_max > 8.0e-13:
        raise AuditError(
            f"production O6 replay mismatch {production_max} exceeds roundoff gate")

    changed_coarse_value_count = int(sum(
        np.count_nonzero(np.abs(coarse_current[m, :, 0, j, i] -
                                coarse_received[m, :, 0, j, i]) >
                         ROUND_GATE * np.maximum(
                             1.0, np.abs(coarse_received[m, :, 0, j, i])))
        for m, j, i in (refresh_cells & same_cells) & need))

    summary = {
        "schema": SCHEMA,
        "event": str(event),
        "topology": {"blocks": len(blocks), "relations": len(links),
                     "coarser_neighbor_relations": coarser_relation_count,
                     "prolongation_parent_targets": len(target_rows)},
        "o6_consumers": {
            "prolongation_parent_targets": len(target_rows),
            "required_unique_coarse_cells": len(need),
            "required_cells_by_authoritative_source": source_counts,
            "required_cells_with_no_pre_refresh_source": missing,
            "required_cells_with_multiple_pre_refresh_writers": duplicate,
            "required_cells_overwritten_by_refresh": overwritten_needed,
        },
        "same_level_refresh": {
            "received_overlap_cells": len(same_cells),
            "refresh_cells": len(refresh_cells),
            "refresh_received_intersection": overwritten,
            "changed_consumed_coarse_variable_values": changed_coarse_value_count,
            "fine_variable_values_changed_by_preserve_received": difference_count,
            "minimum_distance_to_known_seam":
                None if math.isinf(min_seam_distance) else min_seam_distance,
        },
        "production_replay": {
            "compared_fine_variable_values": production_count,
            "max_abs_current_replay_minus_captured": production_max,
            "qualification": "roundoff_reproduction_pass",
        },
        "constraint_scope": {
            "production_constraint_port_used": False,
            "claim": "no independent constraint-causality claim",
        },
    }
    output.mkdir(parents=True, exist_ok=True)
    strict_dump(output / "coarse_cache_ownership_summary.json", summary)
    write_csv(output / "coarse_cache_ownership_cells.csv", ownership_rows,
              list(ownership_rows[0]))
    write_csv(output / "changed_consumed_coarse_values.csv", changed_consumed_rows,
              list(changed_consumed_rows[0]) if changed_consumed_rows else ["receiver_gid"])
    write_csv(output / "preserve_received_fine_differences.csv", fine_rows,
              list(fine_rows[0]) if fine_rows else ["receiver_gid"])
    print("COARSE_CACHE_OWNERSHIP_AUDIT_PASS")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
