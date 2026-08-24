#!/usr/bin/env python3
"""Reconstruct AthenaK's event-3 Z-order and relocation maps from AMR history."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


Location = tuple[int, int, int, int]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def z_order(sorted_leaves: list[list[int]], dimension: int) -> list[Location]:
    """Undo AMRHistory::CurrentLeaves sorting and reproduce tree child traversal."""
    leaves = {tuple(value) for value in sorted_leaves}
    children = 1 << dimension

    def has_descendant(node: Location) -> bool:
        level, lx1, lx2, lx3 = node
        return any(
            candidate_level > level
            and candidate_lx1 >> (candidate_level - level) == lx1
            and candidate_lx2 >> (candidate_level - level) == lx2
            and candidate_lx3 >> (candidate_level - level) == lx3
            for candidate_level, candidate_lx1, candidate_lx2, candidate_lx3
            in leaves
        )

    result: list[Location] = []

    def visit(node: Location) -> None:
        if node in leaves:
            result.append(node)
            return
        level, lx1, lx2, lx3 = node
        for child in range(children):
            candidate = (
                level + 1,
                2 * lx1 + (child & 1),
                2 * lx2 + ((child >> 1) & 1) if dimension >= 2 else 0,
                2 * lx3 + ((child >> 2) & 1) if dimension >= 3 else 0,
            )
            if candidate in leaves or has_descendant(candidate):
                visit(candidate)

    visit((0, 0, 0, 0))
    if set(result) != leaves or len(result) != len(leaves):
        raise RuntimeError("tree traversal did not reproduce the accepted leaf set")
    return result


def lower_child(parent: Location) -> Location:
    level, lx1, lx2, lx3 = parent
    return level + 1, 2 * lx1, 2 * lx2, 2 * lx3


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    args = parser.parse_args()

    records = [json.loads(line) for line in args.history.read_text().splitlines()]
    header = next(record for record in records if record.get("type") == "header")
    events = {record["event"]: record for record in records
              if record.get("type") == "event"}
    before = z_order(events[2]["leaves"], header["dimension"])
    after = z_order(events[3]["leaves"], header["dimension"])
    before_gid = {location: gid for gid, location in enumerate(before)}

    newtoold: list[int] = []
    for location in after:
        if location in before_gid:
            newtoold.append(before_gid[location])
        else:
            newtoold.append(before_gid[lower_child(location)])

    nleaf = 1 << header["dimension"]
    oldtonew: list[int | None] = [None] * len(before)
    oldtonew[0] = 0
    old_index = 1
    for new_gid in range(1, len(after)):
        if newtoold[new_gid] == newtoold[new_gid - 1] + 1:
            oldtonew[old_index] = new_gid
            old_index += 1
        elif newtoold[new_gid] == newtoold[new_gid - 1] + nleaf:
            for _ in range(nleaf - 1):
                oldtonew[old_index] = new_gid - 1
                old_index += 1
            oldtonew[old_index] = new_gid
            old_index += 1
    while old_index < len(before):
        oldtonew[old_index] = len(after) - 1
        old_index += 1
    if any(value is None for value in oldtonew):
        raise RuntimeError("oldtonew reconstruction left an unmapped old GID")

    removed = [location for location in before if location not in set(after)]
    added = [location for location in after if location not in set(before)]
    families = []
    for parent in added:
        source = before_gid[lower_child(parent)]
        families.append({
            "parent": list(parent),
            "old_children": list(range(source, source + nleaf)),
            "new_parent_gid": after.index(parent),
            "source_base": source,
            "destination_m_one_rank": after.index(parent),
            "slot_shift_one_rank": after.index(parent) - source,
        })

    source_files = [
        "src/mesh/mesh_refinement.cpp",
        "src/mesh/mesh_refinement_vc.cpp",
        "src/mesh/load_balance.cpp",
        "src/mesh/meshblock_tree.cpp",
        "src/mesh/vertex_amr.hpp",
        "src/bvals/bvals_vc.cpp",
        "src/z4c/z4c_tasks.cpp",
        "src/z4c/z4c_vertex_topology.cpp",
        "tst/unit/z4c/z4c_vertex_dynamic_amr_test.py",
        "tst/unit/z4c/z4c_vertex_dynamic_linear_wave_test.py",
    ]
    output = {
        "schema": 1,
        "authority_history": str(args.history),
        "authority_history_sha256": sha256(args.history),
        "transition": {
            "before_event": 2,
            "after_event": 3,
            "time": events[3]["time"],
            "time_hex": events[3]["time_hex"],
            "dimension": header["dimension"],
            "nleaf": nleaf,
            "old_leaf_count": len(before),
            "new_leaf_count": len(after),
        },
        "families": families,
        "removed_locations": [list(value) for value in removed],
        "added_locations": [list(value) for value in added],
        "old_z_order": [{"old_gid": gid, "location": list(location)}
                        for gid, location in enumerate(before)],
        "new_z_order": [{"new_gid": gid, "location": list(location),
                         "newtoold": newtoold[gid]}
                        for gid, location in enumerate(after)],
        "oldtonew": [{"old_gid": gid, "location": list(before[gid]),
                      "new_gid": oldtonew[gid]}
                     for gid in range(len(before))],
        "source_sha256": {
            name: sha256(args.source_root / name) for name in source_files
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
