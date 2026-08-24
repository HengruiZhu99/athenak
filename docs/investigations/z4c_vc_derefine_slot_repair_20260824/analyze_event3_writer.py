#!/usr/bin/env python3
"""Summarize and fail closed on the bounded event-3 VC writer evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_PHASES = ["A4", "A5", "A6", "A8", "A14", "A15", "A16", "R0", "U0"]
PROJECTED = {
    "z4c_gxx", "z4c_gxy", "z4c_gyy", "z4c_gzz",
    "z4c_Axx", "z4c_Axy", "z4c_Ayy", "z4c_Azz",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_map(row: dict, key: str = "checkpoint_parent") -> dict[str, str]:
    return {entry["name"]: entry["hash"] for entry in row["hashes"][key]}


def variable_map(row: dict, key: str) -> dict[str, object]:
    return {entry["name"]: entry[key] for entry in row["variables"]}


def differences(left: dict[str, object], right: dict[str, object]) -> list[str]:
    assert left.keys() == right.keys()
    return sorted(name for name in left if left[name] != right[name])


def summarize(run_dir: Path) -> dict:
    writer_path = run_dir / "vc_derefine_writer_event3.jsonl"
    lifecycle_path = run_dir / "vc_amr_lifecycle_event3.jsonl"
    writer = [json.loads(line) for line in writer_path.read_text().splitlines()]
    lifecycle = [json.loads(line) for line in lifecycle_path.read_text().splitlines()]
    assert len(writer) == 18
    assert len(lifecycle) == 32

    family_keys = sorted({
        tuple(row["parent_location"][key] for key in ("level", "lx1", "lx2", "lx3"))
        for row in writer
    })
    assert len(family_keys) == 2
    families = []
    for family_key in family_keys:
        rows = [row for row in writer if tuple(
            row["parent_location"][key] for key in ("level", "lx1", "lx2", "lx3")
        ) == family_key]
        assert [row["phase"] for row in rows] == EXPECTED_PHASES
        by_phase = {row["phase"]: row for row in rows}
        oracle = hash_map(by_phase["A4"], "independent_restriction_oracle")
        assert len(oracle) == 25
        comparisons = {}
        for phase in ("A5", "A6", "A8", "A14"):
            comparisons[f"{phase}_vs_oracle"] = differences(
                hash_map(by_phase[phase]), oracle
            )
            assert not comparisons[f"{phase}_vs_oracle"]
        comparisons["A15_vs_A14"] = differences(
            hash_map(by_phase["A15"]), hash_map(by_phase["A14"])
        )
        assert set(comparisons["A15_vs_A14"]) == PROJECTED
        comparisons["A16_vs_A15"] = differences(
            hash_map(by_phase["A16"]), hash_map(by_phase["A15"])
        )
        assert not comparisons["A16_vs_A15"]
        assert hash_map(by_phase["A5"], "post_a5_staging") == oracle
        assert hash_map(by_phase["A6"], "post_a6_final_parent") == oracle
        assert all(not row["first_oracle_mismatch"]["found"] for row in rows)
        assert all(not row["maximum_oracle_mismatch"]["found"] for row in rows)
        relocation_exact = all(
            survivor["exact"]
            for survivor in by_phase["A6"]["post_a6_relocation_survivors"]
        )
        assert relocation_exact
        families.append({
            "parent_location": dict(zip(("level", "lx1", "lx2", "lx3"), family_key)),
            "old_lower_child_gid": rows[0]["old_lower_child_gid"],
            "old_lower_child_local_slot": rows[0]["old_lower_child_local_slot"],
            "new_parent_gid": rows[0]["new_parent_gid"],
            "new_parent_local_slot": rows[0]["new_parent_local_slot"],
            "signed_slot_shift": rows[0]["signed_slot_shift"],
            "all_siblings_local": rows[0]["all_siblings_local"],
            "comparisons": comparisons,
            "relocation_survivors_exact": relocation_exact,
        })

    lifecycle_by_phase = {row["phase"]: row for row in lifecycle}
    assert "R0" in lifecycle_by_phase and "U0" in lifecycle_by_phase
    nonfinite = {
        phase: sum(variable_map(lifecycle_by_phase[phase], "nonfinite").values())
        for phase in ("A4", "A5", "A6", "A8", "A14", "A15", "A16", "R0", "U0")
    }
    assert all(value == 0 for value in nonfinite.values())
    a15_to_a16_full = differences(
        variable_map(lifecycle_by_phase["A16"], "hash"),
        variable_map(lifecycle_by_phase["A15"], "hash"),
    )
    a15_to_a16_interior = differences(
        variable_map(lifecycle_by_phase["A16"], "block_strict_interior_hash"),
        variable_map(lifecycle_by_phase["A15"], "block_strict_interior_hash"),
    )
    assert set(a15_to_a16_full) == PROJECTED
    assert not a15_to_a16_interior
    assert lifecycle_by_phase["R0"]["stage"] == 1
    assert lifecycle_by_phase["U0"]["stage"] == 1

    return {
        "run_directory": str(run_dir),
        "writer_sha256": sha256(writer_path),
        "lifecycle_sha256": sha256(lifecycle_path),
        "event_cycle": writer[0]["cycle"],
        "event_time": writer[0]["time"],
        "writer_records": len(writer),
        "lifecycle_records": len(lifecycle),
        "families": families,
        "nonfinite_counts": nonfinite,
        "global_A16_vs_A15_full_active_changes": a15_to_a16_full,
        "global_A16_vs_A15_block_strict_interior_changes": a15_to_a16_interior,
        "first_post_event_rhs_stage": lifecycle_by_phase["R0"]["stage"],
        "first_post_event_update_stage": lifecycle_by_phase["U0"]["stage"],
        "status": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", type=Path, nargs="+")
    args = parser.parse_args()
    result = {
        "schema": "athenak_vc_event3_writer_summary_v1",
        "runs": [summarize(path) for path in args.run_dirs],
        "status": "PASS",
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
