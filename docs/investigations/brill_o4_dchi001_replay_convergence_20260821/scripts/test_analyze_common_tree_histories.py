#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import pathlib
import subprocess
import sys
import tempfile


HERE = pathlib.Path(__file__).resolve().parent
ANALYZER = HERE / "analyze_common_tree_histories.py"
FIELDS = [
    "time", "dt", "C-norm2", "H-norm2", "M-norm2", "Z-norm2",
    "max_abs_K", "nmb_total", "maxAbsKret", "maxRefLev", "cycle",
    "axisLapse", "axisTau", "axisKret",
]


def write_history(path: pathlib.Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    header = "# " + " ".join(f"[{index}]={name}" for index, name in enumerate(FIELDS, 1))
    path.write_text(
        header + "\n" + "\n".join(" ".join(format(value, ".17g") for value in row) for row in rows) + "\n",
        encoding="utf-8",
    )


def history_row(time: float, cycle: int, error: float) -> list[float]:
    base = 1.0 + time
    return [
        time, 0.01, base + error, 0.5 * base + error, 0.25 * base + error,
        0.125 * base + error, 2 + time, 32 + cycle, 3 + time, 1 + cycle // 4,
        cycle, 0.9, time, 0.1 + time,
    ]


def write_jsonl(path: pathlib.Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def replay_row(event: int, time_hex: str, checksum: str) -> dict:
    return {
        "action": "replay", "event": event, "time_hex": time_hex,
        "leaves": 32 + 6 * event, "max_level": 3 + event,
        "tree_checksum": checksum, "ranks": 1, "exact_match": True,
        "authority_time_hex": time_hex, "actual_mesh_time_hex": time_hex,
        "signed_time_difference": 0, "ulp_difference": 0,
        "candidate_dt_hex": float(0.02).hex(), "applied_dt_hex": float(0.01).hex(),
        "preceding_timestep_clipped": True,
    }


def shadow_row(cycle: int, gid: int, event: int, classification: str) -> dict:
    return {
        "schema": "athenak_amr_native_shadow_v1", "cycle": cycle,
        "time": cycle / 4, "time_hex": float(cycle / 4).hex(), "tau_c": cycle / 4,
        "root_nx1": 64, "cells_per_meshblock": [16, 16, 1], "gid": gid,
        "logical_location": [3, gid, 0, 0], "relative_level": 0, "dx": 0.25,
        "raw_dchi": 0.01 + 0.001 * gid, "dchi_over_dx": 0.04 + 0.004 * gid,
        "native_action": "refine", "authority_event": True,
        "authority_event_index": event, "authority_event_time_hex": float(cycle / 4).hex(),
        "authority_action": "same", "classification": classification,
        "record_scope": "requested_or_authority", "strongest_cell_ordinal": 0,
        "strongest_cell_offset": [0, 0, 0], "strongest_physical_location": [0.1, 0.2],
        "block_center": [1, 2],
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = pathlib.Path(tmp)
        errors = {"n128": 16e-4, "n256": 1e-4, "n512": 1e-4 / 16}
        for case, error in errors.items():
            case_root = root / case
            rows = [history_row(0.0, 0, error), history_row(1.0, 4, error)]
            write_history(case_root / "segment0" / f"{case}.z4c.user.hst", rows)
            rows = [history_row(1.0, 4, error), history_row(2.0, 8, error)]
            write_history(case_root / "segment1" / f"{case}.z4c.user.hst", rows)

        authority = [
            {"type": "event", "event": 0, "time": "0", "time_hex": float(0).hex(),
             "leaf_count": 32, "max_level": 3, "created": 0, "deleted": 0,
             "tree_checksum": "initial"},
            {"type": "event", "event": 1, "time": "0.5", "time_hex": float(0.5).hex(),
             "leaf_count": 38, "max_level": 4, "created": 6, "deleted": 0,
             "tree_checksum": "event1"},
            {"type": "event", "event": 2, "time": "1.5", "time_hex": float(1.5).hex(),
             "leaf_count": 44, "max_level": 5, "created": 6, "deleted": 0,
             "tree_checksum": "event2"},
        ]
        authority_path = root / "authority.jsonl"
        write_jsonl(authority_path, authority)
        for case in ("n128", "n512"):
            one = replay_row(1, float(0.5).hex(), "event1")
            two = replay_row(2, float(1.5).hex(), "event2")
            write_jsonl(root / case / "segment0" / f"{case}.amr_history_replay.jsonl", [one])
            write_jsonl(root / case / "segment1" / f"{case}.amr_history_replay.jsonl", [one, two])
            shadow1 = shadow_row(2, 1, 1, "WOULD_REFINE_EARLIER")
            shadow2 = shadow_row(6, 2, 2, "AGREES")
            write_jsonl(root / case / "segment0" / f"{case}.amr_native_shadow.rank0000.jsonl", [shadow1])
            write_jsonl(root / case / "segment1" / f"{case}.amr_native_shadow.rank0000.jsonl", [shadow1, shadow2])

        output = root / "output"
        subprocess.run([
            sys.executable, str(ANALYZER), "--authority", str(authority_path),
            "--n128", str(root / "n128"), "--n256", str(root / "n256"),
            "--n512", str(root / "n512"), "--output", str(output),
            "--trusted-tau-max", "2",
        ], check=True)

        summary = json.loads((output / "comparison_summary.history_only.json").read_text())
        assert summary["replay"]["n128"]["exact_executed_prefix"] is True
        assert summary["replay"]["n512"]["max_abs_ulp_difference"] == 0
        with (output / "data/constraint_convergence.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        c_rows = [row for row in rows if row["field"] == "C-norm2"]
        assert c_rows and all(abs(float(row["p"]) - 4.0) < 1e-10 for row in c_rows)
        with (output / "data/native_amr_shadow_n128.csv").open(newline="") as stream:
            shadow = list(csv.DictReader(stream))
        assert len(shadow) == 2
        assert shadow[0]["would_refine_earlier"] == "1"
        assert shadow[1]["agrees"] == "1"
        expected = {
            "constraints_vs_t.png", "constraints_vs_tau.png",
            "curvature_and_timestep_vs_tau.png", "amr_vs_tau.png",
            "curvature_vs_tau.png", "timestep_vs_tau.png",
            "constraint_convergence_order.png", "native_amr_shadow.png",
            "native_amr_sensor_vs_tau.png", "authority_event_jump_convergence.png",
        }
        assert expected == {path.name for path in (output / "figures").glob("*.png")}
    print("COMMON_TREE_HISTORY_ANALYZER_TEST_PASS")


if __name__ == "__main__":
    main()
