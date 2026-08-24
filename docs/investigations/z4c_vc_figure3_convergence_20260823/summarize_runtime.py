#!/usr/bin/env python3
"""Compact fail-closed runtime, replay, shadow, and axis telemetry."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


CASES = ("n128", "n256", "n512")


def lines(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def text(path: Path) -> str | None:
    return path.read_text().strip() if path.exists() else None


def summarize_case(root: Path, case: str) -> dict:
    with (root / "z4c_vertex_axis_regularity.csv").open(newline="") as stream:
        axis = list(csv.DictReader(stream))
    replay_files = list(root.glob("*.amr_history_replay.jsonl"))
    replay = lines(replay_files[0]) if replay_files else []
    shadow_files = list(root.glob("*.amr_native_shadow.rank*.jsonl"))
    shadow = [row for path in shadow_files for row in lines(path)]
    failure_path = root / "z4c_state_failure.json"
    failure = json.loads(failure_path.read_text()) if failure_path.exists() else None
    result = {
        "root": str(root),
        "disposition": text(root / "disposition"),
        "run_status": text(root / "run-status"),
        "command": text(root / "command.txt"),
        "axis_regularity": {
            "rows": len(axis),
            "max_abs": max(float(row["max_abs"]) for row in axis),
            "max_scaled": max(float(row["max_scaled"]) for row in axis),
            "nonfinite_rows": sum(int(row["nonfinite"]) != 0 for row in axis),
        },
        "replay": None if not replay else {
            "events": len(replay),
            "all_exact": all(row.get("exact_match") is True for row in replay),
            "last_event": replay[-1].get("event"),
            "last_tree_checksum": replay[-1].get("tree_checksum"),
        },
        "native_shadow": {
            "rows": len(shadow),
            "native_action_counts": dict(Counter(row.get("native_action") for row in shadow)),
            "classification_counts": dict(Counter(row.get("classification") for row in shadow)),
        },
        "state_failure": failure,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    for case in CASES:
        parser.add_argument(f"--{case}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"schema": "z4c_vc_figure3_runtime_summary_v1",
              "cases": {case: summarize_case(getattr(args, case), case)
                        for case in CASES}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
