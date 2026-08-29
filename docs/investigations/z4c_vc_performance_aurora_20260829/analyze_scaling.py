#!/usr/bin/env python3
"""Summarize matched Aurora N512 scaling runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re


def last(pattern: str, text: str, label: str) -> float:
    values = re.findall(pattern, text)
    if not values:
        raise RuntimeError(f"missing {label}")
    return float(values[-1])


def parse(root: Path) -> dict[str, float | int | str]:
    match = re.search(r"scaling(?:_[a-z]+)*_(\d+)_tiles$", root.name)
    if root.name == "optimized_one_tile_v9":
        tiles = 1
    elif match is not None:
        tiles = int(match.group(1))
    else:
        raise RuntimeError(f"cannot infer tile count from {root}")
    text = (root / "stdout.log").read_text(encoding="utf-8")
    command = (root / "command.txt").read_text(encoding="utf-8")
    ranks_match = re.search(r"mpiexec\s+-n\s+(\d+)", command)
    if ranks_match is None or int(ranks_match.group(1)) != tiles:
        raise RuntimeError(f"rank/tile mismatch in {root}")
    disposition = (root / "disposition").read_text(encoding="utf-8").strip()
    if disposition != "REACHED_TLIM":
        raise RuntimeError(f"non-authoritative disposition {disposition}: {root}")
    return {
        "tiles": tiles,
        "ranks": tiles,
        "meshblocks": int(last(
            r"Current number of MeshBlocks = ([0-9]+)", text, "meshblocks")),
        "meshblock_cycles": int(last(
            r"MeshBlock-cycles = ([0-9]+)", text, "meshblock cycles")),
        "execution_seconds": last(
            r"cpu time used\s*= ([0-9.eE+-]+)", text, "execution time"),
        "zone_cycles_per_second": last(
            r"zone-cycles/cpu_second = ([0-9.eE+-]+)", text, "throughput"),
        "output_seconds": last(
            r"output wall time = ([0-9.eE+-]+)", text, "output time"),
        "root": str(root),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()

    rows = sorted((parse(root) for root in args.runs), key=lambda row: row["tiles"])
    if not rows or rows[0]["tiles"] != 1:
        raise RuntimeError("scaling set must include one tile")
    baseline = float(rows[0]["zone_cycles_per_second"])
    for row in rows:
        tiles = int(row["tiles"])
        speedup = float(row["zone_cycles_per_second"]) / baseline
        row["speedup"] = speedup
        row["parallel_efficiency"] = speedup / tiles
        row["zone_cycles_per_second_per_tile"] = (
            float(row["zone_cycles_per_second"]) / tiles
        )
        row["meshblocks_per_rank"] = float(row["meshblocks"]) / tiles

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "tiles", "ranks", "meshblocks", "meshblocks_per_rank",
        "meshblock_cycles", "execution_seconds", "output_seconds",
        "zone_cycles_per_second", "zone_cycles_per_second_per_tile",
        "speedup", "parallel_efficiency", "root",
    ]
    with args.csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    args.json.write_text(json.dumps({
        "schema": "z4c_vc_n512_aurora_scaling_v1",
        "rows": rows,
    }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
