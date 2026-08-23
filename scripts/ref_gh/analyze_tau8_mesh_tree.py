#!/usr/bin/env python3
"""Create compact exact evidence from AthenaK mesh_structure.dat."""

import argparse
import json
import math
import re
from pathlib import Path


def parse(path):
    lines = path.read_text().splitlines()
    blocks = []
    index = 0
    while index < len(lines):
        match = re.match(r"#MeshBlock (\d+) on rank=(\d+) with cost=(\S+)",
                         lines[index])
        if match is None:
            index += 1
            continue
        level_match = re.match(
            r"#  Logical level (\d+), location = \((\d+) (\d+) (\d+)\)",
            lines[index + 1])
        if level_match is None:
            raise RuntimeError(f"invalid logical-location row after line {index + 1}")
        points = []
        cursor = index + 2
        while cursor < len(lines) and lines[cursor] and not lines[cursor].startswith("#"):
            points.append(tuple(float(value) for value in lines[cursor].split()))
            cursor += 1
        if len(points) != 17:
            raise RuntimeError(f"MeshBlock {match.group(1)} has {len(points)} vertices")
        blocks.append({
            "gid": int(match.group(1)), "rank": int(match.group(2)),
            "cost": float(match.group(3)), "logical_level": int(level_match.group(1)),
            "logical_location": [int(level_match.group(i)) for i in range(2, 5)],
            "bounds": [min(point[axis] for point in points) for axis in range(3)]
                      + [max(point[axis] for point in points) for axis in range(3)],
        })
        index = cursor
    if not blocks:
        raise RuntimeError(f"no MeshBlocks parsed from {path}")
    return blocks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mesh_structure", type=Path)
    parser.add_argument("--root-blocks-per-direction", type=int, default=3)
    parser.add_argument("--domain-min", type=float, default=-12.0)
    parser.add_argument("--domain-max", type=float, default=12.0)
    parser.add_argument("--output-prefix", type=Path, required=True)
    args = parser.parse_args()

    blocks = parse(args.mesh_structure)
    root_level = math.ceil(math.log2(args.root_blocks_per_direction))
    for block in blocks:
        block["physical_level"] = block["logical_level"] - root_level
        xmin, ymin, zmin, xmax, ymax, zmax = block["bounds"]
        block["width"] = xmax - xmin
        if not (math.isclose(block["width"], ymax - ymin)
                and math.isclose(block["width"], zmax - zmin)):
            raise RuntimeError(f"non-cubic MeshBlock {block['gid']}")

    counts = {}
    coverage = {}
    widths = {}
    for level in sorted({block["physical_level"] for block in blocks}):
        selected = [block for block in blocks if block["physical_level"] == level]
        counts[str(level)] = len(selected)
        coverage[str(level)] = [
            min(block["bounds"][axis] for block in selected) for axis in range(3)
        ] + [max(block["bounds"][axis] for block in selected) for axis in range(3, 6)]
        widths[str(level)] = sorted({block["width"] for block in selected})

    finest_level = max(block["physical_level"] for block in blocks)
    finest = [block for block in blocks if block["physical_level"] == finest_level]
    resolutions = {}
    for label, cells in (("coarse", 32), ("medium", 48), ("fine", 64)):
        dx_by_level = {str(level): values[0] / cells
                       for level, values in ((int(key), value)
                                             for key, value in widths.items())}
        resolutions[label] = {
            "cells_per_meshblock": cells, "dx_by_physical_level": dx_by_level,
            "dx_min": min(block["width"] for block in finest) / cells,
            "puncture_vertex_all_active_levels": all(
                math.isclose(-args.domain_min / dx, round(-args.domain_min / dx),
                             rel_tol=0.0, abs_tol=1.0e-12)
                for dx in dx_by_level.values()),
        }

    payload = {
        "schema": "ref-gh-tau8-authoritative-mesh-tree-v1",
        "source": str(args.mesh_structure),
        "domain": [args.domain_min, args.domain_max] * 3,
        "root_blocks_per_direction": args.root_blocks_per_direction,
        "root_logical_level": root_level, "total_meshblocks": len(blocks),
        "counts_by_physical_level": counts,
        "coverage_by_physical_level": coverage,
        "block_widths_by_physical_level": widths,
        "finest_level": finest_level,
        "transition_shell_0p30_to_0p60_on_finest": all(
            coverage[str(finest_level)][axis] <= -0.6
            and coverage[str(finest_level)][axis + 3] >= 0.6
            for axis in range(3)),
        "resolutions": resolutions, "blocks": blocks,
    }
    json_path = Path(str(args.output_prefix) + ".json")
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tsv_path = Path(str(args.output_prefix) + ".tsv")
    with tsv_path.open("w") as stream:
        stream.write("gid\trank\tphysical_level\tlogical_level\tlx1\tlx2\tlx3\t"
                     "xmin\txmax\tymin\tymax\tzmin\tzmax\twidth\n")
        for block in blocks:
            xmin, ymin, zmin, xmax, ymax, zmax = block["bounds"]
            stream.write("\t".join(str(value) for value in (
                block["gid"], block["rank"], block["physical_level"],
                block["logical_level"], *block["logical_location"], xmin, xmax,
                ymin, ymax, zmin, zmax, block["width"])) + "\n")
    print(json_path)
    print(tsv_path)


if __name__ == "__main__":
    main()
