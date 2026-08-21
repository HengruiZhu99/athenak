#!/usr/bin/env python3
"""Separate replay-event clipping from the underlying Z4c timestep candidate."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


THRESHOLDS = (1.0e-3, 1.0e-4, 1.0e-5)


def analyze(path: Path, cfl: float) -> dict:
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {"cycle", "time", "dt_spatial", "dt_source", "dt_final"}
    if not rows or not required <= rows[0].keys():
        raise RuntimeError(f"invalid timestep contract: {path}")
    parsed = []
    for row in rows:
        values = {name: float(row[name]) for name in required - {"cycle"}}
        values["cycle"] = int(row["cycle"])
        if not all(math.isfinite(value) for value in values.values()):
            raise RuntimeError(f"nonfinite timestep contract row: {path}")
        candidate = cfl * min(values["dt_spatial"], values["dt_source"])
        values["unclipped_z4c_candidate"] = candidate
        values["event_or_external_clip"] = values["dt_final"] < candidate * (1.0 - 2.0e-13)
        parsed.append(values)
    landmarks = {}
    for threshold in THRESHOLDS:
        key = format(threshold, ".0e")
        landmarks[key] = {}
        for quantity in ("dt_final", "unclipped_z4c_candidate"):
            match = next((row for row in parsed if row[quantity] <= threshold), None)
            landmarks[key][quantity] = None if match is None else {
                "cycle": match["cycle"], "time": match["time"],
                "value": match[quantity],
                "event_or_external_clip": match["event_or_external_clip"],
            }
    return {
        "path": str(path), "rows": len(parsed),
        "clip_count": sum(row["event_or_external_clip"] for row in parsed),
        "terminal": parsed[-1], "landmarks": landmarks,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfl", type=float, required=True)
    for case in ("n128", "n256", "n512"):
        parser.add_argument(f"--{case}", type=Path, required=True)
    args = parser.parse_args()
    payload = {
        "schema": "brill_common_tree_timestep_landmarks_v1",
        "cfl": args.cfl,
        "cases": {case: analyze(getattr(args, case), args.cfl)
                  for case in ("n128", "n256", "n512")},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
