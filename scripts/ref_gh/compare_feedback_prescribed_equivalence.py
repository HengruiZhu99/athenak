#!/usr/bin/env python3
"""Compare legacy-time and prescribed-xi compact Ref-GH field payloads."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np


def load_reader(source_root: Path):
    path = source_root / "scripts/ref_gh/analyze_perturbed_trumpet_convergence.py"
    spec = importlib.util.spec_from_file_location("refgh_cbin_reader", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.read_cbin


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("legacy", type=Path)
    parser.add_argument("prescribed", type=Path)
    parser.add_argument("--source-root", type=Path, default=Path(__file__).parents[2])
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    read_cbin = load_reader(args.source_root.resolve())
    legacy = read_cbin(args.legacy)
    prescribed = read_cbin(args.prescribed)
    if legacy["variables"] != prescribed["variables"]:
        raise SystemExit("variable lists differ")
    difference = np.abs(legacy["data"] - prescribed["data"])
    result = {
        "schema": "ref-gh-prescribed-equivalence-v1",
        "legacy": str(args.legacy),
        "prescribed": str(args.prescribed),
        "legacy_time": legacy["time"],
        "prescribed_time": prescribed["time"],
        "legacy_cycle": legacy["cycle"],
        "prescribed_cycle": prescribed["cycle"],
        "payload_linf": float(np.max(difference)),
        "payload_bitwise_equal": bool(np.array_equal(
            legacy["data"], prescribed["data"])),
    }
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(text)
    print(text, end="")
    if not result["payload_bitwise_equal"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
