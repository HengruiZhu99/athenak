#!/usr/bin/env python3
"""Verify exact implicit/default versus explicit-CC equality without toolchain hashes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    sys.path.insert(0, str(args.source / "tst/unit/z4c"))
    sys.path.insert(0, str(args.source / "vis/python"))
    import z4c_cc_selector_equivalence_test as cc  # pylint: disable=import-error
    import bin_convert  # pylint: disable=import-error

    implicit = args.work_dir / "implicit"
    explicit = args.work_dir / "explicit"
    require(implicit.is_dir() and explicit.is_dir(),
            "the authoritative selector test did not leave both completed runs")
    basename = "z4c_kerr_half_plane_init"
    evidence: dict[str, object] = {
        "schema": 1,
        "disposition": "exact_relative_pass",
        "byte_equal": {},
        "binary_payload_sha256": {},
        "restart_payload_sha256": {},
    }

    for name in (f"{basename}.z4c.user.hst", "z4c_timestep_contract.csv"):
        left = (implicit / name).read_bytes()
        right = (explicit / name).read_bytes()
        require(left == right, f"implicit/explicit {name} bytes differ")
        evidence["byte_equal"][name] = hashlib.sha256(left).hexdigest()

    for variable in ("z4c", "con"):
        for output in (0, 1, 2):
            relative = Path("bin") / f"{basename}.{variable}.{output:05d}.bin"
            left_path = implicit / relative
            right_path = explicit / relative
            require(left_path.is_file() and right_path.is_file(),
                    f"missing paired binary output {relative}")
            digest = cc.compare_binary(
                bin_convert.read_binary(str(left_path)),
                bin_convert.read_binary(str(right_path)), str(relative))
            evidence["binary_payload_sha256"][str(relative)] = digest

    left_restarts = sorted((implicit / "rst").glob(f"{basename}.*.rst"))
    right_restarts = sorted((explicit / "rst").glob(f"{basename}.*.rst"))
    require(left_restarts and
            [path.name for path in left_restarts] ==
            [path.name for path in right_restarts],
            "implicit/explicit restart inventories differ")
    for left_path, right_path in zip(left_restarts, right_restarts):
        left = cc.restart_payload(left_path)
        right = cc.restart_payload(right_path)
        require(left == right,
                f"restart payload differs for {left_path.name}")
        evidence["restart_payload_sha256"][left_path.name] = \
            hashlib.sha256(left).hexdigest()

    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                           encoding="utf-8")
    print("PASS: implicit/default and explicit cell-centered Z4c are exact on this toolchain")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
