#!/usr/bin/env python3
"""Fail-closed source contract for AMR cadence and tracker configuration."""

import argparse
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    refinement = (root / "src/mesh/mesh_refinement.cpp").read_text(encoding="utf-8")
    header = (root / "src/mesh/amr_cadence.hpp").read_text(encoding="utf-8")
    z4c_amr = (root / "src/z4c/z4c_amr.cpp").read_text(encoding="utf-8")
    require('GetOrAddInteger("mesh_refinement", "ncycle_check", 1)' in refinement,
            "ncycle_check is not read as an integer")
    require('GetOrAddInteger("mesh_refinement", "refinement_interval", 5)' in refinement,
            "refinement_interval is not read as an integer")
    require("IsValidAMRCadence(ncyc_check_amr, refinement_interval)" in refinement and
            "positive integers" in refinement,
            "AMR cadence lacks a positive-integer fail-closed gate")
    require("inline constexpr bool IsValidAMRCadence" in header,
            "cadence predicate is not independently unit-testable")
    tracker = z4c_amr[z4c_amr.index('ref_method == "tracker"'):
                      z4c_amr.index('ref_method == "chi"')]
    require('DoesParameterExist("z4c", "co_0_type")' in tracker and
            "requires at least one" in tracker,
            "empty tracker configuration does not fail before refinement")
    print("Z4C_AMR_CONFIGURATION_STATIC_TEST_PASS")


if __name__ == "__main__":
    main()
