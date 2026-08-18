#!/usr/bin/env python3
"""Fail-closed source contract for the dchi derefinement hysteresis."""

import argparse
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    source = (args.source_root / "src/z4c/z4c_amr.cpp").read_text(encoding="utf-8")
    header = (args.source_root / "src/z4c/z4c_amr.hpp").read_text(encoding="utf-8")
    require('GetOrAddReal("z4c_amr", "dchi_derefine_factor", 0.25)' in source,
            "dchi derefinement default is not 0.25")
    require("team_dmax < dchi_derefine_factor * dchi_thresh" in source,
            "dchi derefinement does not use the configured factor")
    require("0.5 * dchi_thresh" not in source,
            "legacy 0.5 derefinement threshold remains")
    require("Real dchi_derefine_factor" in header,
            "dchi derefinement factor is not retained by Z4c_AMR")
    require("dchi_derefine_factor > 0.0 && dchi_derefine_factor < 1.0" in source,
            "dchi derefinement factor lacks strict range validation")
    print("Z4C_AMR_DEREFINE_FACTOR_STATIC_TEST_PASS")


if __name__ == "__main__":
    main()
