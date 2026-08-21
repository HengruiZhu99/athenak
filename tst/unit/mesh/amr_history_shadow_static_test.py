#!/usr/bin/env python3
"""Fail-closed source contract for replay-only native-AMR shadow evidence."""

import argparse
from pathlib import Path


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    args = parser.parse_args()
    history = (args.source_root / "src/mesh/amr_history.cpp").read_text(
        encoding="utf-8")
    amr = (args.source_root / "src/z4c/z4c_amr.cpp").read_text(
        encoding="utf-8")
    for token in (
        "AppendShadowLedger();",
        "athenak_amr_native_shadow_v1",
        "raw_dchi",
        "dchi_over_dx",
        "native_action",
        "authority_action",
        "strongest_physical_location",
        "WOULD_REFINE_EARLIER",
        "WOULD_NOT_REFINE",
        "WOULD_DEREFINE",
    ):
        require(token in history, f"native replay shadow field is missing: {token}")
    require("capture_replay_dchi" in amr and "Z4c_AMR::DchiArgmax" in amr,
            "exact per-block dchi maximum/location capture is missing")
    require('mode") == "replay"' in amr,
            "dchi shadow capture is not replay-only")
    require("candidate_dt_hex" in history and "applied_dt_hex" in history and
            "ulp_difference" in history,
            "replay event timestep alignment evidence is incomplete")
    print("AMR_HISTORY_SHADOW_STATIC_PASS")


if __name__ == "__main__":
    main()
