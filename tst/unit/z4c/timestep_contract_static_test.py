#!/usr/bin/env python3
"""Fail-closed structural checks for the two-part Z4c timestep contract."""

import argparse
from pathlib import Path
import sys


def require(text: str, token: str, label: str) -> None:
    if token not in text:
        raise AssertionError(f"missing {label}: {token}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True)
    args = parser.parse_args()
    source_dir = Path(args.source_dir)
    mesh = (source_dir / "src/mesh/mesh.cpp").read_text()
    newdt = (source_dir / "src/z4c/z4c_newdt.cpp").read_text()
    header = (source_dir / "src/z4c/timestep_contract.hpp").read_text()

    require(mesh, "(cfl_no)*(pmb_pack->pz4c->dt_spatial)", "spatial CFL application")
    require(mesh, "dt = std::min(dt, pmb_pack->pz4c->dt_source);",
            "unscaled source ceiling")
    if "(cfl_no)*(pmb_pack->pz4c->dt_source)" in mesh:
        raise AssertionError("source ceiling is incorrectly multiplied by spatial CFL")
    require(mesh, "WriteTimestepContractRecord(dt)", "final timestep record")
    for token, label in (
        ("ScaleInvariantTelegraphCoefficients", "shared telegraph helper"),
        ("LocalChiGradientNormTelegraphMu", "local-gradient telegraph source rate"),
        ("shift_mode != Z4cShiftMode::prescribed_zero", "zero-shift exclusion"),
        ("dt_spatial", "spatial diagnostic"),
        ("dt_source", "source diagnostic"),
        ("max_source_rate", "source-rate diagnostic"),
        ("max_coordinate_speed", "speed diagnostic"),
        ("z4c_timestep_contract.csv", "debug record"),
    ):
        require(newdt, token, label)
    require(header, "ExplicitRKStabilityPolynomial", "actual-RK stability polynomial")
    require(header, "CoordinateCharacteristicSpeed", "coordinate-speed composition helper")
    print("Z4C_TIMESTEP_CONTRACT_STATIC_PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except AssertionError as error:
        print(f"timestep contract static test: {error}", file=sys.stderr)
        raise SystemExit(1)
