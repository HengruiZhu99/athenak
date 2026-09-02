#!/usr/bin/env python3
"""Verify that PC-GH and matched Z4c one-puncture Cartesian data agree at t=0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from plot_qualification import read_cartesian


NATIVE_FIELD_PAIRS = (
    ("chi", "pcgh_chi", "z4c_chi", "identity"),
    ("gtilde_xx", "pcgh_gtxx", "z4c_gxx", "identity"),
    ("gtilde_xy", "pcgh_gtxy", "z4c_gxy", "identity"),
    ("gtilde_xz", "pcgh_gtxz", "z4c_gxz", "identity"),
    ("gtilde_yy", "pcgh_gtyy", "z4c_gyy", "identity"),
    ("gtilde_yz", "pcgh_gtyz", "z4c_gyz", "identity"),
    ("gtilde_zz", "pcgh_gtzz", "z4c_gzz", "identity"),
    ("K", "pcgh_K", "z4c_Khat", "identity"),
    ("Atilde_xx", "pcgh_Atxx", "z4c_Axx", "identity"),
    ("Atilde_xy", "pcgh_Atxy", "z4c_Axy", "identity"),
    ("Atilde_xz", "pcgh_Atxz", "z4c_Axz", "identity"),
    ("Atilde_yy", "pcgh_Atyy", "z4c_Ayy", "identity"),
    ("Atilde_yz", "pcgh_Atyz", "z4c_Ayz", "identity"),
    ("Atilde_zz", "pcgh_Atzz", "z4c_Azz", "identity"),
    ("Gamma_x", "pcgh_Lamx", "z4c_Gamx", "identity"),
    ("Gamma_y", "pcgh_Lamy", "z4c_Gamy", "identity"),
    ("Gamma_z", "pcgh_Lamz", "z4c_Gamz", "identity"),
    ("alpha_squared", "pcgh_A", "z4c_alpha", "square_rhs"),
    ("beta_x", "pcgh_betax", "z4c_betax", "identity"),
    ("beta_y", "pcgh_betay", "z4c_betay", "identity"),
    ("beta_z", "pcgh_betaz", "z4c_betaz", "identity"),
)

ADM_FIELD_PAIRS = tuple(
    (name.removeprefix("adm_"), name, name, "identity")
    for name in (
        "adm_gxx", "adm_gxy", "adm_gxz", "adm_gyy", "adm_gyz", "adm_gzz",
        "adm_Kxx", "adm_Kxy", "adm_Kxz", "adm_Kyy", "adm_Kyz", "adm_Kzz",
        "adm_psi4",
    )
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pcgh", type=Path)
    parser.add_argument("z4c", type=Path)
    parser.add_argument("--tolerance", type=float, default=2.0e-6)
    args = parser.parse_args()

    pcgh = read_cartesian(args.pcgh)
    z4c = read_cartesian(args.z4c)
    if abs(float(pcgh["time"])) > 1.0e-12 or abs(float(z4c["time"])) > 1.0e-12:
        raise SystemExit("both inputs must be t=0 outputs")
    for coordinate in ("x", "y", "z"):
        if not np.array_equal(pcgh[coordinate], z4c[coordinate]):
            raise SystemExit(f"Cartesian {coordinate} grids differ")

    pcgh_fields = pcgh["data"]
    z4c_fields = z4c["data"]
    field_pairs = (ADM_FIELD_PAIRS if "adm_gxx" in pcgh_fields
                   else NATIVE_FIELD_PAIRS)
    report = {}
    passed = True
    for label, pcgh_name, z4c_name, transform in field_pairs:
        lhs = np.asarray(pcgh_fields[pcgh_name], dtype=float)
        rhs = np.asarray(z4c_fields[z4c_name], dtype=float)
        if transform == "square_rhs":
            rhs = rhs*rhs
        difference = np.abs(lhs - rhs)
        scale = max(1.0, float(np.nanmax(np.abs(lhs))), float(np.nanmax(np.abs(rhs))))
        normalized = float(np.nanmax(difference))/scale
        report[label] = {
            "max_abs": float(np.nanmax(difference)),
            "normalized_max_abs": normalized,
        }
        passed &= np.all(np.isfinite(lhs)) and np.all(np.isfinite(rhs)) \
            and normalized <= args.tolerance
    result = {"passed": bool(passed), "tolerance": args.tolerance, "fields": report}
    print(json.dumps(result, indent=2))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
