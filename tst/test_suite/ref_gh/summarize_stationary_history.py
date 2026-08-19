#!/usr/bin/env python3
"""Summarize Ref-GH stationary-trumpet AthenaK history files.

The Ref-GH history records squared L2 integrals.  This deliberately reports
their square roots rather than calling them L2 norms, which keeps the compact
stationary t=20 comparison explicit about the normalization convention.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path


FIELDS = (
    "time",
    "dt",
    "gh_l2sq",
    "reduction_l2sq",
    "curl_l2sq",
    "psi_error_l2sq",
    "pi_l2sq",
    "phi_l2sq",
    "gh_near_l2sq",
    "reduction_near_l2sq",
    "curl_near_l2sq",
    "volume",
    "alpha_max",
    "minus_alpha_min",
    "regular_max",
    "g_condition",
    "coordinate_ricci_max",
    "characteristic_speed_max",
    "effective_cfl",
    "minus_detg_min",
    "near_volume",
    "bad_state",
)


def final_row(path: Path) -> list[float]:
    rows = [line.split() for line in path.read_text().splitlines()
            if line and not line.startswith("#")]
    if not rows:
        raise ValueError(f"no data rows in {path}")
    values = [float(value) for value in rows[-1]]
    if len(values) != len(FIELDS):
        raise ValueError(f"expected {len(FIELDS)} columns in {path}, got {len(values)}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("history", nargs="+", type=Path)
    args = parser.parse_args()

    columns = ("history", "time", "dt", "gh_l2", "reduction_l2", "curl_l2",
               "psi_error_l2", "pi_l2", "phi_l2", "gh_near_l2",
               "reduction_near_l2", "curl_near_l2", "regular_max", "g_condition",
               "characteristic_speed_max", "effective_cfl", "bad_state")
    print("\t".join(columns))
    for path in args.history:
        values = dict(zip(FIELDS, final_row(path), strict=True))
        row = (
            str(path),
            values["time"], values["dt"],
            math.sqrt(values["gh_l2sq"]),
            math.sqrt(values["reduction_l2sq"]),
            math.sqrt(values["curl_l2sq"]),
            math.sqrt(values["psi_error_l2sq"]),
            math.sqrt(values["pi_l2sq"]),
            math.sqrt(values["phi_l2sq"]),
            math.sqrt(values["gh_near_l2sq"]),
            math.sqrt(values["reduction_near_l2sq"]),
            math.sqrt(values["curl_near_l2sq"]),
            values["regular_max"], values["g_condition"],
            values["characteristic_speed_max"], values["effective_cfl"],
            int(values["bad_state"]),
        )
        print("\t".join(str(value) for value in row))


if __name__ == "__main__":
    main()
