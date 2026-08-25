#!/usr/bin/env python3
"""Check the linearized transverse GH-constraint damping eigenmode."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


CASES = (
    (0.25, 0.0, "g0p25_d0p0"),
    (0.5, 0.0, "g0p5_d0p0"),
    (1.0, 0.0, "g1p0_d0p0"),
    (0.25, 0.02, "g0p25_d0p02"),
    (0.5, 0.02, "g0p5_d0p02"),
    (1.0, 0.02, "g1p0_d0p02"),
)


def read_history(path: Path) -> list[list[float]]:
    rows = [
        [float(value) for value in line.split()]
        for line in path.read_text().splitlines()
        if line and not line.startswith("#")
    ]
    if len(rows) < 2 or any(len(row) < 5 for row in rows):
        raise ValueError(f"invalid Ref-GH history: {path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--nx", type=int, default=32)
    parser.add_argument("--length", type=float, default=1.0)
    parser.add_argument("--fd-order", type=int, default=4, choices=(2, 4, 6))
    parser.add_argument("--absolute-tolerance", type=float, default=2.0e-5)
    parser.add_argument("--purity-tolerance", type=float, default=1.0e-14)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    args = parser.parse_args()

    spacing = args.length/args.nx
    phase = 2.0*math.pi/args.nx
    diss_order = args.fd_order//2 + 1
    records = []
    all_pass = True
    for gamma0, dissipation, tag in CASES:
        path = args.root/tag/f"{tag}.ref_gh.hst"
        rows = read_history(path)
        initial_gh2 = rows[0][2]
        if not math.isfinite(initial_gh2) or not initial_gh2 > 0.0:
            raise ValueError(f"invalid initial GH norm: {path}")
        ko_rate = dissipation*math.sin(0.5*phase)**(2*diss_order)/spacing
        errors = []
        for row in rows:
            observed = math.sqrt(row[2]/initial_gh2)
            predicted = math.exp(-(0.5*gamma0 + ko_rate)*row[0])
            errors.append(abs(observed-predicted))
        max_reduction = math.sqrt(max(row[3] for row in rows))
        max_curl = math.sqrt(max(row[4] for row in rows))
        passed = (
            max(errors) <= args.absolute_tolerance
            and max_reduction <= args.purity_tolerance
            and max_curl <= args.purity_tolerance
        )
        all_pass = all_pass and passed
        records.append({
            "tag": tag,
            "gamma0": gamma0,
            "dissipation": dissipation,
            "final_time": rows[-1][0],
            "ko_rate": ko_rate,
            "final_gh_growth": math.sqrt(rows[-1][2]/initial_gh2),
            "final_gh_predicted": math.exp(
                -(0.5*gamma0 + ko_rate)*rows[-1][0]),
            "max_gh_absolute_error": max(errors),
            "max_reduction_l2": max_reduction,
            "max_curl_l2": max_curl,
            "pass": passed,
        })

    result = {
        "schema": "ref-gh-gamma0-transverse-mode-v1",
        "linearized_prediction": (
            "C_y L2(t) = C_y L2(0) exp[-(gamma0/2 + lambda_KO)t]"
        ),
        "primary_reference_equation": (
            "Lindblom et al. 2006 Eq. (21), transverse short-wavelength mode"
        ),
        "grid": {
            "nx": [args.nx, 8, 8],
            "length": args.length,
            "fd_order": args.fd_order,
            "dissipation_operator_order": 2*diss_order,
        },
        "absolute_tolerance": args.absolute_tolerance,
        "purity_tolerance": args.purity_tolerance,
        "records": records,
        "overall_pass": all_pass,
    }
    args.output_json.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    with args.output_csv.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(records[0]))
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
