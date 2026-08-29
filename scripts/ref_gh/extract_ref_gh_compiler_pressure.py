#!/usr/bin/env python3
"""Extract Intel PVC register/spill reports for Ref-GH CalcRHS kernels."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


GENERIC_LAMBDA_NAMES = {
    "": "psi_rhs",
    "0": "generic_gauge_driver",
    "1": "scalar_source_and_pi_rhs",
    "2": "compatible_phi_rhs",
    "3": "standard_phi_rhs",
    "4": "gamma2_reduction_damping",
    "5": "dissipation",
}

ANALYTIC_LAMBDA_NAMES = {
    "": "psi_rhs",
    "0": "staged_physical_geometry_and_gauge",
    "1": "compatible_phi_rhs",
    "2": "standard_phi_rhs",
    "3": "gamma2_reduction_damping",
    "4": "dissipation",
}

ANALYTIC_FLAT_NAMES = {
    "0": "staged_coordinate_partial_source_components",
    "1": "staged_coordinate_frame_transform_components",
    "2": "staged_pi_principal_components",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("build_log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = ["line\tfdng\tanalytic\tlambda\tkernel\tsimd\tregs\tspill_reals"]
    par_for_pattern = re.compile(
        r"CalcRHSImplILi(?P<fdng>\d+)ELb(?P<analytic>[01]).*?"
        r"EUliiiiE(?P<lambda>\d*)_.*?compiled SIMD(?P<simd>\d+) "
        r"allocated (?P<regs>\d+) regs and spilled around (?P<spill>\d+)")
    flat_pattern = re.compile(
        r"CalcRHSImplILi(?P<fdng>\d+)ELb1EEE.*?EUliE(?P<lambda>\d+)_.*?"
        r"compiled SIMD(?P<simd>\d+) allocated (?P<regs>\d+) regs and "
        r"spilled around (?P<spill>\d+)")
    for line_number, line in enumerate(
            args.build_log.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1):
        match = flat_pattern.search(line)
        policy = "flat"
        if match is None:
            match = par_for_pattern.search(line)
            policy = "par_for"
        if match is None:
            continue
        values = match.groupdict()
        lambda_id = values["lambda"]
        analytic = values.get("analytic", "1")
        if policy == "flat":
            kernel = ANALYTIC_FLAT_NAMES.get(lambda_id, "unknown_flat")
        elif analytic == "1":
            kernel = ANALYTIC_LAMBDA_NAMES.get(lambda_id, "unknown")
        else:
            kernel = GENERIC_LAMBDA_NAMES.get(lambda_id, "unknown")
        rows.append("\t".join((
            str(line_number), values["fdng"], analytic,
            lambda_id or "base", kernel,
            values["simd"], values["regs"], values["spill"])))
    rendered = "\n".join(rows) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
