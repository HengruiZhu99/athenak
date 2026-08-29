#!/usr/bin/env python3
"""Extract Intel PVC register/spill reports for Ref-GH CalcRHS kernels."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


LAMBDA_NAMES = {
    "": "psi_rhs",
    "0": "generic_gauge_driver",
    "1": "scalar_source_and_pi_rhs",
    "2": "compatible_phi_rhs",
    "3": "standard_phi_rhs",
    "4": "gamma2_reduction_damping",
    "5": "dissipation",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("build_log", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = ["line\tfdng\tanalytic\tlambda\tkernel\tsimd\tregs\tspill_reals"]
    pattern = re.compile(
        r"CalcRHSImplILi(?P<fdng>\d+)ELb(?P<analytic>[01]).*?"
        r"EUliiiiE(?P<lambda>\d*)_.*?compiled SIMD(?P<simd>\d+) "
        r"allocated (?P<regs>\d+) regs and spilled around (?P<spill>\d+)")
    for line_number, line in enumerate(
            args.build_log.read_text(encoding="utf-8", errors="replace").splitlines(),
            start=1):
        match = pattern.search(line)
        if match is None:
            continue
        values = match.groupdict()
        lambda_id = values["lambda"]
        rows.append("\t".join((
            str(line_number), values["fdng"], values["analytic"],
            lambda_id or "base", LAMBDA_NAMES.get(lambda_id, "unknown"),
            values["simd"], values["regs"], values["spill"])))
    rendered = "\n".join(rows) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
