#!/usr/bin/env python3
"""Guard nvcc's public-enclosing-method requirement for KOKKOS_LAMBDA."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", required=True, type=Path)
    args = parser.parse_args()
    root = args.source_dir.resolve()
    header = (root / "src/z4c/cartoon_m0_fastflow.hpp").read_text(
        encoding="utf-8")
    source = (root / "src/z4c/cartoon_m0_fastflow.cpp").read_text(
        encoding="utf-8")

    class_match = re.search(
        r"class CartoonM0FastFlow \{(?P<body>.*?)\n\};", header, re.DOTALL)
    require(class_match is not None, "CartoonM0FastFlow declaration is missing")
    body = class_match.group("body")
    public = body.split(" private:", 1)[0]
    private = body.split(" private:", 1)[1]
    for declaration in (
        "M0AdmSample SampleAdm(Real rho, Real z) const;",
        "M0AxisSample SampleAxisLapse(Real z) const;",
    ):
        require(public.count(declaration) == 1,
                f"nvcc kernel-enclosing method is not public: {declaration}")
        require(declaration not in private,
                f"nvcc kernel-enclosing method remains private: {declaration}")
    require("M0CandidateSummary SearchCandidate" in private and
            "void Restore();" in private and "void Capture();" in private,
            "unrelated FastFlow implementation methods escaped private access")

    definitions = tuple(re.finditer(
        r"(?:M0AdmSample|M0AxisSample) CartoonM0FastFlow::"
        r"(?P<name>SampleAdm|SampleAxisLapse)\([^)]*\) const \{(?P<body>.*?)\n\}",
        source, re.DOTALL))
    require(len(definitions) == 2 and
            {match.group("name") for match in definitions} ==
            {"SampleAdm", "SampleAxisLapse"},
            "kernel-launching sampler definition inventory changed")
    require(all("KOKKOS_LAMBDA" in match.group("body") for match in definitions) and
            source.count("KOKKOS_LAMBDA") == 2,
            "KOKKOS_LAMBDA launch inventory changed")
    print("Cartoon m=0 FastFlow CUDA access static checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
