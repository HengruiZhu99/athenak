#!/usr/bin/env python3
"""Generate and verify the half-plane SO(2) direct-operator inventory."""

from __future__ import annotations

import argparse
from fractions import Fraction
from pathlib import Path


ROWS = {
    2: {0: (Fraction(-1, 2), Fraction(1, 2))},
    3: {
        0: (Fraction(-2, 3), Fraction(3, 4), Fraction(-1, 12)),
        1: (Fraction(-1, 3), Fraction(1, 4), Fraction(1, 12)),
    },
    4: {
        0: (Fraction(-3, 4), Fraction(9, 10), Fraction(-1, 6), Fraction(1, 60)),
        1: (Fraction(-5, 18), Fraction(3, 20), Fraction(5, 36), Fraction(-1, 90)),
        2: (Fraction(1, 6), Fraction(-9, 20), Fraction(1, 4), Fraction(1, 30)),
    },
}


OPERATORS = (
    ("scalar `d_y`, active-suppressed mixed", "`F(s)`", "zero by SO(2)",
     "none", "`0`"),
    ("scalar `d_y^2`", "EvenScalar `F(s)`", "`2 F_s`", "`D_s[F]`",
     "`2 D_s[F]`"),
    ("vector `d_y V^rho`, `d_y V^y`", "OddLinear `V^rho=rho A`, `V^y=rho B`",
     "`-B`, `A`", "coefficient value", "`-V^y/rho`, `V^rho/rho`"),
    ("vector `d_y V^z`", "EvenScalar `C(s)`", "zero", "none", "`0`"),
    ("vector `d_y^2 V^z`", "EvenScalar `C(s)`", "`2 C_s`", "`D_s[C]`",
     "`2 D_s[C]`"),
    ("vector `d_y^2 V^rho`, `d_y^2 V^y`",
     "OddLinear `rho A(s)`, `rho B(s)`", "`2 rho A_s`, `2 rho B_s`",
     "`D_s[A]`, `D_s[B]`", "`2 rho D_s[A]`, `2 rho D_s[B]`"),
    ("vector `d_rho d_y V^rho`, `d_rho d_y V^y`",
     "rotated OddLinear coefficient", "`-2 rho B_s`, `2 rho A_s`",
     "`D_s[B]`, `D_s[A]`", "`-2 rho D_s[B]`, `2 rho D_s[A]`"),
    ("vector `d_z d_y V^rho`, `d_z d_y V^y`",
     "rotated OddLinear coefficient", "`-B_z`, `A_z`", "centered `d_z`",
     "`-(d_z V^y)/rho`, `(d_z V^rho)/rho`"),
    ("tensor `d_y T_rhorho`, `d_y T_yy`", "TensorSwirl `T_rhoy=s R`",
     "`-2 rho R`, `2 rho R`", "coefficient value",
     "`-2 T_rhoy/rho`, `2 T_rhoy/rho`"),
    ("tensor `d_y T_rhoy`", "TensorPlanarPair `T_rhorho-T_yy=s Q`",
     "`rho Q`", "paired coefficient value", "`(T_rhorho-T_yy)/rho`"),
    ("tensor `d_y T_rhoz`, `d_y T_yz`", "OddLinear `rho U`, `rho V`",
     "`-V`, `U`", "coefficient value", "`-T_yz/rho`, `T_rhoz/rho`"),
    ("tensor `d_y^2 T_rhorho`", "TensorPlanarPair `P+s Q`, `P`",
     "`2(P_s+s Q_s)`", "`D_s[T_rhorho]` and paired `Q`",
     "`2 D_s[T_rhorho]-2 Q`"),
    ("tensor `d_y^2 T_yy`", "TensorPlanarPair `P+s Q`, `P`",
     "`2(P_s+Q)`", "`D_s[T_yy]` and paired `Q`",
     "`2 D_s[T_yy]+2 Q`"),
    ("tensor `d_y^2 T_rhoy`", "TensorSwirl `s R`", "`-2 R+2 s R_s`",
     "`D_s[R]` and coefficient value", "`2 s D_s[R]-2 R`"),
    ("tensor `d_y^2 T_rhoz`, `d_y^2 T_yz`", "OddLinear `rho U`, `rho V`",
     "`2 rho U_s`, `2 rho V_s`", "`D_s[U]`, `D_s[V]`",
     "`2 rho D_s[U]`, `2 rho D_s[V]`"),
    ("tensor `d_y^2 T_zz`", "EvenScalar `W(s)`", "`2 W_s`", "`D_s[W]`",
     "`2 D_s[W]`"),
    ("tensor `d_rho d_y T_rhorho`, `d_rho d_y T_yy`", "TensorSwirl `s R`",
     "`-2 R-4 s R_s`, `2 R+4 s R_s`", "`D_s[R]` and coefficient value",
     "`-2 R-4 s D_s[R]`, `2 R+4 s D_s[R]`"),
    ("tensor `d_rho d_y T_rhoy`", "TensorPlanarPair `s Q`",
     "`Q+2 s Q_s`", "paired `D_s[Q]` and coefficient value",
     "`Q+2 s D_s[Q]`"),
    ("tensor `d_rho d_y T_rhoz`, `d_rho d_y T_yz`",
     "OddLinear `rho U`, `rho V`", "`-2 rho V_s`, `2 rho U_s`",
     "`D_s[V]`, `D_s[U]`", "`-2 rho D_s[V]`, `2 rho D_s[U]`"),
    ("tensor `d_z d_y` nonzero branches", "regular `R,Q,U,V` coefficients",
     "`-2 rho R_z`, `2 rho R_z`, `rho Q_z`, `-V_z`, `U_z`",
     "centered `d_z`", "analytic quotient of centered `d_z`"),
    ("vector divergence", "OddLinear radial component plus even axial component",
     "`2 A + d_z V^z` on axis", "coefficient value / centered derivatives",
     "`d_rho V^rho+d_z V^z+V^rho/rho`"),
    ("active rho/z first, second, mixed, advection, dissipation",
     "parity-extended scalar/vector/tensor", "ordinary Cartesian action",
     "none", "unchanged centered `Dx/Dxx/Dxy/Lx/Diss`"),
)


SOURCE_MARKERS = (
    "RegularCoefficientDerivative", "EvenCoefficientDerivative",
    "OddCoefficientDerivative", "QuadraticCoefficientDerivative",
    "QuadraticDifferenceCoefficientDerivative", "VectorSecondSuppressed",
    "VectorMixedSuppressed", "TensorSecondSuppressed", "TensorMixedSuppressed",
    "return NGHOST - 1;", "-3.0 / 4.0 * samples[0]",
    "1.0 / 30.0 * samples[3]",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def fraction(value: Fraction) -> str:
    return (str(value.numerator) if value.denominator == 1 else
            f"{value.numerator}/{value.denominator}")


def verify_rows() -> None:
    for nghost, targets in ROWS.items():
        require(set(targets) == set(range(nghost - 1)),
                f"NGHOST={nghost}: target inventory changed")
        sample_s = [Fraction((2 * point + 1) ** 2, 4) for point in range(nghost)]
        for target, weights in targets.items():
            require(len(weights) == nghost, f"NGHOST={nghost}: row width changed")
            target_s = Fraction((2 * target + 1) ** 2, 4)
            for degree in range(nghost):
                observed = sum(weight * value ** degree
                               for weight, value in zip(weights, sample_s, strict=True))
                expected = (Fraction(0) if degree == 0 else
                            degree * target_s ** (degree - 1))
                require(observed == expected,
                        f"NGHOST={nghost} target={target} degree={degree} is inexact")


def render() -> str:
    lines = [
        "# Generated half-plane SO(2) operator inventory", "",
        "This file is generated by `tst/unit/z4c/generate_z4c_cartoon_operator_table.py`.",
        "Do not edit it by hand. Here `s=rho^2`; `D_s` is the fixed rational",
        "regular-coefficient derivative below. Direct rows apply only to target",
        "layers `0,...,NGHOST-2`, where the ordinary centered radial stencil crosses",
        "the axis. The first wholly active row and all outer rows use the algebraically",
        "equivalent bulk SO(2) identity with centered radial derivatives.", "",
        "## Exact regular-coefficient rows", "",
        "For `rho_l=(l+1/2)h`,",
        "`D_s F(s_t)=h^-2 sum_l w[t,l] F(s_l)`. The rows reproduce every",
        "polynomial in `s` through degree `NGHOST-1` exactly.", "",
        "| order | NGHOST | target layer | positive half-cell support | weights |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for nghost, targets in ROWS.items():
        order = 2 * (nghost - 1)
        support = ", ".join(f"{point + 0.5:g}h" for point in range(nghost))
        for target, weights in targets.items():
            lines.append(f"| O{order} | {nghost} | {target} | `{support}` | "
                         f"`[{', '.join(fraction(value) for value in weights)}]` |")
    lines += ["", "## Provider branch inventory", "",
              "The tensor representation is `T_rhorho=P+sQ`, `T_yy=P`,",
              "`T_rhoy=sR`, `T_rhoz=rho U`, `T_yz=rho V`, and `T_zz=W`.", "",
              "| provider branch | regularity class | regular continuum form | axis-row primitive | implemented direct form |",
              "| --- | --- | --- | --- | --- |"]
    lines.extend(f"| {branch} | {regularity} | {continuum} | {primitive} | {direct} |"
                 for branch, regularity, continuum, primitive, direct in OPERATORS)
    lines += ["", "All fixed rows use exactly `NGHOST` positive half-cell samples, require no",
              "runtime solve or allocation, and have formal O2/O4/O6 accuracy for",
              "`NGHOST=2/3/4`. `Q=(T_rhorho-T_yy)/rho^2` and",
              "`R=T_rhoy/rho^2` are paired regularity coefficients, not independent fields.",
              "The diagnostic-axis branches use analytic limits and are not active evolution",
              "cell closures because no active cell lies at `rho=0`.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    root = args.source_dir.resolve()
    source = (root / "src/z4c/cartoon_derivatives.hpp").read_text(encoding="utf-8")
    for marker in SOURCE_MARKERS:
        require(marker in source, f"production provider lacks inventory marker {marker!r}")
    verify_rows()
    destination = root / "docs/z4c_cartoon_half_plane_operator_table.md"
    expected = render()
    if args.write:
        destination.write_text(expected, encoding="utf-8")
    else:
        require(destination.is_file(), "generated operator table is missing")
        require(destination.read_text(encoding="utf-8") == expected,
                "generated operator table is stale")
    print("half-plane SO(2) direct-operator inventory verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
