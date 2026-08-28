#!/usr/bin/env python3
"""Generate exact radial-q reference geometry from the production coframe.

This generator deliberately starts from the coframe rather than copying the
generic ReferenceGeometry implementation.  It differentiates a compact radial
two-jet algebra, constructs the Levi-Civita and Cartan objects, and applies
deterministic common-subexpression elimination.  The generated header is an
independent oracle and the source for later bounded production component groups.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import sympy as sp


Index = Tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class Jet2:
    value: sp.Expr
    first: Tuple[sp.Expr, ...]
    second: Tuple[Tuple[sp.Expr, ...], ...]

    @staticmethod
    def constant(value: sp.Expr | int | float) -> "Jet2":
        scalar = sp.sympify(value)
        return Jet2(scalar, (sp.S.Zero,) * 4, ((sp.S.Zero,) * 4,) * 4)

    def __add__(self, other: "Jet2 | sp.Expr | int | float") -> "Jet2":
        right = as_jet(other)
        return Jet2(
            self.value + right.value,
            tuple(self.first[p] + right.first[p] for p in range(4)),
            tuple(tuple(self.second[p][q] + right.second[p][q]
                        for q in range(4)) for p in range(4)),
        )

    __radd__ = __add__

    def __neg__(self) -> "Jet2":
        return Jet2(-self.value, tuple(-v for v in self.first),
                    tuple(tuple(-v for v in row) for row in self.second))

    def __sub__(self, other: "Jet2 | sp.Expr | int | float") -> "Jet2":
        return self + (-as_jet(other))

    def __rsub__(self, other: "Jet2 | sp.Expr | int | float") -> "Jet2":
        return as_jet(other) - self

    def __mul__(self, other: "Jet2 | sp.Expr | int | float") -> "Jet2":
        right = as_jet(other)
        return Jet2(
            self.value * right.value,
            tuple(self.first[p] * right.value
                  + self.value * right.first[p] for p in range(4)),
            tuple(tuple(
                self.second[p][q] * right.value
                + self.value * right.second[p][q]
                + self.first[p] * right.first[q]
                + self.first[q] * right.first[p]
                for q in range(4)) for p in range(4)),
        )

    __rmul__ = __mul__

    def reciprocal(self) -> "Jet2":
        inverse = 1 / self.value
        derivative = -inverse**2
        second_derivative = 2 * inverse**3
        return Jet2(
            inverse,
            tuple(derivative * self.first[p] for p in range(4)),
            tuple(tuple(derivative * self.second[p][q]
                        + second_derivative * self.first[p] * self.first[q]
                        for q in range(4)) for p in range(4)),
        )

    def __truediv__(self, other: "Jet2 | sp.Expr | int | float") -> "Jet2":
        return self * as_jet(other).reciprocal()

    def __rtruediv__(self, other: "Jet2 | sp.Expr | int | float") -> "Jet2":
        return as_jet(other) * self.reciprocal()


def as_jet(value: Jet2 | sp.Expr | int | float) -> Jet2:
    return value if isinstance(value, Jet2) else Jet2.constant(value)


def radial_jet(value: sp.Expr, dt: sp.Expr, dr: sp.Expr,
               dtt: sp.Expr, dtr: sp.Expr, drr: sp.Expr,
               displacement: Sequence[sp.Symbol], radius: sp.Symbol) -> Jet2:
    normal = [coordinate / radius for coordinate in displacement]
    first = [dt] + [dr * component for component in normal]
    second: List[List[sp.Expr]] = [[sp.S.Zero] * 4 for _ in range(4)]
    second[0][0] = dtt
    for i in range(3):
        second[0][i + 1] = second[i + 1][0] = dtr * normal[i]
        for j in range(3):
            transverse = (sp.S.One if i == j else sp.S.Zero) - normal[i] * normal[j]
            second[i + 1][j + 1] = (
                drr * normal[i] * normal[j] + dr * transverse / radius
            )
    return Jet2(value, tuple(first), tuple(tuple(row) for row in second))


def coordinate_jet(value: sp.Symbol, direction: int) -> Jet2:
    first = [sp.S.Zero] * 4
    first[direction] = sp.S.One
    return Jet2(value, tuple(first), ((sp.S.Zero,) * 4,) * 4)


def all_indices(rank: int) -> Iterable[Index]:
    return itertools.product(range(4), repeat=rank)


def build_geometry() -> Dict[str, Dict[Index, sp.Expr]]:
    radius = sp.Symbol("radius", positive=True)
    displacement = sp.symbols("X0:3", real=True)
    alpha_symbols = sp.symbols("alpha alpha_r alpha_rr")
    l_symbols = sp.symbols("L L_t L_r L_tt L_tr L_rr")
    b_symbols = sp.symbols("B B_r B_rr")
    alpha = radial_jet(alpha_symbols[0], 0, alpha_symbols[1], 0, 0,
                       alpha_symbols[2], displacement, radius)
    scale = radial_jet(*l_symbols, displacement, radius)
    shift_b = radial_jet(b_symbols[0], 0, b_symbols[1], 0, 0,
                         b_symbols[2], displacement, radius)
    coordinates = [coordinate_jet(displacement[i], i + 1) for i in range(3)]

    coframe = {index: Jet2.constant(0) for index in all_indices(2)}
    frame = {index: Jet2.constant(0) for index in all_indices(2)}
    coframe[0, 0] = alpha
    frame[0, 0] = 1 / alpha
    for i in range(3):
        coframe[i + 1, 0] = scale * shift_b * coordinates[i]
        coframe[i + 1, i + 1] = scale
        frame[0, i + 1] = -shift_b * coordinates[i] / alpha
        frame[i + 1, i + 1] = 1 / scale

    metric: Dict[Index, Jet2] = {}
    inverse_metric: Dict[Index, Jet2] = {}
    for a, b in all_indices(2):
        metric[a, b] = -coframe[0, a] * coframe[0, b]
        inverse_metric[a, b] = -frame[0, a] * frame[0, b]
        for capital in range(1, 4):
            metric[a, b] += coframe[capital, a] * coframe[capital, b]
            inverse_metric[a, b] += frame[capital, a] * frame[capital, b]

    christoffel: Dict[Index, sp.Expr] = {}
    d_christoffel: Dict[Index, sp.Expr] = {}
    for a, b, c in all_indices(3):
        christoffel[a, b, c] = sum(
            inverse_metric[a, ell].value * (
                metric[ell, c].first[b] + metric[ell, b].first[c]
                - metric[b, c].first[ell]) / 2
            for ell in range(4)
        )
        for p in range(4):
            d_christoffel[p, a, b, c] = sum(
                inverse_metric[a, ell].first[p] * (
                    metric[ell, c].first[b] + metric[ell, b].first[c]
                    - metric[b, c].first[ell]) / 2
                + inverse_metric[a, ell].value * (
                    metric[ell, c].second[p][b]
                    + metric[ell, b].second[p][c]
                    - metric[b, c].second[p][ell]) / 2
                for ell in range(4)
            )

    raw_spin: Dict[Index, sp.Expr] = {}
    d_raw_spin: Dict[Index, Tuple[sp.Expr, ...]] = {}
    for capital_a, capital_b, capital_c in all_indices(3):
        value = sp.S.Zero
        derivative = [sp.S.Zero] * 4
        for a, c in all_indices(2):
            covariant = frame[capital_b, a].first[c] + sum(
                christoffel[a, c, d] * frame[capital_b, d].value
                for d in range(4)
            )
            value += (coframe[capital_a, a].value
                      * frame[capital_c, c].value * covariant)
            for p in range(4):
                d_covariant = frame[capital_b, a].second[p][c] + sum(
                    d_christoffel[p, a, c, d] * frame[capital_b, d].value
                    + christoffel[a, c, d] * frame[capital_b, d].first[p]
                    for d in range(4)
                )
                derivative[p] += (
                    (coframe[capital_a, a].first[p]
                     * frame[capital_c, c].value
                     + coframe[capital_a, a].value
                     * frame[capital_c, c].first[p]) * covariant
                    + coframe[capital_a, a].value
                    * frame[capital_c, c].value * d_covariant
                )
        raw_spin[capital_a, capital_b, capital_c] = value
        d_raw_spin[capital_a, capital_b, capital_c] = tuple(derivative)

    spin: Dict[Index, sp.Expr] = {}
    spin_derivative: Dict[Index, sp.Expr] = {}
    for capital_a, capital_b, capital_c in all_indices(3):
        eta_product = (-1 if capital_a == 0 else 1) * (
            -1 if capital_b == 0 else 1)
        spin[capital_a, capital_b, capital_c] = (
            raw_spin[capital_a, capital_b, capital_c]
            - eta_product * raw_spin[capital_b, capital_a, capital_c]) / 2
        for capital_d in range(4):
            spin_derivative[capital_d, capital_a, capital_b, capital_c] = sum(
                frame[capital_d, p].value * (
                    d_raw_spin[capital_a, capital_b, capital_c][p]
                    - eta_product
                    * d_raw_spin[capital_b, capital_a, capital_c][p]) / 2
                for p in range(4)
            )

    structure4: Dict[Index, sp.Expr] = {}
    for capital_e, capital_c, capital_d in all_indices(3):
        structure4[capital_e, capital_c, capital_d] = sum(
            coframe[capital_e, a].value * (
                frame[capital_c, p].value * frame[capital_d, a].first[p]
                - frame[capital_d, p].value * frame[capital_c, a].first[p])
            for a, p in all_indices(2)
        )
    spatial_frame = {(capital_i, i): (
        frame[capital_i + 1, i + 1].value if capital_i == i else sp.S.Zero)
        for capital_i in range(3) for i in range(3)}
    spatial_coframe = {(capital_i, i): (
        coframe[capital_i + 1, i + 1].value if capital_i == i else sp.S.Zero)
        for capital_i in range(3) for i in range(3)}
    dt_spatial_frame = {(capital_i, i): (
        frame[capital_i + 1, i + 1].first[0]
        if capital_i == i else sp.S.Zero)
        for capital_i in range(3) for i in range(3)}
    structure = {}
    for capital_i, capital_j, capital_k in itertools.product(range(3), repeat=3):
        structure[capital_i, capital_j, capital_k] = (
            frame[capital_i + 1, capital_i + 1].first[capital_i + 1]
            if capital_j == capital_k else sp.S.Zero)
        structure[capital_i, capital_j, capital_k] -= (
            frame[capital_j + 1, capital_j + 1].first[capital_j + 1]
            if capital_i == capital_k else sp.S.Zero)
    riemann: Dict[Index, sp.Expr] = {}
    for capital_a, capital_b, capital_c, capital_d in all_indices(4):
        riemann[capital_a, capital_b, capital_c, capital_d] = (
            spin_derivative[capital_c, capital_a, capital_b, capital_d]
            - spin_derivative[capital_d, capital_a, capital_b, capital_c]
            + sum(
                spin[capital_a, capital_e, capital_c]
                * spin[capital_e, capital_b, capital_d]
                - spin[capital_a, capital_e, capital_d]
                * spin[capital_e, capital_b, capital_c]
                - structure4[capital_e, capital_c, capital_d]
                * spin[capital_a, capital_b, capital_e]
                for capital_e in range(4)
            )
        )
    ricci = {(capital_b, capital_d): sum(
        riemann[capital_a, capital_b, capital_a, capital_d]
        for capital_a in range(4)) for capital_b, capital_d in all_indices(2)}

    fields: Dict[str, Dict[Index, sp.Expr]] = {
        "metric": {index: value.value for index, value in metric.items()},
        "inverse_metric": {
            index: value.value for index, value in inverse_metric.items()},
        "d_metric": {(p,) + index: value.first[p]
                     for index, value in metric.items() for p in range(4)},
        "dd_metric": {(p, q) + index: value.second[p][q]
                      for index, value in metric.items()
                      for p in range(4) for q in range(4)},
        "coframe": {index: value.value for index, value in coframe.items()},
        "frame": {index: value.value for index, value in frame.items()},
        "d_frame": {(p,) + index: value.first[p]
                    for index, value in frame.items() for p in range(4)},
        "dd_frame": {(p, q) + index: value.second[p][q]
                     for index, value in frame.items()
                     for p in range(4) for q in range(4)},
        "christoffel": christoffel,
        "d_christoffel": d_christoffel,
        "spatial_frame": spatial_frame,
        "spatial_coframe": spatial_coframe,
        "dt_spatial_frame": dt_spatial_frame,
        "structure": structure,
        "spin": spin,
        "spin_derivative": spin_derivative,
        "structure4": structure4,
        "riemann_frame": riemann,
        "ricci_frame": ricci,
    }
    return fields


def cpp(expression: sp.Expr) -> str:
    return sp.ccode(expression, standard="C99").replace("pow(", "Kokkos::pow(")


def emit(output: Path) -> None:
    fields = build_geometry()
    names: List[Tuple[str, Index]] = []
    expressions: List[sp.Expr] = []
    for field, values in fields.items():
        for index, expression in values.items():
            names.append((field, index))
            expressions.append(expression)
    replacements, reduced = sp.cse(
        expressions, symbols=sp.numbered_symbols("ref_cse_"), order="canonical")
    generator_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    lines = [
        "// Generated by scripts/ref_gh/generate_analytic_radial_q_geometry.py.",
        "// Do not edit by hand.",
        f"// generator_sha256={generator_sha}",
        f"// sympy_version={sp.__version__}",
        "#ifndef REF_GH_GENERATED_ANALYTIC_RADIAL_Q_GEOMETRY_HPP_",
        "#define REF_GH_GENERATED_ANALYTIC_RADIAL_Q_GEOMETRY_HPP_",
        "",
        "#include \"athena.hpp\"",
        "#include \"ref_gh/reference_analytic_radial_q.hpp\"",
        "#include \"ref_gh/reference_geometry.hpp\"",
        "",
        "namespace ref_gh {",
        "",
        "KOKKOS_INLINE_FUNCTION",
        "void PopulateGeneratedAnalyticRadialQGeometry(",
        "    const AnalyticRadialScalar &alpha_jet,",
        "    const AnalyticRadialScalar &l_jet,",
        "    const AnalyticRadialScalar &b_jet, const Real displacement[3],",
        "    const Real radius, ReferenceGeometry &reference) {",
        "  const Real X0 = displacement[0];",
        "  const Real X1 = displacement[1];",
        "  const Real X2 = displacement[2];",
        "  const Real alpha = alpha_jet.value;",
        "  const Real alpha_r = alpha_jet.dr;",
        "  const Real alpha_rr = alpha_jet.drr;",
        "  const Real L = l_jet.value;",
        "  const Real L_t = l_jet.dt;",
        "  const Real L_r = l_jet.dr;",
        "  const Real L_tt = l_jet.dtt;",
        "  const Real L_tr = l_jet.dtr;",
        "  const Real L_rr = l_jet.drr;",
        "  const Real B = b_jet.value;",
        "  const Real B_r = b_jet.dr;",
        "  const Real B_rr = b_jet.drr;",
        "  ZeroReferenceGeometry(reference);",
    ]
    for symbol, expression in replacements:
        lines.append(f"  const Real {symbol} = {cpp(expression)};")
    for (field, index), expression in zip(names, reduced):
        indices = "".join(f"[{value}]" for value in index)
        lines.append(f"  reference.{field}{indices} = {cpp(expression)};")
    lines.extend(["}", "", "}  // namespace ref_gh", "",
                  "#endif  // REF_GH_GENERATED_ANALYTIC_RADIAL_Q_GEOMETRY_HPP_", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("src/ref_gh/generated/analytic_radial_q_geometry.hpp"))
    arguments = parser.parse_args()
    emit(arguments.output)


if __name__ == "__main__":
    main()
