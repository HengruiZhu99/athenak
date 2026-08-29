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
    # mixed_third[i][q] = d_t d_{i+1} d_q, the only third slice needed by
    # the moving-reference gauge baseline.
    mixed_third: Tuple[Tuple[sp.Expr, ...], ...]

    @staticmethod
    def constant(value: sp.Expr | int | float) -> "Jet2":
        scalar = sp.sympify(value)
        return Jet2(scalar, (sp.S.Zero,) * 4, ((sp.S.Zero,) * 4,) * 4,
                    ((sp.S.Zero,) * 4,) * 3)

    def __add__(self, other: "Jet2 | sp.Expr | int | float") -> "Jet2":
        right = as_jet(other)
        return Jet2(
            self.value + right.value,
            tuple(self.first[p] + right.first[p] for p in range(4)),
            tuple(tuple(self.second[p][q] + right.second[p][q]
                        for q in range(4)) for p in range(4)),
            tuple(tuple(self.mixed_third[i][q] + right.mixed_third[i][q]
                        for q in range(4)) for i in range(3)),
        )

    __radd__ = __add__

    def __neg__(self) -> "Jet2":
        return Jet2(
            -self.value, tuple(-v for v in self.first),
            tuple(tuple(-v for v in row) for row in self.second),
            tuple(tuple(-v for v in row) for row in self.mixed_third))

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
            tuple(tuple(
                self.mixed_third[i][q] * right.value
                + self.second[0][i + 1] * right.first[q]
                + self.second[0][q] * right.first[i + 1]
                + self.second[i + 1][q] * right.first[0]
                + self.first[0] * right.second[i + 1][q]
                + self.first[i + 1] * right.second[0][q]
                + self.first[q] * right.second[0][i + 1]
                + self.value * right.mixed_third[i][q]
                for q in range(4)) for i in range(3)),
        )

    __rmul__ = __mul__

    def reciprocal(self) -> "Jet2":
        inverse = 1 / self.value
        derivative = -inverse**2
        second_derivative = 2 * inverse**3
        third_derivative = -6 * inverse**4
        return Jet2(
            inverse,
            tuple(derivative * self.first[p] for p in range(4)),
            tuple(tuple(derivative * self.second[p][q]
                        + second_derivative * self.first[p] * self.first[q]
                        for q in range(4)) for p in range(4)),
            tuple(tuple(
                derivative * self.mixed_third[i][q]
                + second_derivative * (
                    self.second[0][i + 1] * self.first[q]
                    + self.second[0][q] * self.first[i + 1]
                    + self.second[i + 1][q] * self.first[0])
                + third_derivative * self.first[0]
                  * self.first[i + 1] * self.first[q]
                for q in range(4)) for i in range(3)),
        )

    def __truediv__(self, other: "Jet2 | sp.Expr | int | float") -> "Jet2":
        return self * as_jet(other).reciprocal()

    def __rtruediv__(self, other: "Jet2 | sp.Expr | int | float") -> "Jet2":
        return as_jet(other) * self.reciprocal()


def as_jet(value: Jet2 | sp.Expr | int | float) -> Jet2:
    return value if isinstance(value, Jet2) else Jet2.constant(value)


def radial_jet(value: sp.Expr, dt: sp.Expr, dr: sp.Expr,
               dtt: sp.Expr, dtr: sp.Expr, drr: sp.Expr,
               dttr: sp.Expr, dtrr: sp.Expr,
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
    mixed: List[List[sp.Expr]] = [[sp.S.Zero] * 4 for _ in range(3)]
    for i in range(3):
        mixed[i][0] = dttr * normal[i]
        for j in range(3):
            transverse = (sp.S.One if i == j else sp.S.Zero) - normal[i] * normal[j]
            mixed[i][j + 1] = (
                dtrr * normal[i] * normal[j] + dtr * transverse / radius
            )
    return Jet2(value, tuple(first), tuple(tuple(row) for row in second),
                tuple(tuple(row) for row in mixed))


def coordinate_jet(value: sp.Symbol, direction: int) -> Jet2:
    first = [sp.S.Zero] * 4
    first[direction] = sp.S.One
    return Jet2(value, tuple(first), ((sp.S.Zero,) * 4,) * 4,
                ((sp.S.Zero,) * 4,) * 3)


def all_indices(rank: int) -> Iterable[Index]:
    return itertools.product(range(4), repeat=rank)


def build_geometry() -> Dict[str, Dict[Index, sp.Expr]]:
    radius = sp.Symbol("radius", positive=True)
    displacement = sp.symbols("X0:3", real=True)
    alpha_symbols = sp.symbols("alpha alpha_r alpha_rr")
    l_symbols = sp.symbols("L L_t L_r L_tt L_tr L_rr L_ttr L_trr")
    b_symbols = sp.symbols("B B_r B_rr")
    alpha = radial_jet(alpha_symbols[0], 0, alpha_symbols[1], 0, 0,
                       alpha_symbols[2], 0, 0, displacement, radius)
    scale = radial_jet(*l_symbols, displacement, radius)
    shift_b = radial_jet(b_symbols[0], 0, b_symbols[1], 0, 0,
                         b_symbols[2], 0, 0, displacement, radius)
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

    # Ordinary reference GH gauge baseline and its exact moving-frame time
    # derivative.  These expressions use the mixed third jet only through
    # d_t d_i Gamma, matching the production gauge-subtraction mathematics.
    h_upper: Dict[Index, sp.Expr] = {}
    d_h_upper: Dict[Index, sp.Expr] = {}
    for a in range(4):
        h_upper[a,] = -sum(
            inverse_metric[b, c].value*christoffel[a, b, c]
            for b, c in all_indices(2))
        for p in range(4):
            d_h_upper[p, a] = -sum(
                inverse_metric[b, c].first[p]*christoffel[a, b, c]
                + inverse_metric[b, c].value*d_christoffel[p, a, b, c]
                for b, c in all_indices(2))
    h_lower: Dict[Index, sp.Expr] = {}
    d_h_lower: Dict[Index, sp.Expr] = {}
    for a in range(4):
        h_lower[a,] = sum(
            metric[a, b].value*h_upper[b,] for b in range(4))
        for p in range(4):
            d_h_lower[p, a] = sum(
                metric[a, b].first[p]*h_upper[b,]
                + metric[a, b].value*d_h_upper[p, b]
                for b in range(4))
    gauge_hhat: Dict[Index, sp.Expr] = {}
    gauge_d_hhat: Dict[Index, sp.Expr] = {}
    gauge_reference_k: Dict[Index, sp.Expr] = {}
    frame_motion: Dict[Index, sp.Expr] = {}
    dt_frame_motion: Dict[Index, sp.Expr] = {}
    for capital_a in range(4):
        gauge_hhat[capital_a,] = sum(
            frame[capital_a, a].value*h_lower[a,] for a in range(4))
        for p in range(4):
            gauge_d_hhat[p, capital_a] = sum(
                frame[capital_a, a].first[p]*h_lower[a,]
                + frame[capital_a, a].value*d_h_lower[p, a]
                for a in range(4))
        # Kref_iA = partial_i Href_A - Omega_Ai^B Href_B
        #          = e_A^a partial_i Href_a.
        # Emit the second expression directly.  Forming the first expression
        # in production would subtract independently reconstructed singular
        # reference terms at the puncture.
        for spatial in range(3):
            gauge_reference_k[spatial, capital_a] = sum(
                frame[capital_a, a].value*d_h_lower[spatial + 1, a]
                for a in range(4))
        for lam in range(4):
            for capital_b in range(4):
                frame_motion[capital_a, lam, capital_b] = sum(
                    frame[capital_a, a].first[lam]
                    * coframe[capital_b, a].value for a in range(4))
                dt_frame_motion[capital_a, lam, capital_b] = sum(
                    frame[capital_a, a].second[0][lam]
                    * coframe[capital_b, a].value
                    + frame[capital_a, a].first[lam]
                    * coframe[capital_b, a].first[0]
                    for a in range(4))
    reference_shift = [shift_b.value*displacement[i] for i in range(3)]
    gauge_theta: Dict[Index, sp.Expr] = {}
    for capital_a in range(4):
        value = -sum(reference_shift[i]*gauge_d_hhat[i + 1, capital_a]
                     for i in range(3))
        value -= sum(
            (frame_motion[capital_a, 0, capital_b]
             - sum(reference_shift[i]
                   * frame_motion[capital_a, i + 1, capital_b]
                   for i in range(3))) * gauge_hhat[capital_b,]
            for capital_b in range(4))
        gauge_theta[capital_a,] = value

    dt_di_h_upper: Dict[Index, sp.Expr] = {}
    for spatial in range(3):
        s = spatial + 1
        for a in range(4):
            value = sp.S.Zero
            for b, c in all_indices(2):
                dt_di_christoffel = sp.S.Zero
                for ell in range(4):
                    first = (
                        metric[ell, c].first[b]
                        + metric[ell, b].first[c]
                        - metric[b, c].first[ell]) / 2
                    first_t = (
                        metric[ell, c].second[0][b]
                        + metric[ell, b].second[0][c]
                        - metric[b, c].second[0][ell]) / 2
                    first_i = (
                        metric[ell, c].second[s][b]
                        + metric[ell, b].second[s][c]
                        - metric[b, c].second[s][ell]) / 2
                    first_ti = (
                        metric[ell, c].mixed_third[spatial][b]
                        + metric[ell, b].mixed_third[spatial][c]
                        - metric[b, c].mixed_third[spatial][ell]) / 2
                    dt_di_christoffel += (
                        inverse_metric[a, ell].second[0][s]*first
                        + inverse_metric[a, ell].first[s]*first_t
                        + inverse_metric[a, ell].first[0]*first_i
                        + inverse_metric[a, ell].value*first_ti)
                value -= (
                    inverse_metric[b, c].second[0][s]
                      * christoffel[a, b, c]
                    + inverse_metric[b, c].first[s]
                      * d_christoffel[0, a, b, c]
                    + inverse_metric[b, c].first[0]
                      * d_christoffel[s, a, b, c]
                    + inverse_metric[b, c].value*dt_di_christoffel)
            dt_di_h_upper[spatial, a] = value
    dt_di_h_lower: Dict[Index, sp.Expr] = {}
    for spatial in range(3):
        s = spatial + 1
        for a in range(4):
            dt_di_h_lower[spatial, a] = sum(
                metric[a, b].second[0][s]*h_upper[b,]
                + metric[a, b].first[s]*d_h_upper[0, b]
                + metric[a, b].first[0]*d_h_upper[s, b]
                + metric[a, b].value*dt_di_h_upper[spatial, b]
                for b in range(4))
    dt_di_hhat: Dict[Index, sp.Expr] = {}
    for spatial in range(3):
        s = spatial + 1
        for capital_a in range(4):
            dt_di_hhat[spatial, capital_a] = sum(
                frame[capital_a, a].second[0][s]*h_lower[a,]
                + frame[capital_a, a].first[s]*d_h_lower[0, a]
                + frame[capital_a, a].first[0]*d_h_lower[s, a]
                + frame[capital_a, a].value*dt_di_h_lower[spatial, a]
                for a in range(4))
    gauge_dt_theta: Dict[Index, sp.Expr] = {}
    # alpha and B are stationary, so the exact coordinate reference shift has
    # zero time derivative even though L and the frame evolve.
    for capital_a in range(4):
        value = -sum(reference_shift[i]*dt_di_hhat[i, capital_a]
                     for i in range(3))
        for capital_b in range(4):
            motion = frame_motion[capital_a, 0, capital_b] - sum(
                reference_shift[i]
                * frame_motion[capital_a, i + 1, capital_b]
                for i in range(3))
            dt_motion = dt_frame_motion[capital_a, 0, capital_b] - sum(
                reference_shift[i]
                * dt_frame_motion[capital_a, i + 1, capital_b]
                for i in range(3))
            value -= (dt_motion*gauge_hhat[capital_b,]
                      + motion*gauge_d_hhat[0, capital_b])
        gauge_dt_theta[capital_a,] = value

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
        "gauge_hhat": gauge_hhat,
        "gauge_d_hhat": gauge_d_hhat,
        "gauge_reference_k": gauge_reference_k,
        "gauge_theta": gauge_theta,
        "gauge_dt_theta": gauge_dt_theta,
        "frame_motion": frame_motion,
        "dt_frame_motion": dt_frame_motion,
    }
    return fields


def cpp(expression: sp.Expr) -> str:
    return sp.ccode(expression, standard="C99").replace("pow(", "Kokkos::pow(")


def stable_oracle_primitives() -> List[str]:
    """Emit an oracle-only Cartesian two-jet geometry front end.

    Flattening the primitive metric two-jet with symbolic CSE changes the
    binary64 association of strongly cancelling near-puncture expressions.
    The independent oracle therefore uses a small generated two-jet type and
    preserves the algebraic staging of the mature generic implementation.
    This deliberately expensive type is local to this generated oracle header;
    it is forbidden in the compact production backend.
    """
    return r"""
namespace analytic_radial_q_oracle_detail {

struct CartesianTwoJet {
  Real value;
  Real d[4];      // NOLINT(runtime/arrays)
  Real dd[4][4];  // NOLINT(runtime/arrays)
};

KOKKOS_INLINE_FUNCTION
CartesianTwoJet Constant(const Real value) {
  CartesianTwoJet result;
  result.value = value;
  for (int p = 0; p < 4; ++p) {
    result.d[p] = 0.0;
    for (int q = 0; q < 4; ++q) result.dd[p][q] = 0.0;
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
CartesianTwoJet Coordinate(const Real value, const int direction) {
  CartesianTwoJet result = Constant(value);
  result.d[direction] = 1.0;
  return result;
}

KOKKOS_INLINE_FUNCTION
CartesianTwoJet Radial(const AnalyticRadialScalar &radial,
                       const Real displacement[3], const Real radius) {
  CartesianTwoJet result;
  result.value = radial.value;
  for (int p = 0; p < 4; ++p) {
    result.d[p] = radial.D(displacement, radius, p);
    for (int q = 0; q < 4; ++q) {
      result.dd[p][q] = radial.DD(displacement, radius, p, q);
    }
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
CartesianTwoJet Add(const CartesianTwoJet &left,
                    const CartesianTwoJet &right) {
  CartesianTwoJet result;
  result.value = left.value + right.value;
  for (int p = 0; p < 4; ++p) {
    result.d[p] = left.d[p] + right.d[p];
    for (int q = 0; q < 4; ++q) {
      result.dd[p][q] = left.dd[p][q] + right.dd[p][q];
    }
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
CartesianTwoJet Negate(const CartesianTwoJet &value) {
  CartesianTwoJet result;
  result.value = -value.value;
  for (int p = 0; p < 4; ++p) {
    result.d[p] = -value.d[p];
    for (int q = 0; q < 4; ++q) result.dd[p][q] = -value.dd[p][q];
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
CartesianTwoJet Multiply(const CartesianTwoJet &left,
                         const CartesianTwoJet &right) {
  CartesianTwoJet result;
  result.value = left.value*right.value;
  for (int p = 0; p < 4; ++p) {
    result.d[p] = left.d[p]*right.value + left.value*right.d[p];
    for (int q = 0; q < 4; ++q) {
      result.dd[p][q] = left.dd[p][q]*right.value
                        + left.value*right.dd[p][q]
                        + left.d[p]*right.d[q]
                        + left.d[q]*right.d[p];
    }
  }
  return result;
}

KOKKOS_INLINE_FUNCTION
CartesianTwoJet Reciprocal(const CartesianTwoJet &input) {
  const Real inverse = 1.0/input.value;
  const Real inverse2 = inverse*inverse;
  const Real first = -inverse2;
  const Real second = 2.0*inverse2*inverse;
  CartesianTwoJet result;
  result.value = inverse;
  for (int p = 0; p < 4; ++p) {
    result.d[p] = first*input.d[p];
    for (int q = 0; q < 4; ++q) {
      result.dd[p][q] = first*input.dd[p][q]
                        + second*input.d[p]*input.d[q];
    }
  }
  return result;
}

}  // namespace analytic_radial_q_oracle_detail
""".strip("\n").splitlines()


def stable_oracle_geometry_frontend() -> List[str]:
    """Emit the staged primitive geometry used before Cartan completion."""
    return r"""
  using analytic_radial_q_oracle_detail::Add;
  using analytic_radial_q_oracle_detail::CartesianTwoJet;
  using analytic_radial_q_oracle_detail::Constant;
  using analytic_radial_q_oracle_detail::Coordinate;
  using analytic_radial_q_oracle_detail::Multiply;
  using analytic_radial_q_oracle_detail::Negate;
  using analytic_radial_q_oracle_detail::Radial;
  using analytic_radial_q_oracle_detail::Reciprocal;
  const CartesianTwoJet alpha = Radial(alpha_jet, displacement, radius);
  const CartesianTwoJet psi2 = Radial(l_jet, displacement, radius);
  const CartesianTwoJet shift_q = Radial(b_jet, displacement, radius);
  const CartesianTwoJet inverse_alpha = Reciprocal(alpha);
  const CartesianTwoJet inverse_psi2 = Reciprocal(psi2);
  const CartesianTwoJet coordinates[3] = {
      Coordinate(displacement[0], 1), Coordinate(displacement[1], 2),
      Coordinate(displacement[2], 3)};
  CartesianTwoJet coframe[4][4];  // NOLINT(runtime/arrays)
  CartesianTwoJet frame[4][4];    // NOLINT(runtime/arrays)
  for (int A = 0; A < 4; ++A) {
    for (int a = 0; a < 4; ++a) {
      coframe[A][a] = Constant(0.0);
      frame[A][a] = Constant(0.0);
    }
  }
  coframe[0][0] = alpha;
  frame[0][0] = inverse_alpha;
  for (int I = 0; I < 3; ++I) {
    coframe[I + 1][0] = Multiply(Multiply(psi2, shift_q), coordinates[I]);
    coframe[I + 1][I + 1] = psi2;
    frame[0][I + 1] = Negate(
        Multiply(Multiply(shift_q, coordinates[I]), inverse_alpha));
    frame[I + 1][I + 1] = inverse_psi2;
  }
  CartesianTwoJet metric[4][4];          // NOLINT(runtime/arrays)
  CartesianTwoJet inverse_metric[4][4];  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      metric[a][b] = Negate(Multiply(coframe[0][a], coframe[0][b]));
      inverse_metric[a][b] = Negate(Multiply(frame[0][a], frame[0][b]));
      for (int I = 1; I < 4; ++I) {
        metric[a][b] = Add(
            metric[a][b], Multiply(coframe[I][a], coframe[I][b]));
        inverse_metric[a][b] = Add(
            inverse_metric[a][b], Multiply(frame[I][a], frame[I][b]));
      }
      reference.metric[a][b] = metric[a][b].value;
      reference.inverse_metric[a][b] = inverse_metric[a][b].value;
      reference.coframe[a][b] = coframe[a][b].value;
      reference.frame[a][b] = frame[a][b].value;
      for (int p = 0; p < 4; ++p) {
        reference.d_metric[p][a][b] = metric[a][b].d[p];
        reference.d_frame[p][a][b] = frame[a][b].d[p];
        for (int q = 0; q < 4; ++q) {
          reference.dd_metric[p][q][a][b] = metric[a][b].dd[p][q];
          reference.dd_frame[p][q][a][b] = frame[a][b].dd[p][q];
        }
      }
    }
  }
  for (int I = 0; I < 3; ++I) {
    reference.spatial_frame[I][I] = inverse_psi2.value;
    reference.spatial_coframe[I][I] = psi2.value;
    reference.dt_spatial_frame[I][I] = inverse_psi2.d[0];
    for (int J = 0; J < 3; ++J) {
      for (int K = 0; K < 3; ++K) {
        reference.structure[I][J][K] =
            ((J == K) ? inverse_psi2.d[I + 1] : 0.0)
            - ((I == K) ? inverse_psi2.d[J + 1] : 0.0);
      }
    }
  }
  Real first_kind[4][4][4];  // NOLINT(runtime/arrays)
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      for (int c = 0; c < 4; ++c) {
        first_kind[a][b][c] = 0.5*(reference.d_metric[b][a][c]
                                    + reference.d_metric[c][a][b]
                                    - reference.d_metric[a][b][c]);
      }
    }
  }
  for (int a = 0; a < 4; ++a) {
    for (int b = 0; b < 4; ++b) {
      for (int c = 0; c < 4; ++c) {
        for (int ell = 0; ell < 4; ++ell) {
          reference.christoffel[a][b][c] +=
              reference.inverse_metric[a][ell]*first_kind[ell][b][c];
        }
        for (int p = 0; p < 4; ++p) {
          for (int ell = 0; ell < 4; ++ell) {
            const Real d_first = 0.5*(reference.dd_metric[p][b][ell][c]
                                      + reference.dd_metric[p][c][ell][b]
                                      - reference.dd_metric[p][ell][b][c]);
            reference.d_christoffel[p][a][b][c] +=
                inverse_metric[a][ell].d[p]*first_kind[ell][b][c]
                + reference.inverse_metric[a][ell]*d_first;
          }
        }
      }
    }
  }
""".strip("\n").splitlines()


def stable_oracle_completion() -> List[str]:
    """Emit a staged binary64-stable Cartan completion for the full oracle.

    The fully flattened symbolic spin-derivative and curvature expressions are
    mathematically exact but lose digits at the smallest qualification radii.
    These generated loops consume the independently generated primitive
    frame/connection fields and preserve their natural contraction order.  This
    path remains oracle-only and is never a production implementation.
    """
    return r"""
  Real generated_d_coframe[4][4][4];  // NOLINT(runtime/arrays)
  for (int p = 0; p < 4; ++p) {
    for (int A = 0; A < 4; ++A) {
      for (int a = 0; a < 4; ++a) {
        generated_d_coframe[p][A][a] = 0.0;
        for (int B = 0; B < 4; ++B) {
          for (int b = 0; b < 4; ++b) {
            generated_d_coframe[p][A][a] -= reference.coframe[B][a]
                *reference.d_frame[p][B][b]*reference.coframe[A][b];
          }
        }
      }
    }
  }
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int C = 0; C < 4; ++C) {
        for (int a = 0; a < 4; ++a) {
          for (int c = 0; c < 4; ++c) {
            Real derivative = reference.d_frame[c][B][a];
            for (int d = 0; d < 4; ++d) {
              derivative += reference.christoffel[a][c][d]
                            *reference.frame[B][d];
            }
            reference.spin[A][B][C] += reference.coframe[A][a]
                *reference.frame[C][c]*derivative;
          }
        }
      }
    }
  }
  for (int A = 0; A < 4; ++A) {
    const Real eta_A = (A == 0) ? -1.0 : 1.0;
    for (int B = A; B < 4; ++B) {
      const Real eta_B = (B == 0) ? -1.0 : 1.0;
      for (int C = 0; C < 4; ++C) {
        const Real projected = 0.5*(eta_A*reference.spin[A][B][C]
                                    - eta_B*reference.spin[B][A][C]);
        reference.spin[A][B][C] = eta_A*projected;
        reference.spin[B][A][C] = -eta_B*projected;
      }
    }
  }
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int C = 0; C < 4; ++C) {
        for (int p = 0; p < 4; ++p) {
          Real coordinate_derivative = 0.0;
          for (int a = 0; a < 4; ++a) {
            for (int c = 0; c < 4; ++c) {
              Real frame_covariant_derivative = reference.d_frame[c][B][a];
              Real d_frame_covariant_derivative =
                  reference.dd_frame[p][c][B][a];
              for (int d = 0; d < 4; ++d) {
                frame_covariant_derivative +=
                    reference.christoffel[a][c][d]*reference.frame[B][d];
                d_frame_covariant_derivative +=
                    reference.d_christoffel[p][a][c][d]
                      *reference.frame[B][d]
                    + reference.christoffel[a][c][d]
                      *reference.d_frame[p][B][d];
              }
              coordinate_derivative +=
                  (generated_d_coframe[p][A][a]*reference.frame[C][c]
                   + reference.coframe[A][a]*reference.d_frame[p][C][c])
                    *frame_covariant_derivative
                  + reference.coframe[A][a]*reference.frame[C][c]
                    *d_frame_covariant_derivative;
            }
          }
          for (int D = 0; D < 4; ++D) {
            reference.spin_derivative[D][A][B][C] +=
                reference.frame[D][p]*coordinate_derivative;
          }
        }
      }
    }
  }
  for (int A = 0; A < 4; ++A) {
    const Real eta_A = (A == 0) ? -1.0 : 1.0;
    for (int B = A; B < 4; ++B) {
      const Real eta_B = (B == 0) ? -1.0 : 1.0;
      for (int C = 0; C < 4; ++C) {
        for (int D = 0; D < 4; ++D) {
          const Real projected = 0.5*(
              eta_A*reference.spin_derivative[D][A][B][C]
              - eta_B*reference.spin_derivative[D][B][A][C]);
          reference.spin_derivative[D][A][B][C] = eta_A*projected;
          reference.spin_derivative[D][B][A][C] = -eta_B*projected;
        }
      }
    }
  }
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int C = B; C < 4; ++C) {
        Real value = 0.0;
        for (int a = 0; a < 4; ++a) {
          for (int p = 0; p < 4; ++p) {
            value += reference.coframe[A][a]
                *(reference.frame[B][p]*reference.d_frame[p][C][a]
                  - reference.frame[C][p]*reference.d_frame[p][B][a]);
          }
        }
        reference.structure4[A][B][C] = value;
        reference.structure4[A][C][B] = -value;
      }
    }
  }
  for (int A = 0; A < 4; ++A) {
    for (int B = 0; B < 4; ++B) {
      for (int C = 0; C < 4; ++C) {
        for (int D = 0; D < 4; ++D) {
          Real value = reference.spin_derivative[C][A][B][D]
                       - reference.spin_derivative[D][A][B][C];
          for (int E = 0; E < 4; ++E) {
            value += reference.spin[A][E][C]*reference.spin[E][B][D]
                     - reference.spin[A][E][D]*reference.spin[E][B][C]
                     - reference.structure4[E][C][D]
                       *reference.spin[A][B][E];
          }
          reference.riemann_frame[A][B][C][D] = value;
        }
      }
    }
  }
  for (int B = 0; B < 4; ++B) {
    for (int D = 0; D < 4; ++D) {
      for (int A = 0; A < 4; ++A) {
        reference.ricci_frame[B][D] +=
            reference.riemann_frame[A][B][A][D];
      }
    }
  }
""".strip("\n").splitlines()


def emit(output: Path) -> None:
    generator_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    lines = [
        "// Generated by scripts/ref_gh/generate_analytic_radial_q_geometry.py.",
        "// Do not edit by hand.",
        "// ORACLE ONLY: forbidden in CalcRHS, q measurement, boundaries,",
        "// timestep calculation, and ordinary production reference updates.",
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
    ]
    lines.extend(stable_oracle_primitives())
    lines.extend([
        "",
        "KOKKOS_INLINE_FUNCTION",
        "void PopulateGeneratedAnalyticRadialQGeometry(",
        "    const AnalyticRadialScalar &alpha_jet,",
        "    const AnalyticRadialScalar &l_jet,",
        "    const AnalyticRadialScalar &b_jet, const Real displacement[3],",
        "    const Real radius, ReferenceGeometry &reference) {",
        "  ZeroReferenceGeometry(reference);",
    ])
    lines.extend(stable_oracle_geometry_frontend())
    lines.extend(stable_oracle_completion())
    lines.extend(["}", "", "}  // namespace ref_gh", "",
                  "#endif  // REF_GH_GENERATED_ANALYTIC_RADIAL_Q_GEOMETRY_HPP_", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def compact_scalar_declarations() -> List[str]:
    return [
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
        "  const Real L_ttr = l_jet.dttr;",
        "  const Real L_trr = l_jet.dtrr;",
        "  const Real B = b_jet.value;",
        "  const Real B_r = b_jet.dr;",
        "  const Real B_rr = b_jet.drr;",
    ]


def emit_component_switch(lines: List[str], function_name: str,
                          values: Dict[Index, sp.Expr]) -> None:
    lines.extend([
        "KOKKOS_INLINE_FUNCTION",
        f"Real {function_name}(",
        "    const AnalyticRadialScalar &alpha_jet,",
        "    const AnalyticRadialScalar &l_jet,",
        "    const AnalyticRadialScalar &b_jet, const Real displacement[3],",
        "    const Real radius, const int A, const int lambda,",
        "    const int capital_B) {",
    ])
    lines.extend(compact_scalar_declarations())
    lines.append("  switch (16*A + 4*lambda + capital_B) {")
    for index, expression in values.items():
        if expression == 0:
            continue
        capital_a, lam, capital_b = index
        packed = 16*capital_a + 4*lam + capital_b
        lines.append(f"    case {packed}: return {cpp(expression)};")
    lines.extend(["    default: return 0.0;", "  }", "}", ""])


def emit_gauge(output: Path) -> None:
    fields = build_geometry()
    generator_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    ordered_names = (
        "gauge_hhat", "gauge_d_hhat", "gauge_reference_k",
        "gauge_theta", "gauge_dt_theta")
    expressions: List[sp.Expr] = []
    targets: List[str] = []
    for field_name in ordered_names:
        for index, expression in fields[field_name].items():
            expressions.append(expression)
            if field_name == "gauge_hhat":
                targets.append(f"result.hhat[{index[0]}]")
            elif field_name == "gauge_d_hhat":
                targets.append(f"result.d_hhat[{index[0]}][{index[1]}]")
            elif field_name == "gauge_reference_k":
                targets.append(f"result.reference_k[{index[0]}][{index[1]}]")
            elif field_name == "gauge_theta":
                targets.append(f"result.theta[{index[0]}]")
            else:
                targets.append(f"result.dt_theta[{index[0]}]")
    # Eliminate exact rational cancellations before CSE.  Applying CSE to the
    # unsimplified tensor contractions preserves enormous near-puncture terms
    # that cancel only after evaluation and loses binary64 digits in dH and
    # dtTheta.  This algebraic pass changes neither the ansatz nor equations.
    compact_expressions = [sp.factor(sp.cancel(value))
                           for value in expressions]
    replacements, reduced = sp.cse(
        compact_expressions, symbols=sp.numbered_symbols("ref_gauge_cse_"),
        order="canonical")
    lines = [
        "// Generated by scripts/ref_gh/generate_analytic_radial_q_geometry.py.",
        "// Do not edit by hand.",
        "// Compact production contractions: this header never materializes",
        "// ReferenceGeometry, spin, spin-derivative, or Riemann arrays.",
        f"// generator_sha256={generator_sha}",
        f"// sympy_version={sp.__version__}",
        "#ifndef REF_GH_GENERATED_ANALYTIC_RADIAL_Q_GAUGE_HPP_",
        "#define REF_GH_GENERATED_ANALYTIC_RADIAL_Q_GAUGE_HPP_",
        "",
        "#include \"athena.hpp\"",
        "#include \"ref_gh/reference_analytic_radial_q.hpp\"",
        "",
        "namespace ref_gh {",
        "",
        "struct AnalyticRadialQGaugeBaseline {",
        "  Real hhat[4];        // NOLINT(runtime/arrays)",
        "  Real theta[4];       // NOLINT(runtime/arrays)",
        "  Real d_hhat[4][4];   // NOLINT(runtime/arrays)",
        "  Real reference_k[3][4];  // NOLINT(runtime/arrays)",
        "  Real dt_theta[4];    // NOLINT(runtime/arrays)",
        "  bool valid;",
        "};",
        "",
        "KOKKOS_INLINE_FUNCTION",
        "AnalyticRadialQGaugeBaseline PopulateGeneratedAnalyticRadialQGauge(",
        "    const AnalyticRadialScalar &alpha_jet,",
        "    const AnalyticRadialScalar &l_jet,",
        "    const AnalyticRadialScalar &b_jet, const Real displacement[3],",
        "    const Real radius) {",
        "  AnalyticRadialQGaugeBaseline result{};",
    ]
    lines.extend(compact_scalar_declarations())
    for symbol, expression in replacements:
        lines.append(f"  const Real {symbol} = {cpp(expression)};")
    for target, expression in zip(targets, reduced):
        lines.append(f"  {target} = {cpp(expression)};")
    lines.extend([
        "  result.valid = true;",
        "  for (int A = 0; A < 4; ++A) {",
        "    result.valid = result.valid && Kokkos::isfinite(result.hhat[A])",
        "                   && Kokkos::isfinite(result.theta[A])",
        "                   && Kokkos::isfinite(result.dt_theta[A]);",
        "    for (int p = 0; p < 4; ++p) {",
        "      result.valid = result.valid",
        "                     && Kokkos::isfinite(result.d_hhat[p][A]);",
        "    }",
        "    for (int i = 0; i < 3; ++i) {",
        "      result.valid = result.valid",
        "                     && Kokkos::isfinite(result.reference_k[i][A]);",
        "    }",
        "  }",
        "  return result;",
        "}",
        "",
    ])
    emit_component_switch(
        lines, "GeneratedAnalyticRadialQFrameMotion", fields["frame_motion"])
    emit_component_switch(
        lines, "GeneratedAnalyticRadialQDtFrameMotion",
        fields["dt_frame_motion"])
    lines.extend([
        "}  // namespace ref_gh", "",
        "#endif  // REF_GH_GENERATED_ANALYTIC_RADIAL_Q_GAUGE_HPP_", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def source_contractions(fields: Dict[str, Dict[Index, sp.Expr]]) -> Tuple[
        Dict[Index, sp.Expr], Dict[Index, sp.Expr], Dict[Index, sp.Expr]]:
    psi = {(a, b): sp.Symbol(f"psi[{a}][{b}]")
           for a, b in all_indices(2)}
    inverse = {(a, b): sp.Symbol(f"inverse[{a}][{b}]")
               for a, b in all_indices(2)}
    p = {(a, b, c): sp.Symbol(f"p[{a}][{b}][{c}]")
         for a, b, c in all_indices(3)}
    q = {(a, b, c): sp.Symbol(f"q[{a}][{b}][{c}]")
         for a, b, c in all_indices(3)}
    delta_upper = {
        (a, b, c): sp.Symbol(f"delta_upper[{a}][{b}][{c}]")
        for a, b, c in all_indices(3)}
    q_correction: Dict[Index, sp.Expr] = {}
    for capital_c, capital_a, capital_b in all_indices(3):
        q_correction[capital_c, capital_a, capital_b] = sum(
            fields["spin"][capital_d, capital_a, capital_c]
              * psi[capital_d, capital_b]
            + fields["spin"][capital_d, capital_b, capital_c]
              * psi[capital_a, capital_d]
            for capital_d in range(4))
    curvature: Dict[Index, sp.Expr] = {}
    frame_correction: Dict[Index, sp.Expr] = {}
    for capital_a in range(4):
        for capital_b in range(capital_a, 4):
            curvature[capital_a, capital_b] = -sum(
                inverse[capital_c, capital_d] * (
                    fields["riemann_frame"][
                        capital_e, capital_c, capital_d, capital_a]
                      * psi[capital_b, capital_e]
                    + fields["riemann_frame"][
                        capital_e, capital_c, capital_d, capital_b]
                      * psi[capital_a, capital_e])
                for capital_c, capital_d, capital_e in all_indices(3))
            value = sp.S.Zero
            for capital_c, capital_d in all_indices(2):
                inner = sp.S.Zero
                for capital_e in range(4):
                    inner -= (
                        fields["spin"][capital_e, capital_d, capital_c]
                        + delta_upper[capital_e, capital_d, capital_c]
                    ) * p[capital_e, capital_a, capital_b]
                    inner += (
                        fields["spin_derivative"][
                            capital_c, capital_e, capital_a, capital_d]
                          * psi[capital_e, capital_b]
                        + fields["spin"][capital_e, capital_a, capital_d]
                          * p[capital_c, capital_e, capital_b]
                        + fields["spin_derivative"][
                            capital_c, capital_e, capital_b, capital_d]
                          * psi[capital_a, capital_e]
                        + fields["spin"][capital_e, capital_b, capital_d]
                          * p[capital_c, capital_a, capital_e]
                        + fields["spin"][capital_e, capital_d, capital_c]
                          * q[capital_e, capital_a, capital_b]
                        + fields["spin"][capital_e, capital_a, capital_c]
                          * q[capital_d, capital_e, capital_b]
                        + fields["spin"][capital_e, capital_b, capital_c]
                          * q[capital_d, capital_a, capital_e])
                value += inverse[capital_c, capital_d]*inner
            frame_correction[capital_a, capital_b] = value
    return q_correction, curvature, frame_correction


def physical_gauge_contractions(
        fields: Dict[str, Dict[Index, sp.Expr]]) -> Tuple[
            Dict[Index, sp.Expr], Dict[Index, sp.Expr]]:
    inverse = {(a, b): sp.Symbol(f"physical_inverse[{a}][{b}]")
               for a, b in all_indices(2)}
    d_inverse = {
        (p, a, b): sp.Symbol(f"physical_d_inverse[{p}][{a}][{b}]")
        for p, a, b in all_indices(3)}
    h_upper: Dict[Index, sp.Expr] = {}
    d_h_upper: Dict[Index, sp.Expr] = {}
    for a in range(4):
        h_upper[a,] = -sum(
            inverse[b, c]*fields["christoffel"][a, b, c]
            for b, c in all_indices(2))
        for p in range(4):
            d_h_upper[p, a] = -sum(
                d_inverse[p, b, c]*fields["christoffel"][a, b, c]
                + inverse[b, c]*fields["d_christoffel"][p, a, b, c]
                for b, c in all_indices(2))
    return h_upper, d_h_upper


def emit_contracted_group(lines: List[str], values: Dict[Index, sp.Expr],
                          prefix: str, target: str,
                          symmetric_output: bool = False) -> None:
    names = list(values.keys())
    replacements, reduced = sp.cse(
        [values[index] for index in names],
        symbols=sp.numbered_symbols(prefix), order="canonical")
    for symbol, expression in replacements:
        lines.append(f"  const Real {symbol} = {cpp(expression)};")
    for index, expression in zip(names, reduced):
        indices = "".join(f"[{value}]" for value in index)
        lines.append(f"  {target}{indices} = {cpp(expression)};")
        if symmetric_output and index[0] != index[1]:
            lines.append(
                f"  {target}[{index[1]}][{index[0]}] = {target}{indices};")


def emit_source(output: Path) -> None:
    fields = build_geometry()
    q_correction, curvature, frame_correction = source_contractions(fields)
    physical_h_upper, physical_d_h_upper = physical_gauge_contractions(fields)
    generator_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    lines = [
        "// Generated by scripts/ref_gh/generate_analytic_radial_q_geometry.py.",
        "// Do not edit by hand.",
        "// Compact production contractions: no spin, spin-derivative, or",
        "// Riemann tensor is materialized or recursively reconstructed.",
        f"// generator_sha256={generator_sha}",
        f"// sympy_version={sp.__version__}",
        "#ifndef REF_GH_GENERATED_ANALYTIC_RADIAL_Q_SOURCE_HPP_",
        "#define REF_GH_GENERATED_ANALYTIC_RADIAL_Q_SOURCE_HPP_",
        "",
        "#include \"athena.hpp\"",
        "#include \"ref_gh/reference_analytic_radial_q.hpp\"",
        "",
        "namespace ref_gh {",
        "",
        "KOKKOS_INLINE_FUNCTION",
        "void GeneratedAnalyticRadialQQCorrection(",
        "    const AnalyticRadialScalar &alpha_jet,",
        "    const AnalyticRadialScalar &l_jet,",
        "    const AnalyticRadialScalar &b_jet, const Real displacement[3],",
        "    const Real radius, const Real psi[4][4],",
        "    Real correction[4][4][4]) {",
    ]
    lines.extend(compact_scalar_declarations())
    emit_contracted_group(lines, q_correction, "ref_q_cse_", "correction")
    lines.extend([
        "}", "", "KOKKOS_INLINE_FUNCTION",
        "void GeneratedAnalyticRadialQCurvatureSource(",
        "    const AnalyticRadialScalar &alpha_jet,",
        "    const AnalyticRadialScalar &l_jet,",
        "    const AnalyticRadialScalar &b_jet, const Real displacement[3],",
        "    const Real radius, const Real inverse[4][4],",
        "    const Real psi[4][4], Real source[4][4]) {",
    ])
    lines.extend(compact_scalar_declarations())
    emit_contracted_group(
        lines, curvature, "ref_curvature_cse_", "source", True)
    lines.extend([
        "}", "", "KOKKOS_INLINE_FUNCTION",
        "void GeneratedAnalyticRadialQFrameCorrection(",
        "    const AnalyticRadialScalar &alpha_jet,",
        "    const AnalyticRadialScalar &l_jet,",
        "    const AnalyticRadialScalar &b_jet, const Real displacement[3],",
        "    const Real radius, const Real inverse[4][4],",
        "    const Real psi[4][4], const Real p[4][4][4],",
        "    const Real q[4][4][4], const Real delta_upper[4][4][4],",
        "    Real source[4][4]) {",
    ])
    lines.extend(compact_scalar_declarations())
    emit_contracted_group(
        lines, frame_correction, "ref_frame_cse_", "source", True)
    lines.extend([
        "}", "", "KOKKOS_INLINE_FUNCTION",
        "void GeneratedAnalyticRadialQPhysicalGaugeUpper(",
        "    const AnalyticRadialScalar &alpha_jet,",
        "    const AnalyticRadialScalar &l_jet,",
        "    const AnalyticRadialScalar &b_jet, const Real displacement[3],",
        "    const Real radius, const Real physical_inverse[4][4],",
        "    const Real physical_d_inverse[4][4][4], Real h_upper[4],",
        "    Real d_h_upper[4][4]) {",
    ])
    lines.extend(compact_scalar_declarations())
    emit_contracted_group(
        lines, physical_h_upper, "ref_physical_h_cse_", "h_upper")
    emit_contracted_group(
        lines, physical_d_h_upper, "ref_physical_dh_cse_", "d_h_upper")
    lines.extend([
        "}", "", "}  // namespace ref_gh", "",
        "#endif  // REF_GH_GENERATED_ANALYTIC_RADIAL_Q_SOURCE_HPP_", ""])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path,
        default=Path("src/ref_gh/generated/analytic_radial_q_geometry.hpp"))
    parser.add_argument("--gauge-output", type=Path)
    parser.add_argument("--source-output", type=Path)
    arguments = parser.parse_args()
    emit(arguments.output)
    if arguments.gauge_output is not None:
        emit_gauge(arguments.gauge_output)
    if arguments.source_output is not None:
        emit_source(arguments.source_output)


if __name__ == "__main__":
    main()
