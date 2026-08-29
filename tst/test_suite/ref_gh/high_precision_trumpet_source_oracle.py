#!/usr/bin/env python3
"""Arbitrary-precision stationary-trumpet coordinate-source oracle.

This intentionally does not read the generated binary64 table.  It evaluates
the n=2 trumpet from its implicit areal-radius equation, reconstructs one exact
coordinate 2-jet, and mirrors StandardGhPartialWaveSource plus the reference-
frame product-rule transform.  The purpose is to distinguish an algebraic
residual from binary64 cancellation in the legacy coordinate oracle.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import mpmath as mp
from scipy.optimize import brentq

try:
    from . import generate_trumpet_table as double_trumpet
except ImportError:
    import generate_trumpet_table as double_trumpet


N = mp.mpf(2)


def constants():
    alpha_c = mp.sqrt((mp.sqrt(4 + 9 * N**2) - 3 * N)
                      / (mp.sqrt(4 + 9 * N**2) + 3 * N))
    radius_c = (3 * N**2 + mp.sqrt(4 * N**2 + 9 * N**4)) / (4 * N**2)
    c_squared = ((3 * N + mp.sqrt(4 + 9 * N**2))**3 / (128 * N**3)
                 * mp.exp(-2 * alpha_c / N))
    return alpha_c, radius_c, c_squared


ALPHA_C, RADIUS_C, C_SQUARED = constants()


def bisect(function, lower, upper, iterations):
    f_lower = function(lower)
    f_upper = function(upper)
    if f_lower == 0:
        return lower
    if f_upper == 0:
        return upper
    if f_lower * f_upper > 0:
        raise ValueError("root is not bracketed")
    for _ in range(iterations):
        middle = (lower + upper) / 2
        f_middle = function(middle)
        if f_middle == 0:
            return middle
        if f_lower * f_middle <= 0:
            upper = middle
            f_upper = f_middle
        else:
            lower = middle
            f_lower = f_middle
    return (lower + upper) / 2


def implicit(alpha, radius):
    return ((1 - alpha**2) * radius**4 - 2 * radius**3
            + C_SQUARED * mp.exp(2 * alpha / N))


def radius_from_alpha(alpha):
    if abs(alpha - ALPHA_C) < mp.eps * 100:
        return RADIUS_C
    function = lambda radius: implicit(alpha, radius)
    derivative = lambda radius: (4 * (1 - alpha**2) * radius**3
                                 - 6 * radius**2)
    iterations = int(mp.mp.dps * 3.6) + 32
    if abs(alpha - ALPHA_C) > mp.mpf("1e-12"):
        if alpha < ALPHA_C:
            radius0 = radius_from_alpha.zero_radius
            guess = radius0 + (RADIUS_C - radius0) * alpha / ALPHA_C
        else:
            guess = (RADIUS_C + 1 / (1 - alpha)
                     - 1 / (1 - ALPHA_C))
        try:
            root = mp.findroot(function, guess, df=derivative, solver="newton",
                               verify=False)
            scale = max(mp.mpf(1), abs(C_SQUARED * mp.exp(2 * alpha / N)))
            if (root > 0 and ((root < RADIUS_C) == (alpha < ALPHA_C))
                    and abs(function(root)) < scale * mp.mpf(10) ** (-(mp.mp.dps - 10))):
                return root
        except (ValueError, ZeroDivisionError):
            pass
    if alpha < ALPHA_C:
        return bisect(function, mp.mpf(0), RADIUS_C, iterations)
    upper = max(2 * RADIUS_C, 4 / (1 - alpha**2))
    return bisect(function, RADIUS_C, upper, iterations)


radius_from_alpha.zero_radius = bisect(
    lambda radius: implicit(mp.mpf(0), radius), mp.mpf(0), RADIUS_C, 400)


def radius_alpha_derivatives(alpha, radius):
    exponential = C_SQUARED * mp.exp(2 * alpha / N)
    f_r = 4 * (1 - alpha**2) * radius**3 - 6 * radius**2
    f_a = -2 * alpha * radius**4 + 2 * exponential / N
    if abs(alpha - ALPHA_C) < mp.sqrt(mp.eps):
        f_rr = 12 * (1 - alpha**2) * radius**2 - 12 * radius
        twice_f_ra = -16 * alpha * radius**3
        f_aa = -2 * radius**4 + 4 * exponential / N**2
        roots = mp.polyroots([f_rr, twice_f_ra, f_aa], maxsteps=1000)
        first = max(root.real for root in roots if abs(root.imag) < mp.sqrt(mp.eps))
    else:
        first = -f_a / f_r
    f_rr = 12 * (1 - alpha**2) * radius**2 - 12 * radius
    f_ra = -8 * alpha * radius**3
    f_aa = -2 * radius**4 + 4 * exponential / N**2
    second = -(f_rr * first**2 + 2 * f_ra * first + f_aa) / f_r
    return first, second


def normalization_constant():
    """Compute C0 with a compactified alpha->1 tail at current precision."""
    start = -mp.log(mp.mpf("0.9"))
    # exp(-umax) is below the requested 80-digit reporting floor while still
    # representable at the default 100 decimal digits.
    maximum = mp.mpf(min(220, int(2.15 * mp.mp.dps)))

    def integrand(u):
        alpha = 1 - mp.exp(-u)
        return (mp.log(radius_from_alpha(alpha)) / alpha**2) * mp.exp(-u)

    points = [start, mp.mpf(1), mp.mpf(3), mp.mpf(8), mp.mpf(20),
              mp.mpf(50), mp.mpf(100), maximum]
    points = sorted(set(point for point in points if start <= point <= maximum))
    return mp.quad(integrand, points)


def alpha_at_isotropic_radius(radius, c0):
    split = mp.mpf("0.1")
    split_radius = radius_from_alpha(split)

    def d_radius_d_alpha(alpha, areal):
        return radius_alpha_derivatives(alpha, areal)[0]

    def log_isotropic(alpha):
        integral = mp.quad(
            lambda value: -d_radius_d_alpha(value, radius_from_alpha(value))
                          / (value * radius_from_alpha(value)),
            [alpha, split],
        )
        return mp.log(split_radius) / split + integral - c0

    target = mp.log(radius)
    # A binary64 implicit-solution solve is used only as a Newton seed; neither
    # the generated table nor any binary64 value contributes to the final jet.
    seed = brentq(
        lambda lapse: double_trumpet.isotropic_radius_from_alpha(lapse)
                      - float(radius),
        0.0, 1.0 - 1.0e-12, xtol=1.0e-14,
    )
    alpha = mp.mpf(repr(seed))
    for _ in range(5):
        areal = radius_from_alpha(alpha)
        derivative = radius_alpha_derivatives(alpha, areal)[0] / (alpha * areal)
        correction = (log_isotropic(alpha) - target) / derivative
        alpha -= correction
        if abs(correction) < mp.mpf(10) ** (-(mp.mp.dps - 12)):
            break
    if abs(log_isotropic(alpha) - target) > mp.mpf(10) ** (-(mp.mp.dps - 10)):
        raise AssertionError("high-precision isotropic-radius inversion failed")
    return alpha


@dataclass
class Jet:
    value: mp.mpf
    first: list
    second: list

    @staticmethod
    def constant(value):
        return Jet(mp.mpf(value), [mp.mpf(0) for _ in range(4)],
                   [[mp.mpf(0) for _ in range(4)] for _ in range(4)])

    @staticmethod
    def coordinate(value, direction):
        result = Jet.constant(value)
        result.first[direction] = mp.mpf(1)
        return result

    def __add__(self, other):
        other = other if isinstance(other, Jet) else Jet.constant(other)
        return Jet(self.value + other.value,
                   [self.first[a] + other.first[a] for a in range(4)],
                   [[self.second[a][b] + other.second[a][b]
                     for b in range(4)] for a in range(4)])

    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.value, [-value for value in self.first],
                   [[-value for value in row] for row in self.second])

    def __sub__(self, other):
        return self + (-other if isinstance(other, Jet) else -Jet.constant(other))

    def __rsub__(self, other):
        return Jet.constant(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Jet) else Jet.constant(other)
        return Jet(
            self.value * other.value,
            [self.first[a] * other.value + self.value * other.first[a]
             for a in range(4)],
            [[self.second[a][b] * other.value
              + self.value * other.second[a][b]
              + self.first[a] * other.first[b]
              + self.first[b] * other.first[a]
              for b in range(4)] for a in range(4)],
        )

    __rmul__ = __mul__

    def reciprocal(self):
        inverse = 1 / self.value
        return Jet(
            inverse,
            [-self.first[a] * inverse**2 for a in range(4)],
            [[2 * self.first[a] * self.first[b] * inverse**3
              - self.second[a][b] * inverse**2
              for b in range(4)] for a in range(4)],
        )

    def __truediv__(self, other):
        return self * (other.reciprocal() if isinstance(other, Jet)
                       else mp.mpf(1) / other)


def radial_jet(value, first, second, coordinates):
    radius = mp.sqrt(sum(item**2 for item in coordinates))
    result = Jet.constant(value)
    for i in range(3):
        ni = coordinates[i] / radius
        result.first[i + 1] = first * ni
        for j in range(3):
            nj = coordinates[j] / radius
            result.second[i + 1][j + 1] = (
                second * ni * nj
                + first * ((1 if i == j else 0) - ni * nj) / radius
            )
    return result


def trumpet_jets(isotropic_radius, c0):
    alpha_value = alpha_at_isotropic_radius(isotropic_radius, c0)
    areal = radius_from_alpha(alpha_value)
    radius_alpha, radius_alpha2 = radius_alpha_derivatives(alpha_value, areal)
    alpha_radius = 1 / radius_alpha
    alpha_radius2 = -radius_alpha2 / radius_alpha**3
    areal_r = alpha_value * areal / isotropic_radius
    alpha_r = alpha_radius * areal_r
    areal_rr = ((alpha_r * areal + alpha_value * areal_r) / isotropic_radius
                - alpha_value * areal / isotropic_radius**2)
    alpha_rr = alpha_radius2 * areal_r**2 + alpha_radius * areal_rr
    q_value = mp.sqrt(C_SQUARED) * mp.exp(alpha_value / N) / areal**3
    log_q_r = alpha_r / N - 3 * areal_r / areal
    log_q_rr = (alpha_rr / N
                - 3 * (areal_rr / areal - (areal_r / areal)**2))
    q_r = q_value * log_q_r
    q_rr = q_value * (log_q_r**2 + log_q_rr)
    psi2 = areal / isotropic_radius
    psi2_r = areal_r / isotropic_radius - areal / isotropic_radius**2
    psi2_rr = (areal_rr / isotropic_radius
               - 2 * areal_r / isotropic_radius**2
               + 2 * areal / isotropic_radius**3)
    coordinates = [isotropic_radius, mp.mpf(0), mp.mpf(0)]
    return (radial_jet(alpha_value, alpha_r, alpha_rr, coordinates),
            radial_jet(psi2, psi2_r, psi2_rr, coordinates),
            radial_jet(q_value, q_r, q_rr, coordinates),
            alpha_value, areal)


def reference_geometry(isotropic_radius, c0):
    alpha, psi2, q, alpha_value, areal = trumpet_jets(isotropic_radius, c0)
    coordinates = [Jet.coordinate(isotropic_radius, 1),
                   Jet.coordinate(0, 2), Jet.coordinate(0, 3)]
    shift = [q * coordinate for coordinate in coordinates]
    coframe = [[Jet.constant(0) for _ in range(4)] for _ in range(4)]
    frame = [[Jet.constant(0) for _ in range(4)] for _ in range(4)]
    coframe[0][0] = alpha
    frame[0][0] = alpha.reciprocal()
    for i in range(3):
        coframe[i + 1][0] = psi2 * shift[i]
        coframe[i + 1][i + 1] = psi2
        frame[0][i + 1] = -(shift[i] / alpha)
        frame[i + 1][i + 1] = psi2.reciprocal()
    metric = [[Jet.constant(0) for _ in range(4)] for _ in range(4)]
    inverse = [[Jet.constant(0) for _ in range(4)] for _ in range(4)]
    for a in range(4):
        for b in range(4):
            metric[a][b] = -(coframe[0][a] * coframe[0][b])
            inverse[a][b] = -(frame[0][a] * frame[0][b])
            for A in range(1, 4):
                metric[a][b] += coframe[A][a] * coframe[A][b]
                inverse[a][b] += frame[A][a] * frame[A][b]
    return metric, inverse, frame, coframe, alpha_value, areal


def coordinate_geometry(metric_jet, inverse_jet):
    metric = [[metric_jet[a][b].value for b in range(4)] for a in range(4)]
    inverse = [[inverse_jet[a][b].value for b in range(4)] for a in range(4)]
    dmetric = [[[metric_jet[a][b].first[p] for b in range(4)]
                for a in range(4)] for p in range(4)]
    ddmetric = [[[[metric_jet[a][b].second[p][q] for b in range(4)]
                  for a in range(4)] for q in range(4)] for p in range(4)]
    first_kind = [[[mp.mpf(0) for _ in range(4)] for _ in range(4)]
                  for _ in range(4)]
    christoffel = [[[mp.mpf(0) for _ in range(4)] for _ in range(4)]
                   for _ in range(4)]
    d_christoffel = [[[[mp.mpf(0) for _ in range(4)] for _ in range(4)]
                      for _ in range(4)] for _ in range(4)]
    for a in range(4):
        for b in range(4):
            for c in range(4):
                first_kind[a][b][c] = (dmetric[b][a][c] + dmetric[c][a][b]
                                       - dmetric[a][b][c]) / 2
    for a in range(4):
        for b in range(4):
            for c in range(4):
                christoffel[a][b][c] = sum(
                    inverse[a][ell] * first_kind[ell][b][c]
                    for ell in range(4))
                for p in range(4):
                    d_christoffel[p][a][b][c] = sum(
                        inverse_jet[a][ell].first[p] * first_kind[ell][b][c]
                        + inverse[a][ell] * (
                            ddmetric[p][b][ell][c]
                            + ddmetric[p][c][ell][b]
                            - ddmetric[p][ell][b][c]) / 2
                        for ell in range(4))
    return metric, inverse, dmetric, first_kind, christoffel, d_christoffel


def legacy_source(isotropic_radius, c0, gamma0=mp.mpf(1)):
    (metric_jet, inverse_jet, frame_jet, _, alpha_exact,
     areal) = reference_geometry(isotropic_radius, c0)
    metric, inverse, dmetric, first, christoffel, dchristoffel = coordinate_geometry(
        metric_jet, inverse_jet)
    alpha = 1 / mp.sqrt(-inverse[0][0])
    shift = [alpha**2 * inverse[0][i + 1] for i in range(3)]
    normal_up = [1 / alpha] + [-value / alpha for value in shift]
    normal_down = [-alpha, mp.mpf(0), mp.mpf(0), mp.mpf(0)]
    contracted_first = [sum(inverse[b][c] * first[a][b][c]
                            for b in range(4) for c in range(4))
                        for a in range(4)]
    contracted_upper = [sum(inverse[b][c] * christoffel[a][b][c]
                            for b in range(4) for c in range(4))
                        for a in range(4)]
    h_upper = [-sum(inverse[b][c] * christoffel[a][b][c]
                    for b in range(4) for c in range(4)) for a in range(4)]
    h_lower = [sum(metric[a][b] * h_upper[b] for b in range(4))
               for a in range(4)]
    constraint = [h_lower[a] + contracted_first[a] for a in range(4)]
    d_inverse = [[[inverse_jet[a][b].first[p] for b in range(4)]
                  for a in range(4)] for p in range(4)]
    d_h_upper = [[-sum(d_inverse[p][b][c] * christoffel[a][b][c]
                          + inverse[b][c] * dchristoffel[p][a][b][c]
                          for b in range(4) for c in range(4))
                  for a in range(4)] for p in range(4)]
    d_h_lower = [[sum(dmetric[p][a][b] * h_upper[b]
                           + metric[a][b] * d_h_upper[p][b]
                           for b in range(4))
                  for a in range(4)] for p in range(4)]
    coordinate_source = [[mp.mpf(0) for _ in range(4)] for _ in range(4)]
    for a in range(4):
        for b in range(4):
            nabla_h_ab = d_h_lower[a][b] - sum(
                christoffel[c][a][b] * h_lower[c] for c in range(4))
            nabla_h_ba = d_h_lower[b][a] - sum(
                christoffel[c][b][a] * h_lower[c] for c in range(4))
            value = -nabla_h_ab - nabla_h_ba
            for c in range(4):
                for d in range(4):
                    for e in range(4):
                        for f in range(4):
                            value += 2 * inverse[c][d] * inverse[e][f] * (
                                dmetric[e][c][a] * dmetric[f][d][b]
                                - first[a][c][e] * first[b][d][f])
            for c in range(4):
                projector = ((normal_down[b] if c == a else 0)
                             + (normal_down[a] if c == b else 0)
                             - metric[a][b] * normal_up[c])
                value += gamma0 * projector * constraint[c]
            coordinate_source[a][b] = value
    scalar_source = [[mp.mpf(0) for _ in range(4)] for _ in range(4)]
    for A in range(4):
        for B in range(4):
            value = mp.mpf(0)
            for a in range(4):
                for b in range(4):
                    tensor = frame_jet[A][a].value * frame_jet[B][b].value
                    value += tensor * coordinate_source[a][b]
                    for c in range(4):
                        d_tensor = (frame_jet[A][a].first[c]
                                    * frame_jet[B][b].value
                                    + frame_jet[A][a].value
                                    * frame_jet[B][b].first[c])
                        for d in range(4):
                            dd_tensor = (
                                frame_jet[A][a].second[c][d]
                                * frame_jet[B][b].value
                                + frame_jet[A][a].first[c]
                                * frame_jet[B][b].first[d]
                                + frame_jet[A][a].first[d]
                                * frame_jet[B][b].first[c]
                                + frame_jet[A][a].value
                                * frame_jet[B][b].second[c][d])
                            value += (2 * inverse[c][d] * d_tensor
                                      * dmetric[d][a][b]
                                      + inverse[c][d] * dd_tensor * metric[a][b])
            scalar_source[A][B] = value
    ricci = [[mp.mpf(0) for _ in range(4)] for _ in range(4)]
    for a in range(4):
        for b in range(4):
            for c in range(4):
                ricci[a][b] += (dchristoffel[c][c][a][b]
                                - dchristoffel[b][c][a][c])
                for d in range(4):
                    ricci[a][b] += (christoffel[c][c][d] * christoffel[d][a][b]
                                    - christoffel[c][b][d] * christoffel[d][a][c])
    maximum = lambda array: max(abs(value) for row in array for value in row)
    return {
        "alpha": alpha_exact,
        "areal_radius": areal,
        "coordinate_ricci_linf": maximum(ricci),
        "gauge_constraint_linf": max(abs(value) for value in constraint),
        "coordinate_source_linf": maximum(coordinate_source),
        "scalar_source_linf": maximum(scalar_source),
    }


def stationary_gauge_identity(isotropic_radius, c0, nu=mp.mpf("0.83")):
    """Verify F_ref=H_ref and conformal Gamma_ref=0 independently.

    The target is the exact advective 1+log/conformal-Gamma target used by the
    C++ driver.  The trumpet comes from the implicit arbitrary-precision
    solution rather than the generated binary64 table.
    """
    (metric_jet, inverse_jet, frame_jet, _, _,
     _) = reference_geometry(isotropic_radius, c0)
    metric, inverse, dmetric, _, christoffel, _ = coordinate_geometry(
        metric_jet, inverse_jet)
    spatial_metric = [[metric[i + 1][j + 1] for j in range(3)]
                      for i in range(3)]
    spatial_inverse = inverse_matrix(spatial_metric)
    spatial_determinant = (
        spatial_metric[0][0] * (
            spatial_metric[1][1] * spatial_metric[2][2]
            - spatial_metric[1][2] * spatial_metric[2][1])
        - spatial_metric[0][1] * (
            spatial_metric[1][0] * spatial_metric[2][2]
            - spatial_metric[1][2] * spatial_metric[2][0])
        + spatial_metric[0][2] * (
            spatial_metric[1][0] * spatial_metric[2][1]
            - spatial_metric[1][1] * spatial_metric[2][0]))
    lapse = 1 / mp.sqrt(-inverse[0][0])
    shift = [lapse**2 * inverse[0][i + 1] for i in range(3)]
    d_inverse = [[[inverse_jet[a][b].first[p] for b in range(4)]
                  for a in range(4)] for p in range(4)]
    d_lapse = [lapse**3 * d_inverse[p + 1][0][0] / 2
               for p in range(3)]
    d_shift = [[
        2 * lapse * d_lapse[p] * inverse[0][i + 1]
        + lapse**2 * d_inverse[p + 1][0][i + 1]
        for i in range(3)] for p in range(3)]
    trace_k = -lapse * sum(
        spatial_inverse[i][j] * christoffel[0][i + 1][j + 1]
        for i in range(3) for j in range(3))
    determinant_factor = spatial_determinant ** (mp.mpf(1) / 3)
    conformal_gamma = [mp.mpf(0) for _ in range(3)]
    for i in range(3):
        conformal_gamma[i] = determinant_factor * sum(
            (spatial_inverse[i][k] * spatial_inverse[j][ell]
             - spatial_inverse[i][j] * spatial_inverse[k][ell] / 3)
            * dmetric[j + 1][k + 1][ell + 1]
            for j in range(3) for k in range(3) for ell in range(3))
    desired_dt_shift = [
        nu * conformal_gamma[i]
        - sum(shift[p] * d_shift[p][i] for p in range(3))
        for i in range(3)]
    target_coordinate = [mp.mpf(0) for _ in range(4)]
    for i in range(3):
        contracted_spatial_connection = sum(
            spatial_inverse[j][k] * (
                dmetric[j + 1][i + 1][k + 1]
                + dmetric[k + 1][i + 1][j + 1]
                - dmetric[i + 1][j + 1][k + 1]) / 2
            for j in range(3) for k in range(3))
        target_coordinate[i + 1] = (
            d_lapse[i] / lapse - contracted_spatial_connection
            + sum(metric[i + 1][j + 1] * desired_dt_shift[j]
                  / lapse**2 for j in range(3)))
    normal_target = (2 / lapse - 1) * trace_k
    target_coordinate[0] = (
        lapse * normal_target
        + sum(shift[i] * target_coordinate[i + 1] for i in range(3)))
    target_frame = [sum(frame_jet[A][a].value * target_coordinate[a]
                        for a in range(4)) for A in range(4)]
    h_upper = [-sum(inverse[b][c] * christoffel[a][b][c]
                    for b in range(4) for c in range(4))
               for a in range(4)]
    h_lower = [sum(metric[a][b] * h_upper[b] for b in range(4))
               for a in range(4)]
    h_frame = [sum(frame_jet[A][a].value * h_lower[a] for a in range(4))
               for A in range(4)]
    return {
        "target_minus_h_linf": max(
            abs(target_frame[A] - h_frame[A]) for A in range(4)),
        "conformal_gamma_linf": max(abs(value) for value in conformal_gamma),
    }


def inverse_matrix(matrix):
    """Small arbitrary-precision Gauss-Jordan inverse for the frame oracle."""
    size = len(matrix)
    augmented = [[matrix[row][column] for column in range(size)]
                 + [mp.mpf(row == column) for column in range(size)]
                 for row in range(size)]
    for column in range(size):
        pivot = max(range(column, size),
                    key=lambda row: abs(augmented[row][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        diagonal = augmented[column][column]
        if diagonal == 0:
            raise AssertionError("singular frame metric in covariant oracle")
        augmented[column] = [entry / diagonal for entry in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [augmented[row][item] - factor * augmented[column][item]
                              for item in range(2 * size)]
    return [[augmented[row][column + size] for column in range(size)]
            for row in range(size)]


def frame_geometry(metric_jet, inverse_jet, coframe_jet, frame_jet):
    """Evaluate the exact same spin/Cartan construction as the device provider.

    This intentionally starts from the arbitrary-precision coordinate 2-jet and
    never reads the generated interpolation table.  The construction is kept
    independent of the production C++ implementation while using the documented
    frame/spin definitions.
    """
    _, _, _, _, christoffel, dchristoffel = coordinate_geometry(
        metric_jet, inverse_jet)
    theta = [[coframe_jet[A][a].value for a in range(4)] for A in range(4)]
    tetrad = [[frame_jet[A][a].value for a in range(4)] for A in range(4)]
    dtheta = [[[coframe_jet[A][a].first[p] for a in range(4)]
               for A in range(4)] for p in range(4)]
    dframe = [[[frame_jet[A][a].first[p] for a in range(4)]
               for A in range(4)] for p in range(4)]
    ddframe = [[[[frame_jet[A][a].second[p][q] for a in range(4)]
                 for A in range(4)] for q in range(4)] for p in range(4)]
    omega = [[[mp.mpf(0) for _ in range(4)] for _ in range(4)]
             for _ in range(4)]
    coordinate_domega = [[[[mp.mpf(0) for _ in range(4)] for _ in range(4)]
                           for _ in range(4)] for _ in range(4)]
    for A in range(4):
        for B in range(4):
            for C in range(4):
                for a in range(4):
                    for c in range(4):
                        covariant = dframe[c][B][a] + sum(
                            christoffel[a][c][d] * tetrad[B][d]
                            for d in range(4))
                        omega[A][B][C] += theta[A][a] * tetrad[C][c] * covariant
                        for p in range(4):
                            d_covariant = ddframe[p][c][B][a] + sum(
                                dchristoffel[p][a][c][d] * tetrad[B][d]
                                + christoffel[a][c][d] * dframe[p][B][d]
                                for d in range(4))
                            coordinate_domega[p][A][B][C] += (
                                (dtheta[p][A][a] * tetrad[C][c]
                                 + theta[A][a] * dframe[p][C][c]) * covariant
                                + theta[A][a] * tetrad[C][c] * d_covariant)
    # Project only the exact metric-compatibility antisymmetry, as production does.
    for A in range(4):
        eta_a = -1 if A == 0 else 1
        for B in range(A, 4):
            eta_b = -1 if B == 0 else 1
            for C in range(4):
                projected = (eta_a * omega[A][B][C]
                             - eta_b * omega[B][A][C]) / 2
                omega[A][B][C] = eta_a * projected
                omega[B][A][C] = -eta_b * projected
    domega = [[[[mp.mpf(0) for _ in range(4)] for _ in range(4)]
               for _ in range(4)] for _ in range(4)]
    for D in range(4):
        for A in range(4):
            for B in range(4):
                for C in range(4):
                    domega[D][A][B][C] = sum(
                        tetrad[D][p] * coordinate_domega[p][A][B][C]
                        for p in range(4))
    for A in range(4):
        eta_a = -1 if A == 0 else 1
        for B in range(A, 4):
            eta_b = -1 if B == 0 else 1
            for C in range(4):
                for D in range(4):
                    projected = (eta_a * domega[D][A][B][C]
                                 - eta_b * domega[D][B][A][C]) / 2
                    domega[D][A][B][C] = eta_a * projected
                    domega[D][B][A][C] = -eta_b * projected
    structure = [[[mp.mpf(0) for _ in range(4)] for _ in range(4)]
                 for _ in range(4)]
    for A in range(4):
        for B in range(4):
            for C in range(B, 4):
                value = sum(theta[A][a] * (
                    tetrad[B][p] * dframe[p][C][a]
                    - tetrad[C][p] * dframe[p][B][a])
                    for a in range(4) for p in range(4))
                structure[A][B][C] = value
                structure[A][C][B] = -value
    riemann = [[[[mp.mpf(0) for _ in range(4)] for _ in range(4)]
                for _ in range(4)] for _ in range(4)]
    for A in range(4):
        for B in range(4):
            for C in range(4):
                for D in range(4):
                    riemann[A][B][C][D] = (
                        domega[C][A][B][D] - domega[D][A][B][C]
                        + sum(omega[A][E][C] * omega[E][B][D]
                              - omega[A][E][D] * omega[E][B][C]
                              - structure[E][C][D] * omega[A][B][E]
                              for E in range(4)))
    return theta, tetrad, omega, domega, riemann


def covariant_reference_source(isotropic_radius, c0):
    """High-precision Q=Delta=S=0 identity for the exact trumpet reference."""
    alpha, psi2, q, _, _ = trumpet_jets(isotropic_radius, c0)
    coordinates = [Jet.coordinate(isotropic_radius, 1),
                   Jet.coordinate(0, 2), Jet.coordinate(0, 3)]
    shift = [q * coordinate for coordinate in coordinates]
    coframe = [[Jet.constant(0) for _ in range(4)] for _ in range(4)]
    frame = [[Jet.constant(0) for _ in range(4)] for _ in range(4)]
    coframe[0][0] = alpha
    frame[0][0] = alpha.reciprocal()
    for i in range(3):
        coframe[i + 1][0] = psi2 * shift[i]
        coframe[i + 1][i + 1] = psi2
        frame[0][i + 1] = -(shift[i] / alpha)
        frame[i + 1][i + 1] = psi2.reciprocal()
    eta = [[mp.mpf(-1) if A == B == 0 else mp.mpf(A == B)
            for B in range(4)] for A in range(4)]
    metric = [[sum(eta[A][A] * coframe[A][a] * coframe[A][b]
                   for A in range(4)) for b in range(4)] for a in range(4)]
    inverse = [[sum(eta[A][A] * frame[A][a] * frame[A][b]
                    for A in range(4)) for b in range(4)] for a in range(4)]
    _, tetrad, omega, domega, riemann = frame_geometry(
        metric, inverse, coframe, frame)
    p = [[[mp.mpf(0) for _ in range(4)] for _ in range(4)] for _ in range(4)]
    q_cov = [[[p[C][A][B] - sum(
        omega[D][A][C] * eta[D][B] + omega[D][B][C] * eta[A][D]
        for D in range(4)) for B in range(4)] for A in range(4)] for C in range(4)]
    delta_lower = [[[((q_cov[B][A][C] + q_cov[C][A][B]
                       - q_cov[A][B][C]) / 2)
                    for C in range(4)] for B in range(4)] for A in range(4)]
    delta = [sum(eta[B][B] * delta_lower[A][B][B] for B in range(4))
             for A in range(4)]
    curvature = [[-sum(eta[C][C] * (
        riemann[E][C][C][A] * eta[B][E]
        + riemann[E][C][C][B] * eta[A][E])
        for C in range(4) for E in range(4)) for B in range(4)] for A in range(4)]
    correction = [[sum(eta[C][C] * (
        sum(domega[C][E][A][C] * eta[E][B]
            + domega[C][E][B][C] * eta[A][E] for E in range(4)))
        for C in range(4)) for B in range(4)] for A in range(4)]
    source = [[curvature[A][B] + correction[A][B]
               for B in range(4)] for A in range(4)]
    maximum = lambda array: max(abs(value) for row in array for value in row)
    maximum3 = lambda array: max(abs(value) for plane in array
                                 for row in plane for value in row)
    return {
        "covariant_q_linf": maximum3(q_cov),
        "covariant_delta_linf": max(maximum3(delta_lower), max(abs(v) for v in delta)),
        "covariant_scalar_source_linf": maximum(source),
        "covariant_frame_ricci_linf": max(abs(sum(riemann[A][B][A][D]
                                                     for A in range(4)))
                                          for B in range(4) for D in range(4)),
        "covariant_spin_antisymmetry_linf": max(
            abs(((-1 if A == 0 else 1) * omega[A][B][C]
                 + (-1 if B == 0 else 1) * omega[B][A][C]))
            for A in range(4) for B in range(4) for C in range(4)),
        "covariant_tetrad_duality_linf": max(abs(sum(
            [[coframe[A][a].value for a in range(4)] for A in range(4)][A][a]
            * tetrad[B][a] for a in range(4)) - (1 if A == B else 0))
            for A in range(4) for B in range(4)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dps", type=int, default=100)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--gauge-only", action="store_true",
        help="run only the expanded stationary gauge-identity matrix")
    args = parser.parse_args()
    if not 80 <= args.dps <= 150:
        raise ValueError("--dps must be in [80,150]")
    mp.mp.dps = args.dps
    global N, ALPHA_C, RADIUS_C, C_SQUARED
    N = mp.mpf(2)
    ALPHA_C, RADIUS_C, C_SQUARED = constants()
    radius_from_alpha.zero_radius = bisect(
        lambda radius: implicit(mp.mpf(0), radius),
        mp.mpf(0), RADIUS_C, int(mp.mp.dps * 3.6) + 32)
    c0 = normalization_constant()
    results = []
    if not args.gauge_only:
        radii = [mp.mpf(1) / denominator
                 for denominator in (8, 12, 16, 24, 32, 48, 64, 96, 128)]
        for radius in radii:
            row = legacy_source(radius, c0)
            row.update(covariant_reference_source(radius, c0))
            row["radius"] = radius
            results.append({key: mp.nstr(value, args.dps)
                            for key, value in row.items()})
    gauge_radii = [mp.mpf(value) for value in (
        "0.03", "0.05", "0.08", "0.125", "0.2", "0.4", "0.8", "1.5",
        "3", "5")]
    gauge_results = []
    for radius in gauge_radii:
        row = stationary_gauge_identity(radius, c0)
        row["radius"] = radius
        gauge_results.append({key: mp.nstr(value, args.dps)
                              for key, value in row.items()})
    payload = {"decimal_digits": args.dps,
               "normalization_constant": mp.nstr(c0, args.dps),
               "results": results,
               "stationary_gauge_identity": gauge_results}
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")
    if not args.gauge_only:
        maximum = max(mp.mpf(row["scalar_source_linf"]) for row in results)
        if maximum > mp.mpf(10) ** (-(args.dps // 2)):
            raise AssertionError(
                f"finite continuum scalar-source residual: {maximum}")
        for key in ("covariant_q_linf", "covariant_delta_linf",
                    "covariant_scalar_source_linf",
                    "covariant_frame_ricci_linf"):
            maximum = max(mp.mpf(row[key]) for row in results)
            if maximum > mp.mpf(10) ** (-(args.dps // 2)):
                raise AssertionError(
                    f"finite covariant reference residual {key}: {maximum}")
    for key in ("target_minus_h_linf", "conformal_gamma_linf"):
        maximum = max(mp.mpf(row[key]) for row in gauge_results)
        if maximum > mp.mpf(10) ** (-(args.dps // 2)):
            raise AssertionError(
                f"finite stationary gauge identity residual {key}: {maximum}")


if __name__ == "__main__":
    main()
