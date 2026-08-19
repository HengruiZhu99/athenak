#!/usr/bin/env python3
"""Independent frame-covariant GH lower-order source oracles."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np

try:
    from .standard_gh_source_audit import coordinate_wave_source
except ImportError:
    from standard_gh_source_audit import coordinate_wave_source


def frame_covariant_source(psi: np.ndarray, p: np.ndarray,
                           omega: np.ndarray, omega_derivative: np.ndarray,
                           riemann: np.ndarray, gamma0: float) -> tuple[np.ndarray, dict]:
    """Evaluate Eq. (24) of arXiv:1312.0701 and the scalar-frame correction."""
    inverse = np.linalg.inv(psi)
    q = np.zeros((4, 4, 4))
    for C in range(4):
        for A in range(4):
            for B in range(4):
                q[C, A, B] = p[C, A, B]
                for D in range(4):
                    q[C, A, B] -= (omega[D, A, C] * psi[D, B]
                                    + omega[D, B, C] * psi[A, D])
    delta_lower = np.zeros((4, 4, 4))
    for A in range(4):
        for B in range(4):
            for C in range(4):
                delta_lower[A, B, C] = 0.5 * (
                    q[B, A, C] + q[C, A, B] - q[A, B, C])
    delta_upper = np.einsum("ad,dbc->abc", inverse, delta_lower)
    delta = np.einsum("bc,abc->a", inverse, delta_lower)

    lapse = (-inverse[0, 0])**-0.5
    shift = lapse**2 * inverse[0, 1:]
    normal_up = np.concatenate(([1.0 / lapse], -shift / lapse))
    normal_down = psi @ normal_up
    tensor = np.zeros((4, 4))
    curvature_sector = np.zeros((4, 4))
    qq_sector = np.zeros((4, 4))
    delta_sector = np.zeros((4, 4))
    damping_sector = np.zeros((4, 4))
    for A in range(4):
        for B in range(4):
            for C in range(4):
                for D in range(4):
                    for E in range(4):
                        curvature_sector[A, B] -= inverse[C, D] * (
                            riemann[E, C, D, A] * psi[B, E]
                            + riemann[E, C, D, B] * psi[A, E])
                        for F in range(4):
                            qq_sector[A, B] += (
                                2 * inverse[C, D] * inverse[E, F]
                                * q[E, C, A] * q[F, D, B])
                            delta_sector[A, B] -= (
                                2 * inverse[C, D] * inverse[E, F]
                                * delta_lower[A, C, E]
                                * delta_lower[B, D, F])
                projector = ((normal_down[B] if C == A else 0.0)
                             + (normal_down[A] if C == B else 0.0)
                             - psi[A, B] * normal_up[C])
                damping_sector[A, B] += gamma0 * projector * delta[C]
            tensor[A, B] = (curvature_sector[A, B] + qq_sector[A, B]
                            + delta_sector[A, B] + damping_sector[A, B])

    correction = np.zeros((4, 4))
    for A in range(4):
        for B in range(4):
            for C in range(4):
                for D in range(4):
                    f_cdab = 0.0
                    for E in range(4):
                        f_cdab -= (omega[E, D, C] + delta_upper[E, D, C]) \
                                    * p[E, A, B]
                        f_cdab += omega_derivative[C, E, A, D] * psi[E, B]
                        f_cdab += omega[E, A, D] * p[C, E, B]
                        f_cdab += omega_derivative[C, E, B, D] * psi[A, E]
                        f_cdab += omega[E, B, D] * p[C, A, E]
                        f_cdab += omega[E, D, C] * q[E, A, B]
                        f_cdab += omega[E, A, C] * q[D, E, B]
                        f_cdab += omega[E, B, C] * q[D, A, E]
                    correction[A, B] += inverse[C, D] * f_cdab
    return tensor + correction, {
        "q": q,
        "delta_lower": delta_lower,
        "delta_upper": delta_upper,
        "delta": delta,
        "tensor": tensor,
        "curvature_sector": curvature_sector,
        "qq_sector": qq_sector,
        "delta_sector": delta_sector,
        "damping_sector": damping_sector,
        "frame_correction": correction,
    }


def random_lorentzian(rng: np.random.Generator) -> np.ndarray:
    seed = rng.normal(scale=0.15, size=(3, 3))
    spatial = seed.T @ seed + np.diag(rng.uniform(0.7, 1.4, size=3))
    lapse = rng.uniform(0.6, 1.4)
    shift = rng.normal(scale=0.08, size=3)
    metric = np.zeros((4, 4))
    metric[1:, 1:] = spatial
    metric[0, 1:] = spatial @ shift
    metric[1:, 0] = metric[0, 1:]
    metric[0, 0] = -lapse**2 + shift @ spatial @ shift
    return metric


@dataclass
class Jet:
    value: float
    first: np.ndarray
    second: np.ndarray

    @staticmethod
    def constant(value: float):
        return Jet(float(value), np.zeros(4), np.zeros((4, 4)))

    def __add__(self, other):
        other = other if isinstance(other, Jet) else Jet.constant(other)
        return Jet(self.value + other.value, self.first + other.first,
                   self.second + other.second)

    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.value, -self.first, -self.second)

    def __sub__(self, other):
        return self + (-other if isinstance(other, Jet) else -Jet.constant(other))

    def __rsub__(self, other):
        return Jet.constant(other) - self

    def __mul__(self, other):
        other = other if isinstance(other, Jet) else Jet.constant(other)
        return Jet(self.value * other.value,
                   self.first * other.value + self.value * other.first,
                   self.second * other.value + self.value * other.second
                   + np.outer(self.first, other.first)
                   + np.outer(other.first, self.first))

    __rmul__ = __mul__

    def reciprocal(self):
        inverse = 1.0 / self.value
        return Jet(inverse, -self.first * inverse**2,
                   2.0 * np.outer(self.first, self.first) * inverse**3
                   - self.second * inverse**2)

    def __truediv__(self, other):
        return self * (other.reciprocal() if isinstance(other, Jet)
                       else 1.0 / other)


def inverse_jet_matrix(matrix: list[list[Jet]]) -> list[list[Jet]]:
    augmented = [[matrix[row][column]
                  for column in range(4)]
                 + [Jet.constant(1.0 if row == column else 0.0)
                    for column in range(4)] for row in range(4)]
    for column in range(4):
        pivot = max(range(column, 4),
                    key=lambda row: abs(augmented[row][column].value))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        diagonal = augmented[column][column]
        augmented[column] = [entry / diagonal for entry in augmented[column]]
        for row in range(4):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [augmented[row][item] - factor * augmented[column][item]
                              for item in range(8)]
    return [[augmented[row][column + 4] for column in range(4)]
            for row in range(4)]


def coordinate_connection(metric_jet, inverse_jet):
    metric = np.array([[metric_jet[a][b].value for b in range(4)]
                       for a in range(4)])
    inverse = np.array([[inverse_jet[a][b].value for b in range(4)]
                        for a in range(4)])
    dmetric = np.array([[[metric_jet[a][b].first[p] for b in range(4)]
                         for a in range(4)] for p in range(4)])
    ddmetric = np.array([[[[metric_jet[a][b].second[p, q] for b in range(4)]
                            for a in range(4)] for q in range(4)]
                          for p in range(4)])
    first = np.zeros((4, 4, 4))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                first[a, b, c] = 0.5 * (
                    dmetric[b, a, c] + dmetric[c, a, b] - dmetric[a, b, c])
    christoffel = np.einsum("ad,dbc->abc", inverse, first)
    dchristoffel = np.zeros((4, 4, 4, 4))
    for p in range(4):
        for a in range(4):
            for b in range(4):
                for c in range(4):
                    for ell in range(4):
                        d_first = 0.5 * (
                            ddmetric[p, b, ell, c] + ddmetric[p, c, ell, b]
                            - ddmetric[p, ell, b, c])
                        dchristoffel[p, a, b, c] += (
                            inverse_jet[a][ell].first[p] * first[ell, b, c]
                            + inverse[a, ell] * d_first)
    return metric, inverse, dmetric, christoffel, dchristoffel


def manufactured_reference(rng: np.random.Generator, kind: str) -> dict:
    base = np.eye(4)
    first = np.zeros((4, 4, 4))
    second = np.zeros((4, 4, 4, 4))
    if kind == "diagonal":
        base += np.diag(rng.normal(scale=0.08, size=4))
        for A in range(4):
            first[:, A, A] = rng.normal(scale=0.08, size=4)
            raw = rng.normal(scale=0.05, size=(4, 4))
            second[:, :, A, A] = 0.5 * (raw + raw.T)
    else:
        base += rng.normal(scale=0.05, size=(4, 4))
        first = rng.normal(scale=0.06, size=(4, 4, 4))
        raw = rng.normal(scale=0.04, size=(4, 4, 4, 4))
        second = 0.5 * (raw + raw.swapaxes(0, 1))
        # The reference tetrad used by ref_gh is foliation adapted:
        # theta^0=alpha_ref dt. Nonzero shift lives in theta^I_t.
        base[0, 1:] = 0.0
        first[:, 0, 1:] = 0.0
        second[:, :, 0, 1:] = 0.0
        if kind == "off_diagonal":
            base[1:, 0] = 0.0
            first[:, 1:, 0] = 0.0
            second[:, :, 1:, 0] = 0.0
        elif kind == "shift":
            base[1:, 0] += rng.normal(scale=0.12, size=3)
    # The source-repair gate is for the stationary reference implemented now.
    first[0, :, :] = 0.0
    second[0, :, :, :] = 0.0
    second[:, 0, :, :] = 0.0
    coframe = [[Jet(base[A, a], first[:, A, a], second[:, :, A, a])
                for a in range(4)] for A in range(4)]
    inverse_coframe = inverse_jet_matrix(coframe)
    frame = [[inverse_coframe[a][A] for a in range(4)] for A in range(4)]
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    metric_jet = [[sum(eta[A, A] * coframe[A][a] * coframe[A][b]
                       for A in range(4)) for b in range(4)] for a in range(4)]
    inverse_jet = [[sum(eta[A, A] * frame[A][a] * frame[A][b]
                        for A in range(4)) for b in range(4)] for a in range(4)]
    metric, inverse, dmetric, christoffel, dchristoffel = coordinate_connection(
        metric_jet, inverse_jet)
    theta = np.array([[coframe[A][a].value for a in range(4)] for A in range(4)])
    tetrad = np.array([[frame[A][a].value for a in range(4)] for A in range(4)])
    dframe = np.array([[[frame[A][a].first[p] for a in range(4)]
                        for A in range(4)] for p in range(4)])
    ddframe = np.array([[[[frame[A][a].second[p, q] for a in range(4)]
                           for A in range(4)] for q in range(4)]
                         for p in range(4)])
    dtheta = np.array([[[coframe[A][a].first[p] for a in range(4)]
                        for A in range(4)] for p in range(4)])
    omega = np.zeros((4, 4, 4))
    coordinate_domega = np.zeros((4, 4, 4, 4))
    for A in range(4):
        for B in range(4):
            for C in range(4):
                for a in range(4):
                    for c in range(4):
                        covariant = dframe[c, B, a] + sum(
                            christoffel[a, c, d] * tetrad[B, d] for d in range(4))
                        omega[A, B, C] += theta[A, a] * tetrad[C, c] * covariant
                        for p in range(4):
                            d_covariant = ddframe[p, c, B, a] + sum(
                                dchristoffel[p, a, c, d] * tetrad[B, d]
                                + christoffel[a, c, d] * dframe[p, B, d]
                                for d in range(4))
                            coordinate_domega[p, A, B, C] += (
                                (dtheta[p, A, a] * tetrad[C, c]
                                 + theta[A, a] * dframe[p, C, c]) * covariant
                                + theta[A, a] * tetrad[C, c] * d_covariant)
    omega_derivative = np.einsum("Cp,pABD->CABD", tetrad, coordinate_domega)
    structure = np.zeros((4, 4, 4))
    for A in range(4):
        for B in range(4):
            for C in range(4):
                structure[A, B, C] = sum(
                    theta[A, a] * (tetrad[B, p] * dframe[p, C, a]
                                   - tetrad[C, p] * dframe[p, B, a])
                    for a in range(4) for p in range(4))
    riemann = np.zeros((4, 4, 4, 4))
    for A in range(4):
        for B in range(4):
            for C in range(4):
                for D in range(4):
                    riemann[A, B, C, D] = (omega_derivative[C, A, B, D]
                                            - omega_derivative[D, A, B, C])
                    for E in range(4):
                        riemann[A, B, C, D] += (
                            omega[A, E, C] * omega[E, B, D]
                            - omega[A, E, D] * omega[E, B, C]
                            - structure[E, C, D] * omega[A, B, E])
    coordinate_ricci = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                coordinate_ricci[a, b] += (dchristoffel[c, c, a, b]
                                            - dchristoffel[b, c, a, c])
                for d in range(4):
                    coordinate_ricci[a, b] += (
                        christoffel[c, c, d] * christoffel[d, a, b]
                        - christoffel[c, b, d] * christoffel[d, a, c])
    frame_ricci = np.einsum("Aa,Bb,ab->AB", tetrad, tetrad, coordinate_ricci)
    cartan_ricci = np.einsum("abad->bd", riemann)
    return {"coframe": coframe, "frame": frame, "theta": theta,
            "tetrad": tetrad, "metric": metric, "inverse": inverse,
            "metric_jet": metric_jet,
            "christoffel": christoffel, "dchristoffel": dchristoffel,
            "omega": omega, "omega_derivative": omega_derivative,
            "riemann": riemann, "frame_ricci": frame_ricci,
            "cartan_ricci": cartan_ricci}


def legacy_nonflat_source(reference: dict, psi: np.ndarray, p: np.ndarray,
                          gamma0: float, return_details: bool = False):
    theta = reference["theta"]
    tetrad = reference["tetrad"]
    coframe = reference["coframe"]
    frame = reference["frame"]
    dpsi = np.einsum("Cp,CAB->pAB", theta, p)
    metric = np.einsum("AB,Aa,Bb->ab", psi, theta, theta)
    dmetric = np.zeros((4, 4, 4))
    for pcoord in range(4):
        for a in range(4):
            for b in range(4):
                dmetric[pcoord, a, b] = sum(
                    dpsi[pcoord, A, B] * theta[A, a] * theta[B, b]
                    + psi[A, B] * coframe[A][a].first[pcoord] * theta[B, b]
                    + psi[A, B] * theta[A, a] * coframe[B][b].first[pcoord]
                    for A in range(4) for B in range(4))
    inverse = np.linalg.inv(metric)
    first = np.zeros((4, 4, 4))
    for a in range(4):
        for b in range(4):
            for c in range(4):
                first[a, b, c] = 0.5 * (
                    dmetric[b, a, c] + dmetric[c, a, b] - dmetric[a, b, c])
    christoffel = np.einsum("ad,dbc->abc", inverse, first)
    ref_gamma = reference["christoffel"]
    ref_dgamma = reference["dchristoffel"]
    h_upper = -np.einsum("bc,abc->a", inverse, ref_gamma)
    h_lower = metric @ h_upper
    contracted_first = np.einsum("bc,abc->a", inverse, first)
    constraint = h_lower + contracted_first
    d_inverse = -np.einsum("ac,bd,pcd->pab", inverse, inverse, dmetric)
    d_h_upper = -np.einsum("pbc,abc->pa", d_inverse, ref_gamma) \
                    - np.einsum("bc,pabc->pa", inverse, ref_dgamma)
    d_h_lower = np.einsum("pab,b->pa", dmetric, h_upper) \
                    + np.einsum("ab,pb->pa", metric, d_h_upper)
    lapse = (-inverse[0, 0])**-0.5
    shift = lapse**2 * inverse[0, 1:]
    normal_up = np.concatenate(([1.0 / lapse], -shift / lapse))
    normal_down = metric @ normal_up
    quadratic = (2 * np.einsum("cd,ef,eca,fdb->ab", inverse, inverse,
                                dmetric, dmetric)
                 - 2 * np.einsum("cd,ef,ace,bdf->ab", inverse, inverse,
                                  first, first))
    partial = np.zeros((4, 4))
    for a in range(4):
        for b in range(4):
            nabla_ab = d_h_lower[a, b] - np.dot(christoffel[:, a, b], h_lower)
            nabla_ba = d_h_lower[b, a] - np.dot(christoffel[:, b, a], h_lower)
            value = -nabla_ab - nabla_ba
            value += quadratic[a, b]
            for c in range(4):
                projector = ((normal_down[b] if c == a else 0.0)
                             + (normal_down[a] if c == b else 0.0)
                             - metric[a, b] * normal_up[c])
                value += gamma0 * projector * constraint[c]
            partial[a, b] = value
    contracted_upper = np.einsum("bc,abc->a", inverse, christoffel)
    source = np.zeros((4, 4))
    for A in range(4):
        for B in range(4):
            for a in range(4):
                for b in range(4):
                    source[A, B] += tetrad[A, a] * tetrad[B, b] * partial[a, b]
                    for c in range(4):
                        d_tensor = (frame[A][a].first[c] * tetrad[B, b]
                                    + tetrad[A, a] * frame[B][b].first[c])
                        for d in range(4):
                            dd_tensor = (frame[A][a].second[c, d] * tetrad[B, b]
                                         + frame[A][a].first[c]
                                         * frame[B][b].first[d]
                                         + frame[A][a].first[d]
                                         * frame[B][b].first[c]
                                         + tetrad[A, a] * frame[B][b].second[c, d])
                            source[A, B] += (
                                2 * inverse[c, d] * d_tensor * dmetric[d, a, b]
                                + inverse[c, d] * dd_tensor * metric[a, b])
            source[A, B] -= np.dot(contracted_upper, dpsi[:, A, B])
    if return_details:
        return source, {"partial": partial, "metric": metric,
                        "inverse": inverse, "dmetric": dmetric,
                        "christoffel": christoffel}
    return source


def nonflat_reference_audit(samples_per_kind: int = 16,
                            seed: int = 0xC0A1312) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    maximum_source = 0.0
    maximum_ricci = 0.0
    minimum_curvature = np.inf
    minimum_spin = np.inf
    kinds = ("diagonal", "off_diagonal", "shift", "generic")
    for kind in kinds:
        for _ in range(samples_per_kind):
            reference = manufactured_reference(rng, kind)
            maximum_ricci = max(maximum_ricci, float(np.max(np.abs(
                reference["frame_ricci"] - reference["cartan_ricci"]))))
            minimum_curvature = min(minimum_curvature, float(np.max(np.abs(
                reference["riemann"]))))
            minimum_spin = min(minimum_spin, float(np.max(np.abs(reference["omega"]))))
            psi = random_lorentzian(rng)
            p = rng.normal(scale=0.04, size=(4, 4, 4))
            p = 0.5 * (p + p.swapaxes(1, 2))
            gamma0 = rng.uniform(0.2, 2.0)
            new, _ = frame_covariant_source(
                psi, p, reference["omega"], reference["omega_derivative"],
                reference["riemann"], gamma0)
            old = legacy_nonflat_source(reference, psi, p, gamma0)
            maximum_source = max(maximum_source,
                                 float(np.max(np.abs(new - old))))
    return {"nonflat_samples": float(samples_per_kind * len(kinds)),
            "nonflat_source_max_error": maximum_source,
            "frame_vs_coordinate_ricci_error": maximum_ricci,
            "minimum_reference_curvature": minimum_curvature,
            "minimum_reference_spin": minimum_spin}


def flat_reference_audit(samples: int = 1000, seed: int = 13120701) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    maximum = 0.0
    l2 = 0.0
    zero = np.zeros((4, 4, 4))
    zero4 = np.zeros((4, 4, 4, 4))
    for _ in range(samples):
        psi = random_lorentzian(rng)
        p = rng.normal(scale=0.05, size=(4, 4, 4))
        p = 0.5 * (p + p.swapaxes(1, 2))
        gamma0 = rng.uniform(0.2, 2.0)
        new, _ = frame_covariant_source(psi, p, zero, zero4, zero4, gamma0)
        old_partial, geometry = coordinate_wave_source(psi, p, gamma0)
        contracted = np.einsum("bc,abc->a", geometry["inverse"],
                               geometry["second"])
        old = old_partial - np.einsum("c,cab->ab", contracted, p)
        difference = new - old
        maximum = max(maximum, float(np.max(np.abs(difference))))
        l2 = max(l2, float(np.linalg.norm(difference)))
    return {
        "flat_samples": float(samples),
        "flat_source_max_error": maximum,
        "flat_source_l2_error": l2,
    }


def exact_reference_identity_audit(samples: int = 1000,
                                   seed: int = 2401312) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    maximum_q = 0.0
    maximum_delta = 0.0
    for _ in range(samples):
        # Any metric-compatible antisymmetric spin connection annihilates eta.
        eta = np.diag([-1.0, 1.0, 1.0, 1.0])
        omega_lower = rng.normal(scale=0.3, size=(4, 4, 4))
        omega_lower = 0.5 * (omega_lower - omega_lower.swapaxes(0, 1))
        omega = np.einsum("a,abc->abc", np.diag(eta), omega_lower)
        zero = np.zeros((4, 4, 4))
        zero4 = np.zeros((4, 4, 4, 4))
        _, sectors = frame_covariant_source(eta, zero, omega, zero4, zero4, 1.0)
        maximum_q = max(maximum_q, float(np.max(np.abs(sectors["q"]))))
        maximum_delta = max(
            maximum_delta,
            float(np.max(np.abs(sectors["delta_lower"]))),
            float(np.max(np.abs(sectors["delta"]))),
        )
    return {"exact_reference_q_error": maximum_q,
            "exact_reference_delta_error": maximum_delta}


def main():
    results = flat_reference_audit()
    results.update(exact_reference_identity_audit())
    results.update(nonflat_reference_audit())
    print(json.dumps(results, indent=2, sort_keys=True))
    assert results["flat_source_max_error"] < 1.0e-11
    assert results["flat_source_l2_error"] < 2.0e-11
    assert results["exact_reference_q_error"] < 2.0e-15
    assert results["exact_reference_delta_error"] < 2.0e-15
    assert results["nonflat_source_max_error"] < 2.0e-11
    assert results["frame_vs_coordinate_ricci_error"] < 2.0e-11
    assert results["minimum_reference_curvature"] > 1.0e-5
    assert results["minimum_reference_spin"] > 1.0e-5


if __name__ == "__main__":
    main()
