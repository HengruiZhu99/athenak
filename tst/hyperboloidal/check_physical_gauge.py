#!/usr/bin/env python3
"""Verify the actual 20-field residue and exact Hurwitz conditions at scri.

This is a frozen lower-order audit at S=1 with varying curvature radius, not
a PDE stability proof or nonlinear boundary closure. The determinant and
trace-free constraints are enforced in the compiled extractor's tangent basis.
"""
import json
import subprocess
import sys

import numpy as np
import sympy as sp


def expected_matrix(kappa, d, physical):
    expected = np.zeros((20, 20))
    expected[:4, :4] = [
        [-d if physical else 3, 0, -1, 0],
        [-2, 0, 2 / 3, 4 / 3],
        [0, -3, -2, kappa - 4],
        [0, -3, -2, -1 - 2 * kappa],
    ]
    expected[0, 4] = 1 if physical else 3
    expected[1, 4] = -2
    expected[2, 7] = expected[3, 7] = 3
    expected[12, 17] = -4 / 3
    expected[13, 18] = expected[14, 19] = -1
    for i in range(5):
        expected[12 + i, 7 + i] = 2
        expected[12 + i, 12 + i] = -2
    for i in range(3):
        expected[17 + i, 17 + i] = 2 - kappa
        expected[17 + i, 12 + i] = 4
    return expected


def check_compiled(executable):
    samples = json.loads(subprocess.check_output(
        [executable, "--pole"], text=True))
    maximum = 0.
    for sample in samples:
        a = sample["a"]
        raw_matrix = np.array(sample["M"])
        # Relative lapse/shift and a-scaled trace variables at S=1;
        # A and Lambda have unit scale. Normalize time by c=1/a^2.
        diagonal = np.ones(20)
        diagonal[[0, 2, 3, 4, 5, 6]] = 1 / a
        matrix = raw_matrix * diagonal[None, :] / diagonal[:, None] * a**2
        kappa, xi = sample["kappa"] * a**2, sample["xi"] * a
        d = 1 + 2 * xi
        expected = expected_matrix(kappa, d, sample["physical"])
        error = np.max(np.abs(matrix - expected))
        maximum = max(maximum, error)
        assert error < 2e-8, (a, sample["physical"], kappa, xi, error)
        if sample["physical"] and kappa > 1:
            spectrum = np.linalg.eigvals(matrix)
            negative = spectrum[np.abs(spectrum) > 1e-6]
            assert len(negative) == 12
            assert np.max(negative.real) < 0
            # The eight zero roots (beta and metric) have no Jordan chains.
            assert 20 - np.linalg.matrix_rank(matrix, 1e-6) == 8
        elif sample["physical"] and kappa < 1:
            assert np.max(np.linalg.eigvals(matrix[:4, :4]).real) > 0
        elif not sample["physical"]:
            assert np.max(np.linalg.eigvals(matrix[:4, :4]).real) > 0
        if sample["physical"] and sample["xi"] == 1.5 and sample["kappa"] == 5:
            actual_roots = np.linalg.eigvals(raw_matrix)
            negative_roots = actual_roots[np.abs(actual_roots) > 1e-6]
            print(f"curvature_radius={a:g} effective_kappa={kappa:g} "
                  f"normalized_matrix_error={error:.3g} "
                  f"max_nonzero_pole_real={np.max(negative_roots.real):.9g}")
    print(f"PASS {len(samples)} compiled 20-field pole samples: "
          f"maximum matrix error {maximum:.3g}")
    return samples


def check_hurwitz():
    d, kappa = sp.symbols("d kappa", real=True)
    scalar = sp.Matrix([
        [-d, 0, -1, 0],
        [-2, 0, sp.Rational(2, 3), sp.Rational(4, 3)],
        [0, -3, -2, kappa - 4],
        [0, -3, -2, -1 - 2 * kappa],
    ])
    poly = scalar.charpoly()
    a1, a2, a3, a4 = poly.all_coeffs()[1:]
    assert sp.simplify(a4 - 6 * (d + 3) * (kappa - 1)) == 0
    dd, kk = sp.symbols("dd kk", nonnegative=True)
    # Quartic Routh-Hurwitz. For d>=1, kappa>1, the shifted
    # coefficients and determinants are strictly positive polynomials.
    for expression in (a1, a2, a3, a1 * a2 - a3,
                       a1 * a2 * a3 - a3**2 - a1**2 * a4):
        shifted = sp.Poly(sp.expand(expression.subs(
            {d: 1 + dd, kappa: 1 + kk})), dd, kk)
        assert all(coefficient > 0 for coefficient in shifted.coeffs())
        assert shifted.coeff_monomial(1) > 0
    lam = poly.gen
    longitudinal = sp.Matrix([[-2, -sp.Rational(4, 3)], [4, 2 - kappa]])
    transverse = sp.Matrix([[-2, -1], [4, 2 - kappa]])
    for block, constant in ((longitudinal, 2 * kappa + sp.Rational(4, 3)),
                            (transverse, 2 * kappa)):
        block_poly = block.charpoly()
        target = lam**2 + kappa * lam + constant
        assert sp.expand(block_poly.as_expr().subs(
            block_poly.gen, lam) - target) == 0
    print("PASS exact scalar Routh-Hurwitz: xi>=0 and kappa>1")
    print("PASS A/Lambda quadratics: kappa>0; two tensor roots -2")


def check_hypothetical_projection(executable):
    """Reject an instantaneous preferred source that reopens the old pole."""
    samples = json.loads(subprocess.check_output(
        [executable, "--projection-pole"], text=True))
    maximum = 0.
    for sample in samples:
        kappa, d = sample["kappa"], 1 + 2 * sample["xi"]
        matrix = np.array(sample["M"])
        expected = expected_matrix(kappa, d, True)
        expected[4, 0], expected[4, 4] = d + 3, 2
        error = np.max(np.abs(matrix - expected))
        maximum = max(maximum, error)
        assert error < 2e-8, (kappa, d, error)
        cubic_roots = np.roots([1, 2 * kappa, -9, -12 * kappa])
        expected_positive = np.max(cubic_roots.real)
        roots = np.linalg.eigvals(matrix)
        assert abs(np.max(roots.real) - expected_positive) < 2e-8
        if kappa > 0:
            assert np.sqrt(6) < expected_positive < 3
            assert sum(np.abs(roots) < 1e-6) == 8
            assert 20 - np.linalg.matrix_rank(matrix, 1e-6) == 8
    d, kappa = sp.symbols("d kappa", real=True)
    block = sp.Matrix([
        [-d, 0, -1, 0, 1],
        [-2, 0, sp.Rational(2, 3), sp.Rational(4, 3), -2],
        [0, -3, -2, kappa - 4, 0],
        [0, -3, -2, -1 - 2 * kappa, 0],
        [d + 3, 0, 0, 0, 2],
    ])
    poly = block.charpoly()
    lam = poly.gen
    target = lam * (lam + d + 1) * (
        lam**3 + 2 * kappa * lam**2 - 9 * lam - 12 * kappa)
    assert sp.expand(poly.as_expr() - target) == 0
    selected = next(s for s in samples if s["xi"] == 1.5 and
                    s["kappa"] == 5)
    positive = np.max(np.linalg.eigvals(np.array(selected["M"])).real)
    assert abs(positive - 2.57170948731154) < 2e-8
    print(f"PASS negative candidate audit: {len(samples)} hypothetical "
          f"preferred-projection matrices, error {maximum:.3g}; "
          f"positive pole {positive:.11g}/Omega")
    print("PASS exact scalar/radial-shift factor: "
          "lambda*(lambda+d+1)*(lambda^3+2k*lambda^2-9lambda-12k)")


def check_frozen_scalar_normal():
    """Classify the preferred projection's frozen scalar pole at S=a=1.

    Here u=delta(alpha)+delta(beta_radial), the metric tensor perturbation is
    zero, and all perturbation spatial derivatives are frozen. These leading
    conditions do not establish a full nonlinear constraint manifold.
    """
    kappa = sp.symbols("kappa", real=True)
    matrix = sp.Matrix([
        [3, 0, -1, 0],
        [-2, 0, sp.Rational(2, 3), sp.Rational(4, 3)],
        [0, -3, -2, kappa - 4],
        [0, -3, -2, -1 - 2 * kappa],
    ])
    # In this sector C=delta(|DOmega|^2-wn^2)=delta(chi)+2u,
    # T=delta(P-3wn)=delta(P)-3u, and Theta is physical Theta.
    transform = sp.Matrix([
        [1, 0, 0, 0], [2, 1, 0, 0], [-3, 0, 1, 0], [0, 0, 0, 1],
    ])
    normal = sp.simplify(transform * matrix * transform.inv())
    expected = sp.Matrix([
        [0, 0, -1, 0],
        [0, 0, -sp.Rational(4, 3), sp.Rational(4, 3)],
        [0, -3, 1, kappa - 4],
        [0, -3, -2, -1 - 2 * kappa],
    ])
    assert normal == expected
    tangent = sp.Matrix([1, -2, 3, 0])
    assert matrix * tangent == sp.zeros(4, 1)
    assert transform * tangent == sp.Matrix([1, 0, 0, 0])
    # The leading physical Hamiltonian variation is -6chi-4P-8Theta;
    # a longitudinal metric perturbation would add +6h00.
    hamiltonian = sp.Matrix([[-6, -4, -8]])
    assert hamiltonian * tangent[1:, :] == sp.zeros(1, 1)
    residue = normal[1:, 1:]
    poly = residue.charpoly()
    lam = poly.gen
    assert sp.expand(poly.as_expr() - (
        lam**3 + 2 * kappa * lam**2 - 9 * lam - 12 * kappa)) == 0
    print("PASS frozen scalar tangent (u,chi,P,Theta)=(1,-2,3,0): zero pole")
    print("PASS frozen scalar normal (C,T,Theta): unstable cubic; "
          "leading H_phys=-6chi-4P-8Theta")


def main():
    samples = check_compiled(sys.argv[1])
    check_hurwitz()
    check_hypothetical_projection(sys.argv[1])
    check_frozen_scalar_normal()
    selected = next(s for s in samples if s["physical"] and s["a"] == 1 and
                    s["xi"] == 1.5 and s["kappa"] == 5)
    roots = np.linalg.eigvals(np.array(selected["M"])[:4, :4])
    print("Scalar pole roots at xi=1.5, kappa=5:", roots)
    print("PASS unprojected physical gauge: 12 negative pole roots "
          "and 8 semisimple zeros; "
          "global evolution acceptance remains separate")


if __name__ == "__main__":
    main()
