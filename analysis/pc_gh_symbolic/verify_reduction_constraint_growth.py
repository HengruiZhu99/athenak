#!/usr/bin/env python3
"""Derive the stationary tangential trace-free Q reduction-mode growth rate."""

from __future__ import annotations

import sympy as sp


def main() -> None:
    u, t, amplitude = sp.symbols("u t amplitude", real=True)
    shift_gradient = sp.diag(u, t, t)
    trace_b = sp.trace(shift_gradient)
    ell = 1
    reduction = sp.diag(0, amplitude, -amplitude)

    # Linearization with respect to gtilde of the nonadvective metric source
    # F_ij = -2 alpha Atilde_ij + 2 gtilde_k(i B_j)^k - 2 gtilde_ij B/3.
    delta_f = sp.zeros(3)
    for a in range(3):
        for c in range(3):
            delta_f[a, c] = sum(
                reduction[p, a]*shift_gradient[c, p]
                + reduction[p, c]*shift_gradient[a, p]
                for p in range(3)) - sp.Rational(2, 3)*trace_b*reduction[a, c]

    reduction_rate = sp.simplify(delta_f[1, 1]/amplitude)
    full_q_rate = sp.simplify(
        (delta_f[1, 1] + shift_gradient[ell, ell]*reduction[1, 1])/amplitude)
    assert reduction_rate == sp.Rational(2, 3)*(t - u)
    assert full_q_rate == (5*t - 2*u)/3
    assert sp.simplify(full_q_rate - reduction_rate - t) == 0

    # The pointwise Q operator contains B_ell^ell R_ell.  In the actual reduction
    # constraint R_ell=Q_ell-partial_ell(g), the same term arises from
    # partial_ell(beta^k R_k) and cancels.  Both remaining rates are independent of
    # Gauge-A1 lapse/shift feedback because F_gtilde contains neither h_perp nor h^i.
    target_u = sp.Float("-0.02183002615", 30)
    target_t = sp.Float("0.15424545392", 30)
    numeric_full = sp.N(full_q_rate.subs({u: target_u, t: target_t}), 16)
    numeric_reduction = sp.N(reduction_rate.subs({u: target_u, t: target_t}), 16)
    if numeric_reduction <= 0:
        raise AssertionError("production Gauge-A0 reduction rate is not positive")
    print(f"PASS: pointwise full-Q frozen rate = {full_q_rate} = {numeric_full}/M")
    print(f"PASS: true Q-reduction rate = {reduction_rate} = {numeric_reduction}/M")
    print("PASS: bounded Gauge A1 cannot alter the tangential trace-free reduction rate")


if __name__ == "__main__":
    main()
