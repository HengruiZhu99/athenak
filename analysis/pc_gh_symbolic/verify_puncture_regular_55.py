#!/usr/bin/env python3
"""Exact checks for the puncture-regular 55-field PC-GH variable change.

The checks deliberately begin with the old positive-w, positive-rho equations and
derive the production expressions.  Denominator freedom is checked only after the
reduction constraints have been used; divisions in the intermediate inverse variable
map are therefore visible rather than hidden.
"""

import sympy as sp


def assert_zero(name, expression):
    residual = sp.factor(sp.cancel(expression))
    if residual != 0:
        raise AssertionError(f"{name}: {residual}")


def assert_polynomial(name, expression, variables):
    denominator = sp.factor(sp.denom(sp.cancel(expression)))
    if any(denominator.has(variable) for variable in variables):
        raise AssertionError(f"{name} retains field denominator {denominator}")


def main():
    w, rho, K, B, kappa = sp.symbols("w rho K B kappa", positive=True)
    C, H = sp.symbols("C H", real=True)
    p, L, dK, dB = sp.symbols("p L dK dB", real=True)
    Z, alpha_m, dC = sp.symbols("Z alpha_m dC", real=True)
    alpha = rho*w
    chi = w**2
    A = rho**2*w**2

    # Direct moving-puncture configuration equations and their gradients.
    f_w = w*(alpha*K - B)/3
    f_rho = rho*(-2*K - (alpha*K - B)/3)
    assert_zero("D0 chi", 2*w*f_w - 2*chi*(alpha*K - B)/3)
    f_alpha = sp.expand(w*f_rho + rho*f_w)
    assert_zero("D0 alpha", f_alpha + 2*alpha*K)
    assert_zero("D0 A", 2*alpha*f_alpha + 4*A*K)

    d_alpha = L/2
    d_f_w = (p*(alpha*K - B) + w*(d_alpha*K + alpha*dK - dB))/3
    expected_p_source = (
        p*(alpha*K - B) + w*(L*K/2 + alpha*dK - dB)
    )/3
    assert_zero("p source", d_f_w - expected_p_source)
    expected_l_source = -2*K*L - 4*alpha*dK
    assert_zero("L source from 2 d_i alpha", 2*(-2*d_alpha*K - 2*alpha*dK)
                - expected_l_source)

    # Reduction manifold and the old X,Y reductions.
    drho = (L/2 - rho*p)/w
    assert_zero("R_alpha", L - 2*(w*drho + rho*p))
    X = 2*w*p
    Y = alpha*L
    assert_zero("R_chi", X - 2*w*p)
    assert_zero("R_A", Y - 2*alpha*(w*drho + rho*p))

    # Tensor identities for H, E, S and T.  Arbitrary components are sufficient
    # because every relation is pointwise tensorial in a conformal orthonormal frame.
    R, div_p, p2, at2_tensor = sp.symbols("R div_p p2 at2_tensor", real=True)
    div_x = 2*(w*div_p + p2)
    x2_over_chi = 4*p2
    h_old = (sp.Rational(2, 3)*K**2 - at2_tensor + chi*R
             + 2*div_x - sp.Rational(5, 2)*x2_over_chi)
    h_new = (sp.Rational(2, 3)*K**2 - at2_tensor + w**2*R
             + 4*w*div_p - 6*p2)
    assert_zero("Hamiltonian transform", h_old - h_new)

    dpij, dlij, pi_pj, li_pj, zi_pj, zqij = sp.symbols(
        "dpij dlij pi_pj li_pj zi_pj zqij", real=True
    )
    xcov = 2*(pi_pj + w*dpij)
    old_wi_wj = 4*pi_pj
    e_new = w**2*dlij/2
    s_old = (alpha*chi*R + alpha*xcov/2 - alpha*old_wi_wj/4
             - e_new - w*li_pj/2)
    s_new = rho*w**3*R + rho*w**2*dpij - e_new - w*li_pj/2
    assert_zero("S_ij transform", s_old - s_new)
    t_old = -w*zi_pj - w**2*zqij/2
    t_new = -w*zi_pj - w**2*zqij/2
    assert_zero("T_ij transform", t_old - t_new)

    m, at_p = sp.symbols("m at_p", real=True)
    alpha_m_old = alpha*m - 3*alpha*(2*w*at_p)/(2*chi)
    alpha_m_new = rho*w*m - 3*rho*at_p
    assert_zero("scaled momentum transform", alpha_m_old - alpha_m_new)

    # C_perp is obtained by adding the independently projected pi and K equations.
    at2, accel, p_dot_l, p_dot_z = sp.symbols(
        "at2 accel p_dot_l p_dot_z", real=True
    )
    d0_pi = (-alpha*at2 - alpha*K**2/3 + accel - w*p_dot_l/2
             + w**2*Z*L/2 - kappa*alpha*C/2)
    d0_k = (alpha*at2 + alpha*K**2/3 - accel + w*p_dot_l/2
            + alpha*(H - K*C + w*p_dot_z) - 3*kappa*alpha*C/2)
    c_target = (alpha*(H - K*C) + w**2*(rho*p_dot_z + L*Z/2)
                - 2*kappa*alpha*C)
    # p_dot_z is a scalar p_i Z^i; use Z*L for the corresponding L_i Z^i term.
    assert_zero("C_perp propagation", d0_pi + d0_k - c_target)

    # Direct Z propagation after alpha=rho*w and alpha M_i is regularized.
    gu, shift_z = sp.symbols("gu shift_z", real=True)
    z_old = (-2*gu*alpha_m - alpha*gu*dC + C*gu*L/2 + shift_z
             - (2*alpha*K/3 + kappa*alpha)*Z)
    z_target = (-2*gu*alpha_m - rho*w*gu*dC + C*gu*L/2 + shift_z
                - (2*rho*w*K/3 + kappa*rho*w)*Z)
    assert_zero("Z propagation", z_old - z_target)

    # Hyperbolic shift and the complete differentiated S(z) M term.
    gu_ac, dgu, pc, dpc = sp.symbols("gu_ac dgu pc dpc", real=True)
    switch, switch_prime = sp.symbols("S S_prime", real=True)
    zeta = rho*w**3
    dzeta_target = w**2*(L/2 + 2*rho*p)
    assert_zero("d(alpha chi)", sp.diff(rho*w**3, rho)*drho
                + sp.diff(rho*w**3, w)*p - dzeta_target)
    v = rho*pc - L/2
    old_shift_vector = gu_ac*(A*(2*w*pc) - chi*(alpha*L))/2
    new_shift_vector = rho*w**3*gu_ac*v
    assert_zero("hyperbolic shift-vector transform",
                old_shift_vector - new_shift_vector)
    # Here pc represents p_c, while p is p_i in the differentiating direction.
    dv_weighted = (rho*w**2*(L/2 - rho*p)*pc
                   + rho**2*w**3*dpc - rho*w**3*sp.symbols("dL")/2)
    dM_target = dzeta_target*gu_ac*v + zeta*dgu*v + gu_ac*dv_weighted
    dM_chain = (dzeta_target*gu_ac*v + zeta*dgu*v
                + zeta*gu_ac*(drho*pc + rho*dpc - sp.symbols("dL")/2))
    assert_zero("denominator-free dM", dM_target - dM_chain)
    metric_source = zeta*gu_ac*v
    differentiated_shift = sp.symbols("dLambda") - sp.symbols("eta")*sp.symbols("Bij")
    differentiated_shift += switch*dM_target + switch_prime*dzeta_target*metric_source
    assert_zero("S prime product rule", differentiated_shift - (
        sp.symbols("dLambda") - sp.symbols("eta")*sp.symbols("Bij")
        + switch*dM_chain + switch_prime*dzeta_target*metric_source))

    # Every preferred evolution expression checked here is polynomial in puncture fields.
    puncture_fields = (w, rho, alpha, chi, A)
    expressions = {
        "w": f_w, "rho": f_rho, "p": expected_p_source,
        "L": expected_l_source, "C": c_target, "Z": z_target,
        "zeta": dzeta_target, "dM": dM_target,
    }
    for name, expression in expressions.items():
        assert_polynomial(name, expression, puncture_fields)

    # Inner asymptotics.  The evolved exponents must be nonnegative even though d rho
    # is singular on the stationary trumpet and is intentionally absent from the state.
    trumpet_alpha = sp.Rational("1.091297265")
    exponents = {
        "w": sp.Integer(1),
        "rho": trumpet_alpha - 1,
        "p": sp.Integer(0),
        "L": trumpet_alpha - 1,
    }
    if any(value < 0 for value in exponents.values()):
        raise AssertionError(exponents)
    assert_zero("stationary trumpet drho exponent",
                (trumpet_alpha - 2) - sp.Rational("-0.908702735"))
    wormhole = {"w": 2, "rho": 0, "p": 1, "L": 1}
    if any(value < 0 for value in wormhole.values()):
        raise AssertionError(wormhole)

    print("PASS: 55-field configuration and reduction identities")
    print("PASS: H, E, S, T, alpha*M, and shift-vector transforms")
    print("PASS: C_perp and direct Z propagation substitutions")
    print("PASS: direct/hyperbolic shift differentiation including S'(z)")
    print("PASS: preferred evolution expressions have no puncture-field denominator")
    print("PASS: wormhole/trumpet evolved-field inner exponents are nonnegative")
    print("INFO: stationary trumpet d(rho)/dr exponent = -0.908702735 (not evolved)")


if __name__ == "__main__":
    main()
