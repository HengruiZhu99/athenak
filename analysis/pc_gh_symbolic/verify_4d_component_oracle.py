#!/usr/bin/env python3
"""Exact 4D point-jet oracle for the corrected PC-GH primary equations.

An arbitrary rational 3+1 metric jet is converted to a four-metric jet.  The
first derivatives of an arbitrary GH-constraint covector are then chosen so
that the covariant reduced Einstein equation holds exactly at the point.  The
script independently constructs PC-GH variables from the same jet and compares
their direct D0 derivatives with the corrected K, Atilde, pi, and Lambda RHS.
"""

from dataclasses import dataclass
import sympy as sp


SDIM = 3
DDIM = 4
R = sp.Rational


@dataclass
class Jet2:
    value: object
    grad: list
    hess: list

    @staticmethod
    def constant(value):
        return Jet2(value, [sp.Integer(0)] * DDIM,
                    [[sp.Integer(0)] * DDIM for _ in range(DDIM)])

    def __add__(self, other):
        other = as_jet(other)
        return Jet2(
            self.value + other.value,
            [self.grad[a] + other.grad[a] for a in range(DDIM)],
            [[self.hess[a][b] + other.hess[a][b] for b in range(DDIM)]
             for a in range(DDIM)],
        )

    __radd__ = __add__

    def __neg__(self):
        return Jet2(-self.value, [-v for v in self.grad],
                    [[-v for v in row] for row in self.hess])

    def __sub__(self, other):
        return self + (-as_jet(other))

    def __rsub__(self, other):
        return as_jet(other) - self

    def __mul__(self, other):
        other = as_jet(other)
        return Jet2(
            self.value * other.value,
            [self.grad[a] * other.value + self.value * other.grad[a]
             for a in range(DDIM)],
            [[
                self.hess[a][b] * other.value
                + self.grad[a] * other.grad[b]
                + self.grad[b] * other.grad[a]
                + self.value * other.hess[a][b]
                for b in range(DDIM)
            ] for a in range(DDIM)],
        )

    __rmul__ = __mul__


def as_jet(value):
    return value if isinstance(value, Jet2) else Jet2.constant(value)


def make_jet(value, grad_seed, hess_seed):
    grad = [R(grad_seed(a), 420) for a in range(DDIM)]
    hess = [[R(hess_seed(min(a, b), max(a, b)), 2310)
             for b in range(DDIM)] for a in range(DDIM)]
    return Jet2(value, grad, hess)


def matrix_derivative_inverse(inv, derivative):
    return -inv * derivative * inv


def tf(tensor, metric, inverse):
    return tensor - metric * sp.trace(inverse * tensor) / 3


def assert_zero(name, expr):
    residual = sp.simplify(sp.cancel(expr))
    if residual != 0:
        raise AssertionError(f"{name} failed: {residual}")


def main():
    alpha = make_jet(
        R(3, 2),
        lambda a: (2 * a + 1) * (-1 if a == 2 else 1),
        lambda a, b: (a + 2) * (b + 3) - 5,
    )
    beta_values = [R(1, 7), R(-1, 11), R(1, 13)]
    beta = [
        make_jet(
            beta_values[i],
            lambda a, i=i: (i + 2) * (a + 1) - 4,
            lambda a, b, i=i: (i + 1) * (a + 2) - (b + 1),
        )
        for i in range(SDIM)
    ]

    # Non-diagonal SPD spatial metric with exact unit determinant at the point.
    lower = sp.Matrix([
        [R(2), 0, 0],
        [R(1, 5), R(3), 0],
        [R(-1, 7), R(2, 9), R(1, 6)],
    ])
    gamma_value = lower * lower.T
    assert_zero("det(gamma) at oracle point", gamma_value.det() - 1)
    gamma = [[None] * SDIM for _ in range(SDIM)]
    for i in range(SDIM):
        for j in range(i, SDIM):
            entry = make_jet(
                gamma_value[i, j],
                lambda a, i=i, j=j: (a + 1) * (i + 2) - (j + 1),
                lambda a, b, i=i, j=j: (a + i + 2) * (b + j + 1) - 3,
            )
            gamma[i][j] = gamma[j][i] = entry

    # Construct the four-metric as exact jets.
    g4 = [[Jet2.constant(0) for _ in range(DDIM)] for _ in range(DDIM)]
    g4[0][0] = -alpha * alpha
    for i in range(SDIM):
        for j in range(SDIM):
            g4[0][0] += gamma[i][j] * beta[i] * beta[j]
    for i in range(SDIM):
        g4[0][i + 1] = sum(gamma[i][j] * beta[j] for j in range(SDIM))
        g4[i + 1][0] = g4[0][i + 1]
        for j in range(SDIM):
            g4[i + 1][j + 1] = gamma[i][j]

    gv = sp.Matrix(DDIM, DDIM, lambda mu, nu: g4[mu][nu].value)
    gu = sp.simplify(gv.inv())
    dg = [sp.Matrix(DDIM, DDIM, lambda mu, nu, a=a: g4[mu][nu].grad[a])
          for a in range(DDIM)]
    ddg = [[sp.Matrix(DDIM, DDIM,
                      lambda mu, nu, a=a, b=b: g4[mu][nu].hess[a][b])
            for b in range(DDIM)] for a in range(DDIM)]
    dgu = [matrix_derivative_inverse(gu, dg[a]) for a in range(DDIM)]

    gamma4 = sp.MutableDenseNDimArray.zeros(DDIM, DDIM, DDIM)
    dgamma4 = sp.MutableDenseNDimArray.zeros(DDIM, DDIM, DDIM, DDIM)
    for rho in range(DDIM):
        for mu in range(DDIM):
            for nu in range(DDIM):
                first_kind = [
                    (dg[mu][sigma, nu] + dg[nu][sigma, mu] - dg[sigma][mu, nu]) / 2
                    for sigma in range(DDIM)
                ]
                gamma4[rho, mu, nu] = sum(gu[rho, sigma] * first_kind[sigma]
                                          for sigma in range(DDIM))
                for a in range(DDIM):
                    d_first_kind = [
                        (ddg[a][mu][sigma, nu] + ddg[a][nu][sigma, mu]
                         - ddg[a][sigma][mu, nu]) / 2
                        for sigma in range(DDIM)
                    ]
                    dgamma4[a, rho, mu, nu] = sum(
                        dgu[a][rho, sigma] * first_kind[sigma]
                        + gu[rho, sigma] * d_first_kind[sigma]
                        for sigma in range(DDIM)
                    )

    ricci4 = sp.MutableDenseMatrix(DDIM, DDIM, lambda mu, nu: 0)
    for mu in range(DDIM):
        for nu in range(DDIM):
            ricci4[mu, nu] = sum(
                dgamma4[rho, rho, mu, nu] - dgamma4[nu, rho, mu, rho]
                for rho in range(DDIM)
            ) + sum(
                gamma4[rho, mu, nu] * gamma4[sigma, rho, sigma]
                - gamma4[sigma, mu, rho] * gamma4[rho, nu, sigma]
                for rho in range(DDIM) for sigma in range(DDIM)
            )

    n_cov = sp.Matrix([-alpha.value, 0, 0, 0])
    n_up = sp.simplify(gu * n_cov)
    for mu, expected in enumerate(
        [1 / alpha.value] + [-beta[i].value / alpha.value for i in range(SDIM)]
    ):
        assert_zero(f"normal component {mu}", n_up[mu] - expected)

    c_cov = sp.Matrix([R(2, 17), R(-1, 19), R(3, 23), R(1, 29)])
    c_perp = sp.simplify((n_up.T * c_cov)[0])
    kappa = R(2, 5)
    dc = sp.MutableDenseNDimArray.zeros(DDIM, DDIM)
    for mu in range(DDIM):
        for nu in range(DDIM):
            damping = kappa * (
                (n_cov[mu] * c_cov[nu] + n_cov[nu] * c_cov[mu]) / 2
                - gv[mu, nu] * c_perp / 2
            )
            covariant_sym_derivative = ricci4[mu, nu] + damping
            connection_piece = sum(gamma4[rho, mu, nu] * c_cov[rho]
                                   for rho in range(DDIM))
            # Choose zero antisymmetric partial derivative at the point.
            dc[mu, nu] = covariant_sym_derivative + connection_piece

    # Verify the starting covariant reduced equation exactly.
    for mu in range(DDIM):
        for nu in range(DDIM):
            nabla_sym = (dc[mu, nu] + dc[nu, mu]) / 2 - sum(
                gamma4[rho, mu, nu] * c_cov[rho] for rho in range(DDIM)
            )
            bracket = (
                (n_cov[mu] * c_cov[nu] + n_cov[nu] * c_cov[mu]) / 2
                - gv[mu, nu] * c_perp / 2
            )
            assert_zero(
                f"covariant reduced equation ({mu},{nu})",
                ricci4[mu, nu] - nabla_sym + kappa * bracket,
            )

    G = gamma_value
    Ginv = sp.simplify(G.inv())
    dG = [sp.Matrix(SDIM, SDIM,
                    lambda i, j, a=a: gamma[i][j].grad[a])
          for a in range(DDIM)]
    ddG = [[sp.Matrix(SDIM, SDIM,
                      lambda i, j, a=a, b=b: gamma[i][j].hess[a][b])
            for b in range(DDIM)] for a in range(DDIM)]
    dGinv = [matrix_derivative_inverse(Ginv, dG[a]) for a in range(DDIM)]

    tr_g = [sp.trace(Ginv * dG[a]) for a in range(DDIM)]
    chi = sp.Integer(1)
    dchi = [-tr_g[a] / 3 for a in range(DDIM)]
    ddchi = [[sp.Integer(0)] * DDIM for _ in range(DDIM)]
    for a in range(DDIM):
        for b in range(DDIM):
            dtrace = sp.trace(dGinv[b] * dG[a] + Ginv * ddG[b][a])
            ddchi[a][b] = tr_g[a] * tr_g[b] / 9 - dtrace / 3

    gt = G
    gtu = Ginv
    dgt = [dchi[a] * G + dG[a] for a in range(DDIM)]
    ddgt = [[
        ddchi[a][b] * G + dchi[a] * dG[b] + dchi[b] * dG[a] + ddG[a][b]
        for b in range(DDIM)
    ] for a in range(DDIM)]
    dgtu = [matrix_derivative_inverse(gtu, dgt[a]) for a in range(DDIM)]

    # ADM extrinsic curvature and first derivatives.
    kval = sp.MutableDenseMatrix(SDIM, SDIM, lambda i, j: 0)
    dk = [sp.MutableDenseMatrix(SDIM, SDIM, lambda i, j: 0) for _ in range(DDIM)]
    for i in range(SDIM):
        for j in range(SDIM):
            tij = dG[0][i, j] - sum(
                beta[k].value * dG[k + 1][i, j]
                + G[i, k] * beta[k].grad[j + 1]
                + G[j, k] * beta[k].grad[i + 1]
                for k in range(SDIM)
            )
            kval[i, j] = -tij / (2 * alpha.value)
            for a in range(DDIM):
                dtij = ddG[a][0][i, j] - sum(
                    beta[k].grad[a] * dG[k + 1][i, j]
                    + beta[k].value * ddG[a][k + 1][i, j]
                    + dG[a][i, k] * beta[k].grad[j + 1]
                    + G[i, k] * beta[k].hess[a][j + 1]
                    + dG[a][j, k] * beta[k].grad[i + 1]
                    + G[j, k] * beta[k].hess[a][i + 1]
                    for k in range(SDIM)
                )
                dk[a][i, j] = (
                    -dtij / (2 * alpha.value)
                    + tij * alpha.grad[a] / (2 * alpha.value**2)
                )

    K = sp.trace(Ginv * kval)
    dK = [sp.trace(dGinv[a] * kval + Ginv * dk[a]) for a in range(DDIM)]
    at = kval - G * K / 3
    dat = [
        dchi[a] * (kval - G * K / 3)
        + dk[a] - dG[a] * K / 3 - G * dK[a] / 3
        for a in range(DDIM)
    ]
    at_u = sp.simplify(gtu * at * gtu)
    at2 = sp.trace(gtu * at * gtu * at)

    A = alpha.value**2
    Y = sp.Matrix([2 * alpha.value * alpha.grad[i + 1] for i in range(SDIM)])
    Lvec = Y / alpha.value
    X = sp.Matrix([dchi[i + 1] for i in range(SDIM)])
    W = X
    Bmat = sp.Matrix(SDIM, SDIM,
                     lambda i, j: beta[j].grad[i + 1])
    Btrace = sp.trace(Bmat)

    # Conformal Christoffels and derivatives.
    tg = sp.MutableDenseNDimArray.zeros(SDIM, SDIM, SDIM)
    dtg = sp.MutableDenseNDimArray.zeros(DDIM, SDIM, SDIM, SDIM)
    for m in range(SDIM):
        for i in range(SDIM):
            for j in range(SDIM):
                first = [
                    (dgt[i + 1][ell, j] + dgt[j + 1][ell, i]
                     - dgt[ell + 1][i, j]) / 2
                    for ell in range(SDIM)
                ]
                tg[m, i, j] = sum(gtu[m, ell] * first[ell] for ell in range(SDIM))
                for a in range(DDIM):
                    dfirst = [
                        (ddgt[a][i + 1][ell, j] + ddgt[a][j + 1][ell, i]
                         - ddgt[a][ell + 1][i, j]) / 2
                        for ell in range(SDIM)
                    ]
                    dtg[a, m, i, j] = sum(
                        dgtu[a][m, ell] * first[ell] + gtu[m, ell] * dfirst[ell]
                        for ell in range(SDIM)
                    )
    contracted = sp.Matrix([
        sum(gtu[k, ell] * tg[m, k, ell]
            for k in range(SDIM) for ell in range(SDIM))
        for m in range(SDIM)
    ])
    dcontracted = [sp.Matrix([
        sum(
            dgtu[a][k, ell] * tg[m, k, ell]
            + gtu[k, ell] * dtg[a, m, k, ell]
            for k in range(SDIM) for ell in range(SDIM)
        ) for m in range(SDIM)
    ]) for a in range(DDIM)]

    dcperp = []
    dnup = []
    for a in range(DDIM):
        derivative = sp.Matrix([
            -alpha.grad[a] / alpha.value**2,
            *[
                -beta[i].grad[a] / alpha.value
                + beta[i].value * alpha.grad[a] / alpha.value**2
                for i in range(SDIM)
            ],
        ])
        dnup.append(derivative)
        dcperp.append(sum(
            derivative[mu] * c_cov[mu] + n_up[mu] * dc[a, mu]
            for mu in range(DDIM)
        ))

    Z = sp.simplify(gtu * c_cov[1:4, 0])
    dZ = [sp.simplify(dgtu[a] * c_cov[1:4, 0]
                      + gtu * sp.Matrix([dc[a, i + 1] for i in range(SDIM)]))
          for a in range(DDIM)]
    Lambda = contracted - Z
    dLambda = [dcontracted[a] - dZ[a] for a in range(DDIM)]
    pi = c_perp - K
    dpi = [dcperp[a] - dK[a] for a in range(DDIM)]

    def d0_scalar(derivatives):
        return derivatives[0] - sum(beta[k].value * derivatives[k + 1]
                                    for k in range(SDIM))

    d0K = d0_scalar(dK)
    d0pi = d0_scalar(dpi)
    d0at = dat[0] - sum(
        (beta[k].value * dat[k + 1] for k in range(SDIM)),
        sp.zeros(SDIM),
    )
    d0Lambda = dLambda[0] - sum(
        (beta[k].value * dLambda[k + 1] for k in range(SDIM)),
        sp.zeros(SDIM, 1),
    )

    # Independent Gauss-normal identity sanity check for the metric jet.
    gamma_phys = sp.MutableDenseNDimArray.zeros(SDIM, SDIM, SDIM)
    for m in range(SDIM):
        for i in range(SDIM):
            for j in range(SDIM):
                gamma_phys[m, i, j] = sum(
                    Ginv[m, ell]
                    * (dG[i + 1][ell, j] + dG[j + 1][ell, i]
                       - dG[ell + 1][i, j]) / 2
                    for ell in range(SDIM)
                )
    hess_alpha = sp.MutableDenseMatrix(
        SDIM,
        SDIM,
        lambda i, j: alpha.hess[i + 1][j + 1]
        - sum(gamma_phys[k, i, j] * alpha.grad[k + 1] for k in range(SDIM)),
    )
    lap_alpha = sp.trace(Ginv * hess_alpha)
    rnn = sum(n_up[mu] * n_up[nu] * ricci4[mu, nu]
              for mu in range(DDIM) for nu in range(DDIM))
    assert_zero(
        "Gauss-normal Ricci identity",
        rnn - ((d0K + lap_alpha) / alpha.value - sp.trace(Ginv * kval * Ginv * kval)),
    )
    # Regular conformal geometry.
    calX = sp.MutableDenseMatrix(SDIM, SDIM, lambda i, j: 0)
    calY = sp.MutableDenseMatrix(SDIM, SDIM, lambda i, j: 0)
    for i in range(SDIM):
        for j in range(SDIM):
            calX[i, j] = ddchi[i + 1][j + 1] - sum(
                tg[k, i, j] * X[k] for k in range(SDIM)
            )
            partial_y = (
                2 * alpha.grad[i + 1] * alpha.grad[j + 1]
                + 2 * alpha.value * alpha.hess[i + 1][j + 1]
            )
            calY[i, j] = partial_y - sum(tg[k, i, j] * Y[k]
                                         for k in range(SDIM))
    acal = (calY - Lvec * Lvec.T / 2) / (2 * alpha.value)

    tg_l = sp.MutableDenseNDimArray.zeros(SDIM, SDIM, SDIM)
    for i in range(SDIM):
        for j in range(SDIM):
            for k in range(SDIM):
                tg_l[i, j, k] = sum(gt[i, n] * tg[n, j, k] for n in range(SDIM))
    rcal = sp.MutableDenseMatrix(SDIM, SDIM, lambda i, j: 0)
    for i in range(SDIM):
        for j in range(SDIM):
            principal = -sum(
                gtu[k, ell] * ddgt[k + 1][ell + 1][i, j] / 2
                for k in range(SDIM) for ell in range(SDIM)
            )
            quadratic = 0
            for k in range(SDIM):
                for ell in range(SDIM):
                    quadratic += gtu[k, ell] * sum(
                        tg[m, k, ell] * (tg_l[i, j, m] + tg_l[j, i, m]) / 2
                        + tg[m, k, i] * tg_l[j, m, ell]
                        + tg[m, k, j] * tg_l[i, m, ell]
                        + tg[m, i, k] * tg_l[m, j, ell]
                        for m in range(SDIM)
                    )
            lambda_derivative = sum(
                (gt[k, i] * dLambda[j + 1][k]
                 + gt[k, j] * dLambda[i + 1][k]) / 2
                for k in range(SDIM)
            )
            rcal[i, j] = principal + quadratic + lambda_derivative

    # Independent coordinate-Ricci check of the Brown reduction convention.
    ricci_tilde = sp.MutableDenseMatrix(SDIM, SDIM, lambda i, j: 0)
    for i in range(SDIM):
        for j in range(SDIM):
            ricci_tilde[i, j] = sum(
                dtg[k + 1, k, i, j] - dtg[j + 1, k, i, k]
                for k in range(SDIM)
            ) + sum(
                tg[k, i, j] * tg[ell, k, ell]
                - tg[ell, i, k] * tg[k, j, ell]
                for k in range(SDIM) for ell in range(SDIM)
            )
    brown_derivative = sp.MutableDenseMatrix(
        SDIM,
        SDIM,
        lambda i, j: sum(
            (gt[k, i] * dZ[j + 1][k] + gt[k, j] * dZ[i + 1][k]) / 2
            for k in range(SDIM)
        ),
    )
    for i in range(SDIM):
        for j in range(SDIM):
            assert_zero(
                f"Brown Ricci reduction ({i},{j})",
                rcal[i, j] - ricci_tilde[i, j] + brown_derivative[i, j],
            )

    divX = sp.trace(gtu * calX)
    x2 = (X.T * gtu * X)[0]
    x_dot_l = (X.T * gtu * Lvec)[0]
    H = R(2, 3) * K**2 - at2 + sp.trace(gtu * rcal) + 2 * divX - R(5, 2) * x2
    Sreg = alpha.value * rcal + alpha.value * calX / 2 - alpha.value * (X * X.T) / 4
    Sreg -= acal + (Lvec * X.T + X * Lvec.T) / 4

    # Check the reduced-Hamiltonian relation against an independently built
    # coordinate Ricci tensor of the physical three-metric.
    dgamma_phys = sp.MutableDenseNDimArray.zeros(SDIM, SDIM, SDIM, SDIM)
    for a in range(SDIM):
        aa = a + 1
        for m in range(SDIM):
            for i in range(SDIM):
                for j in range(SDIM):
                    first = [
                        (dG[i + 1][ell, j] + dG[j + 1][ell, i]
                         - dG[ell + 1][i, j]) / 2
                        for ell in range(SDIM)
                    ]
                    dfirst = [
                        (ddG[aa][i + 1][ell, j] + ddG[aa][j + 1][ell, i]
                         - ddG[aa][ell + 1][i, j]) / 2
                        for ell in range(SDIM)
                    ]
                    dgamma_phys[a, m, i, j] = sum(
                        dGinv[aa][m, ell] * first[ell] + Ginv[m, ell] * dfirst[ell]
                        for ell in range(SDIM)
                    )
    ricci3 = sp.MutableDenseMatrix(SDIM, SDIM, lambda i, j: 0)
    for i in range(SDIM):
        for j in range(SDIM):
            ricci3[i, j] = sum(
                dgamma_phys[k, k, i, j] - dgamma_phys[j, k, i, k]
                for k in range(SDIM)
            ) + sum(
                gamma_phys[k, i, j] * gamma_phys[ell, k, ell]
                - gamma_phys[ell, i, k] * gamma_phys[k, j, ell]
                for k in range(SDIM) for ell in range(SDIM)
            )
    d0k = dk[0] - sum(
        (beta[k].value * dk[k + 1] for k in range(SDIM)),
        sp.zeros(SDIM),
    )
    dcov_c = sp.MutableDenseMatrix(
        SDIM,
        SDIM,
        lambda i, j: dc[i + 1, j + 1]
        - sum(gamma_phys[k, i, j] * c_cov[k + 1] for k in range(SDIM)),
    )
    rhs_kij_direct = -hess_alpha + alpha.value * (
        ricci3 + K * kval - 2 * kval * Ginv * kval
        - (dcov_c + dcov_c.T) / 2 - c_perp * kval
        - kappa * G * c_perp / 2
    )
    rhs_kij_direct += kval * Bmat.T + Bmat * kval
    for i in range(SDIM):
        for j in range(SDIM):
            assert_zero(
                f"direct spatial Kij projection ({i},{j})",
                d0k[i, j] - rhs_kij_direct[i, j],
            )
    hadm = sp.trace(Ginv * ricci3) + R(2, 3) * K**2 - at2
    div_z = sum(dZ[i + 1][i] for i in range(SDIM)) + sum(
        tg[i, i, k] * Z[k] for i in range(SDIM) for k in range(SDIM)
    )
    assert_zero("reduced Hamiltonian relation", H - (hadm - div_z))

    rhsK = (
        alpha.value * at2 + alpha.value * K**2 / 3
        - sp.trace(gtu * acal) + x_dot_l / 4
        + alpha.value * (H - K * c_perp + X.dot(Z) / 2)
        - R(3, 2) * kappa * alpha.value * c_perp
    )
    assert_zero("corrected K 4D oracle", d0K - rhsK)

    rhsPi = (
        -alpha.value * at2 - alpha.value * K**2 / 3
        + sp.trace(gtu * acal) - x_dot_l / 4
        + Z.dot(Lvec) / 2 - kappa * alpha.value * c_perp / 2
    )
    assert_zero("pi 4D oracle", d0pi - rhsPi)

    rhsAt = tf(Sreg, gt, gtu)
    rhsAt += at * Bmat.T + Bmat * at - R(2, 3) * at * Btrace
    rhsAt += -2 * alpha.value * at * gtu * at + alpha.value * K * at
    rhsAt += -alpha.value * c_perp * at
    Zcov = gt * Z
    zterm = -(Zcov * X.T + X * Zcov.T) / 2
    zterm -= sum(
        (Z[k] * dgt[k + 1] for k in range(SDIM)),
        sp.zeros(SDIM),
    ) / 2
    direct_reduced_core = tf(
        alpha.value * (ricci3 - (dcov_c + dcov_c.T) / 2) - hess_alpha,
        gt,
        gtu,
    )
    for i in range(SDIM):
        for j in range(SDIM):
            assert_zero(
                f"regular Atilde curvature reduction ({i},{j})",
                direct_reduced_core[i, j]
                - tf(Sreg + alpha.value * zterm, gt, gtu)[i, j],
            )
    rhsAt += alpha.value * tf(zterm, gt, gtu)
    for i in range(SDIM):
        for j in range(SDIM):
            assert_zero(f"corrected Atilde 4D oracle ({i},{j})", d0at[i, j] - rhsAt[i, j])

    rhsLambda = sp.Matrix.zeros(SDIM, 1)
    for i in range(SDIM):
        rhsLambda[i] = sum(
            gtu[k, ell] * beta[i].hess[k + 1][ell + 1]
            for k in range(SDIM) for ell in range(SDIM)
        )
        rhsLambda[i] += sum(
            gtu[i, j] * sum(beta[ell].hess[j + 1][ell + 1]
                            for ell in range(SDIM)) / 3
            for j in range(SDIM)
        )
        rhsLambda[i] += -sum(Lambda[k] * Bmat[k, i] for k in range(SDIM))
        rhsLambda[i] += R(2, 3) * Lambda[i] * Btrace
        rhsLambda[i] += -sum(at_u[i, k] * Lvec[k] for k in range(SDIM))
        rhsLambda[i] += 2 * alpha.value * sum(
            at_u[k, ell] * tg[i, k, ell]
            for k in range(SDIM) for ell in range(SDIM)
        )
        rhsLambda[i] += -3 * alpha.value * sum(at_u[i, k] * X[k]
                                               for k in range(SDIM))
        rhsLambda[i] += -R(4, 3) * alpha.value * sum(
            gtu[i, j] * dK[j + 1] for j in range(SDIM)
        )
        rhsLambda[i] += alpha.value * sum(
            gtu[i, j] * dcperp[j + 1] for j in range(SDIM)
        )
        rhsLambda[i] += -c_perp * sum(gtu[i, j] * Lvec[j]
                                      for j in range(SDIM)) / 2
        rhsLambda[i] += (R(2, 3) * alpha.value * K + kappa * alpha.value) * Z[i]
        assert_zero(f"corrected Lambda 4D oracle {i}", d0Lambda[i] - rhsLambda[i])

    print("PASS: exact covariant reduced equation for all 10 four-tensor components")
    print("PASS: corrected K, Atilde, pi, and Lambda equations against the 4D point jet")


if __name__ == "__main__":
    main()
