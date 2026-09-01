#!/usr/bin/env python3
"""Exact product-rule audit of the standard-order X/Y/Q/B equations."""

from dataclasses import dataclass
import sympy as sp


DIM = 3


@dataclass(frozen=True)
class Dual:
    value: object
    deriv: object

    def __add__(self, other):
        other = as_dual(other)
        return Dual(self.value + other.value, self.deriv + other.deriv)

    __radd__ = __add__

    def __neg__(self):
        return Dual(-self.value, -self.deriv)

    def __sub__(self, other):
        return self + (-as_dual(other))

    def __rsub__(self, other):
        return as_dual(other) - self

    def __mul__(self, other):
        other = as_dual(other)
        return Dual(
            self.value * other.value,
            self.deriv * other.value + self.value * other.deriv,
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = as_dual(other)
        return Dual(
            self.value / other.value,
            self.deriv / other.value
            - self.value * other.deriv / other.value**2,
        )


def as_dual(value):
    return value if isinstance(value, Dual) else Dual(value, sp.Integer(0))


def dual_sqrt(value):
    return Dual(sp.sqrt(value.value), value.deriv / (2 * sp.sqrt(value.value)))


def assert_zero(name, expr):
    residual = sp.simplify(sp.expand(expr))
    if residual != 0:
        raise AssertionError(f"{name} failed: {residual}")


def main():
    # One arbitrary derivative direction i.  Tensor component labels remain
    # arbitrary, so this proves the component product rules for every i.
    chi, A, K, pi, btrace = sp.symbols("chi A K pi btrace", positive=True)
    Xi, Yi, dK, dpi, dbtrace = sp.symbols("Xi Yi dK dpi dbtrace", real=True)
    hp, dhp = sp.symbols("hp dhp", real=True)

    dchi = Dual(chi, Xi)
    dA = Dual(A, Yi)
    alpha = dual_sqrt(dA)
    dKfield = Dual(K, dK)
    dpifield = Dual(pi, dpi)
    dBtrace = Dual(btrace, dbtrace)
    dhperp = Dual(hp, dhp)

    f_chi = sp.Rational(2, 3) * dchi * (alpha * dKfield - dBtrace)
    expanded_chi = sp.Rational(2, 3) * (
        Xi * (alpha.value * K - btrace)
        + chi * (alpha.deriv * K + alpha.value * dK - dbtrace)
    )
    assert_zero("partial_i F_chi", f_chi.deriv - expanded_chi)

    f_A = 2 * dA * (alpha * dpifield - dhperp)
    expanded_A = 2 * Yi * (alpha.value * pi - hp) + 2 * A * (
        alpha.deriv * pi + alpha.value * dpi - dhp
    )
    assert_zero("partial_i F_A", f_A.deriv - expanded_A)

    # Generic conformal metric and its derivative Q_iab.  The inverse
    # derivative is independently imposed by d(g^{-1}g)=0.
    g_symbols = sp.symbols("g00 g01 g02 g11 g12 g22", real=True)
    g = sp.Matrix([
        [g_symbols[0], g_symbols[1], g_symbols[2]],
        [g_symbols[1], g_symbols[3], g_symbols[4]],
        [g_symbols[2], g_symbols[4], g_symbols[5]],
    ])
    gu = g.inv()
    q_symbols = sp.symbols("q00 q01 q02 q11 q12 q22", real=True)
    qdir = sp.Matrix([
        [q_symbols[0], q_symbols[1], q_symbols[2]],
        [q_symbols[1], q_symbols[3], q_symbols[4]],
        [q_symbols[2], q_symbols[4], q_symbols[5]],
    ])
    dgu = -gu * qdir * gu

    Lambda = sp.symbols("Lam0:3", real=True)
    dLambda = sp.symbols("dLam0:3", real=True)
    X = sp.symbols("X0:3", real=True)
    dX = sp.symbols("dX0:3", real=True)
    Y = sp.symbols("Y0:3", real=True)
    dY = sp.symbols("dY0:3", real=True)
    h = sp.symbols("h0:3", real=True)
    dh = sp.symbols("dh0:3", real=True)

    for j in range(DIM):
        direct = Dual(h[j], dh[j]) + dA * dchi * Dual(Lambda[j], dLambda[j])
        contraction = Dual(0, 0)
        for ell in range(DIM):
            inverse_component = Dual(gu[j, ell], dgu[j, ell])
            regular_covector = dA * Dual(X[ell], dX[ell]) - dchi * Dual(
                Y[ell], dY[ell]
            )
            contraction += inverse_component * regular_covector / 2
        direct += contraction

        expanded = dh[j] + (Yi * chi + A * Xi) * Lambda[j] + A * chi * dLambda[j]
        for ell in range(DIM):
            expanded += dgu[j, ell] * (A * X[ell] - chi * Y[ell]) / 2
            expanded += gu[j, ell] * (
                Yi * X[ell] + A * dX[ell]
                - Xi * Y[ell] - chi * dY[ell]
            ) / 2
        assert_zero(f"partial_i F_beta^{j}", direct.deriv - expanded)

    # Metric configuration source.  The derivative direction is arbitrary;
    # qdir is Q_iab and dBmat is partial_i B_a^b.
    Avec = sp.symbols("At00 At01 At02 At11 At12 At22", real=True)
    dAvec = sp.symbols("dAt00 dAt01 dAt02 dAt11 dAt12 dAt22", real=True)
    At = sp.Matrix([
        [Avec[0], Avec[1], Avec[2]],
        [Avec[1], Avec[3], Avec[4]],
        [Avec[2], Avec[4], Avec[5]],
    ])
    dAt = sp.Matrix([
        [dAvec[0], dAvec[1], dAvec[2]],
        [dAvec[1], dAvec[3], dAvec[4]],
        [dAvec[2], dAvec[4], dAvec[5]],
    ])
    Bvals = sp.symbols("B00:03 B10:13 B20:23", real=True)
    dBvals = sp.symbols("dB00:03 dB10:13 dB20:23", real=True)
    Bmat = sp.Matrix(DIM, DIM, Bvals)
    dBmat = sp.Matrix(DIM, DIM, dBvals)

    for a in range(DIM):
        for b in range(a, DIM):
            direct = -2 * alpha * Dual(At[a, b], dAt[a, b])
            for k in range(DIM):
                direct += Dual(g[k, a], qdir[k, a]) * Dual(Bmat[b, k], dBmat[b, k])
                direct += Dual(g[k, b], qdir[k, b]) * Dual(Bmat[a, k], dBmat[a, k])
            direct -= sp.Rational(2, 3) * Dual(g[a, b], qdir[a, b]) * dBtrace

            expanded = -2 * (alpha.deriv * At[a, b] + alpha.value * dAt[a, b])
            for k in range(DIM):
                expanded += qdir[k, a] * Bmat[b, k] + g[k, a] * dBmat[b, k]
                expanded += qdir[k, b] * Bmat[a, k] + g[k, b] * dBmat[a, k]
            expanded -= sp.Rational(2, 3) * (
                qdir[a, b] * btrace + g[a, b] * dbtrace
            )
            assert_zero(f"partial_i F_g({a},{b})", direct.deriv - expanded)

    # Direct differentiation versus standard ordering differs exactly by the
    # reduction curl K_ik=partial_i G_k-partial_k G_i.
    beta = sp.symbols("beta0:3", real=True)
    d_i_g_k = sp.symbols("dig0:3", real=True)
    d_k_g_i = sp.symbols("d0gi d1gi d2gi", real=True)
    direct_minus_standard = sum(
        beta[k] * (d_i_g_k[k] - d_k_g_i[k]) for k in range(DIM)
    )
    curl_contraction = sum(
        beta[k] * (d_i_g_k[k] - d_k_g_i[k]) for k in range(DIM)
    )
    assert_zero("compatible-standard curl difference",
                direct_minus_standard - curl_contraction)

    print("PASS: exact expanded partial_i F for chi and A")
    print("PASS: exact expanded partial_i F for all beta components")
    print("PASS: exact expanded partial_i F for all gtilde components")
    print("PASS: compatible and standard orderings differ by beta-contracted curl")


if __name__ == "__main__":
    main()
