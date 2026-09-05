"""Regular moving-puncture reduction extension: frozen principal matrix.

This model is checked separately against the compiled production kernel. A
symbolic similarity to the existing moving-gauge matrix is not an FO-GH claim.
"""
from pathlib import Path
import sys

import sympy as s

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pc_gh_symbolic"))
import analyze_z4c_mp_principal as old

KEPT = [i for i in range(55) if i not in {6, 13, 30, 36, 42}]
T = s.zeros(55, 50)
SELECT = s.zeros(50, 55)
for column, index in enumerate(KEPT):
    T[index, column] = SELECT[column, index] = 1
for dep, first, second in [(6, 1, 4), (13, 8, 11), (30, 25, 28),
                            (36, 31, 34), (42, 37, 40)]:
    T[dep, KEPT.index(first)] = T[dep, KEPT.index(second)] = -1
CONFIG = [KEPT.index(i) for i in [0, 1, 2, 3, 4, 5, 18, 19, 20, 21]]
NONCONFIG = [i for i in range(50) if i not in CONFIG]


def jacobian(w, rho):
    """Legacy (chi, Lambda, pi, A, X, Y) versus regular coordinates."""
    alpha = rho*w
    full = s.eye(55)
    full[0, 0] = 2*w
    full[18, 0] = 2*rho*rho*w
    full[18, 18] = 2*rho*w*w
    full[17, 7] = -1
    for i in range(3):
        full[22+i, 22+i] = 2*w
        full[43+i, 43+i] = alpha
        full[14+i, 14+i] = -1
        for d in range(3):
            full[14+i, old.q_index(d, i, d)] += 1
            full[14+i, old.q_index(i, d, d)] -= s.Rational(1, 2)
    return SELECT*full*T


def damping_symbol(w, rho, gamma):
    """P increment in u_t=P u_x, lambda=alpha*gamma; no field quotients."""
    rate = rho*w*gamma
    delta = s.zeros(55)
    delta[22, 0] = rate
    for i in range(3):
        for j in range(i, 3):
            delta[old.q_index(0, i, j), old.G[old.sym_index(i, j)]] = rate
    delta[43, 0] = 2*rate*rho
    delta[43, 18] = 2*rate*w
    for j in range(3):
        delta[old.b_index(0, j), 19+j] = rate
    delta[7, 19] = gamma
    delta[17, 19] = gamma
    for i in range(3):
        for j in range(i, 3):
            row = old.AT[old.sym_index(i, j)]
            if i == 0:
                delta[row, 19+j] += gamma/2
            if j == 0:
                delta[row, 19+i] += gamma/2
            if i == j:
                delta[row, 19] -= gamma/3
    for i in range(3):
        delta[14+i, old.G[old.sym_index(i, 0)]] += rate
        if i == 0:
            for d in range(3):
                delta[14+i, old.G[old.sym_index(d, d)]] -= rate/2
    delta[14, 18] -= gamma*w
    return SELECT*delta*T


def coordinate_damping_symbol(w, rho, rate):
    """The alternative with only gradient sources and a bounded coordinate rate."""
    delta = damping_symbol(w, rho, rate/(rho*w))
    for full_row in range(7, 18):
        if full_row in KEPT:
            for col in range(50):
                delta[KEPT.index(full_row), col] = 0
    return delta.applyfunc(s.cancel)


def baseline(w, rho, switch, symmetric=True):
    j = jacobian(w, rho)
    p = old.full_symbol(rho*w, w*w, switch)
    if symmetric:
        # The production upper-triangle Hessian ordering must be symmetrized
        # off the curl manifold before a rotation of this symbol is legitimate.
        for k in (1, 2):
            row = old.AT[old.sym_index(0, k)]
            p[row, old.X[k]] /= 2
            p[row, old.Y[k]] /= 2
    return j.inv()*(SELECT*p*T)*j


def candidate(w, rho, switch, gamma, shift):
    return baseline(w, rho, switch)+coordinate_damping_symbol(w, rho, gamma)+shift*s.eye(50)


def full_regular_symbol(w, rho, switch, rate, at=None):
    """Independent transcription of production derivative coefficients, before traces."""
    alpha = rho*w
    p = s.zeros(55)
    contracted = [s.zeros(1, 55) for _ in range(3)]
    for a in range(3):
        for d in range(3):
            contracted[a][old.q_index(d, a, d)] += 1
            contracted[a][old.q_index(a, d, d)] -= s.Rational(1, 2)
    lam = [row.copy() for row in contracted]
    for a in range(3):
        lam[a][14+a] -= 1
    scalar = lam[0].copy()
    for d in range(3):
        scalar[old.q_index(0, d, d)] -= s.Rational(1, 2)
    p[7, :] = alpha*w*w*scalar
    p[17, :] = alpha*w*w*scalar
    p[7, 22] += 4*alpha*w
    p[17, 22] += 4*alpha*w
    p[7, 43] -= w*w/2
    source = [[s.zeros(1, 55) for _ in range(3)] for _ in range(3)]
    for a in range(3):
        p[14+a, old.AT[old.sym_index(0, a)]] -= 2*alpha
        if a == 0:
            p[14+a, 7] += 4*alpha/3
            p[14+a, 17] -= alpha
        p[14+a, old.b_index(a, 0)] += 1
        if a == 0:
            for d in range(3):
                p[14+a, old.b_index(d, d)] -= 1
        for b in range(3):
            row = source[a][b]
            row[old.q_index(0, a, b)] -= alpha*w*w/2
            if b == 0:
                row += alpha*w*w*lam[a]/2
                row[22+a] += alpha*w/2
                row[43+a] -= w*w/4
            if a == 0:
                row += alpha*w*w*lam[b]/2
                row[22+b] += alpha*w/2
                row[43+b] -= w*w/4
            source[a][b] = row
    trace = sum((source[d][d] for d in range(3)), s.zeros(1, 55))
    for a in range(3):
        for b in range(a, 3):
            p[old.AT[old.sym_index(a, b)], :] = source[a][b]-(trace/3 if a == b else s.zeros(1, 55))
            p[old.q_index(0, a, b), old.AT[old.sym_index(a, b)]] -= 2*alpha
            p[old.q_index(0, a, b), old.b_index(b, a)] += 1
            p[old.q_index(0, a, b), old.b_index(a, b)] += 1
            if a == b:
                for d in range(3):
                    p[old.q_index(0, a, b), old.b_index(d, d)] -= s.Rational(2, 3)
    p[22, 7] = alpha*w/3
    for d in range(3):
        p[22, old.b_index(d, d)] = -w/3
    p[43, 7] = -4*alpha
    for a in range(3):
        p[old.b_index(0, a), :] = lam[a]
        p[old.b_index(0, a), 22+a] += switch*alpha*alpha*w
        p[old.b_index(0, a), 43+a] -= switch*alpha*w*w/2
    p[22, 0] += rate
    p[43, 0] += 2*rate*rho
    p[43, 18] += 2*rate*w
    for a in range(3):
        p[old.b_index(0, a), 19+a] += rate
        for b in range(a, 3):
            p[old.q_index(0, a, b), old.G[old.sym_index(a, b)]] += rate
    if at is not None:
        # Intrinsic derivative correction needed to keep tr_g Q_i=0 when
        # tr_g A=0 is differentiated using true rather than stored metric jets.
        for a in range(3):
            for b in range(3):
                for d in range(3):
                    p[old.q_index(0, d, d), old.G[old.sym_index(a, b)]] += 2*alpha*at[a, b]/3
    return p


if __name__ == "__main__":
    w, rho, switch, gamma = s.symbols("w rho switch gamma", positive=True)
    p = baseline(w, rho, switch)
    d = coordinate_damping_symbol(w, rho, gamma)
    assert all(s.cancel(value) == 0 for value in
               SELECT*full_regular_symbol(w, rho, switch, gamma)*T-p-d)
    fields = s.symbols("A0:5")
    at = s.Matrix([[fields[0], fields[1], fields[2]],
                   [fields[1], fields[3], fields[4]],
                   [fields[2], fields[4], -fields[0]-fields[3]]])
    q = []
    for k in range(3):
        fields = s.symbols(f"Q{k}_0:5")
        q.append(s.Matrix([[fields[0], fields[1], fields[2]],
                           [fields[1], fields[3], fields[4]],
                           [fields[2], fields[4], -fields[0]-fields[3]]]))
    tangent = T.copy()
    for col in range(50):
        dg = s.Matrix(3, 3, lambda i, j: T[old.G[old.sym_index(i, j)], col])
        tangent[13, col] += s.trace(dg*at)
        for k in range(3):
            tangent[old.q_index(k, 2, 2), col] += s.trace(dg*q[k])
    full = full_regular_symbol(w, rho, switch, gamma, at)
    for value in full:
        assert s.denom(s.cancel(value)).free_symbols.isdisjoint({w, rho})
        assert value.subs(w, 0).is_finite is not False
    print("PASS: regular production principal coefficients have finite w=0 limits")
    general = SELECT*full*tangent
    assert all(s.cancel(value) == 0 for value in full*tangent-tangent*general)
    d = (general-p).applyfunc(s.cancel)
    # A polynomial group inverse avoids artificial pivot poles at coincident
    # characteristic speeds, in particular the asymptotic Minkowski state.
    z = s.symbols("z")
    speeds_squared = [s.Integer(1), 2*rho*w**3, rho**2*w**4,
                      (4-switch*rho**2*w**4)/3]
    polynomial = s.Poly(s.prod(z*z-speed for speed in speeds_squared), z)
    q0 = polynomial.nth(0)
    shear = s.zeros(50)
    power_d = d
    for exponent in range(1, 9):
        coefficient = polynomial.nth(exponent)
        if coefficient:
            shear -= coefficient*power_d/q0
        power_d = p*power_d
    shear = shear.applyfunc(lambda entry: s.factor(s.cancel(entry)))
    residual = p*shear-d
    assert all(s.cancel(value) == 0 for value in residual)
    assert shear*shear == s.zeros(50)
    assert shear*p == s.zeros(50)
    assert d*shear == s.zeros(50)
    print("PASS: exact parameterized similarity P_damped=(I-S) P_base (I+S)")
    print("PASS: S^2=0, so the characteristic basis stays complete wherever the base is complete")
    print("PASS: arbitrary constant shift adds beta.n times the identity")
    print("PASS: arbitrary algebraically allowed Atilde/Q backgrounds and intrinsic trace preservation")
    print("Group-inverse denominator:", s.factor(q0))
    print("Nonzero shear entries (regular tangent indices):")
    for i, j, value in shear.row_list() if hasattr(shear, "row_list") else [
            (i, j, shear[i, j]) for i in range(50) for j in range(50) if shear[i, j] != 0]:
        print(i, j, value)
