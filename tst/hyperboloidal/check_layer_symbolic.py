"""Independent symbolic audit of the source-derived conformal Z4c symbol.

Exact constrained-symbol proof; pair with check_kernel_symbol.py, which
extracts the symbol from the compiled final tensor/gauge RHS. Requires sympy.
"""
import sympy as s

f, mu, ea, ec, lam = s.symbols("f mu ea ec lam", real=True)
q = (4 * mu - 2 * ec) / 3

# Local conformal spatial orthonormal frame; alpha=chi=1 by frozen scaling.
# State: d=alpha_s, c=chi_s, h=g_ss,s, k=delta P/Omega,
#        t=delta Theta_phys/Omega, A=A_ss, L=Lambda_s, b=beta_s,s.
M = s.zeros(8)
M[0, 3] = -f
M[1, 3], M[1, 4], M[1, 7] = s.Rational(2, 3), s.Rational(4, 3), -s.Rational(2, 3)
M[2, 5], M[2, 7] = -2, s.Rational(4, 3)
M[3, 0] = -1
M[4, 1], M[4, 6] = 1, s.Rational(1, 2)
M[5, 0], M[5, 1] = -s.Rational(2, 3), s.Rational(1, 3)
M[5, 2], M[5, 6] = -s.Rational(1, 2), s.Rational(2, 3)
M[6, 3], M[6, 4], M[6, 7] = -s.Rational(4, 3), -s.Rational(2, 3), s.Rational(4, 3)
M[7, 0], M[7, 1], M[7, 6] = -ea, ec, mu

expected = (lam**2 - f) * (lam**2 - 1)**2 * (lam**2 - q)
cp = M.charpoly()
assert s.simplify(cp.as_expr().subs(cp.gen, lam) - expected) == 0
print("scalar charpoly:", s.factor(expected))

V = s.Matrix([[0, -2, 0, 1], [-s.Rational(1, 2), 0, s.Rational(1, 2), 0],
              [0, 0, 0, 1], [0, 0, mu, 0]])
vp = V.charpoly()
assert s.simplify(vp.as_expr().subs(vp.gen, lam) - (lam**2 - 1) * (lam**2 - mu)) == 0
print("vector charpoly:", s.factor(vp.as_expr()))

for label, coefficients in [
    ("old harmonic/Gamma scri", {f: 1, mu: s.Rational(3, 4), ea: 0, ec: 0}),
    ("harmonic lapse/sub-light Gamma", {f: 1, mu: s.Rational(3, 8), ea: 0, ec: 0}),
    ("harmonic lapse/harmonic shift", {f: 1, mu: 1, ea: 1, ec: s.Rational(1, 2)}),
]:
    eig = M.subs(coefficients).eigenvects()
    print(label, [(x, multiplicity, len(v)) for x, multiplicity, v in eig])

# Explicit left eigenbasis. p=sqrt(f), z=sqrt(q) avoid branch simplification.
p, z, E_a, E_c = s.symbols("p z E_a E_c", nonzero=True, real=True)
F, Q = p**2, z**2
Mu = (3 * Q + 2 * E_c) / 4
rows, eigenvalues = [], []
for sign in [-1, 1]:
    rows.append([0, -sign, -s.Rational(sign, 2), -s.Rational(2, 3),
                 -s.Rational(4, 3), 1, 0, 0])
    rows.append([0, 2, 0, 0, 2 * sign, 0, 1, 0])
    eigenvalues.extend([sign, sign])
for sign in [-1, 1]:
    rows.append([-sign / p, 0, 0, 1, 0, 0, 0, 0])
    eigenvalues.append(sign * p)
for sign in [-1, 1]:
    eigenvalue = sign * z
    rows.append([eigenvalue * (E_a - 1) / (F - Q),
                 eigenvalue * (E_c - s.Rational(1, 2)) / (Q - 1), 0,
                 (Q - F * E_a) / (F - Q), (E_c - Q / 2) / (Q - 1), 0,
                 eigenvalue * (Mu - 1) / (Q - 1), 1])
    eigenvalues.append(eigenvalue)
L = s.Matrix(rows)
MM = M.subs({f: F, mu: Mu, ea: E_a, ec: E_c})
assert all(s.cancel(value) == 0 for value in L * MM - s.diag(*eigenvalues) * L)
assert s.factor(L.det(method="domain-ge")) == -24 * z / p
print("left eigenbasis determinant:", -24 * z / p)

# Smooth Gamma -> harmonic family. All apparent singular ratios cancel.
W, Fex, q0 = s.symbols("W Fex q0", real=True)
blend_f = 1 + (1 - W) * Fex
blend_q = q0 + (1 - q0) * W
blend_mu = (1 - W) * 3 * q0 / 4 + W
blend_ea, blend_ec = W, W / 2
ratios = [
    (blend_ea - 1) / (blend_f - blend_q),
    (blend_ec - s.Rational(1, 2)) / (blend_q - 1),
    (blend_q - blend_f * blend_ea) / (blend_f - blend_q),
    (blend_ec - blend_q / 2) / (blend_q - 1),
    (blend_mu - 1) / (blend_q - 1),
]
print("blend canceled shift eigenfield ratios:")
for value in ratios:
    print(" ", s.factor(value))
assert s.cancel(ratios[0] + 1 / (Fex + 1 - q0)) == 0

# Full limiting left basis remains nonsingular at W=1 (f=q=mu=1).
limit_rows = []
for sign in [-1, 1]:
    limit_rows.append([0, -sign, -s.Rational(sign, 2), -s.Rational(2, 3),
                       -s.Rational(4, 3), 1, 0, 0])
    limit_rows.append([0, 2, 0, 0, 2 * sign, 0, 1, 0])
for sign in [-1, 1]:
    limit_rows.append([-sign, 0, 0, 1, 0, 0, 0, 0])
for sign in [-1, 1]:
    limit_rows.append([-sign / (Fex + 1 - q0), sign / (2 * (1 - q0)), 0,
                       (q0 - Fex) / (Fex + 1 - q0), q0 / (2 * (1 - q0)), 0,
                       sign * (1 - s.Rational(3, 4) * q0) / (1 - q0), 1])
limitL = s.Matrix(limit_rows)
assert s.factor(limitL.det(method="domain-ge")) == -24
assert all(s.cancel(value) == 0 for value in
           limitL * M.subs({f: 1, mu: 1, ea: 1, ec: s.Rational(1, 2)})
           - s.diag(-1, -1, 1, 1, -1, 1, -1, 1) * limitL)
print("full harmonic blend limiting eigenbasis determinant: -24")

print("All symbolic assertions passed.")
