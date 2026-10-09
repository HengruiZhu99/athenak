"""Reference residue identities and the unfinished live scri closure. Requires sympy."""
import sympy as s

r = s.symbols('r', positive=True)
omega_symbol, b = s.Function('omega_symbol')(r), s.Function('b')(r)
L = omega_symbol - r * s.diff(omega_symbol, r)
A2 = omega_symbol**2 + b**2
q = L**2 / A2
w = b * s.diff(omega_symbol, r) / L
kr, kt = -s.diff(b, r) / L, -b / (r * L)

# Exact reference conformal residues, computed from the ADM geometry.
Srr_num = s.diff(omega_symbol, r, 2) - s.diff(q, r) * \
    s.diff(omega_symbol, r) / (2 * q) + w * q * kr
Srr = s.diff(omega_symbol, r, 2) / L + s.diff(omega_symbol, r)**2 / A2
assert s.simplify(Srr_num - omega_symbol * Srr) == 0
Stt_num = r * s.diff(omega_symbol, r) / q + w * r**2 * kt
assert s.simplify(Stt_num - r * omega_symbol**2 * s.diff(omega_symbol, r) / L**2) == 0
Nnum = s.diff(omega_symbol, r)**2 / q - w**2
assert s.simplify(Nnum - omega_symbol**2 * s.diff(omega_symbol, r)**2 / L**2) == 0

# 4D divergence expression for Box(Omega), with sqrt(-g)=L*r^2*sin(theta).
Box = s.diff(r**2 * omega_symbol**2 * s.diff(omega_symbol, r) / L, r) / (L * r**2)
Wref = 2 * s.diff(omega_symbol, r)**2 / L**2 + omega_symbol * (
    s.diff(omega_symbol, r, 2) / L**2 + 2 * s.diff(omega_symbol, r) / (r * L**2)
    - s.diff(omega_symbol, r) * s.diff(L, r) / L**3)
assert s.simplify(Box - omega_symbol * Wref) == 0

# Source projection uses a Euclidean nonzero gradient norm, never the null 4-norm.
nx, ny, nz, Fx, Fy, Fz, H, Omega, W = s.symbols('nx ny nz Fx Fy Fz H Omega W')
n = s.Matrix([nx, ny, nz])
F = s.Matrix([Fx, Fy, Fz])
v = n / (n.dot(n))
projected = F + v * (H - Omega * W - n.dot(F))
assert s.simplify(n.dot(projected) - (H - Omega * W)) == 0
print('PASS: reference Hessian/shear and null residues, factored '
      'Box(Omega), preferred-conformal source projection')

# Transverse full eigenbasis stays complete even at the harmonic endpoint.
mu, z = s.symbols('mu z', positive=True)
V = s.Matrix([[0, -2, 0, 1], [-s.Rational(1, 2), 0, s.Rational(1, 2), 0],
              [0, 0, 0, 1], [0, 0, mu, 0]])
B = s.Matrix([[s.Rational(1, 2), 1, -s.Rational(1, 2), 0],
              [-s.Rational(1, 2), 1, s.Rational(1, 2), 0],
              [0, 0, -z, 1], [0, 0, z, 1]])
assert B * V.subs(mu, z**2) == s.diag(-1, 1, -z, z) * B
assert s.simplify(B.det() + 2 * z) == 0
assert B.subs(z, 1).rank() == 4
print('PASS: transverse characteristic fields and complete harmonic endpoint')

# Physical-Theta equation as implemented in ConformalRHS (C_Z4c=0).
# Theta=omega_symbol*tau and P=omega_symbol*Q+3*wn. D0Theta includes -alpha*wn*tau
# from beta.grad(omega_symbol*tau). Damping k1 is the kernel argument, not its input knob.
omega_symbol, wn, Q, tau, k1, k2, alpha, lap, N, Rtheta, advtau = s.symbols(
    'omega_symbol wn Q tau k1 k2 alpha lap N Rtheta advtau')
K = omega_symbol * (Q + 2 * tau) + 3 * wn
gradient2 = wn**2 + omega_symbol**2 * N
rhs = (omega_symbol * alpha * Rtheta + 2 * alpha * lap
       + alpha * (K**2 / 3 - 3 * gradient2
                  - (3 * wn + k1 * (2 + k2)) * omega_symbol * tau) / omega_symbol
       - alpha * wn * tau + omega_symbol * advtau)
factored = alpha * (2 * lap + 2 * wn * Q - k1 * (2 + k2) * tau) + omega_symbol * (
    alpha * (Rtheta - 3 * N + (Q + 2 * tau)**2 / 3) + advtau)
assert s.simplify(rhs - factored) == 0
print('Theta/Omega boundary numerator:', alpha *
      (2 * lap + 2 * wn * Q - k1 * (2 + k2) * tau))

# Exact physical-trace mass matrix and the two transformed pole numerators.
rP, sP, ra, sa, bx, by, bz, Ox, Oy, Oz = s.symbols('rP sP ra sa bx by bz Ox Oy Oz')
Qdot = (rP + sP / omega_symbol + 3 * (Ox * bx + Oy * by + Oz * bz) /
        alpha + 3 * wn * (ra + sa / omega_symbol) / alpha) / omega_symbol
E2 = sP + 3 * wn * sa / alpha
E1 = rP + 3 * (Ox * bx + Oy * by + Oz * bz) / alpha + 3 * wn * ra / alpha
assert s.simplify(Qdot - E2 / omega_symbol**2 - E1 / omega_symbol) == 0
print('Qdot double-pole numerator:', E2)
print('Qdot single-pole numerator:', E1)

# Smooth live null data do not suffice for regular Q evolution. On flat outer
# reference spatial geometry with unperturbed alpha,beta but delta(P)=omega_symbol*deltaQ,
# the lapse pole is omega_symbol and the stabilized trace pole is omega_symbol;
# nonetheless generic
# deltaQ gives a nonzero Qdot residue. At S=a=1: alpha=1, wn=-1, Qhat=-3.
# This is off the shear/constraint manifold, not admissible asymptotic data.
dq = s.symbols('dq')
omega_symbol = s.symbols('omega_symbol', positive=True)
r = s.symbols('r', positive=True)
o = (1 - r * r) / 2
a = (1 + r * r) / 2
wn = -r * r / a
P = -3 + o * dq
# Unmodified flat spatial geometry, alpha and shift; all derivatives of dq=0.
# dtP: regular=-3 + beta.dP + 3 grad(alpha).grad(omega_symbol), pole=alpha*(P^2/3-3r^2)
rdotP = -3 - 3 * r * r + r * r * dq
sdotP = a * (P * P / 3 - 3 * r * r)
dotalpha = -a * a * dq
# Preferred source changes radial shift algebraically but at the reference
# gauge state this source has no dq dependence and dtbeta=0.
qdot = s.factor((rdotP + sdotP / o + 3 * wn * dotalpha / a) / o)
limit = s.simplify(s.limit(o * qdot, r, 1, dir='-'))
assert s.simplify(limit - 2 * dq) == 0
print('Concrete unclosed physical-trace limit: Omega*dtQ -> 2*deltaQ')
print('PASS reference identities and exact transformed residues; live '
      'closure remains unresolved')

# Independent lower-order audit, distinct from the principal polynomial.
# At S=a=1, scri, kappa1=5/alpha and kappa2=0; freeze all perturbation derivatives.
m = s.Matrix([[3, 0, -1, 0], [-2, 0, s.Rational(2, 3), s.Rational(4, 3)],
              [0, -3, -2, 1], [0, -3, -2, -11]])
lambda0 = s.symbols('lambda0')
cp = m.charpoly(lambda0).as_expr()
assert s.expand(cp - lambda0 * (lambda0**3 + 10 * lambda0**2 - 9 * lambda0 - 60)) == 0
roots = s.nroots(cp)
positive = [float(s.re(v)) for v in roots if abs(float(s.im(v))) < 1e-12 and s.re(v) > 0]
assert len(positive) == 1 and abs(positive[0] - 2.57170948731154) < 1e-12
print('Frozen off-manifold pole block positive eigenvalue:', positive[0], '/Omega')
print('This is a local lower-order growth obstruction, not a global '
      'PDE/discrete spectrum proof.')

# Positive leading pole eigenvalue persists for every nonnegative kappa1 when
# kappa2=0; increasing finite restoring rates cannot change a 1/Omega residue.
k, lam0 = s.symbols('k lam0', real=True)
mk = s.Matrix([[3, 0, -1, 0], [-2, 0, s.Rational(2, 3), s.Rational(4, 3)],
               [0, -3, -2, k - 4], [0, -3, -2, -1 - 2 * k]])
cpk = mk.charpoly()
assert s.expand(cpk.as_expr().subs(cpk.gen, lam0) - lam0 *
                (lam0**3 + 2 * k * lam0**2 - 9 * lam0 - 12 * k)) == 0
cubic = lam0**3 + 2 * k * lam0**2 - 9 * lam0 - 12 * k
assert s.simplify(cubic.subs(lam0, s.sqrt(6))) == -3 * s.sqrt(6)
assert s.simplify(cubic.subs(lam0, 3)) == 6 * k
print('For kappa1>0, a positive pole root lies between sqrt(6) and 3; '
      'at kappa1=0 it is 3.')
