import sympy as s
r = s.symbols('r', positive=True)
omega_symbol = s.Function('omega_symbol')(r)
b = s.Function('b')(r)
L = omega_symbol - r * s.diff(omega_symbol, r)
A = s.sqrt(omega_symbol**2 + b**2)
q = L**2 / A**2
beta = -b * A / L
kr = s.simplify((beta * s.diff(q, r) / q + 2 * s.diff(beta, r)) / (2 * A))
kt = s.simplify(beta / (A * r))
assert s.simplify(kr + b.diff(r) / L) == 0
assert s.simplify(kt + b / (r * L)) == 0
assert s.simplify(A**2 / L**2 - beta**2 / A**2 - omega_symbol**2 / L**2) == 0
assert s.simplify(beta * (-1) + A**2 / L - A * (A + b) / L) == 0
assert s.simplify(-beta - A**2 / L + A * omega_symbol**2 / (L * (A + b))) == 0
Kp = omega_symbol * (kr + 2 * kt) + 3 * b * s.diff(omega_symbol, r) / L
assert s.simplify(Kp + (omega_symbol * (b.diff(r) + 2 * b / r) -
                  3 * b * omega_symbol.diff(r)) / L) == 0
w = s.Function('w')(r)
a, S = s.symbols('a S', positive=True)
Oo = (S**2 - r**2) / (2 * a * S)
Ob = 1 - w + w * Oo
bb = r * w / a
Lb = Ob - r * s.diff(Ob, r)
assert s.simplify(Lb - (1 - w + w * (S**2 + r**2) /
                  (2 * a * S) + r * w.diff(r) * (1 - Oo))) == 0
Kpb = -(Ob * (bb.diff(r) + 2 * bb / r) - 3 * bb * Ob.diff(r)) / Lb
assert s.simplify(Kpb + (3 * w + r * Ob * w.diff(r) / Lb) / a) == 0
assert s.simplify(((S**2 + r**2) / (2 * a * S))**2 - Oo**2 - (r / a)**2) == 0
print('PASS: ADM extrinsic curvature, exact inverse radial metric, both '
      'factored light speeds, physical trace transform, '
      'layer L and K, outer CMC identity')
