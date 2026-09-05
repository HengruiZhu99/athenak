"""Exact real-spectrum and semisimplicity checks for the new principal ordering."""
import sympy as s

from symbol import SELECT, T, baseline, old


def symmetric_legacy(alpha, chi):
    p = old.full_symbol(alpha, chi, s.Integer(1))
    for k in (1, 2):
        row = old.AT[old.sym_index(0, k)]
        p[row, old.X[k]] /= 2
        p[row, old.Y[k]] /= 2
    return SELECT*p*T


w, rho, switch = s.symbols("w rho switch", positive=True)
p = baseline(w, rho, switch).applyfunc(s.cancel)
z = s.symbols("z")
speeds2 = [s.Integer(1), 2*rho*w**3, rho**2*w**4,
           (4-switch*rho**2*w**4)/3]
expected = z**30*(z*z-speeds2[0])**2*(z*z-speeds2[1])*(z*z-speeds2[2])**6*(z*z-speeds2[3])
assert s.factor(p.charpoly(z).as_expr()-expected) == 0
remainder = p
p2 = p*p
for speed2 in speeds2:
    remainder = ((p2-speed2*s.eye(50))*remainder).applyfunc(s.cancel)
assert remainder == s.zeros(50)
print("PASS: exact characteristic polynomial and square-free annihilator off speed coincidences")

# For 0<alpha<2 and alpha^2*chi<4, the transition is complete at
# zeta=1/2. In the transition (zeta<1/2), the four positive speeds
# are distinct. The complete list of possible nonzero coincidences follows.
a = s.symbols("a", positive=True)
checks = [
    (symmetric_legacy(a, 1/(2*a)), s.Integer(1), 3, "lapse/transverse shift, zeta=1/2"),
    (symmetric_legacy(a, 4/(a*(6+a))), s.sqrt(8/(6+a)), 2,
     "lapse/longitudinal shift, zeta=4/(6+alpha)"),
    (symmetric_legacy(a, 1/(a*a)), s.Integer(1), 9,
     "light and both shift families, alpha^2*chi=1"),
]
for matrix, eigenvalue, multiplicity, label in checks:
    for sign in (1, -1):
        nullity = 50-(matrix-sign*eigenvalue*s.eye(50)).rank()
        assert nullity == multiplicity, (label, sign, nullity)
    assert 50-matrix.rank() == 30
    print(f"PASS: {label}, both signs semisimple; zero eigenspace complete")
print("DOMAIN: SPD conformal metric, w>0, rho>0, 0<alpha<2, 0<alpha^2*chi<4")
print("DOMAIN: the implemented switch is completed by zeta=0.5; no uniform puncture-limit theorem")
print("The general-background nilpotent similarity is checked separately in symbol.py.")
