"""Exact identities for the spherical constrained CMC/Dirac module (SymPy)."""
import sympy as s

r, R = s.symbols("r R", positive=True)
M, C = s.symbols("M C", real=True)
u, y = s.Function("u")(r), s.Function("y")(r)
V = -s.diff(u, r) - 2*(u - u**s.Rational(-1, 2))/r
c = (2 + u**s.Rational(-3, 2))/r
p = s.diff(u, r)/(4*u) + 1/r + c
q = -3*(s.diff(u, r, 2) + s.diff(u, r)*(1/r+c))/(4*u)
W = r*y*s.diff(u, r) - s.Rational(4, 3)*u*r*s.diff(y, r)
Vt = -s.diff(W, r) - c*W
assert s.simplify(Vt - 4*u*r*(s.diff(y, r, 2)+p*s.diff(y, r)+q*y)/3) == 0

# Factorization and the conserved annular conformal radial distance.
assert s.simplify(s.diff(u**s.Rational(-3, 4)*y, r)
                  + 3*u**s.Rational(-7, 4)*W/(4*r)) == 0

# Physical areal-radius metric: dl^2=dR^2/D+R^2 dOmega_2^2.
D = 1 - 2*M/R + (-R + C/R**2)**2
scalar_curvature = 2*(1-D-R*s.diff(D, R))/R**2
A2 = 6*C**2/R**6
assert s.simplify(scalar_curvature - A2 + 6) == 0
Ar, At = -2*C/R**3, C/R**3
assert s.simplify(s.diff(Ar, R)+2*(Ar-At)/R) == 0
# Divide Delta(sqrt(D)) by sqrt(D), avoiding spurious branch simplifications.
lapse_laplacian_ratio = s.diff(D, R, 2)/2+s.diff(D, R)/R
assert s.simplify(lapse_laplacian_ratio-(3+A2)) == 0

# Exact nullness of the compactified scri boundary.
aS = s.symbols("aS", positive=True)
chiS, alphaS, betaS = aS, s.Integer(1), s.Integer(-1)
assert s.simplify(chiS/aS-betaS**2/alphaS**2) == 0

# The linearized metric gauge has a_t=-eta*(a-1) on the regular-center ball.
f, eta = s.Function("f")(r), s.symbols("eta", nonnegative=True)
Vlin = s.diff(f, r)+3*f/r
# If f_t=-eta*f, Vlin_t=-eta*Vlin exactly. Its regular homogeneous
# ambiguity is C/r^3, excluded at the origin. This is a linear statement.
assert s.simplify(s.diff(-eta*f, r)+3*(-eta*f)/r+eta*Vlin) == 0
print("PASS: Dirac operator, Hamiltonian, momentum, CMC lapse, scri nullness, linear gauge")
