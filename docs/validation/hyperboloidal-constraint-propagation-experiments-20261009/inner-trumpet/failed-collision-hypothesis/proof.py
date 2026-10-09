"""Exact necessary local stationary balances; not a global trumpet solution."""
import json
from pathlib import Path
import sympy as s

M, R, R0, c, alpha, aprime, p, v, eta, mu = s.symbols(
    "M R R0 c alpha aprime p v eta mu", positive=True)
F = 1 - 2*M/R
S = s.sqrt(alpha**2/c**2 - F)
Sr = (alpha*aprime/c**2-M/R**2)/S
K = Sr + 2*S/R
beta = alpha*S
gamma = c**2/alpha**2
# Stationary Schwarzschild spacetime in T=c*t+h(R). Choose the sign of h'
# giving beta>0; standard ADM K=-1/2 L_n gamma, matching production.
hprime = -beta*gamma/(c*F)
assert s.simplify(-c**2*F - (-alpha**2+gamma*beta**2)) == 0
assert s.simplify(1/F-F*hprime**2-gamma) == 0
assert s.simplify(-c*F*hprime-gamma*beta) == 0
assert s.simplify(beta/(alpha*R)-S/R) == 0
assert s.simplify((s.diff(beta, R)+s.diff(beta, alpha)*aprime
                  -beta*aprime/alpha)/alpha - Sr) == 0
# Exact BM alphaDot=beta*alpha_R-alpha(alpha+2)K in the Omega=1 core.
lapse = beta*aprime-alpha*(alpha+2)*K
root_ap = s.simplify(s.solve(lapse/alpha, aprime)[0])
K0 = (3*M-2*R0)/(R0**2*s.sqrt(2*M/R0-1))
alpha_R0 = 2*(3*M-2*R0)/(R0*(2*M-R0))
assert s.simplify(root_ap.subs({alpha: 0, R: R0})-alpha_R0) == 0
assert s.simplify(alpha_R0*s.sqrt(2*M/R0-1)-2*K0) == 0
# General c is allowed by stationary Schwarzschild geometry. The lapse-zero
# derivative is independent of c. c=1 is used by the numerical local examples.
C4 = (alpha**2/c**2-F)*R**4/(1+alpha/2)**2
total_C4 = s.diff(C4, R)+s.diff(C4, alpha)*root_ap
assert s.simplify(total_C4) == 0
assert s.simplify(C4.subs({alpha: 0, R: R0})-R0**3*(2*M-R0)) == 0
# R-R0=k*r^p, alpha~alpha_R0*k*r^p gives beta^r~v*r.
assert s.simplify(alpha_R0*s.sqrt(2*M/R0-1)/p-2*K0/p) == 0
zeta = c*p/(alpha_R0*R0)
gr, gt = zeta**s.Rational(4, 3), zeta**s.Rational(-2, 3)
assert s.simplify(gr*gt**2-1) == 0
# det(gtilde)=1 and conormal radial expansion imply Gamma^r=2(1/gt-1/gr)/r.
# With alpha=A*r^p, chi=C*r^2, its driver is O(r^(2p+1)), o(r), for p>0.
# Dividing shift RHS by r leaves v(v-eta). No positivity assumption on eta
# is used to solve the equation; v>0 is geometrically required for future BH.
assert s.factor(v**2-eta*v) == v*(v-eta)
leading_p = s.solve(v*p-2*K0, p)[0]
assert s.simplify(leading_p-2*K0/v) == 0
# Subleading necessity is deliberately not promoted to sufficiency. For
# beta=v*r+b*r^(1+sigma), v=eta, first advective-damping correction is
# v*b*(1+sigma)*r^(1+sigma). It requires a coordinate ODE at all orders.
r, b, sigma = s.symbols("r b sigma", positive=True)
be = v*r+b*r**(1+sigma)
subleading = s.expand(be*s.diff(be, r)-v*be)
assert s.simplify(subleading-(v*b*(1+sigma)*r**(1+sigma)
                             +b*b*(1+sigma)*r**(1+2*sigma))) == 0
# A hypothetical constant-coordinate Gamma coefficient requires replacing mu
# by k/(alpha^2*chi). It is a principal change. Check explicit collision
# obstructions in the actual W=0 scalar/vector blocks, not borrowed gauges.
from importlib.util import spec_from_file_location, module_from_spec
spec = spec_from_file_location("kernel_basis", Path(__file__).resolve().parents[2]
                              / "tst/hyperboloidal/check_kernel_symbol.py")
module = module_from_spec(spec)
spec.loader.exec_module(module)
scalar = s.Matrix(module.scalar_matrix(3, s.Rational(3, 4), 0, 0)).applyfunc(
    lambda x: s.Rational(float(x)).limit_denominator())
scalar_lapse_collision = s.Matrix(module.scalar_matrix(3, s.Rational(9, 4), 0, 0)).applyfunc(
    lambda x: s.Rational(float(x)).limit_denominator())
vector = s.Matrix([[0,-2,0,1],[-s.Rational(1,2),0,s.Rational(1,2),0],
                   [0,0,0,1],[0,0,1,0]])
collisions = []
for name, matrix, eigenvalue, size in [("q=1,f=3", scalar, 1, 8),
        ("q=f=3", scalar_lapse_collision, s.sqrt(3), 8),
        ("mu=1 vector", vector, 1, 4)]:
    multiplicity = sum(mult for value, mult in matrix.eigenvals().items()
                       if s.simplify(value-eigenvalue) == 0)
    geometric = size-(matrix-eigenvalue*s.eye(size)).rank()
    assert geometric < multiplicity
    collisions.append({"case":name,"positive_root_algebraic_multiplicity":multiplicity,
                       "positive_root_geometric_multiplicity":geometric})
out = {"passed": True, "assumptions": ["stationary radial Schwarzschild geometry",
    "future/outward shift branch, positive lapse, c>0", "Omega=1 exact reference core",
    "Theta_phys=0; P=K; lapse_inner=0", "regular radial conormal/polyhomogeneous expansion",
    "nondegenerate conformal metric, chi~Cchi*r^2, alpha~A*r^p, p>0"],
    "K0": str(K0), "alpha_R0": str(alpha_R0), "v": "2*K0/p",
    "eta_required": "eta_inner=v>0", "radius_restriction": "0<R0<3*M/2",
    "default_eta0_obstruction": "betaDot/r -> v^2>0; alpha^2*chi*Lambda/r ->0",
    "hypothetical_unsuppressed_driver_principal_collisions": collisions,
    "not_proved": ["existence", "stability", "global matching", "dynamic formation",
                   "uniform puncture hyperbolicity", "all-order stationary coordinate ODE"]}
Path(__file__).with_name("proof-result.json").write_text(json.dumps(out, indent=2)+"\n")
print(json.dumps(out, indent=2))
