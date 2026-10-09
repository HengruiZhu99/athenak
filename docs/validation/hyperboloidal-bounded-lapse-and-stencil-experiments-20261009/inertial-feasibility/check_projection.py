"""Independent exact general-t projection and 100-digit outer limit check.

No evolution, private RHS replacement, or added falloff is used here.
"""
from pathlib import Path
import hashlib
import itertools
import json
import subprocess

import mpmath as mp
import sympy as sp

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
mp.mp.dps = 100
A, W, theta, rho, kap = sp.symbols('A W theta rho kap', real=True)
beta = sp.Matrix(sp.symbols('b0:3', real=True))
V = sp.Matrix(sp.symbols('v0:3', real=True))
Z = sp.Matrix(sp.symbols('z0:3', real=True))
g11, g12, g13, g22, g23, g33 = sp.symbols('g11 g12 g13 g22 g23 g33')
gamma = sp.Matrix([[g11, g12, g13], [g12, g22, g23],
                   [g13, g23, g33]])
g = sp.zeros(4)
g[0, 0] = -A**2 + (beta.T * gamma * beta)[0]
for i in range(3):
    g[0, i+1] = g[i+1, 0] = (gamma * beta)[i]
    for j in range(3):
        g[i+1, j+1] = gamma[i, j]
n = sp.Matrix([1/A, *(-beta/A)])
t = W*n + sp.Matrix([0, *V])
zcov = sp.Matrix([-A*theta + (beta.T*Z)[0], *Z])
tcov = g*t
J = (V.T*Z)[0]
tz = (t.T*zcov)[0]
assert sp.simplify(tz + W*theta-J) == 0
S = tcov*zcov.T + zcov*tcov.T - (1+rho)*g*tz
D = tcov*zcov.T + zcov*tcov.T + rho*g*tz
assert sp.simplify((n.T*g*n)[0]+1) == 0
assert sp.simplify((n.T*tcov)[0]+W) == 0
assert sp.simplify((n.T*zcov)[0]+theta) == 0
for i in range(3):
    assert sp.simplify((n.T*D)[i+1]+W*Z[i]+(gamma*V)[i]*theta) == 0
    for j in range(3):
        expected = (gamma*V)[i]*Z[j]+(gamma*V)[j]*Z[i]
        expected += (1+rho)*gamma[i, j]*(W*theta-J)
        assert sp.simplify(S[i+1, j+1]-expected) == 0
Dnn = (n.T*D*n)[0]
assert sp.simplify(Dnn-((2+rho)*W*theta-rho*J)) == 0
# Trace of the spatial Ricci tensor source, and P=K-2Theta.
trS = 3*(1+rho)*W*theta-(1+3*rho)*J
Pdot = -A*kap*trS+2*A*kap*((2+rho)*W*theta-rho*J)
assert sp.simplify(Pdot-A*kap*((1-rho)*W*theta+(1+rho)*J)) == 0

o, alpha, chi, kin = sp.symbols('Omega alpha chi kin', positive=True)
br, zr = sp.symbols('beta_r Z_r', real=True)
# Exact radial CMC reference: gtilde=I, chi=1; t=partial_t.
Srad, arad, om = sp.symbols('S a omega', positive=True)
r = sp.sqrt(Srad**2-2*arad*Srad*om)
alphar = (Srad**2+r**2)/(2*arad*Srad)
betar = -r/arad
assert sp.simplify(alphar**2-betar**2-om**2) == 0
limits = {
    'Omega2_dTheta_dTheta': sp.limit(-2*kin*alphar, om, 0),
    'Omega2_dLambda_dLambda': sp.limit(-kin*alphar, om, 0),
    'Omega3_dLambda_n_dTheta': sp.limit(-2*kin*betar, om, 0),
    'Omega2_dA_nn_dLambda_n': sp.limit(-sp.Rational(2, 3)*kin*betar, om, 0),
    'Omega2_dP_dTheta': sp.limit(kin*alphar, om, 0),
}
assert limits['Omega2_dTheta_dTheta'] == -2*kin*Srad/arad
assert limits['Omega3_dLambda_n_dTheta'] == 2*kin*Srad/arad
# General live-data counterexample: all lapse/metric values remain positive,
# alpha0 and beta0 retain their reference values, while partial_t is spacelike.
alpha_bad2 = betar**2-om**2
assert sp.limit(alpha_bad2, om, 0) == Srad**2/arad**2
assert sp.simplify((-alpha_bad2+betar**2)/om**2) == 1

rows = []
for ss, aa, kk in itertools.product(('1', '2'), ('.5', '1', '2'), ('1', '10')):
    s, a, k = map(mp.mpf, (ss, aa, kk))
    for exponent in (5, 20, 40):
        omega = mp.mpf(10)**-exponent
        radius = mp.sqrt(s*s-2*a*s*omega)
        alp = (s*s+radius*radius)/(2*a*s)
        bet = -radius/a
        # Factor rather than subtract alpha^2-beta^2 at small Omega.
        null_identity = ((alp-bet)*(alp+bet)-omega*omega)
        assert abs(null_identity) < mp.mpf('1e-95')
        th = -2*k*alp
        ll = -k*alp
        lt = -2*k*bet
        assert abs(th+2*k*s/a) < 3*k*omega
        assert abs(ll+k*s/a) < 2*k*omega
        assert abs(lt-2*k*s/a) < 5*k*omega
        rows.append({'S': ss, 'a': aa, 'kinput': kk, 'Omega_exponent': exponent,
                     'Omega2_Theta_coefficient': mp.nstr(th, 90),
                     'Omega2_Lambda_diagonal_coefficient': mp.nstr(ll, 90),
                     'Omega3_Lambda_Theta_coefficient': mp.nstr(lt, 90)})

# Illustrative current pole cap, not an actual candidate full20/native timestep.
native = []
for N in (24, 36, 48):
    h = 2.1/N
    coords = [-1.05+(i+.5)*h for i in range(N)]
    radii2 = [sum(x*x for x in xyz)
              for xyz in itertools.product(coords, repeat=3)]
    omega = 1-max(x for x in radii2 if x < 1)
    alpha0 = 2-omega
    z = -2*10*alpha0/omega**2*(.03*omega)
    native.append({'N': N, 'span': 2.1, 'Omega_min': omega,
                   'illustrative_dt': .03*omega,
                   'Theta_damping_block_RK_argument': z,
                   'RK3_absolute_amplification': abs(1+z+z*z/2+z**3/6)})

report = {'status': 'PASS', 'scope': 'Exact projection and finiteOmega/asymptotic feasibility only',
          'source': 'https://arxiv.org/pdf/gr-qc/0504114v2',
          'equations': [2, 3, 13, 16, 17, 19], 'pdf_pages_one_based': [2, 3],
          'general_spatial_metric_projection': True, 'mpmath_digits': mp.mp.dps,
          'limits': {key: str(value) for key, value in limits.items()},
          'limit_rows': rows, 'illustrative_fast_damping_block': native,
          'timelike_counterexample': 'alpha^2=beta_ref^2-Omega^2 gives g_phys(partial_t,partial_t)=+1 with positive alpha near scri',
          'versions': {'sympy': sp.__version__, 'mpmath': mp.__version__},
          'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
(HERE/'projection.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
print('PASS generic 4D projections, 36 hundred-digit limits, live timelike counterexample')
