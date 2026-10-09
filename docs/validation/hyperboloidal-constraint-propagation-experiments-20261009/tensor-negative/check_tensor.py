"""Negative exact resonance gate for a hypothetical paper-scheme transfer."""
import json
from pathlib import Path
import importlib.util

import numpy as np
import sympy as sp

p = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('bulk', p.parent/'discrete-bianchi/check_discrete.py')
bulk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bulk)


def scalar(v, w, f, mu):
    m = sp.zeros(8)
    m[0, 3] = -f
    m[1, 3], m[1, 4], m[1, 7] = sp.Rational(2, 3), sp.Rational(4, 3), -sp.Rational(2, 3)*v
    m[2, 5], m[2, 7] = -2, sp.Rational(4, 3)*v
    m[3, 0], m[4, 1], m[4, 6] = -1, 1, v/2
    m[5, 0], m[5, 1], m[5, 2], m[5, 6] = -sp.Rational(2, 3)*v*v, v*v/3, -sp.Rational(1, 2), sp.Rational(2, 3)*v
    m[6, 3], m[6, 4], m[6, 7] = -sp.Rational(4, 3)*v, -sp.Rational(2, 3)*v, 1+v*v/3
    m[7, 0], m[7, 1], m[7, 6] = -v*w, v*w/2, mu
    return m


v, w, f, mu, lam = sp.symbols('v w f mu lambda')
m = scalar(v, w, f, mu)
characteristic = sp.factor(m.charpoly(lam).as_expr())
expected = (lam**2-f)*(lam**2-1)*(3*lam**4-lam**2*mu*v*v
            -3*lam**2*mu+lam**2*v*v*w+lam**2*v*v-4*lam**2
            +4*mu-v*v*w)/3
assert sp.simplify(characteristic-expected) == 0
exact = scalar(sp.sqrt(sp.Rational(7, 17)), sp.Rational(9, 10),
               sp.Rational(6, 5), sp.Rational(15, 16))
factor = sp.factor(exact.charpoly(lam).as_expr())
assert sp.simplify(factor-(lam**2-1)*(5*lam**2-6)**2*(408*lam**2-383)/10200) == 0
a = exact**2-sp.Rational(6, 5)*sp.eye(8)
assert 8-a.rank() == 2
assert 8-(a*a).rank() == 4

case = json.loads((p/'kernel.json').read_text())
L, Q, d, S = [bulk.z(case[k]) for k in ['L', 'Q', 'D', 'S']]
case.update(alpha=1., chi=1., spd=False)
T, smod = bulk.transform(case, d.imag)
ell = np.sqrt(-np.trace(S).real)
positions = [0, 1, 2, 7, 8, 11, 12, 15, 16, 18]
T[positions] *= ell/smod
reduced = np.einsum('ij,jk,kl->il', T, L, np.linalg.inv(T))
M = np.zeros((20, 20))
ratio = smod/ell
M[:8, :8] = np.array(scalar(ratio, case['W'], case['f'], case['mu'])).astype(float)
for start in [8, 12]:
    M[start:start+4, start:start+4] = [[0, -2, 0, ratio],
        [-.5, 0, ratio/2, 0], [0, 0, 0, 1], [0, 0, case['mu'], 0]]
for start in [16, 18]:
    M[start:start+2, start:start+2] = [[0, -2], [-.5, 0]]
target = 1j*ell*M+1j*np.dot([.3, -.2, .1], d.imag)*np.eye(20)
error = float(np.max(abs(reduced-target)))
assert error < 1e-10, error
observed_M = (reduced-1j*np.dot([.3, -.2, .1], d.imag)*np.eye(20))/(1j*ell)
exact_error = float(np.max(abs(observed_M[:8, :8]-np.array(exact).astype(float))))
assert exact_error < 1e-11

# Existing actual static constraint map and unchanged first derivatives: the
# transfer also leaves gauge constraint leakage, as expected from the paper.
CL = np.einsum('ij,jk->ik', Q, L)
delta = np.diag(S)-d*d
wanted = np.zeros((8, 4), complex)
for i in range(3):
    wanted[1+i, 0] = sp.Rational(2, 3)*d[i]*sum(delta)
    wanted[0, 1+i] = -2*d[i]*sum(delta[j] for j in range(3) if j != i)
    wanted[4+i, 1+i] = sum(delta)/2
gauge_error = float(np.max(abs(CL[:, [0, 4, 5, 6]]-wanted)))
assert gauge_error < 1e-9, gauge_error
report = {
    'passed_negative_audit': True,
    'candidate_accepted': False,
    'source': 'https://arxiv.org/pdf/1111.2177, Eqs27-29,40-46',
    'symbolic_scalar_characteristic': str(characteristic),
    'resonance_parameters': {'W': '.9', 'alpha': 1, 'f': '6/5',
                             'mu': '15/16', 'nu_squared': '7/17'},
    'exact_characteristic': str(factor),
    'M_squared_minus_f_nullity': 2,
    'M_squared_minus_f_squared_nullity': 4,
    'actual_native_symbol_theta': case['theta'],
    'actual_transition_radius': case['r'],
    'actual_nu_squared': ratio**2,
    'actual_full20_matrix_error': error,
    'actual_scalar_exact_resonance_matrix_error': exact_error,
    'actual_gauge_constraint_defect_formula_error': gauge_error,
    'actual_lapse_momentum_constraint_source': [[x.real, x.imag] for x in CL[1:4, 0]],
    'scope': 'Hypothetical constant-flat tensor-discretization transfer only. '
             'No production/native source change. No complete basis at the '
             'displayed resonance. KO is a separate dissipative modification; '
             'the paper itself does not claim discrete constraint closure.'}
(p/'check-report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
