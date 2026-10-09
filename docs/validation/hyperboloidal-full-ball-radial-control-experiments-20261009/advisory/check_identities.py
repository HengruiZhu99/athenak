"""Symbolic trial-space identities only; no nodes or PDE operator constructed."""
from pathlib import Path
import hashlib
import importlib.util
import json
import sympy as s
import sys

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
B = ROOT/'build-layer-research/continuum/total-j-harmonic-basis/immutable-Cartesian-total-J-basis-20261009'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(B/'index.json') == '414f241e986e0c46b7d94820fb060b704166d973284854c53d2e8c64ee6c489e'
spec = importlib.util.spec_from_file_location('frozen_basis', B/'total_j_basis.py')
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)
q = s.symbols('q', positive=True)
W = s.Function('W')
cases = []
for l in range(5):
    for m in sorted(set([0,l])):
        H = b.solid(l,m)
        field = H*W(b.rho)
        direct = sum(s.diff(field,x,2)for x in b.xyz)
        expected = H*(4*b.rho*s.diff(W(q),q,2).subs(q,b.rho)
                      +(4*l+6)*s.diff(W(q),q).subs(q,b.rho))
        assert s.simplify(s.expand(direct-expected)) == 0
        cases.append({'L':l,'m':m,'exact_factored_Laplacian':True})
r,L = s.symbols('r L',positive=True)
w,wp,wpp = s.symbols('w wp wpp')
u = r**L*W(r*r)
assert s.simplify(s.diff(u,r)-r**(L-1)*(L*W(r*r)+2*r*r*s.diff(W(q),q).subs(q,r*r))) == 0
assert s.simplify(s.diff(u,r,2)-r**(L-2)*(L*(L-1)*W(r*r)
                  +2*(2*L+1)*r*r*s.diff(W(q),q).subs(q,r*r)
                  +4*r**4*s.diff(W(q),q,2).subs(q,r*r))) == 0
factor_op = 4*q*s.diff(W(q),q,2)+(4*L+6)*s.diff(W(q),q)
divergence = 4*q**(-L-s.Rational(1,2))*s.diff(q**(L+s.Rational(3,2))*s.diff(W(q),q),q)
assert s.simplify(factor_op-divergence) == 0
U,V = s.Function('U'),s.Function('V')
# Exact integration-by-parts integrand identity for the envelope operator.
assert s.simplify(s.Rational(1,2)*q**(L+s.Rational(1,2))*U(q)
                 *(4*q*s.diff(V(q),q,2)+(4*L+6)*s.diff(V(q),q))
                 -s.diff(2*q**(L+s.Rational(3,2))*U(q)*s.diff(V(q),q),q)
                 +2*q**(L+s.Rational(3,2))*s.diff(U(q),q)*s.diff(V(q),q)) == 0
report = {'passed_symbolic_representation_identities':True,
          'frozen_basis_index_sha256':sha(B/'index.json'),
          'factored_Laplacian_cases':cases,
          'r_derivative_to_rho_formulas_exact':True,
          'weighted_envelope_divergence_and_weak_boundary_identity_exact':True,
          'common_quadrature_weight':'rho^(1/2)',
          'orbital_L_spatial_L2_envelope_weight':'(1/2)*rho^(L+1/2)',
          'implicit_polynomial_endpoint_BC':'None of Dirichlet/Neumann/outflow follows from interpolation.',
          'scope':'Advisory mathematical/source analysis. No quadrature nodes, differentiation/PDE matrix, radial boundary condition, spectrum, propagation or stability calculation.',
          'python':sys.version,'sympy_version':s.__version__,
          'source_pins':{'basis_source':sha(B/'total_j_basis.py'),
                         'scalar_radial_SBP':sha(ROOT/'src/z4c/hyperboloidal/radial_sbp.hpp')}}
(P/'identity-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
