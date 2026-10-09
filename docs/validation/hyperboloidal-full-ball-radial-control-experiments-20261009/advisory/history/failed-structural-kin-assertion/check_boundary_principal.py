"""Frozen harmonic principal algebra only: no radial PDE operator or solver."""
from pathlib import Path
import hashlib
import json
import sympy as s
import sys

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
GATE = (ROOT / 'build-layer-research/continuum/conformal-q-null-feedback'
        '/immutable-conformal-Q-null-feedback-local-20261009')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(GATE / 'index.json') == (
    'dac759becbe666c71443328b61324a50aeaf0f72e76b4e751a9cabfa3a87cd96')
R = s.Rational
A = s.zeros(20)
# Exact W=1 matrix in the actual kernel_symbol.cpp normalized field ordering.
A[:8, :8] = s.Matrix([
    [0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, R(2, 3), R(4, 3), 0, 0, -R(2, 3)],
    [0, 0, 0, 0, 0, -2, 0, R(4, 3)],
    [-1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, R(1, 2), 0],
    [-R(2, 3), R(1, 3), -R(1, 2), 0, 0, 0, R(2, 3), 0],
    [0, 0, 0, -R(4, 3), -R(2, 3), 0, 0, R(4, 3)],
    [-1, R(1, 2), 0, 0, 0, 0, 1, 0]])
vector = s.Matrix([[0, -2, 0, 1], [-R(1, 2), 0, R(1, 2), 0],
                   [0, 0, 0, 1], [0, 0, 1, 0]])
for k in [8, 12]:
    A[k:k + 4, k:k + 4] = vector
for k in [16, 18]:
    A[k:k + 2, k:k + 2] = s.Matrix([[0, -2], [-R(1, 2), 0]])
I = s.eye(20)
assert A * A == I
assert A.trace() == 0
H = I + A.T * A
assert H * A == A + A.T
assert H * A == (H * A).T
plus, minus = (I + A) / 2, (I - A) / 2
assert plus * plus == plus and minus * minus == minus
assert plus * minus == s.zeros(20) and plus + minus == I
assert plus.rank() == minus.rank() == 10
assert plus.T * H == H * plus
assert plus.T * H * minus == s.zeros(20)

# q means normalized normal configuration derivatives, not raw stored values.
q_indices = [0, 1, 2, 7, 8, 11, 12, 15, 16, 18]
v_indices = [3, 4, 5, 6, 9, 10, 13, 14, 17, 19]
E = I[:, v_indices]
assert A.extract(q_indices, v_indices).det() == R(128, 3)
# If C+/- are H-orthonormal characteristic rows, these are G+/-^T G+/-.
gram_plus, gram_minus = E.T * H * plus * E, E.T * H * minus * E
assert gram_plus == gram_plus.T and gram_minus == gram_minus.T
leading_minors = [gram_plus[:k, :k].det() for k in range(1, 11)]
assert all(x > 0 for x in leading_minors)
z = s.symbols('z')
poly = s.factor((gram_minus - z * gram_plus).det())
expected_poly = (R(25, 41472) * (z - 1)**4
                 * (11 * z**2 - 23 * z + 11)**2
                 * (4487 * z**2 - 13821 * z + 4487))
assert s.expand(poly - expected_poly) == 0
roots = s.solve(expected_poly, z)
R2 = (13821 + s.sqrt(110487365)) / 8974
assert all(s.N(R2 - x, 80) >= 0 for x in roots)

# Retained actual kernel extraction records; this is not a new kernel run.
actual = json.loads((GATE / 'principal.json').read_text())
harmonic = [row for row in actual if row['W'] == 1]
assert harmonic
max_actual_error = max(
    abs(float(A[i, j]) - row['M'][i][j])
    for row in harmonic for i in range(20) for j in range(20))
assert max_actual_error < 2e-12

S, a, r = s.symbols('S a r', positive=True)
alpha, beta = (S*S + r*r)/(2*a*S), -r/a
kin, kout = s.factor(beta + alpha), s.factor(beta - alpha)
assert kin == (S - r)**2/(2*a*S)
assert kout == -(S + r)**2/(2*a*S)
samples = []
for rb in [R(98, 100), R(995, 1000)]:
    vals = {S: 1, a: R(1, 2), r: rb}
    om = ((S*S-r*r)/(2*a*S)).subs(vals)
    samples.append({
        'S': 1, 'a': .5, 'rb': float(rb), 'Omega': float(om),
        'incoming_outward_speed': float((-kin).subs(vals)),
        'outgoing_outward_speed': float((-kout).subs(vals)),
        'incoming_R2_over_outgoing': float((kin/(-kout)).subs(vals)*R2),
        'kappa10_over_Omega': float(10/om)})

report = {
    'passed_frozen_boundary_principal_algebra': True,
    'scope': ('Advisory local harmonic principal algebra and retained extraction '
              'record check. No radial differentiation/PDE operator, SAT bulk '
              'certificate, constraint boundary condition or evolution.'),
    'python': sys.version, 'sympy_version': s.__version__,
    'actual_retained_harmonic_rows': len(harmonic),
    'max_retained_extraction_error': max_actual_error,
    'exact_A_squared_I': True, 'exact_HA_AplusAT': True,
    'positive_symmetrizer_reason': 'y^T H y=||y||^2+||A y||^2',
    'projector_ranks': [10, 10],
    'q_normal_derivative_indices': q_indices,
    'V_stored_momentum_indices': v_indices,
    'det_A_qV': str(A.extract(q_indices, v_indices).det()),
    'det_Gplus_gram': str(gram_plus.det()),
    'Gplus_positive_leading_minors': [str(x) for x in leading_minors],
    'R_singular_values_squared_polynomial': str(poly),
    'R_norm_squared_exact': str(R2),
    'R_norm_squared': float(R2), 'R_norm': float(s.sqrt(R2)),
    'boundary_negative_semidefinite_condition': 'k_in*||R||^2 <= |k_out|',
    'reference_finite_boundary_examples': samples,
    'source_pins': {
        str(p.relative_to(ROOT)): sha(p)
        for p in [ROOT/'tst/hyperboloidal/kernel_symbol.cpp',
                  ROOT/'tst/hyperboloidal/check_kernel_symbol.py',
                  ROOT/'src/z4c/hyperboloidal/layer_gauge.hpp',
                  ROOT/'src/z4c/hyperboloidal/cmc_reference.hpp',
                  GATE/'index.json', GATE/'principal.json']}}
(P/'boundary-principal-report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
