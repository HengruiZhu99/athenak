"""Read-only exact reconstruction check of already computed outer matrices.

This does not evaluate or rerun the actual tensor kernel. Exact statements
apply to the rationally reconstructed matrices, with the raw reconstruction
error retained. No finite-radius Fourier or continuum stability inference.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time

import sympy as s

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
start = time.monotonic()
data = json.loads(args.input.read_text())
lam, z, K = s.symbols('lambda z K')
cubic = z**3+(2*K+10)*z**2+(20*K+11)*z+28*K-30
cases = []
reconstruction_error = 0.0
for row in data['poles']:
    if row['sigma'] != 5:
        continue
    a = s.Rational(str(row['a']))
    kap = s.Rational(str(row['kappa']))
    raw = row['M']
    matrix = s.Matrix([[s.Rational(float(v)).limit_denominator(1000000)
                        for v in line] for line in raw])
    reconstruction_error = max(reconstruction_error,
                               max(abs(float(matrix[i, j])-raw[i][j])
                                   for i in range(20) for j in range(20)))
    normalized_K = kap*a*a
    expected = (lam**9*(lam+2/a**2)**2
                *(lam**2+kap*lam+2*kap/a**2)**2
                *(lam**2+kap*lam+2*kap/a**2+4/(3*a**4))
                *cubic.subs({z: a*a*lam, K: normalized_K})/a**6)
    assert s.expand(matrix.charpoly(lam).as_expr()-expected) == 0
    assert matrix.rank() == 11 and (matrix*matrix).rank() == 11
    # Independently expose the closed scalar block from the actual full20
    # matrices. C=a^2 deltaNraw, T=a delta(P-3omega_n), E=a deltaTheta.
    project = s.zeros(3, 20)
    project[0, 1], project[0, 7] = 1, -1
    project[0, 0] = project[0, 4] = 2*a
    project[1, 2] = a
    project[1, 0] = project[1, 4] = -3*a
    project[2, 3] = a
    scalar = s.Matrix([[-10, -s.Rational(4, 3), s.Rational(4, 3)],
                       [12, 1, normalized_K-4],
                       [-3, -2, -1-2*normalized_K]])
    assert a*a*project*matrix == scalar*project
    assert s.expand(scalar.charpoly(z).as_expr()
                    -cubic.subs(K, normalized_K)) == 0
    assert normalized_K > s.Rational(15, 14)
    # For sigma=5 all quadratics have positive coefficients. The cubic's
    # Routh determinant is strictly positive for K>0; its constant imposes
    # precisely K>15/14. This supplies 11 negative-real-part nonzero roots.
    coefficients = s.Poly(cubic, z).all_coeffs()
    assert s.expand(coefficients[1]*coefficients[2]-coefficients[3]) == (
        40*K*K+194*K+140)
    cases.append({'a': str(a), 'kappa_input': str(kap), 'K': str(normalized_K),
                  'rank': 11, 'rank_square': 11, 'semisimple_zero_count': 9,
                  'strict_negative_real_part_nonzero_count': 11})
assert len(cases) == 8 and reconstruction_error < 1e-12
out = {'status': 'PASS_RECONSTRUCTED_REFERENCE_OUTER_POLES',
       'input': str(args.input.resolve()),
       'input_sha256': hashlib.sha256(args.input.read_bytes()).hexdigest(),
       'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
       'seconds': time.monotonic()-start, 'sympy': s.__version__,
       'rational_reconstruction_max_error': reconstruction_error,
       'sigma5_normalized_cubic': str(cubic), 'cases': cases,
       'exactness_scope': 'Exact algebra on rationally reconstructed analytic reference matrices; raw floating reconstruction error retained.',
       'admission_scope': 'Outer value-only pole spectrum and zero semisimplicity; no finite-Fourier transition, nonlinear/PDE closure, uniform lapse, native or BH claim.'}
args.output.write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
