from pathlib import Path
import json
import math
import sympy as s
import total_j_basis as b

P = Path(__file__).resolve().parent
release = (P/'probe-release.json').read_bytes()
debug = (P/'probe-debug.json').read_bytes()
assert release == debug
data = json.loads(release)
points = [(0, 0, 0), (.125, -.25, .375), (.36, -.48, .8), (-.2, .3, -.4)]
flatten = lambda expr: [expr]+[s.diff(expr, x) for x in b.xyz]+[s.diff(expr, x, y) for x in b.xyz for y in b.xyz]
oracles = {}
for row in data['basis_values']:
    key = (row['J'], row['spin'], row['L'], row['W'])
    if key not in oracles:
        W = [1, b.rho, b.rho**2, 1+b.rho+b.rho**2][row['W']]
        exprs = [flatten(s.expand(v*W)) for v in b.basis(row['J'], 0, row['spin'], row['L'])]
        oracles[key] = s.lambdify(b.xyz, exprs, 'math', cse=True)
errors = []
for row in data['basis_values']:
    expected = oracles[(row['J'], row['spin'], row['L'], row['W'])](*points[row['point']])
    for x, y in zip(row['jets'], expected):
        for actual, exact in zip(x, y):
            assert math.isfinite(actual) and math.isfinite(exact)
            errors.append(abs(actual-exact)/max(1.0, abs(exact)))
assert max(errors) < 3e-14
x, y, z = b.xyz
B = s.diag(1+x*x, 1+y*y, 1+z*z)
chi = s.det(B)**(-s.Rational(1, 3))
g = chi*B
H = s.Matrix([[x+y, x*z, y+z], [x*z, z+x*y, y*y], [y+z, y*y, 1+z*z]])
T = s.Matrix([[x, z, x*y], [z, y, y*z], [x*y, y*z, -x-y]])
S = s.Matrix([[1, s.Rational(1, 5), s.Rational(3, 10)],
              [s.Rational(1, 5), -s.Rational(2, 5), s.Rational(1, 2)],
              [s.Rational(3, 10), s.Rational(1, 2), -s.Rational(3, 5)]])
gi = g.inv()
A = S-g*s.trace(gi*S)/3
dc = -chi*s.trace(B.inv()*H)/3
dg = chi*H+B*dc
da = T+g*(s.trace(gi*A*gi*dg)-s.trace(gi*T))/3
assert s.simplify(s.trace(gi*dg)) == 0
assert s.simplify(s.trace(gi*da)-s.trace(gi*A*gi*dg)) == 0
assert s.simplify(s.trace(gi*A)) == 0
conversion_oracle = s.lambdify(b.xyz, [flatten(dc),
                          [flatten(s.simplify(q)) for q in dg],
                          [flatten(s.simplify(q)) for q in da]], 'math', cse=True)
conversion_error = 0.0
coupling_nonzero = False
for row in data['conversions']:
    expected = conversion_oracle(*points[row['point']])
    actual = [row['delta_chi'], row['delta_g'], row['delta_A']]
    for got, want in [(actual[0], expected[0])]+list(zip(actual[1], expected[1]))+list(zip(actual[2], expected[2])):
        for v, w in zip(got, want):
            assert math.isfinite(v) and math.isfinite(w)
            conversion_error = max(conversion_error, abs(v-w)/max(1.0, abs(w)))
    gv = gi.subs(dict(zip(b.xyz, points[row['point']])))
    av = s.Matrix(3, 3, [q[0] for q in row['delta_A']])
    coupling_nonzero |= abs(float(s.trace(gv*av))) > 1e-3
assert conversion_error < 3e-13 and coupling_nonzero
report = {'passed_compiled_math_only_basis': True,
          'release_ASan_UBSan_debug_byte_equal': True, 'basis_records': len(data['basis_values']),
          'basis_scaled_value_gradient_hessian_error': max(errors),
          'conversion_points': len(data['conversions']),
          'conversion_scaled_value_gradient_hessian_error': conversion_error,
          'nonzero_A_linearized_trace_coupling_exercised': coupling_nonzero,
          'origin_included_without_division_or_origin_branch': True,
          'scope': 'Standalone polynomial/jet/conversion tests only. No actual continuum kernel, reference evolution, boundary or stability claim.'}
(P/'compiled-check-report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps(report, indent=2))
