"""Read-only review of the frozen angular gate; no kernel rebuild or evolution."""
from pathlib import Path
from fractions import Fraction as F
import hashlib
import json
import sys
import sympy as sp
from sympy.polys.matrices import DomainMatrix

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
GATE = ROOT / 'build-layer-research/continuum/q-null-angular-ideal/immutable-Q-angular-gauge-only-ideal-20261009'
PIN = '8c569fdf6faf0fa9afff76bebfcd8ea888c5b31734b6fb90b092188f5aab3a65'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def j(v=0, d=None, dd=None):
    return (F(v), list(d or [F(0)] * 3),
            [list(row) for row in (dd or [[F(0)] * 3 for _ in range(3)])])


def add(x, y):
    return j(x[0] + y[0], [x[1][i] + y[1][i] for i in range(3)],
             [[x[2][i][k] + y[2][i][k] for k in range(3)] for i in range(3)])


def mul(x, y):
    return j(x[0] * y[0], [x[1][i] * y[0] + x[0] * y[1][i] for i in range(3)],
             [[x[2][i][k] * y[0] + x[1][i] * y[1][k]
               + x[1][k] * y[1][i] + x[0] * y[2][i][k]
               for k in range(3)] for i in range(3)])


def scale(c, x):
    return mul(j(c), x)


def powj(x, m):
    ans = j(1)
    for _ in range(m):
        ans = mul(ans, x)
    return ans


def unpack(col):
    fields = []
    for k in range(4):
        dd = [[F(0)] * 3 for _ in range(3)]
        pair = 0
        for z in range(3):
            for t in range(z, 3):
                dd[z][t] = dd[t][z] = col[16 + 4 * pair + k]
                pair += 1
        fields.append(j(col[k], [col[4 + 4 * z + k] for z in range(3)], dd))
    return fields


def pack(fields):
    ans = [f[0] for f in fields]
    ans += [f[1][z] for z in range(3) for f in fields]
    ans += [f[2][z][t] for z in range(3) for t in range(z, 3) for f in fields]
    return ans


def analytic_map(a, n):
    # Independent exact second-order Cartesian Taylor algebra at r=1.
    ns = [j(n[i], [F(i == z) - n[i] * n[z] for z in range(3)],
            [[-F(i == z) * n[t] - F(i == t) * n[z] - F(z == t) * n[i]
              + 3 * n[i] * n[z] * n[t] for t in range(3)] for z in range(3)])
          for i in range(3)]
    omega = j(0, [-x / a for x in n],
              [[-F(z == t) / a for t in range(3)] for z in range(3)])
    h = j(1 / a, [x / a for x in n],
          [[F(z == t) / a for t in range(3)] for z in range(3)])
    beta = [j(-n[i] / a, [-F(i == z) / a for z in range(3)]) for i in range(3)]
    y, z = ns[1:]
    ang = [j(1), y, z, mul(y, y), mul(y, z), mul(z, z)]
    fs = [j(1), omega, y, z, mul(omega, omega), mul(omega, y),
          mul(omega, z), mul(y, y), mul(y, z), mul(z, z)]
    cols = []
    for f in fs:
        cols.append(pack([mul(h, f)] + [mul(b, f) for b in beta]))
    for m in range(3):
        for k in range(3):
            for f in ang:
                amp = mul(powj(omega, m), f)
                cols.append(pack([j()] + [mul(amp, add(j(i == k),
                             scale(-1, mul(ns[i], ns[k])))) for i in range(3)]))
    for f in ang:
        cols.append(pack([j()] + [mul(mul(powj(omega, 2), f), ni) for ni in ns]))
    return sp.Matrix(40, 70, lambda row, col: cols[col][row]), ns


def compatibility_map(n, ns):
    # b_n = n.delta beta + [2r/(1+r^2)] delta alpha.  At r=1 the
    # coefficient has value 1, zero gradient and Hessian -n_i n_j.
    k = j(1, dd=[[-n[i] * n[t] for t in range(3)] for i in range(3)])
    t1 = [-n[1], n[0], F(0)]
    t2 = [n[1] * t1[2] - n[2] * t1[1],
          n[2] * t1[0] - n[0] * t1[2],
          n[0] * t1[1] - n[1] * t1[0]]
    cols = []
    for c in range(40):
        f = unpack([F(i == c) for i in range(40)])
        b = mul(k, f[0])
        for i in range(3):
            b = add(b, mul(ns[i], f[1 + i]))
        rows = [b[0]] + b[1]
        for u, v in [(t1, t1), (t1, t2), (t2, t2), (n, t1), (n, t2)]:
            rows.append(sum(u[i] * b[2][i][q] * v[q] for i in range(3) for q in range(3)))
        cols.append(rows)
    return sp.Matrix(9, 40, lambda row, col: cols[col][row])


assert sha(GATE / 'index.json') == PIN
idx = json.loads((GATE / 'index.json').read_text())
for item in idx['files']:
    p = GATE / item['path']
    assert p.stat().st_size == item['bytes'] and sha(p) == item['sha256']
raw = json.loads((GATE / 'actual-release.json').read_text())
assert (GATE / 'actual-release.json').read_bytes() == (GATE / 'actual-debug.json').read_bytes()
receipt = json.loads((GATE / 'receipt.json').read_text())
assert len(receipt['commands']) == 5
assert all(c['returncode'] == 0 for c in receipt['commands'])
assert receipt['source_before'] == receipt['source_after']
assert len(receipt['source_before']) == 372
for path, pin in receipt['source_before'].items():
    assert sha(ROOT / path) == pin
matrix_checks = []
exact_maps = []
maxerr = 0.0
for row in raw['input_maps']:
    a = F(str(row['a']))
    n = [F(9, 25), -F(12, 25), F(4, 5)] if row['dir'] else [F(1), F(0), F(0)]
    M, ns = analytic_map(a, n)
    E = compatibility_map(n, ns)
    rankM = DomainMatrix.from_Matrix(M).rank()
    rankE = DomainMatrix.from_Matrix(E).rank()
    assert rankM == 31 and rankE == 9 and E * M == sp.zeros(9, 70)
    err = max(abs(float(M[i, k]) - row['M'][i][k]) for i in range(40) for k in range(70))
    assert err < 1e-12
    maxerr = max(maxerr, err)
    matrix_checks.append({'a': float(a), 'direction': row['dir'],
                          'analytic_basis_rank': rankM, 'independent_compatibility_rank': rankE,
                          'exact_E_times_M_zero': True, 'native_map_max_error': err})
    exact_maps.append({'a': str(a), 'normal': [str(v) for v in n],
                       'E_9x40': [[str(E[i, k]) for k in range(40)] for i in range(9)],
                       'M_40x70': [[str(M[i, k]) for k in range(70)] for i in range(40)]})
assert len(raw['controls']) == 1120
errors = [0.0] * 3
for row in raw['controls']:
    n = [.36, -.48, .8] if row['dir'] else [1, 0, 0]
    ang = [1, n[1], n[2], n[1] ** 2, n[1] * n[2], n[2] ** 2]
    expected = 0 if row['col'] < 64 else 4 * (3 - row['sigma']) * ang[row['col'] - 64] / row['a'] ** 3
    for k in range(3):
        errors[k] = max(errors[k], abs(row['N1_t'][k] - expected))
r, a, sigma = sp.symbols('r a sigma', positive=True)
O = (1 - r**2) / (2 * a)
T = -(r**4 + 6 * r**2 + 1) / (4 * a * r)
C = (r**6 + (16 * sigma - 29) * r**4 + 15 * r**2 - 3) / (4 * a * r**2 * (r**2 - 1))
reaction = sp.factor(C + 2 * T * sp.diff(O, r) / O)
assert sp.simplify(reaction.subs(sigma, 3) + (r**2 + 3) * (3 * r**2 - 1) / (4 * a * r**2)) == 0
assert sp.limit(reaction.subs(sigma, 3), r, 1) == -2 / a
report = {'passed_independent_read_only_review': True, 'scientific_index_sha256': PIN,
          'scientific_files_verified': len(idx['files']), 'source_inputs_verified': 372,
          'release_debug_byte_equal': True, 'commands_all_zero': 5,
          'basis_checks': matrix_checks, 'analytic_native_input_map_error': maxerr,
          'N1_expected_map_errors': errors, 'actual_summary': raw['summary'],
          'next_R0_error': max(x['next_R0_error'] for x in raw['controls']),
          'next_leading_constraints_error': max(x['next_leading_constraints_error'] for x in raw['controls']),
          'scope': 'Exact compatible gauge second-jet basis at fixed reference ADM geometry, plus read-only actual-kernel results. Not full Einstein geometric ideal, nonlinear invariance, finite-Omega growth or sigma3 evolution admission.',
          'python': sys.version, 'sympy_version': sp.__version__,
          'sympy_init_sha256': sha(Path(sp.__file__))}
(OUT / 'review-results.json').write_text(json.dumps(report, indent=2) + '\n')
(OUT / 'exact-compatible-maps.json').write_text(json.dumps(exact_maps, indent=2) + '\n')
print(json.dumps(report, indent=2))
