"""HELD saved-JSON principal checker; standard library only, no eigensolver."""
import itertools
import json
import math
from pathlib import Path
import sys


def mm(a, b):
    return [[sum(x * y for x, y in zip(row, col)) for col in zip(*b)] for row in a]


def eye(n):
    return [[float(i == j) for j in range(n)] for i in range(n)]


def zeros(n):
    return [[0.0] * n for _ in range(n)]


def maxerr(a, b):
    return max(abs(x - y) for ar, br in zip(a, b) for x, y in zip(ar, br))


def scalederr(a, b):
    return max(abs(x - y) / max(1.0, abs(x), abs(y))
               for ar, br in zip(a, b) for x, y in zip(ar, br))


def insert(target, block, start):
    for i, row in enumerate(block):
        for j, x in enumerate(row):
            target[start + i][start + j] = x


def expected(f, mu, ec):
    m = zeros(20)
    insert(m, [[0, 0, 0, -f, 0, 0, 0, 0],
               [0, 0, 0, 2/3, 4/3, 0, 0, -2/3],
               [0, 0, 0, 0, 0, -2, 0, 4/3],
               [-1, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 1/2, 0],
               [-2/3, 1/3, -1/2, 0, 0, 0, 2/3, 0],
               [0, 0, 0, -4/3, -2/3, 0, 0, 4/3],
               [-1, ec, 0, 0, 0, 0, mu, 0]], 0)
    for start in (8, 12):
        insert(m, [[0, -2, 0, 1], [-1/2, 0, 1/2, 0],
                   [0, 0, 0, 1], [0, 0, mu, 0]], start)
    for start in (16, 18):
        insert(m, [[0, -2], [-1/2, 0]], start)
    return m


def scalar_transform(f, C):
    t = [[1, 0, 0, 0, 0, 0, 0, 0],
         [0, 0, 0, -f, 0, 0, 0, 0],
         [0, 1-2*C, 0, 0, 0, 0, -C, 0],
         [0, 0, 0, 2/3, 4/3-2*C, 0, 0, -2/3],
         [0, 2, 1, 0, 0, 0, 0, 0],
         [0, 0, 0, 4/3, 8/3, -2, 0, 0],
         [0, 2, 0, 0, 0, 0, 1, 0],
         [0, 0, 0, 0, 2, 0, 0, 0]]
    inv = [[1, 0, 0, 0, 0, 0, 0, 0],
           [0, 0, 1, 0, 0, 0, C, 0],
           [0, 0, -2, 0, 1, 0, -2*C, 0],
           [0, -1/f, 0, 0, 0, 0, 0, 0],
           [0, 0, 0, 0, 0, 0, 0, 1/2],
           [0, -2/(3*f), 0, 0, 0, -1/2, 0, 2/3],
           [0, 0, -2, 0, 0, 0, 1-2*C, 0],
           [0, -1/f, 0, -3/2, 0, 0, 0, 1-3*C/2]]
    return t, inv


def basis(f, mu, q, C):
    t, ti = scalar_transform(f, C)
    d, di, speeds = zeros(8), zeros(8), []
    for start, speed2 in zip((0, 2, 4, 6), (f, q, 1.0, 1.0)):
        s = math.sqrt(speed2)
        insert(d, [[-s, 1], [s, 1]], start)
        insert(di, [[-1/(2*s), 1/(2*s)], [1/2, 1/2]], start)
        speeds.extend((-s, s))
    b, bi = zeros(20), zeros(20)
    insert(b, mm(d, t), 0)
    insert(bi, mm(ti, di), 0)
    s = math.sqrt(mu)
    for start in (8, 12):
        insert(b, [[1/2, 1, -1/2, 0], [-1/2, 1, 1/2, 0],
                   [0, 0, -s, 1], [0, 0, s, 1]], start)
        insert(bi, [[1, -1, -1/(2*s), 1/(2*s)], [1/2, 1/2, 0, 0],
                    [0, 0, -1/(2*s), 1/(2*s)], [0, 0, 1/2, 1/2]], start)
        speeds.extend((-1, 1, -s, s))
    for start in (16, 18):
        insert(b, [[1/2, 1], [-1/2, 1]], start)
        insert(bi, [[1, -1], [1/2, 1/2]], start)
        speeds.extend((-1, 1))
    return b, bi, speeds


def fixed_cases():
    out = []
    for a, chi, w, g0, oblique in itertools.product(
            (.05, 1., 3.), (.1, 1.), (0., .5, 1.), (.375, .75), (0, 1)):
        out.append((1, a, chi, w, g0, oblique, None, None))
    for w, g0, oblique in itertools.product((0., .5), (.375, .75), (0, 1)):
        out.append((1, 1., g0, w, g0, oblique, None, None))
    a = 54/29
    for oblique in (0, 1):
        out.append((1, a, .375/(2*a*a), 0., .375, oblique, None, None))
    for mu in (.125, .375, .75, 1., 2., 8.):
        ec = 2*mu*mu/(1+mu)**2
        q = (4*mu-2*ec)/3
        for f, oblique in itertools.product((1., 3., q), (0, 1)):
            out.append((0, 1., 1., 0., .375, oblique, f, mu))
    assert len(out) == 118
    return out


def finite(x):
    if isinstance(x, dict):
        return all(finite(v) for v in x.values())
    if isinstance(x, list):
        return all(finite(v) for v in x)
    return not isinstance(x, float) or math.isfinite(x)


def main(path):
    def bad_constant(value):
        raise ValueError('nonfinite JSON token: ' + value)
    data = json.loads(Path(path).read_text(), parse_constant=bad_constant)
    assert finite(data) and len(data) == 118
    maxima = dict(matrix=0., conjugacy_absolute=0., conjugacy_scaled=0.,
                  left_absolute=0., left_scaled=0., basis_inverse=0.,
                  basis_condition_inf=0.)
    collisions = dict(mu1=0, q1=0, f1=0, q_equals_f=0)
    rows = []
    for number, (case, params) in enumerate(zip(data, fixed_cases())):
        candidate, a, chi, w, g0, oblique, f, mu = params
        assert case['candidate'] == candidate and case['oblique'] == oblique
        for name, value in zip(('alpha', 'chi', 'W', 'G0'), (a, chi, w, g0)):
            assert case[name] == value, (number, name, case[name], value)
        if candidate:
            f = 1+2*(1-w)/a
            mu = w+(1-w)*g0/(a*a*chi)
        ec, q = 2*mu*mu/(1+mu)**2, (4*mu-4*mu*mu/(1+mu)**2)/3
        assert f > 0 and mu > 0 and q > 0
        for name, value in zip(('f', 'mu', 'ec', 'q'), (f, mu, ec, q)):
            assert abs(case[name]-value) <= 2e-14*max(1., abs(value)), (number, name)
        m = case['M']
        assert len(m) == 20 and all(len(row) == 20 for row in m)
        matrix_error = maxerr(m, expected(f, mu, ec))
        assert matrix_error <= 2e-12, (number, 'matrix', matrix_error)
        C = 2*(1+mu)**2/(4*mu*mu+5*mu+3)
        t, ti = scalar_transform(f, C)
        n = zeros(8)
        for start, speed2 in zip((0, 2, 4, 6), (f, q, 1., 1.)):
            insert(n, [[0., 1.], [speed2, 0.]], start)
        actual_n = mm(mm(t, [row[:8] for row in m[:8]]), ti)
        ca, cs = maxerr(actual_n, n), scalederr(actual_n, n)
        assert ca <= 1e-10 and cs <= 1e-11, (number, 'conjugacy', ca, cs)
        b, bi, speeds = basis(f, mu, q, C)
        bm = mm(b, m)
        sb = [[speed*x for x in row] for speed, row in zip(speeds, b)]
        la, ls = maxerr(bm, sb), scalederr(bm, sb)
        inv_error = max(maxerr(mm(b, bi), eye(20)), maxerr(mm(bi, b), eye(20)))
        assert la <= 1e-10 and ls <= 1e-10 and inv_error <= 1e-10, (number, 'basis', la, ls, inv_error)
        condition = max(sum(map(abs, r)) for r in b)*max(sum(map(abs, r)) for r in bi)
        for name, value in [('matrix', matrix_error), ('conjugacy_absolute', ca),
                            ('conjugacy_scaled', cs), ('left_absolute', la),
                            ('left_scaled', ls), ('basis_inverse', inv_error),
                            ('basis_condition_inf', condition)]:
            maxima[name] = max(maxima[name], value)
        for name, x, y in [('mu1', mu, 1.), ('q1', q, 1.), ('f1', f, 1.), ('q_equals_f', q, f)]:
            collisions[name] += int(abs(x-y) <= 1e-12*max(1., abs(x), abs(y)))
        rows.append(dict(number=number, candidate=bool(candidate), f=f, mu=mu, q=q,
                         matrix_error=matrix_error, left_residual=la, inverse_error=inv_error))
    assert all(v > 0 for v in collisions.values())
    result = dict(passed=True, actual20_cases=len(data), maxima=maxima, collisions=collisions,
                  full_basis_method='explicit two-sided inverse and analytic left eigenfields; no numerical rank/eigensolve',
                  scope='Frozen positive alpha/chi constant-reference principal only; no nonlinear or puncture admission',
                  cases=rows)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main(sys.argv[1])
