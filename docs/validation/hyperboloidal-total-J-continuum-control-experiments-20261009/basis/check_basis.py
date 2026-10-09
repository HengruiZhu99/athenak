"""Symbolic polynomial/rotation gate and generator. No actual PDE operator."""
from pathlib import Path
import hashlib
import json
import sys
import time
import sympy as s
from sympy.polys.matrices import DomainMatrix
import total_j_basis as b

P = Path(__file__).resolve().parent
t0 = time.monotonic()
records = []
counts = {'norm': 0, 'J2': 0, 'Jz': 0, 'conjugacy': 0,
          'parity': 0, 'homogeneity': 0, 'STF': 0, 'm0_orthogonality': 0}
spin_checks = []
for sr in range(3):
    for m in range(-sr, sr+1):
        v = b.spin(sr, m)
        assert s.simplify(sum(s.conjugate(q)*q for q in v)-1) == 0
        for n in range(-sr, sr+1):
            w = b.spin(sr, n)
            assert s.simplify(sum(s.conjugate(q)*r for q, r in zip(v, w))-int(m == n)) == 0
        if sr == 2:
            assert s.simplify(v[0]+v[4]+v[8]) == 0
            assert all(s.simplify(v[3*i+j]-v[3*j+i]) == 0 for i in range(3) for j in range(3))
        spin_checks.append({'spin': sr, 'm': m, 'norm': 1})
for j in range(3):
    for sr in range(3):
        m0 = []
        for l in b.allowed_l(j, sr):
            for m in range(-j, j+1):
                v = b.basis(j, m, sr, l)
                assert b.inner(v, v) == 1
                counts['norm'] += 1
                j2 = [0]*len(v)
                for axis in range(3):
                    tmp = b.rotation(b.rotation(v, sr, axis), sr, axis)
                    j2 = [s.expand(q+r) for q, r in zip(j2, tmp)]
                assert all(s.simplify(q-j*(j+1)*r) == 0 for q, r in zip(j2, v))
                counts['J2'] += 1
                assert all(s.simplify(q-m*r) == 0 for q, r in zip(b.rotation(v, sr, 2), v))
                counts['Jz'] += 1
                neg = b.basis(j, -m, sr, l)
                assert all(s.simplify(q-(-1)**m*s.conjugate(r)) == 0 for q, r in zip(neg, v))
                counts['conjugacy'] += 1
                assert all(s.simplify(q.subs({b.x: -b.x, b.y: -b.y, b.z: -b.z}, simultaneous=True)-(-1)**l*q) == 0 for q in v)
                counts['parity'] += 1
                assert all(s.simplify(sum(x*s.diff(q, x) for x in b.xyz)-l*q) == 0 for q in v)
                counts['homogeneity'] += 1
                if sr == 2:
                    assert s.simplify(v[0]+v[4]+v[8]) == 0
                    assert all(s.simplify(v[3*i+k]-v[3*k+i]) == 0 for i in range(3) for k in range(3))
                    counts['STF'] += 1
                if m == 0:
                    assert all(s.simplify(s.im(q)) == 0 for q in v)
                    m0.append(v)
                records.append({'J': j, 'm': m, 'spin': sr, 'L': l,
                                'phase': str(s.I**(l+sr-j)),
                                'angular_norm': 1, 'orbital_parity': (-1)**l,
                                'polar_tensor_parity': (-1)**(l+sr),
                                'polynomials': [str(q) for q in v],
                                'terms': b.coefficient_records(v)})
        for i, v in enumerate(m0):
            for k, w in enumerate(m0):
                assert b.inner(v, w) == int(i == k)
                counts['m0_orthogonality'] += 1
angles = [(s.Rational(1), 0, 0), (0, 0, s.Rational(1)),
          (s.Rational(9, 25), -s.Rational(12, 25), s.Rational(4, 5)),
          (s.Rational(2, 3), s.Rational(1, 3), s.Rational(2, 3))]
rank_checks = []
for j in range(3):
    for sr in range(3):
        ls = list(b.allowed_l(j, sr))
        cols = []
        for l in ls:
            v = b.basis(j, 0, sr, l)
            # Removing a nonzero column-wide normalization puts this exact rank
            # calculation over rationals, without altering rank.
            norm = next(c for q in v if q != 0 for c in s.Poly(q, *b.xyz).coeffs() if c != 0)
            col = []
            for n in angles:
                col.extend(s.simplify(q.subs(dict(zip(b.xyz, n)))/norm) for q in v)
            cols.append(col)
        M = s.Matrix(len(cols[0]), len(cols), lambda i, k: cols[k][i])
        assert all(q.is_Rational for q in M)
        rank = DomainMatrix.from_Matrix(M).rank()
        assert rank == len(ls)
        rank_checks.append({'J': j, 'spin': sr, 'L_values': ls,
                            'stacked_rows': M.rows, 'exact_rank': rank})
layouts = {str(j): b.channel_layout(j) for j in range(3)}
assert [len(layouts[str(j)]) for j in range(3)] == [8, 16, 20]
# W=rho^q gives homogeneous polynomial degree L+2q. Every derivative is a
# polynomial, including at the origin; no parity-only extension is used.
polynomial_oracles = []
for j in range(3):
    for sr in range(3):
        for l in b.allowed_l(j, sr):
            for q in (0, 1, 2):
                v = [s.expand(p*b.rho**q) for p in b.basis(j, 0, sr, l)]
                for field in v:
                    assert all(s.simplify(sum(x*s.diff(field, x) for x in b.xyz)-(l+2*q)*field) == 0 for _ in [0])
                    for i in range(3):
                        for k in range(3):
                            assert s.diff(field, b.xyz[i], b.xyz[k]).subs(dict.fromkeys(b.xyz, 0)).is_finite
                polynomial_oracles.append({'J': j, 'spin': sr, 'L': l,
                                           'W_rho_power': q, 'homogeneous_degree': l+2*q})
data = {'conventions': {'spherical_Y': 'Condon--Shortley; sphere integral abs(Y)^2=1',
                         'coupling': 'CG(L,m_L;s,m_s|J,m)',
                         'phase': 'i^(L+s-J)', 'm_conjugacy': 'B(J,-m)=(-1)^m conjugate(B(J,m))',
                         'components': 'scalar one; Cartesian vector xyz; symmetric tensor full row-major 3x3',
                         'radial': 'r^L Y_Lm times W_L(rho), rho=x.x'},
        'records': records, 'channel_layouts': layouts,
        'metric_trace_tensor': 'identity/sqrt(3)',
        'scope': 'Basis/jet mathematics only; no actual full tensor operator, boundary closure, evolution or stability claim.'}
(P/'basis-data.json').write_text(json.dumps(data, indent=2)+'\n')
(P/'total_j_basis.hpp').write_text(b.cpp_header())
report = {'passed_symbolic_harmonic_basis': True, 'checks': counts,
          'spin_normalization_checks': spin_checks, 'stacked_rank_checks': rank_checks,
          'full_m0_channel_counts': [8, 16, 20], 'all_m_records': len(records),
          'regular_polynomial_oracles': polynomial_oracles,
          'seconds': time.monotonic()-t0, 'python': sys.version, 'sympy_version': s.__version__,
          'basis_data_sha256': hashlib.sha256((P/'basis-data.json').read_bytes()).hexdigest(),
          'generated_header_sha256': hashlib.sha256((P/'total_j_basis.hpp').read_bytes()).hexdigest(),
          'scope': data['scope']}
(P/'symbolic-report.json').write_text(json.dumps(report, indent=2)+'\n')
print(json.dumps({'passed': True, 'checks': counts, 'records': len(records),
                  'seconds': report['seconds'], 'header_sha256': report['generated_header_sha256']}, indent=2))
