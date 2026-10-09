"""Read-only independent Cartesian first-jet reconstruction; no kernel run."""
from pathlib import Path
import hashlib
import json
import sympy as s

P = Path(__file__).resolve().parent
G = P.parent/'q-null-jet-followup/immutable-Q-null-firstjet-map-20261009'
ROOT = P.parents[2]
PIN = '5490057dfc04e060bca65ec7ee1a3bb363a34cab0fe5c890f23d477a7bda7ec1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


assert sha(G/'index.json') == PIN
index = json.loads((G/'index.json').read_text())
for name, digest in index['files'].items():
    assert sha(G/name) == digest
assert (G/'actual.json').read_bytes() == (G/'actual-debug.json').read_bytes()
receipt = json.loads((G/'receipt.json').read_text())
assert receipt['source_before'] == receipt['source_after']
assert len(receipt['source_before']) == 372
for name, digest in receipt['source_before'].items():
    assert sha(ROOT/name) == digest
for command in receipt['commands']:
    assert command['returncode'] == 0
    assert (G/command['stderr']).read_text() == ''
for name, entry in index['outside_binaries'].items():
    executable = G.parent/name
    assert sha(executable) == entry['sha256']
    assert executable.stat().st_size == entry['bytes']


def rat(x):
    return s.Rational(x).limit_denominator(1000000)


def reconstruct(a, column):
    """ADM constraints and null jets from flat outer Penrose geometry.

    Independent conventions: M_i=D_j K^j_i-D_i K, K=P+2Theta,
    K^j_i=Omega A^j_i+K delta^j_i/3, physical gamma=Omega^-2 gamma_bar.
    Cartesian first jets are unrestricted at this one north point.
    """
    v = [s.S(0)]*20
    d = [[s.S(0)]*20 for _ in range(3)]
    family, field = divmod(column, 20)
    if family == 0:
        v[field] = s.S(1)
    else:
        d[family-1][field] = -1/a if family == 1 else s.S(1)
    pairs = [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2)]

    def tensor(offset, values):
        h = s.zeros(3)
        for k, (i, j) in enumerate(pairs):
            h[i, j] = h[j, i] = values[offset+k]
        h[2, 2] = -h[0, 0]-h[1, 1]
        return h

    h, A = tensor(7, v), tensor(12, v)
    dh = [tensor(7, row) for row in d]
    c, pp, theta, al, bn = v[1], v[2], v[3], v[0], v[4]
    x, ss = c-h[0, 0], al+bn
    dc = [row[1] for row in d]
    gamma_contract = [sum(dh[j][i, j] for j in range(3)) for i in range(3)]
    delta = [v[17+i]-gamma_contract[i] for i in range(3)]
    H = s.zeros(3)
    for i in range(3):
        for j in range(3):
            # Linearized connection of gamma_bar=gtilde/chi.
            connection = (dh[i][j, 0]+dh[j][i, 0]-dh[0][i, j])/2
            connection -= (int(j == 0)*dc[i]+int(i == 0)*dc[j]-int(i == j)*dc[0])/2
            H[i, j] = h[i, j]+connection
    SA = s.zeros(3)
    for i in range(3):
        for j in range(3):
            ztf = (int(i == 0)*delta[j]+int(j == 0)*delta[i]
                   -s.Rational(2, 3)*int(i == j)*delta[0])/2
            SA[i, j] = 2*(H[i, j]-int(i == j)*s.trace(H)/3-ztf-A[i, j])/a**2
    B = [s.S(0)]*20
    B[0] = (-pp+3*ss)/a**2
    B[1] = 2*(pp+2*theta-3*ss)/(3*a)
    B[2] = -2*(pp+2*theta)/a**2-3*x/a**3+10*theta
    B[3] = -2*pp/a**2-(1/a**2+20)*theta-3*x/a**3
    B[4] = -5*(x+2*a*ss)/a**3
    for k, (i, j) in enumerate(pairs):
        B[12+k] = SA[i, j]
    for i in range(3):
        B[17+i] = -(10-2/a**2)*delta[i]-2*(2*d[i][2]+d[i][3])/(3*a)+4*A[i, 0]/a**2
    E = [s.S(0)]*24
    E[0] = -6*x/a**2-4*(pp+2*theta)/a
    for i in range(3):
        E[1+i] = 2*A[i, 0]/a-2*(d[i][2]+2*d[i][3])/3
        E[4+i] = delta[i]/2
    E[7] = theta
    E[8] = x/a**2+2*ss/a
    # Radial differentiation includes h_ref'(1)=1/a, beta_ref'(1)=-1/a.
    xp = dc[0]-dh[0][0, 0]
    sp = d[0][0]+d[0][4]
    E[9] = -2*x/a-xp/a-2*ss-2*sp
    E[10] = pp-3*ss
    for j in (1, 2):
        # Differentiate n and Omega_i in Cartesian coordinates.
        wn_j = d[j][0]+d[j][4]+v[4+j]
        E[10+j] = (dc[j]-dh[j][0, 0]-2*h[0, j])/a**2+2*wn_j/a
        E[12+j] = d[j][2]-3*wn_j
        E[14+j] = d[j][3]
    P1, T1, x1 = -a*d[0][2], -a*d[0][3], -a*xp
    lap = -3*c/a+dc[0]/(2*a)+sum(dh[j][j, 0] for j in range(3))/a
    E[17] = -4*(P1+2*T1)/a+4*lap-6*(x1/a**2-2*x/a)
    E[18] = T1
    for k, (i, j) in enumerate(pairs):
        E[19+k] = SA[i, j]
    return s.Matrix(B), s.Matrix(E)


actual = json.loads((G/'actual.json').read_text())
rows = []
error = 0.0
for row in actual['maps']:
    a = rat(row['a'])
    columns = [reconstruct(a, k) for k in range(80)]
    B = s.Matrix.hstack(*(b for b, _ in columns))
    E = s.Matrix.hstack(*(e for _, e in columns))
    Braw, Eraw = s.Matrix(row['B']), s.Matrix(row['E'])
    error = max(error, max(abs(float(t)) for t in B-Braw),
                max(abs(float(t)) for t in E-Eraw))
    assert B == Braw.applyfunc(rat)
    assert E == Eraw.applyfunc(rat)
    basic, angular = E[:11, :], E[:18, :]
    minimal = angular.col_join(E[18:19, :]).col_join(E[22:24, :])
    ranks = [B.rank(), basic.rank(), basic.col_join(B).rank(), angular.rank(),
             angular.col_join(B).rank(), minimal.rank(), E.rank(), E.col_join(B).rank()]
    assert ranks == [11, 10, 18, 17, 20, 20, 20, 20]
    assert (B[:, :20]*B[:, :20]).rank() == 11
    assert E[0, :]+4*E[10, :]/a+8*E[7, :]/a+6*E[8, :] == s.zeros(1, 80)
    for i in range(3):
        theta_derivative = -E[18, :]/a if i == 0 else E[14+i, :]
        assert E[1+i, :]-(10*a-2/a)*E[4+i, :]+theta_derivative-a*B[17+i, :]/2 == s.zeros(1, 80)
    rows.append({'a': float(a), 'ranks': ranks, 'R0_nullity': 80-B.rank(),
                 'analytic_B_and_ADM_null_E_exact_after_rational_reconstruction': True})
assert error < 4e-15
# The two controls deliberately do not construct exact Einstein-compatible data.
for t in actual['tests']:
    if t['Omega'] != 0:
        continue
    a, which = rat(t['a']), t['which']
    e, b = list(map(rat, t['E'])), list(map(rat, t['S']))
    if which == 0:
        assert all(q == 0 for q in e[:18]) and e[18] == 1 and b[17] == -2/a**2
    elif which == 1:
        assert all(q == 0 for q in e[:19]) and b[15] == -2/a**2
    elif which in (2, 3):
        assert all(q == 0 for q in b) and all(q == 0 for q in e[:19])
    elif which == 4:
        assert b[12] == -8/(3*a**4) and b[17] == (12-80*a*a)/(3*a**4)
        assert e[9] == -4/(3*a**3)
result = {'passed': True, 'scientific_index_sha256': PIN,
          'verified_index_files': len(index['files']), 'rows': rows,
          'verified_source_inputs': len(receipt['source_before']),
          'verified_successful_commands': len(receipt['commands']),
          'verified_outside_executables': len(index['outside_binaries']),
          'independent_map_max_float_error': error,
          'Release_ASan_UBSan_JSON_byte_identical': True,
          'interpretation': 'Necessary linear first-jet compatibility only. Theta=Omega/P=-2Omega is off the exact Einstein sector; tangential shear is a local leading-data witness; Omega2P violates higher ADM constraints. No invariant exact Einstein ideal conclusion.',
          'no_tensor_compilation_native_propagation_executed': True}
(P/'result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result, indent=2))
