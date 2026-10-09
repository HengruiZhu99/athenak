"""Exact normal-principal algebra; retained kernel records, no kernel execution."""
from pathlib import Path
import hashlib
import json
import sys
import sympy as s

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
GATE = (ROOT / 'build-layer-research/continuum/conformal-q-null-feedback'
        '/immutable-conformal-Q-null-feedback-local-20261009')
R = s.Rational


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(expressions, variables):
    return s.Matrix(expressions).jacobian(variables)


def serialize(matrix):
    return [[str(x) for x in row] for row in matrix.tolist()]


assert sha(GATE / 'index.json') == (
    'dac759becbe666c71443328b61324a50aeaf0f72e76b4e751a9cabfa3a87cd96')
assert sha(GATE / 'principal.json') == (
    '02e4e4b1c4b6d2fccd6f442687d12ad5798e38a85ea52d61fab24b63f162dbc9')

# Exact W=1 matrix in kernel_symbol.cpp ordering, reconstructed independently.
A = s.zeros(20)
A[:8, :8] = s.Matrix([
    [0, 0, 0, -1, 0, 0, 0, 0],
    [0, 0, 0, R(2, 3), R(4, 3), 0, 0, -R(2, 3)],
    [0, 0, 0, 0, 0, -2, 0, R(4, 3)],
    [-1, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, R(1, 2), 0],
    [-R(2, 3), R(1, 3), -R(1, 2), 0, 0, 0, R(2, 3), 0],
    [0, 0, 0, -R(4, 3), -R(2, 3), 0, 0, R(4, 3)],
    [-1, R(1, 2), 0, 0, 0, 0, 1, 0]])
vblock = s.Matrix([[0, -2, 0, 1], [-R(1, 2), 0, R(1, 2), 0],
                   [0, 0, 0, 1], [0, 0, 1, 0]])
for start in [8, 12]:
    A[start:start + 4, start:start + 4] = vblock
for start in [16, 18]:
    A[start:start + 2, start:start + 2] = s.Matrix([[0, -2], [-R(1, 2), 0]])
I = s.eye(20)
assert A * A == I

names = ('a c h p t Ann ln b hT AnT lT bT hU AnU lU bU '
         'hplus Aplus hcross Across').split()
y = s.Matrix(s.symbols(' '.join(names)))
a, c, h, p, t, Ann, ln, b, hT, AnT, lT, bT, hU, AnU, lU, bU = y[:16]

# Independent physical diagnostic derivation in a frozen Penrose orthonormal
# frame: the first derivative of delta gamma_bar is dgtilde/chi-c*I.
dh = s.Matrix([[h, hT, hU],
               [hT, -h/2 + y[16], y[18]],
               [hU, y[18], -h/2 - y[16]]])
dbar = dh - c * s.eye(3)
Hred = s.expand(dbar[0, 0] - s.trace(dbar))
Mred = s.Matrix([Ann - R(2, 3)*(p + 2*t), AnT, AnU])
# Tracefree metric makes Gamma_tilde^i=partial_j h^ij at this order.
Z = (s.Matrix([ln, lT, lU]) - dh[:, 0]) / 2
constraint = s.Matrix([t, Z[0], Z[1], Z[2], Hred, *Mred])
C = rows(constraint, y)
assert Hred == h + 2*c
assert C.rank() == 8
B = s.Matrix([
    [0, 1, 0, 0, R(1, 2), 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 1, 0],
    [0, 0, 0, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, -2, 0, 0],
    [0, 0, 0, 0, -R(1, 2), 0, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 1, 0, 0, 0, 0]])
assert C * A == B * C
assert B * B == s.eye(8)

# Independent coordinate pullback: delta g_ab=Lie_xi eta_ab, negative K sign.
# xi is a normal plane-wave generator with d_tau xi=lambda*d_s xi.
# The four zeta amplitudes are d_s^2 xi^(tau,n,T,U).
ztime, zn, zT, zU = s.symbols('ztime zn zT zU')
zetas = s.Matrix([ztime, zn, zT, zU])
sectors = {}
all_left = []
for sign in [-1, 1]:
    projector = (I + sign*A) / 2
    assert projector.rank() == 10
    assert (C * projector).rank() == 4
    # Unit normal Fourier derivative, with d_tau=sign*d_s. A common second
    # normal derivative factor on xi is absorbed into each zeta amplitude.
    eta = s.diag(-1, 1, 1, 1)
    xi = s.Matrix([ztime, zn, zT, zU])
    derivative = s.Matrix([sign, 1, 0, 0])
    lie_eta = derivative*(eta*xi).T + (eta*xi)*derivative.T
    d_alpha = -lie_eta[0, 0]/2
    d_beta = lie_eta[0, 1:].T
    hbar = lie_eta[1:, 1:]
    d_chi = -s.trace(hbar)/3
    htilde = hbar + d_chi*s.eye(3)
    normal = s.Matrix([1, 0, 0])
    negative_K = -(sign*hbar-normal*d_beta.T-d_beta*normal.T)/2
    assert negative_K == -ztime*normal*normal.T
    trace_K = s.trace(negative_K)
    tracefree_K = negative_K-trace_K*s.eye(3)/3
    contracted_Gamma = htilde[:, 0]
    derived_diffeo = s.Matrix([
        d_alpha, d_chi, htilde[0, 0], trace_K, 0, tracefree_K[0, 0],
        contracted_Gamma[0], d_beta[0],
        htilde[0, 1], tracefree_K[0, 1], contracted_Gamma[1], d_beta[1],
        htilde[0, 2], tracefree_K[0, 2], contracted_Gamma[2], d_beta[2],
        (htilde[1, 1]-htilde[2, 2])/2,
        (tracefree_K[1, 1]-tracefree_K[2, 2])/2,
        htilde[1, 2], tracefree_K[1, 2]])
    diffeo = s.Matrix([
        sign*ztime, -R(2, 3)*zn, R(4, 3)*zn, -ztime, 0,
        -R(2, 3)*ztime, R(4, 3)*zn, sign*zn-ztime,
        zT, 0, zT, sign*zT,
        zU, 0, zU, sign*zU, 0, 0, 0, 0])
    assert derived_diffeo == diffeo
    G = rows(derived_diffeo, zetas)
    assert G.rank() == 4 and A*G == sign*G
    assert C*G == s.zeros(8, 4)
    T = s.zeros(20, 2)
    for col, start in enumerate([16, 18]):
        T[start, col], T[start + 1, col] = 1, -R(sign, 2)
    assert T.rank() == 2 and A*T == sign*T
    assert C*T == s.zeros(8, 2)
    assert G.row_join(T).rank() == 6
    assert s.Matrix.vstack(C, A-sign*I).rank() == 14

    theta, Znormal, Ztangent, Zu, Hr, Mr, Mt, Mu = constraint
    LC = rows([Hr-2*sign*Mr, theta+Mr+sign*Znormal,
               Ztangent+sign*Mt, Zu+sign*Mu], y)
    LG = rows([p-sign*a, b-p-t/2+R(3, 4)*sign*ln,
               lT+sign*bT, lU+sign*bU], y)
    LT = rows([y[17]-R(sign, 2)*y[16],
               y[19]-R(sign, 2)*y[18]], y)
    left = s.Matrix.vstack(LC, LG, LT)
    assert left*A == sign*left and left.rank() == 10
    assert LC*G == s.zeros(4) and LC*T == s.zeros(4, 2)
    assert LG*T == s.zeros(4, 2)
    assert LT*G == s.zeros(2, 4)
    assert LG*G == s.diag(-2, 2*sign, 2, 2)
    assert LT*T == -sign*s.eye(2)
    all_left.append(left)
    sectors[str(sign)] = {
        'projector_rank': projector.rank(),
        'constraint_image_rank': (C*projector).rank(),
        'Einstein_principal_kernel_dimension': 6,
        'coordinate_gauge_rank': G.rank(), 'screen_TT_rank': T.rank(),
        'left_rows_rank': left.rank(),
        'coordinate_gauge_right_vectors': serialize(G),
        'screen_TT_right_vectors': serialize(T),
        'constraint_left_rows': serialize(LC),
        'gauge_left_rows': serialize(LG), 'TT_left_rows': serialize(LT),
        'gauge_left_on_coordinate_vectors': serialize(LG*G)}
combined_left = s.Matrix.vstack(*all_left)
assert combined_left.rank() == 20

# Read historical actual matrices only; no executable or source integration.
actual = json.loads((GATE / 'principal.json').read_text())
harmonic = [row for row in actual if row['W'] == 1]
actual_error = max(abs(float(A[i, j])-row['M'][i][j])
                   for row in harmonic for i in range(20) for j in range(20))
assert len(harmonic) == 288 and actual_error < 2e-12

pinpaths = [Path(__file__), ROOT/'tst/hyperboloidal/kernel_symbol.cpp',
            ROOT/'tst/hyperboloidal/check_kernel_symbol.py',
            ROOT/'src/z4c/hyperboloidal/conformal_rhs.hpp',
            ROOT/'src/z4c/hyperboloidal/conformal_constraints.hpp',
            ROOT/'src/z4c/hyperboloidal/layer_gauge.hpp',
            GATE/'index.json', GATE/'principal.json']
report = {
    'passed_exact_normal_principal_sector_algebra': True,
    'scope': ('Fixed positive Omega, alpha, chi and SPD Penrose metric; W=1 '
              'normal principal/pseudodifferential algebra only. No new kernel '
              'run, PDE operator, boundary condition or energy estimate.'),
    'python': sys.version, 'sympy_version': s.__version__,
    'field_order': names, 'constraint_order':
    ['Theta_phys/Omega', 'Zn', 'ZT', 'ZU', 'Hred', 'Mnred', 'MTred', 'MUred'],
    'physical_scaling': {
        'H_principal': 'Omega^2 D_s Hred',
        'M_Penrose_orthonormal_covector_principal': 'Omega D_s Mred',
        'M_physical_orthonormal_covector_principal': 'Omega^2 D_s Mred',
        'Theta_phys': 'Omega t', 'Z_physical_orthonormal': 'Omega Z',
        'Z_Penrose_orthonormal': 'Z'},
    'pseudodifferential_scope': 'Hred=(ik)^-1 H/Omega^2 and '
    'Mred=(ik)^-1 M_Penrose/Omega at nonzero normal k, principal order only.',
    'A': serialize(A), 'C': serialize(C), 'B': serialize(B),
    'exact_CA_equals_BC': True, 'exact_A_squared_I': True,
    'exact_B_squared_I': True, 'constraint_rank': C.rank(),
    'combined_left_rank': combined_left.rank(), 'sectors': sectors,
    'actual_retained_harmonic_rows': len(harmonic),
    'max_retained_actual_A_error': actual_error,
    'source_pins': {str(path.relative_to(ROOT)): sha(path) for path in pinpaths}}
text = json.dumps(report, indent=2, allow_nan=False)+'\n'
(HERE/'report.json').write_text(text)
print(text, end='')
