"""Bounded synthetic-only preflight; extracts four reviewed helper bodies."""
from pathlib import Path
import ast
import hashlib
import json
import math
import shutil
import sys
import warnings

import numpy as np
import scipy
from scipy.linalg import solve_triangular

warnings.simplefilter('error')
np.seterr(all='raise')
HERE = Path(__file__).resolve().parent
SOURCE = HERE.parent/'analyze_growth.py'
EXPECTED = '85e6b721ad143a73955811ce546afed78ac3ba88c3d8419348cd6e108f2aecda'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(SOURCE) == EXPECTED
assert not (HERE/'result.json').exists()
shutil.copyfile(SOURCE, HERE/'reviewed-analyze-growth-source.py')
module = ast.parse(SOURCE.read_text())
names = ('mm', 'energy_transform', 'energy_form', 'rk3')
functions = [node for node in module.body if isinstance(node, ast.FunctionDef)
             and node.name in names]
assert len(functions) == 4
# No analyzer module import, analyze(), execute flag or matrix loader occurs.
namespace = {'np': np, 'solve_triangular': solve_triangular}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(SOURCE), 'exec'), namespace)
mm, transform, form, rk3 = [namespace[n] for n in names]
np.load = lambda *args, **kwargs: (_ for _ in ()).throw(
    AssertionError('A saved-array loader is forbidden in synthetic preflight'))


def error(a, b):
    return float(np.linalg.norm(a-b)/max(1., np.linalg.norm(a), np.linalg.norm(b)))


E = np.array([[4., 1., .5], [1., 3., .25], [.5, .25, 2.]])
gershgorin = float(np.min(np.diag(E)-(np.sum(np.abs(E), axis=1)-np.diag(E))))
assert gershgorin > 0
L = np.linalg.cholesky(E)
X = np.array([[1., -.3, .2], [.4, .8, -1.], [.2, -.5, .7]])
F = np.array([[1., 2., -3.], [2., 4., .5], [-3., .5, -2.]])
inverse_LT = np.linalg.solve(L.T, np.eye(3))
expected_form = np.linalg.solve(L, F)@inverse_LT
actual_form = form(L, F)
form_error = error(actual_form, expected_form)
quadratic_form_error = error(X.T@F@X, (L.T@X).T@actual_form@(L.T@X))
assert form_error <= 2e-13 and quadratic_form_error <= 2e-13
diagonal = np.diag([-2., -.5, .75])
nilpotent = np.array([[0., 2., 0.], [0., 0., 3.], [0., 0., 0.]])
triangular = -np.eye(3)+nilpotent
cases = (
    ('zero', np.zeros((3, 3)), lambda t: np.eye(3)),
    ('diagonal', diagonal, lambda t: np.diag(np.exp(t*np.diag(diagonal)))),
    ('Jordan_upper_triangular', triangular,
     lambda t: math.exp(-t)*(np.eye(3)+t*nilpotent+(t*t/2)*(nilpotent@nilpotent))),
)
rows = []
T = .5
steps = (8, 16, 32, 64)
for name, J, exponential in cases:
    A = transform(L, J)
    independent_A = L.T@J@inverse_LT
    Z = L.T@X
    action_error = error(A@Z, L.T@(J@X))
    similarity_error = error(A, independent_A)
    work = E@J+J.T@E
    energy_identity = error(A+A.T, form(L, work))
    wrong_transform = L@J@np.linalg.solve(L, np.eye(3))
    wrong_difference = error(wrong_transform, independent_A)
    exact = L.T@(exponential(T)@X)
    numerical = [rk3(A, Z, T, n) for n in steps]
    errors = [error(state, exact) for state in numerical]
    ratios = [a/b if b else None for a, b in zip(errors, errors[1:])]
    orders = [math.log2(v) if v and v > 0 else None for v in ratios]
    assert max(action_error, similarity_error, energy_identity) <= 2e-13
    if name == 'zero':
        assert max(errors) <= 2e-13
    else:
        assert wrong_difference > 1e-3
        assert all(b < a for a, b in zip(errors, errors[1:]))
        assert all(v is not None and 2.7 <= v <= 3.4 for v in orders)
        assert errors[-1] <= 2e-7
    rows.append({'name': name, 'generator_J': J.tolist(),
                 'energy_generator_A': A.tolist(),
                 'similarity_scaled_error': similarity_error,
                 'action_scaled_error': action_error,
                 'symmetric_energy_identity_scaled_error': energy_identity,
                 'wrong_L_vs_LT_transform_difference': wrong_difference,
                 'closed_exponential': exponential(T).tolist(),
                 'T': T, 'steps': list(steps), 'RK_scaled_errors': errors,
                 'RK_error_ratios': ratios, 'RK_observed_orders': orders})
assert sha(SOURCE) == EXPECTED
result = {
    'status': 'PASS_bounded_synthetic_helpers_only',
    'source_sha256': sha(Path(__file__)),
    'reviewed_analyzer_sha256': EXPECTED,
    'extracted_helpers': list(names),
    'helper_extraction': 'Exact four AST function definitions, with actual NumPy and SciPy solve_triangular; analyzer module/analyze()/execute are never invoked.',
    'E': E.tolist(), 'E_strict_diagonal_dominance_lower_bound': gershgorin,
    'E_Cholesky': L.tolist(), 'synthetic_initial_columns_X': X.tolist(),
    'synthetic_symmetric_form_F': F.tolist(),
    'energy_form_scaled_error': form_error,
    'quadratic_form_scaled_error': quadratic_form_error,
    'cases': rows,
    'thresholds': {'helper_algebra': 2e-13, 'zero_RK_roundoff': 2e-13,
                   'nonzero_RK_order_interval': [2.7, 3.4],
                   'nonzero_final_RK_error': 2e-7,
                   'wrong_transpose_sensitivity_minimum': 1e-3},
    'versions': {'python': sys.version, 'executable': sys.executable,
                 'numpy': np.__version__, 'scipy': scipy.__version__,
                 'numpy_path': np.__file__, 'scipy_path': scipy.__file__},
    'actual_scientific_operator_loaded': False,
    'analyzer_analyze_or_execute_called': False,
    'generator_eigensolver_called': False,
    'scope': 'Only helper algebra and third-order time integration on independent three-dimensional analytic examples. No saved finite-rb matrix, expm accuracy, PDE/native/BH result or scientific growth admission follows.'}
(HERE/'result.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({'status': result['status'], 'result_sha256': sha(HERE/'result.json'),
                  'RK_orders': {r['name']: r['RK_observed_orders'] for r in rows}}, indent=2))
