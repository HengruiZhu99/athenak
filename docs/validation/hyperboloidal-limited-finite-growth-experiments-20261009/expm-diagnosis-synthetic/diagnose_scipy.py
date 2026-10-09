"""One isolated original SciPy expm call; failure/trace preserved, no acceptance."""
from pathlib import Path
import ast
import contextlib
import hashlib
import inspect
import io
import json
import os
import platform
import sys
import time
import traceback
import warnings

import numpy as np
import scipy
import scipy.linalg
import scipy.linalg._matfuncs as mf
import scipy.linalg._matfuncs_expm as compiled
from scipy.linalg import solve_triangular

warnings.filterwarnings('error', category=RuntimeWarning)
np.seterr(all='raise', under='ignore')
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
SOURCE = ROOT/'build-layer-research/continuum/finite-rb-limited-matrix-growth-20261009/analyze_growth.py'
FAILED = SOURCE.parent/'J0-N8-rb98-growth001'
OUT = HERE/'isolated-scipy-001'
assert not OUT.exists()
OUT.mkdir()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rec(path):
    return {'path': str(path.resolve()), 'sha256': sha(path), 'bytes': path.stat().st_size}


assert sha(SOURCE) == '7634bf135c20df9588f2e6cc33dd434d6e63925e6716586445443033dc7224e3'
receipt = json.loads((FAILED/'receipt.json').read_text())
assert receipt['error'] == {'type': 'FloatingPointError', 'message': 'divide by zero encountered in matmul'}
assert receipt['input_pins_unchanged'] and not receipt['passed_finite_ODE_numerical_checks']
matrix = next(Path(p) for p in receipt['inputs_before'] if p.endswith('operator.npz'))
assert sha(matrix) == receipt['inputs_before'][str(matrix)]
namespace = {'np': np, 'solve_triangular': solve_triangular}
tree = ast.parse(SOURCE.read_text())
selected = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in ('mm', 'energy_transform')]
assert len(selected) == 2
exec(compile(ast.Module(body=selected, type_ignores=[]), str(SOURCE), 'exec'), namespace)
with np.load(matrix, allow_pickle=False) as saved:
    L = saved['energy_cholesky']
    J = saved['Jbulk']+saved['Jsat']
A = namespace['energy_transform'](L, J)
assert A.shape == (64, 64) and np.isfinite(A).all()
np.savez_compressed(OUT/'isolated-input.npz', A=A, argument=.25*A, L=L, J=J)
(OUT/'scipy-expm-source.py').write_text(inspect.getsource(mf.expm))
config = io.StringIO()
with contextlib.redirect_stdout(config):
    np.show_config()
(OUT/'numpy-configuration.txt').write_text(config.getvalue())
result = {'source': rec(Path(__file__)), 'reviewed_analyzer': rec(SOURCE),
          'original_failed_receipt': rec(FAILED/'receipt.json'), 'matrix': rec(matrix),
          'input': rec(OUT/'isolated-input.npz'), 't': .25,
          'argument_one_norm': float(np.max(np.sum(np.abs(.25*A), axis=0))),
          'argument_maxabs': float(np.max(np.abs(.25*A))),
          'argument_finite': True, 'dtype': str(A.dtype), 'array_flags': str(A.flags),
          'runtime': {'python': sys.version, 'executable': sys.executable,
                      'numpy': np.__version__, 'scipy': scipy.__version__,
                      'platform': platform.platform(), 'machine': platform.machine(),
                      'numpy_file': np.__file__, 'scipy_file': scipy.__file__,
                      'errstate': np.geterr(),
                      'environment': {k: os.environ.get(k) for k in ('PYTHONPATH', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')}},
          'library_pins': [rec(Path(mf.__file__)), rec(Path(compiled.__file__))],
          'original_failure_preserved': True, 'warning_suppression': False,
          'generator_eigensolve': False, 'propagation_acceptance': False}
start = time.monotonic()
try:
    output = scipy.linalg.expm(.25*A)
    result['isolated_call'] = {'returned': True, 'finite': bool(np.isfinite(output).all()),
                               'maxabs': float(np.max(np.abs(output))),
                               'frobenius_norm': float(np.sqrt(np.sum(output*output)))}
    np.savez_compressed(OUT/'unexpected-return.npz', result=output)
except Exception as exc:
    (OUT/'traceback.txt').write_text(traceback.format_exc())
    result['isolated_call'] = {'returned': False, 'error_type': type(exc).__name__,
                               'message': str(exc), 'traceback': rec(OUT/'traceback.txt')}
result['isolated_seconds'] = time.monotonic()-start
(OUT/'receipt.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({'receipt': rec(OUT/'receipt.json'), 'call': result['isolated_call']}, indent=2))
