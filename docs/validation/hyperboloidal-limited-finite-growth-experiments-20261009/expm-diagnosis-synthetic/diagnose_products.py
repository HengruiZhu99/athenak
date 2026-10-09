"""Finite A-power product diagnosis; exceptions remain enabled and preserved."""
from pathlib import Path
import hashlib
import json
import time
import traceback
import warnings

import numpy as np

warnings.filterwarnings('error', category=RuntimeWarning)
np.seterr(all='raise', under='ignore')
HERE = Path(__file__).resolve().parent
OUT = HERE/'products-001'
assert not OUT.exists()
OUT.mkdir()
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
with np.load(HERE/'isolated-scipy-001/isolated-input.npz', allow_pickle=False) as z:
    a = z['argument']
mm = lambda x, y: np.einsum('ik,kj->ij', x, y, optimize=False)
reference2 = mm(a, a)
reference4 = mm(reference2, reference2)
reference6 = mm(reference4, reference2)
assert all(np.isfinite(v).all() for v in (a, reference2, reference4, reference6))
cases = [('A2', a, a, reference2),
         ('A4', reference2, reference2, reference4),
         ('A6', reference4, reference2, reference6),
         ('synthetic_identity', np.eye(64), np.eye(64), np.eye(64)),
         ('synthetic_zero', np.zeros((64, 64)), np.ones((64, 64)), np.zeros((64, 64)))]
rows = []
arrays = {}
for name, left, right, reference in cases:
    output = np.full_like(reference, np.nan)
    row = {'name': name, 'inputs_finite': True,
           'left_maxabs': float(np.max(np.abs(left))),
           'right_maxabs': float(np.max(np.abs(right))),
           'reference_maxabs': float(np.max(np.abs(reference)))}
    start = time.monotonic()
    try:
        np.matmul(left, right, out=output)
        row['returned'] = True
    except Exception as exc:
        row.update({'returned': False, 'error_type': type(exc).__name__, 'message': str(exc)})
        (OUT/(name+'-traceback.txt')).write_text(traceback.format_exc())
    row['seconds'] = time.monotonic()-start
    row['out_array_finite_after_call'] = bool(np.isfinite(output).all())
    if row['out_array_finite_after_call']:
        delta = output-reference
        norm = lambda v: float(np.sqrt(np.sum(v*v)))
        row['out_vs_literal_einsum_scaled'] = norm(delta)/max(1., norm(output), norm(reference))
        row['out_vs_literal_einsum_peak'] = float(np.max(np.abs(delta)))
    arrays[name+'_output'] = output
    arrays[name+'_literal_einsum'] = reference
    rows.append(row)
np.savez_compressed(OUT/'products.npz', **arrays)
result = {'source_sha256': sha(Path(__file__)),
          'input_sha256': sha(HERE/'isolated-scipy-001/isolated-input.npz'),
          'rows': rows, 'errstate': np.geterr(), 'warnings_suppressed': False,
          'propagation_or_generator_eigensolve': False,
          'interpretation': 'Matmul-only finite power products with caught original exceptions. A finite accurate out array after an exception suggests a backend floating-status issue, but does not identify its C-level cause or excuse the failed original exponential.'}
(OUT/'receipt.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({'receipt_sha256': sha(OUT/'receipt.json'), 'rows': rows}, indent=2))
