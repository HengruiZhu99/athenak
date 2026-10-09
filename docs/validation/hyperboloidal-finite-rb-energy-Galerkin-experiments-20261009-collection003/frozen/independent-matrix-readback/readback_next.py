"""Pinned saved-matrix verifier plus energy-only attribution for later J blocks."""
from pathlib import Path
from fractions import Fraction
import argparse
import hashlib
import json
import subprocess
import sys
import time
import warnings

import numpy as np

warnings.simplefilter('error')
np.seterr(all='raise')
HERE = Path(__file__).resolve().parent
VERIFIER_SHA = '855245e3e4429dc713d817285ec79f05dda112ec159b9b821c9983e95987d95b'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
rec = lambda p: {'path': str(p.resolve()), 'sha256': sha(p), 'bytes': p.stat().st_size}
mm = lambda a, b: np.einsum('ik,kj->ij', a, b, optimize=False)
fro = lambda a: float(np.sqrt(np.sum(a*a)))


def error(a, b):
    return {'absolute_frobenius': fro(a-b),
            'scaled_frobenius': fro(a-b)/max(1., fro(a), fro(b)),
            'max_absolute': float(np.max(np.abs(a-b)))}


def solve(L, f):
    return np.linalg.solve(L.T, np.linalg.solve(L, f))


def form(L, f):
    return np.linalg.solve(L, np.linalg.solve(L, f).T).T


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--matrix', type=Path, required=True)
    p.add_argument('--matrix-sha', required=True)
    p.add_argument('--metadata', type=Path, required=True)
    p.add_argument('--metadata-sha', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    verifier = HERE/'verify_operator.py'
    assert sha(verifier) == VERIFIER_SHA
    command = [sys.executable, str(verifier), '--npz', str(a.matrix.resolve()),
               '--expect-sha256', a.matrix_sha,
               '--metadata', str(a.metadata.resolve()),
               '--expect-metadata-sha256', a.metadata_sha,
               '--output', str((a.output/'result.json').resolve())]
    (a.output/'command.json').write_text(json.dumps(command, indent=2)+'\n')
    started = time.monotonic()
    run = subprocess.run(command, capture_output=True)
    (a.output/'stdout').write_bytes(run.stdout)
    (a.output/'stderr').write_bytes(run.stderr)
    primary = json.loads((a.output/'result.json').read_text())
    assert primary['status'] in ('PASS', 'FAIL_preserved')
    assert sha(a.matrix) == a.matrix_sha and sha(a.metadata) == a.metadata_sha
    meta = json.loads(a.metadata.read_text())
    with np.load(a.matrix, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    E, Kw, Ks, G, S = [arrays[k] for k in ('E', 'Kweak', 'Kstrong', 'Gvolume', 'SATload')]
    L = np.linalg.cholesky(E)
    X = arrays['manufactured_X'].reshape(len(E), -1)
    load = arrays['manufactured_load'].reshape(X.shape)
    boundary = arrays['manufactured_boundary_load'].reshape(X.shape)
    defect = solve(L, mm(Kw+S, X)+load+boundary)-X
    quad_defect = solve(L, mm(Kw-Ks, X))
    normal_G = form(L, G)
    normal_flux = form(L, arrays['Fboundary']+S+S.T)
    ge = np.linalg.eigvalsh(normal_G, UPLO='L')
    fe = np.linalg.eigvalsh(normal_flux, UPLO='L')
    trace_norm = float(np.linalg.svd(np.linalg.solve(L, arrays['B'].T).T,
                                     compute_uv=False)[0])
    original_pin = next(r for r in meta['input_pins'] if Path(r['path']).name == 'operator.npz')
    original = Path(original_pin['path'])
    assert sha(original) == original_pin['sha256']
    with np.load(original, allow_pickle=False) as z:
        for key in z.files:
            assert arrays[key].shape == z[key].shape and arrays[key].dtype == z[key].dtype
            assert arrays[key].tobytes() == z[key].tobytes(), key
        original_count = len(z.files)
    principal_pin = next(r for r in meta['input_pins'] if Path(r['path']).name == 'report.json'
                         and 'harmonic-principal-constraint-sectors' in r['path'])
    assert sha(Path(principal_pin['path'])) == principal_pin['sha256']
    sectors = json.loads(Path(principal_pin['path']).read_text())['sectors']['1']
    lefts = []
    for name in ('constraint', 'gauge', 'TT'):
        exact = np.array([[float(Fraction(x)) for x in row]
                          for row in sectors[name+'_left_rows']], dtype=np.float64)
        assert exact.tobytes() == arrays[name+'_left'].tobytes()
        lefts.append(exact)
    assert np.vstack(lefts).tobytes() == arrays['incoming_left'].tobytes()
    auxiliary = {
        'source': rec(Path(__file__)), 'primary_result': rec(a.output/'result.json'),
        'matrix': rec(a.matrix), 'metadata': rec(a.metadata),
        'direct_strong_forcing': error(load, mm(E-Ks, X)),
        'direct_boundary_forcing': error(boundary, -mm(S, X)),
        'weak_minus_strong_defect_prediction': error(defect, quad_defect),
        'G_over_E_minimum': float(ge[0]), 'G_over_E_maximum': float(ge[-1]),
        'G_energy_norm_rate_bound': float(ge[-1]/2),
        'G_normalized_skew': error(normal_G, normal_G.T),
        'post_SAT_flux_over_E_maximum': float(fe[-1]),
        'post_SAT_flux_normalized_skew': error(normal_flux, normal_flux.T),
        'energy_normalized_full_trace_norm': trace_norm,
        'all_original_arrays_bitwise_unchanged': True,
        'original_array_count': original_count,
        'sector_rows_match_exact_frozen_sign_plus': True,
        'scope': 'Fixed finite matrix symmetric energy/trace and forcing readback only; no generator spectrum, CPBC, propagation or uniform energy bound.'}
    (a.output/'auxiliary.json').write_text(json.dumps(auxiliary, indent=2, allow_nan=False)+'\n')
    receipt = {'status': primary['status'], 'command': command,
               'seconds': time.monotonic()-started, 'returncode': run.returncode,
               'stderr_bytes': len(run.stderr), 'source': rec(Path(__file__)),
               'verifier': rec(verifier), 'result': rec(a.output/'result.json'),
               'auxiliary': rec(a.output/'auxiliary.json'),
               'versions': {'python': sys.version, 'numpy': np.__version__},
               'scientific_kernel_or_assembler_run': False,
               'generator_spectrum_or_propagation': False}
    (a.output/'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False)+'\n')
    print(json.dumps(receipt, indent=2))
    return run.returncode


if __name__ == '__main__':
    raise SystemExit(main())
