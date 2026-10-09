"""Saved-array energy/forcing attribution; no operator assembly or propagation."""
from pathlib import Path
from fractions import Fraction
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
ROOT = HERE.parents[2]
P = ROOT / 'build-layer-research/boundary/total-j-finite-rb-control-20261009'
DEST = HERE / 'auxiliary-readback-001'
assert not DEST.exists()
DEST.mkdir()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rec(path):
    return {'path': str(path), 'sha256': sha(path), 'bytes': path.stat().st_size}


def mm(a, b):
    return np.einsum('ik,kj->ij', a, b, optimize=False)


def fro(a):
    return float(np.sqrt(np.sum(a*a)))


def error(a, b):
    return {'absolute_frobenius': fro(a-b),
            'max_absolute': float(np.max(np.abs(a-b))),
            'scaled_frobenius': fro(a-b)/max(1., fro(a), fro(b))}


def esolve(chol, rhs):
    return np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))


def energy_normal(chol, matrix):
    left = np.linalg.solve(chol, matrix)
    return np.linalg.solve(chol, left.T).T


V = HERE / 'verify_operator.py'
assert sha(V) == '855245e3e4429dc713d817285ec79f05dda112ec159b9b821c9983e95987d95b'
q128 = P / 'J0-base-Q128-sector-readback001'
npz128 = 'ca54faeecd0f1952c8e29704c243a4f46a10666d005f543bef8f6f9582dc35ab'
meta128 = '7d22c7abce879393c163cf1219ca94936d4136b0168faaf7c0431a0b2afcfd50'
command = [sys.executable, str(V), '--npz', str(q128/'operator.npz'),
           '--expect-sha256', npz128, '--metadata', str(q128/'matrix-metadata.json'),
           '--expect-metadata-sha256', meta128,
           '--output', str(DEST/'failed-global-Q128-result.json')]
(DEST/'Q128-command.json').write_text(json.dumps(command, indent=2)+'\n')
started = time.monotonic()
run = subprocess.run(command, capture_output=True)
(DEST/'Q128-stdout').write_bytes(run.stdout)
(DEST/'Q128-stderr').write_bytes(run.stderr)
q128_result = json.loads((DEST/'failed-global-Q128-result.json').read_text())
q128_run = {'command': command, 'returncode': run.returncode,
            'seconds': time.monotonic()-started, 'stderr_bytes': len(run.stderr),
            'status': q128_result['status'],
            'source': rec(V), 'result': rec(DEST/'failed-global-Q128-result.json')}

cases = (
    ('global-Q64', P/'J0-N8-rb.98-Q64-a12x24-readback002',
     '26ddb5877c20d4ee5277427989ef9aaf1a9c0ca416a33cf756e53d96f5ae1d6f',
     'fae33d587b03d382f68dcc940f4f2df17d0840d6537681d9ec45f8842875dcc3'),
    ('global-Q128-sector', q128, npz128, meta128),
    ('segmented-Q64-sector', P/'J0-segmented-Q64-sector-readback001',
     '8b4b9a5b33151d86359aa9e35f0ae44dc436ef4ab2d6c17796e77983f9422a27',
     '8449fc9b52f0137498e35b0f5e9b8e35b73998eebe37726bf1eaf4cdd2649d04'),
)
rows = []
for label, folder, npz_sha, meta_sha in cases:
    assert sha(folder/'operator.npz') == npz_sha
    assert sha(folder/'matrix-metadata.json') == meta_sha
    meta = json.loads((folder/'matrix-metadata.json').read_text())
    with np.load(folder/'operator.npz', allow_pickle=False) as z:
        a = {k: z[k] for k in z.files}
    for matrix in a.values():
        assert matrix.dtype == np.float64 and np.isfinite(matrix).all()
    E, Kw, Ks, G, S = [a[k] for k in ('E', 'Kweak', 'Kstrong', 'Gvolume', 'SATload')]
    X = a['manufactured_X'].reshape(E.shape[0], -1)
    forcing = a['manufactured_load'].reshape(X.shape)
    boundary = a['manufactured_boundary_load'].reshape(X.shape)
    chol = np.linalg.cholesky(E)
    rate = esolve(chol, mm(Kw+S, X)+forcing+boundary)
    quad_rate = esolve(chol, mm(Kw-Ks, X))
    strong_residual = forcing-mm(E-Ks, X)
    boundary_residual = boundary+mm(S, X)
    total_prediction = quad_rate+esolve(chol, strong_residual+boundary_residual)
    normal_G = energy_normal(chol, G)
    flux = a['Fboundary']+S+S.T
    normal_flux = energy_normal(chol, flux)
    normal_total = normal_G+normal_flux
    # Small saved skew is reported, never silently averaged away.
    ge = np.linalg.eigvalsh(normal_G, UPLO='L')
    fe = np.linalg.eigvalsh(normal_flux, UPLO='L')
    te = np.linalg.eigvalsh(normal_total, UPLO='L')
    assert np.isfinite(ge).all() and np.isfinite(fe).all() and np.isfinite(te).all()
    trace = np.linalg.solve(chol, a['B'].T).T
    singular = np.linalg.svd(trace, compute_uv=False)
    decoration = None
    if 'incoming_left' in a:
        originals = [Path(r['path']) for r in meta['input_pins']
                     if Path(r['path']).name == 'operator.npz']
        assert len(originals) == 1
        original = originals[0]
        source_pin = next(r for r in meta['input_pins'] if r['path'] == str(original))
        assert sha(original) == source_pin['sha256']
        with np.load(original, allow_pickle=False) as z:
            original_keys = list(z.files)
            for key in original_keys:
                assert a[key].shape == z[key].shape and a[key].dtype == z[key].dtype
                assert a[key].tobytes() == z[key].tobytes(), key
        principal_report = next(Path(r['path']) for r in meta['input_pins']
                                if Path(r['path']).name == 'report.json' and
                                'harmonic-principal-constraint-sectors' in r['path'])
        principal_pin = next(r for r in meta['input_pins'] if r['path'] == str(principal_report))
        assert sha(principal_report) == principal_pin['sha256']
        sectors = json.loads(principal_report.read_text())['sectors']['1']
        lefts = []
        for name in ('constraint', 'gauge', 'TT'):
            exact = np.array([[float(Fraction(x)) for x in row]
                              for row in sectors[name+'_left_rows']], dtype=np.float64)
            assert exact.tobytes() == a[name+'_left'].tobytes()
            lefts.append(exact)
        assert np.vstack(lefts).tobytes() == a['incoming_left'].tobytes()
        decoration = {'all_original_arrays_bitwise_unchanged': True,
                      'original_array_count': len(original_keys),
                      'original': rec(original), 'principal_source': rec(principal_report),
                      'sector_left_rows_match_exact_frozen_sign_plus': True}
    rows.append({
        'name': label, 'matrix': rec(folder/'operator.npz'),
        'metadata': rec(folder/'matrix-metadata.json'),
        'direct_strong_forcing_load': error(forcing, mm(E-Ks, X)),
        'direct_boundary_forcing_load': error(boundary, -mm(S, X)),
        'rate_defect': {'absolute_coefficient_L2': fro(rate-X),
                        'E_norm': fro(mm(chol.T, rate-X))},
        'weak_minus_strong_rate_prediction': error(rate-X, quad_rate),
        'full_rate_defect_decomposition': error(rate-X, total_prediction),
        'symmetric_energy_readback': {
            'G_normalized_skew': error(normal_G, normal_G.T),
            'post_SAT_flux_normalized_skew': error(normal_flux, normal_flux.T),
            'G_over_E_minimum': float(ge[0]), 'G_over_E_maximum': float(ge[-1]),
            'G_energy_norm_rate_bound': float(ge[-1]/2),
            'post_SAT_flux_over_E_maximum': float(fe[-1]),
            'G_plus_post_SAT_flux_over_E_maximum': float(te[-1]),
            'interpretation': 'Symmetric quadratic energy-production spectrum only; not generator eigenvalues, not a uniform/refined stability bound.'},
        'energy_normalized_full_trace_norm': float(singular[0]),
        'energy_normalized_full_trace_norm_squared': float(singular[0]**2),
        'trace_interpretation': '||B L^-T||2 with E=LL^T; finite N=8 and rb=.98 only.',
        'decoration': decoration,
    })
result = {'status': 'PASS_saved_array_attribution_and_energy_readback',
          'source': rec(Path(__file__)), 'Q128_verifier_run': q128_run,
          'cases': rows, 'versions': {'python': sys.version, 'numpy': np.__version__},
          'scientific_kernel_or_assembler_run': False,
          'generator_eigenvalues_or_propagation': False,
          'scope': 'GlobalQ64/Q128 failures remain failures. Segmented output retains its independent matrix-only PASS. Auxiliary values do not establish a PDE/CPBC/scri, uniform energy or nonlinear stability claim.'}
out = DEST/'result.json'
out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
print(json.dumps({'status': result['status'], 'result': rec(out),
                  'Q128_status': q128_result['status'],
                  'energy_G_max': {r['name']: r['symmetric_energy_readback']['G_over_E_maximum'] for r in rows},
                  'trace_norm': {r['name']: r['energy_normalized_full_trace_norm'] for r in rows}}, indent=2))
