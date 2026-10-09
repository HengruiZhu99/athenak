"""Verify original pins and selected subspaces; freeze small evidence once."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import time

import numpy as np
import scipy
from scipy.linalg import svd
from scipy.sparse import load_npz

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
B = ROOT / 'build-layer-research/boundary'
DEST = P / 'immutable-discrete-mode-diagnostic-20261009'
assert not DEST.exists(), 'Refuse to mutate an existing freeze'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
started = time.monotonic()
checks = []


def read_json(path):
    def bad(value):
        raise AssertionError((path, value))
    return json.loads(path.read_text(), parse_constant=bad)


def verify_catalog(path, wanted):
    assert sha(path) == wanted
    data = read_json(path)
    for name, entry in data['files'].items():
        file = path.parent / name
        assert sha(file) == entry['sha256']
        assert file.stat().st_size == entry['bytes']
    checks.append({'path': str(path.relative_to(ROOT)), 'sha256': wanted,
                   'verified_small_files': len(data['files'])})


verify_catalog(B / 'full-tensor-global-final/manifest.json',
               '4f62819e2337c0cb20571071fe10d0e80d8b43895a928ccd4418faec0bc590d2')
verify_catalog(B / 'full-tensor-C0-long-window-20261009/immutable-C0-long-window-20261009/index.json',
               'd1efcbea11e730eec2e784ef7981e64f5e67d64e51bc963c6cdd7ed22834d896')
verify_catalog(B / 'full-tensor-C0-N20-20261009/immutable-C0-N20-default-span-v2-20261009/index.json',
               '8ec4c4dd84898d19b055831696de6e3608542b1fd10a88e04f087b1f689f99aa')
verify_catalog(B / 'full-tensor-mode-finite-step-20261009/immutable-mode-final-step-20261009/index.json',
               '3e212ec5e437c0190f687fcd3b95d52671fe814b36c9e62469924a4a01e241b0')
meta = read_json(P / 'candidate-metadata.json')
diag = read_json(P / 'candidate-diagnostics.json')
suite = read_json(P / 'receipt.json')
assert len(suite['commands']) == 9
assert all(q['returncode'] == 0 for q in suite['commands'])
assert sha(P / 'candidate-vectors.npz') == meta['candidate_vectors_sha256']
assert sha(P / 'candidate-metadata.json') == 'e411042b90498bb4f08eadcdf9214095028c049df8a0c15b46b5dfd4d3f2d6e1'
assert sha(P / 'export_candidates.py') == meta['source_sha256']
assert sha(P / 'classify_candidate.py') == diag['source_sha256']
assert sha(P / 'reduced_modes.py') == suite['source_sha256']
assert sha(P / 'run_suite.py') == suite['runner_sha256']
pins = {}
for report in sorted(P.glob('*.json')):
    data = read_json(report)
    if not isinstance(data, dict):
        continue
    for path, digest in data.get('input_sha256', {}).items():
        assert sha(ROOT / path) == digest
        pins[path] = digest
for key in ('matrix', 'states', 'original_metadata'):
    path = ROOT / meta[key]
    assert sha(path) == meta[key + '_sha256']
    pins[meta[key]] = sha(path)
J = load_npz(ROOT / meta['matrix'])
vectors = np.load(P / 'candidate-vectors.npz')
rechecks = []
for row in meta['candidates']:
    v = vectors[row['key']]
    lam = complex(*row['lambda'])
    assert np.isfinite(v).all() and abs(np.linalg.norm(v) - 1) < 1e-13
    residual = float(np.linalg.norm(J @ v - lam * v))
    assert abs(residual - row['actual_J_residual_generator_units']) < 1e-12
    assert residual < 1e-6 and row['singular_value_fraction_at_rank'] > 1e-10
    rechecks.append({'key': row['key'], 'residual_generator_units': residual})
del J

# Warning-free independent selected N20 reconstruction; no saved output mutation.
N20 = B / 'full-tensor-C0-N20-20261009/full22'
J = load_npz(N20 / 'spatialnorm-projected-J20.npz')
states = np.load(N20 / 'spatialnorm-projected-krylov-m50-80-h0.1-t6.0.npz')
n20 = []
for name, rank in [('n20-late2-gauge-s2', 32),
                   ('n20-late4-gauge-s1', 16),
                   ('n20-late4-gauge-s2', 16)]:
    report = read_json(P / (name + '.json'))
    assert 'DONE ' + name in (P / (name + '.log')).read_text()
    lo, hi = report['window']
    ids = np.flatnonzero((states['times'] >= lo - 1e-12) &
                        (states['times'] <= hi + 1e-12))[::report['stride']]
    X = states['values'][ids[:-1], :, 0].T
    X = X / np.linalg.norm(X, axis=0)
    u, s, vh = svd(X, full_matrices=False, lapack_driver='gesdd')
    U = u[:, :rank]
    JU = J @ U
    small = np.einsum('ki,kj->ij', U, JU, optimize=False)
    assert np.isfinite(small).all()
    eig, eigvec = np.linalg.eig(small)
    k = int(np.argmin(np.abs(eig - (1.9182803 + 6.8055088j))))
    v = np.einsum('ij,j->i', U, eigvec[:, k], optimize=False)
    v /= np.linalg.norm(v)
    residual = float(np.linalg.norm(J @ v - eig[k] * v))
    old = next(q for q in report['results']
               if q['rank'] == rank and q['method'] == 'rayleigh-ritz')
    oldk = int(np.argmin(np.abs(np.array([complex(*z) for z in old['eigenvalues']]) - eig[k])))
    assert abs(residual - old['actual_J_residual_generator_units'][oldk]) < 1e-9
    assert residual < 1e-6 and s[rank - 1] / s[0] > 1e-10
    n20.append({'case': name, 'rank': rank,
                'lambda': [float(eig[k].real), float(eig[k].imag)],
                'residual_generator_units': residual,
                'residual_relative_to_max1_lambda': residual / max(1, abs(eig[k])),
                'singular_fraction': float(s[rank - 1] / s[0]),
                'original_stdout_completion_verified': True,
                'original_full_stderr_not_archived': True})

receipt = {
    'status': 'PASS_READ_ONLY_APPROXIMATE_MODE_DIAGNOSTIC_NOT_EIGENVALUE_CERTIFICATION',
    'head_at_verification': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
    'original_search_head': suite['head'],
    'source_sha256': sha(Path(__file__)), 'versions': {'numpy': np.__version__, 'scipy': scipy.__version__},
    'verified_input_catalogs': checks, 'all_reported_input_pins_reverified': pins,
    'search_suite_commands': 9, 'search_suite_all_returncode_zero': True,
    'search_suite_warning_commands': sum(bool(q['stderr']) for q in suite['commands']),
    'warnings_preserved_in_original_receipt': True,
    'selected_N16_actual_J_rechecks': rechecks,
    'independent_explicit_contraction_N20_rechecks': n20,
    'native_exit': diag['native_exit'], 'actual_RHS_real_imag_probes': 10,
    'actual_constraint_real_imag_probes': 6,
    'no_new_propagation_or_LU_or_evolution': True,
    'residual_never_divided_by_J_norm': True,
    'scope': 'Approximate finite-grid generator pseudomodes; no spectral error bound, globally-rightmost, continuum, or stability claim',
    'seconds': time.monotonic() - started,
}
(P / 'final-verification.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
large = {}
for file in sorted(P.rglob('*')):
    if file.is_file() and file.suffix == '.npz':
        large[str(file.relative_to(ROOT))] = {'sha256': sha(file), 'bytes': file.stat().st_size}
for path, digest in pins.items():
    file = ROOT / path
    if file.suffix in ('.npz', '.bin') or file.name == 'server-spatialnorm':
        large[path] = {'sha256': digest, 'bytes': file.stat().st_size}
(P / 'large-artifacts-metadata-only.json').write_text(json.dumps(large, indent=2) + '\n')
(P / 'verification.log').write_text('PASS: four frozen input catalogs; all source/matrix/history pins; four N16 actual J residuals; three independent explicit-contraction N20 reconstructions.\nNo source, history, or exported-vector mutation. No new propagation.\n')
small = [f for f in sorted(P.rglob('*')) if f.is_file() and f.suffix != '.npz']
DEST.mkdir()
files = {}
for file in small:
    name = str(file.relative_to(P))
    target = DEST / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file, target)
    files[name] = {'sha256': sha(target), 'bytes': target.stat().st_size}
index = {'scope': receipt['scope'], 'files': files,
         'external_large_records': len(large),
         'count': len(files), 'bytes': sum(q['bytes'] for q in files.values())}
(DEST / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
for name, q in files.items():
    assert sha(DEST / name) == q['sha256']
print(json.dumps({'index': str((DEST / 'index.json').relative_to(ROOT)),
                  'sha256': sha(DEST / 'index.json'), 'count': index['count'],
                  'bytes': index['bytes'], 'large_records': len(large),
                  'final_verification_sha256': sha(P / 'final-verification.json')}, indent=2))
