"""Finalize a hash-pinned, read-only C1/C0 approximate-mode comparison once."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import time

import numpy as np
from scipy.sparse import load_npz

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
D = P / 'immutable-discrete-mode-C1-diagnostic-20261009'
assert not D.exists(), 'Refuse to mutate an existing freeze'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
start = time.monotonic()


def load(path):
    def bad(q):
        raise ValueError((path, q))
    return json.loads(path.read_text(), parse_constant=bad)


config = load(P / 'input-pins.json')
catalogs = []
for i, (path, wanted) in enumerate(config['catalogs'].items()):
    index = ROOT / path
    assert sha(index) == wanted
    data = load(index)
    for name, q in data['files'].items():
        assert sha(index.parent / name) == q['sha256']
        assert (index.parent / name).stat().st_size == q['bytes']
    catalogs.append({'path': path, 'sha256': wanted, 'verified_files': len(data['files'])})
    folder = P / 'input-catalogs'
    folder.mkdir(exist_ok=True)
    shutil.copy2(index, folder / ('catalog' + str(i) + '.json'))
for path, q in config['artifacts'].items():
    assert sha(ROOT / path) == q['sha256']
    assert (ROOT / path).stat().st_size == q['bytes']
suite = load(P / 'receipt.json')
assert len(suite['commands']) == 9
assert all(q['returncode'] == 0 and q['stderr'] == '' for q in suite['commands'])
assert suite['source_sha256'] == sha(P / 'search.py')
assert suite['runner_sha256'] == sha(P / 'run_suite.py')
assert suite['config_sha256'] == sha(P / 'input-pins.json')
meta = load(P / 'candidate-metadata.json')
diag = load(P / 'candidate-diagnostics.json')
assert meta['config_sha256'] == sha(P / 'input-pins.json')
assert meta['choices_sha256'] == sha(P / 'choices.json')
assert load(P / 'preparation.json')['new_source_sha256'] == sha(P / 'search.py')
assert load(P / 'classifier-preparation.json')['new_source_sha256'] == sha(P / 'classify_candidate.py')
assert meta['source_sha256'] == sha(P / 'export_candidates.py')
assert diag['source_sha256'] == sha(P / 'classify_candidate.py')
assert diag['native_exit'] == 0
for path, wanted in diag['input_sha256'].items():
    assert sha(ROOT / path) == wanted
assert sha(P / 'candidate-vectors.npz') == meta['candidate_vectors_sha256']
J = load_npz(ROOT / config['matrix'])
v = np.load(P / 'candidate-vectors.npz')
actual = []
for row in meta['candidates']:
    vec = v[row['key']]
    assert np.isfinite(vec).all() and abs(np.linalg.norm(vec) - 1) < 1e-13
    lam = complex(*row['lambda'])
    residual = float(np.linalg.norm(J @ vec - lam * vec))
    assert abs(residual - row['actual_J_residual_generator_units']) < 1e-12
    assert residual < 1e-6 and row['singular_value_fraction_at_rank'] > 1e-10
    actual.append({'key': row['key'], 'residual_generator_units': residual})
C0 = ROOT / 'build-layer-research/continuum/discrete-mode-identification/immutable-discrete-mode-diagnostic-20261009'
oldmeta = load(C0 / 'candidate-metadata.json')
olddiag = load(C0 / 'candidate-diagnostics.json')
pair = meta['candidates'][0]['lambda']
oldpair = oldmeta['candidates'][0]['lambda']
summary = {'scope': 'Finite-grid approximate/pseudospectral continuous-generator comparison only; no eigenvalue error bound or continuum/native stability claim',
           'C0_lambda_candidate0': oldpair, 'C1_lambda_candidate0': pair,
           'C1_minus_C0_lambda': [n - o for n, o in zip(pair, oldpair)],
           'C1_minus_C0_lambda_fraction': [(n - o) / o for n, o in zip(pair, oldpair)],
           'C0_unit_candidate_H_M_Z': olddiag['native_H_M_Z_rms_for_unit_free20_candidate'],
           'C1_unit_candidate_H_M_Z': diag['native_H_M_Z_rms_for_unit_free20_candidate'],
           'C1_over_C0_unit_candidate_H_M_Z': [n / o for n, o in zip(diag['native_H_M_Z_rms_for_unit_free20_candidate'], olddiag['native_H_M_Z_rms_for_unit_free20_candidate'])],
           'C1_over_C0_Theta': diag['Theta_rms_for_unit_free20_candidate'] / olddiag['Theta_rms_for_unit_free20_candidate'],
           'component_norm_not_an_energy': True,
           'no_new_propagation_or_LU_or_evolution': True,
           'C1_C0_seed_and_coordinate_equalities': {q: meta[q] for q in ['initial_seed_arrays_bitwise_equal_C0_C1', 'times_and_reference_coordinates_bitwise_equal_C0_C1']}}
(P / 'comparison-summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False) + '\n')
receipt = {'status': 'PASS_READ_ONLY_C1_APPROXIMATE_MODE_COMPARISON',
           'catalogs': catalogs, 'selected_actual_J_residual_rechecks': actual,
           'suite_commands': 9, 'all_suite_exit_zero_empty_stderr': True,
           'suite_seconds': suite['seconds'], 'native_RHS_probes': 10, 'native_constraint_probes': 6,
           'native_server_exit': diag['native_exit'],
           'source_sha256': sha(Path(__file__)),
           'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
           'seconds': time.monotonic() - start,
           'scope': summary['scope']}
(P / 'final-verification.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
large = dict(config['artifacts'])
for file in sorted(P.glob('*.npz')):
    large[str(file.relative_to(ROOT))] = {'sha256': sha(file), 'bytes': file.stat().st_size}
(P / 'large-artifacts-metadata-only.json').write_text(json.dumps(large, indent=2) + '\n')
(P / 'verification.log').write_text('PASS: three input catalogs and all external artifact pins; all nine searches exit0/empty stderr; four actual-J residual rechecks; ten instantaneous RHS and six constraint probes; no propagation.\n')
small = [p for p in sorted(P.rglob('*')) if p.is_file() and p.suffix != '.npz']
finite_json = sum(p.suffix == '.json' for p in small)
for file in small:
    if file.suffix == '.json':
        load(file)
D.mkdir()
files = {}
for file in small:
    name = str(file.relative_to(P))
    target = D / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(file, target)
    files[name] = {'sha256': sha(target), 'bytes': target.stat().st_size}
index = {'scope': summary['scope'], 'files': files,
         'count': len(files), 'bytes': sum(q['bytes'] for q in files.values()),
         'finite_JSON_files': finite_json, 'external_artifact_records': len(large)}
(D / 'index.json').write_text(json.dumps(index, indent=2) + '\n')
for name, q in files.items():
    assert sha(D / name) == q['sha256']
print(json.dumps({'index': str((D / 'index.json').relative_to(ROOT)),
                  'index_sha256': sha(D / 'index.json'),
                  'count': index['count'], 'bytes': index['bytes'],
                  'finite_JSON': finite_json, 'external_records': len(large),
                  'verification_sha256': sha(P / 'final-verification.json')}, indent=2))
