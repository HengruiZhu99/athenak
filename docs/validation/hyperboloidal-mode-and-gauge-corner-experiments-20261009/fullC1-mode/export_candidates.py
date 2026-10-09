"""Independently rebuild selected C1 approximate vectors and compare frozen C0."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

import numpy as np
from scipy.linalg import svd
from scipy.sparse import load_npz

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
config = json.loads((P / 'input-pins.json').read_text())
choices = json.loads((P / 'choices.json').read_text())
for path, q in config['artifacts'].items():
    assert sha(ROOT / path) == q['sha256']
J = load_npz(ROOT / config['matrix'])
data = np.load(ROOT / config['states'])
values, times = data['values'], data['times']
oldroot = ROOT / 'build-layer-research/continuum/discrete-mode-identification'
oldmeta = json.loads((oldroot / 'candidate-metadata.json').read_text())
assert sha(oldroot / 'candidate-metadata.json') == 'e411042b90498bb4f08eadcdf9214095028c049df8a0c15b46b5dfd4d3f2d6e1'
assert sha(oldroot / 'candidate-vectors.npz') == oldmeta['candidate_vectors_sha256']
c0 = np.load(oldroot / 'candidate-vectors.npz')['candidate0']
c0history = np.load(ROOT / oldmeta['states'])
assert np.array_equal(values[0], c0history['values'][0])
assert np.array_equal(times, c0history['times'])
oldcoords = json.loads((ROOT / oldmeta['original_metadata']).read_text())
newcoords = json.loads((ROOT / config['metadata']).read_text())
assert oldcoords['xyz_omega_volume_ginv_chi'] == newcoords['xyz_omega_volume_ginv_chi']
out, rows = {}, []
started = time.monotonic()
for at, choice in enumerate(choices):
    name, rank, method = choice['case'], choice['rank'], choice['method']
    target = complex(*choice['target_lambda'])
    report = json.loads((P / (name + '.json')).read_text())
    lo, hi = report['window']
    ids = np.flatnonzero((times >= lo - 1e-12) & (times <= hi + 1e-12))[::report['stride']]
    X = np.concatenate([values[ids[:-1], :, i].T for i in (0, 1)], axis=1)
    Y = np.concatenate([values[ids[1:], :, i].T for i in (0, 1)], axis=1)
    if report['pair_column_normalization'] != 'none':
        norm = np.linalg.norm(X, axis=0)
        X, Y = X / norm, Y / norm
    u, s, vh = svd(X, full_matrices=False, lapack_driver='gesdd')
    U = u[:, :rank]
    if method == 'rayleigh-ritz':
        small = np.einsum('ki,kj->ij', U, J @ U, optimize=False)
    else:
        uy = np.einsum('ki,kj->ij', U, Y, optimize=False)
        small = np.einsum('ki,ji->kj', uy, vh[:rank], optimize=False) / s[:rank][None, :]
    assert np.isfinite(small).all()
    lam, eigvec = np.linalg.eig(small)
    if method == 'projected-DMD':
        lam = np.log(lam.astype(complex)) / report['sample_dt']
    k = int(np.argmin(abs(lam - target)))
    v = np.einsum('ij,j->i', U, eigvec[:, k], optimize=False)
    v /= np.linalg.norm(v)
    v *= np.exp(-1j * np.angle(v[np.argmax(abs(v))]))
    residual = float(np.linalg.norm(J @ v - lam[k] * v))
    fraction = float(s[rank - 1] / s[0])
    assert np.isfinite(v).all() and residual < 1e-6 and fraction > 1e-10
    saved = next(q for q in report['results'] if q['rank'] == rank and q['method'] == method)
    oldk = int(np.argmin(abs(np.array([complex(*z) for z in saved['eigenvalues']]) - lam[k])))
    assert abs(residual - saved['actual_J_residual_generator_units'][oldk]) < 1e-9
    key = 'candidate' + str(at)
    out[key] = v
    overlap = np.vdot(c0, v)
    rows.append({'key': key, 'case': name, 'rank': rank, 'method': method,
                 'lambda': [float(lam[k].real), float(lam[k].imag)],
                 'actual_J_residual_generator_units': residual,
                 'actual_J_residual_relative_to_max1_lambda': residual / max(1, abs(lam[k])),
                 'singular_value_fraction_at_rank': fraction,
                 'case_report_sha256': sha(P / (name + '.json')),
                 'lambda_minus_C0_candidate0': [float(lam[k].real - oldmeta['candidates'][0]['lambda'][0]),
                                               float(lam[k].imag - oldmeta['candidates'][0]['lambda'][1])],
                 'phase_aligned_component_distance_to_C0_candidate0': float(np.linalg.norm(v * np.exp(-1j * np.angle(overlap)) - c0))})
np.savez_compressed(P / 'candidate-vectors.npz', **out)
for row in rows:
    v = out[row['key']]
    overlap = np.vdot(out['candidate0'], v)
    row['phase_aligned_component_distance_to_C1_candidate0'] = float(np.linalg.norm(v * np.exp(-1j * np.angle(overlap)) - out['candidate0']))
meta = {'status': 'CONVERGED_APPROXIMATE_C1_GENERATOR_PAIR_NOT_CERTIFIED_EIGENVALUE',
        'scope': 'Finite-grid fullC1 norm N16/span2.2 projected continuous pseudomodes; no globally rightmost/continuum/stability claim',
        'coordinates': 'Complex free20 point-major original active-cell k/j/i order; Euclidean unit norm; maxabs entry phased real positive; conjugates implicit',
        'candidate_vectors_sha256': sha(P / 'candidate-vectors.npz'),
        'config_sha256': sha(P / 'input-pins.json'), 'choices_sha256': sha(P / 'choices.json'),
        'C0_vector_metadata_sha256': sha(oldroot / 'candidate-metadata.json'),
        'C0_vector_sha256': sha(oldroot / 'candidate-vectors.npz'),
        'initial_seed_arrays_bitwise_equal_C0_C1': True,
        'times_and_reference_coordinates_bitwise_equal_C0_C1': True,
        'no_eigenvalue_perturbation_bound': True,
        'conditioning_caveat': 'Small generator residual is pseudospectral evidence; no norm(J) normalization or branch classification',
        'candidates': rows, 'source_sha256': sha(Path(__file__)),
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'seconds': time.monotonic() - started}
(P / 'candidate-metadata.json').write_text(json.dumps(meta, indent=2, allow_nan=False) + '\n')
print('EXPORTED', sha(P / 'candidate-vectors.npz'), sha(P / 'candidate-metadata.json'))
