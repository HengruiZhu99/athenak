"""Export independently reconstructed, generator-residual-checked pseudomodes."""
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
B = ROOT/'build-layer-research/boundary'
OLD = B/'full-tensor-propagator/full22-v2'
LONG = B/'full-tensor-C0-long-window-20261009'
matrix = OLD/'spatialnorm-projected-J20.npz'
states = LONG/'spatialnorm-projected-krylov-m50-80-h0.1-t6.0.npz'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(matrix) == '767b7c998e27db4d598e260f80e2181afe31d427c3f58c9a7bb60e30c35dede6'
assert sha(states) == 'b02bf12087248a96d1b217d164e6d960cfc506791892b1058d7c2df018ed82cf'
J = load_npz(matrix)
d = np.load(states)
values, times = d['values'], d['times']
choices = [('late4-combined-s1', 32, 'rayleigh-ritz'),
           ('late4-combined-s1', 32, 'projected-DMD'),
           ('late2-combined-s2', 80, 'projected-DMD'),
           ('late4-combined-raw-s2', 24, 'rayleigh-ritz')]
out, rows = {}, []
start = time.monotonic()
for at, (name, rank, method) in enumerate(choices):
    report = json.loads((P/(name+'.json')).read_text())
    lower, upper = report['window']
    ids = np.flatnonzero((times >= lower-1e-12) & (times <= upper+1e-12))[::report['stride']]
    X = np.concatenate([values[ids[:-1], :, s].T for s in (0, 1)], axis=1)
    Y = np.concatenate([values[ids[1:], :, s].T for s in (0, 1)], axis=1)
    if report['pair_column_normalization'] != 'none':
        norm = np.linalg.norm(X, axis=0)
        X, Y = X/norm, Y/norm
    u, s, vh = svd(X, full_matrices=False, lapack_driver='gesdd')
    U = u[:, :rank]
    small = U.T@(J@U) if method == 'rayleigh-ritz' else U.T@Y@vh[:rank].T/s[:rank][None, :]
    lam, vec = np.linalg.eig(small)
    if method == 'projected-DMD':
        lam = np.log(lam.astype(complex))/report['sample_dt']
    k = int(np.argmin(np.abs(lam-(1.9163387+6.790995j))))
    v = U@vec[:, k]
    v /= np.linalg.norm(v)
    v *= np.exp(-1j*np.angle(v[np.argmax(abs(v))]))
    residual = float(np.linalg.norm(J@v-lam[k]*v))
    fraction = float(s[rank-1]/s[0])
    assert residual < 1e-6 and fraction > 1e-10
    saved = [r for r in report['results'] if r['rank'] == rank and r['method'] == method][0]
    j = int(np.argmin(np.linalg.norm(np.array(saved['eigenvalues'])-[lam[k].real, lam[k].imag], axis=1)))
    assert abs(saved['actual_J_residual_generator_units'][j]-residual) < 1e-9
    key = 'candidate'+str(at)
    out[key] = v
    rows.append({'key': key, 'case': name, 'rank': rank, 'method': method,
                 'lambda': [float(lam[k].real), float(lam[k].imag)],
                 'actual_J_residual_generator_units': residual,
                 'actual_J_residual_relative_to_max1_lambda': residual/max(1, abs(lam[k])),
                 'singular_value_fraction_at_rank': fraction,
                 'case_report_sha256': sha(P/(name+'.json'))})
np.savez_compressed(P/'candidate-vectors.npz', **out)
principal = out['candidate0']
for row in rows:
    v = out[row['key']]
    overlap = np.vdot(principal, v)
    row['phase_aligned_distance_to_candidate0'] = float(np.linalg.norm(v*np.exp(-1j*np.angle(overlap))-principal))
meta = {'status': 'CONVERGED_APPROXIMATE_CACHED_GENERATOR_PAIR_NOT_CERTIFIED_EIGENVALUE',
        'scope': 'Finite-grid N16 span2.2 C0norm projected-continuous generator pseudomodes from saved histories, no globally-rightmost or continuum claim',
        'coordinates': 'Complex free20 point-major in original active-cell k/j/i order; Euclidean unit norm; largest-magnitude component phased real positive; conjugates implicit',
        'native_field_ids': [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 14, 15, 16, 17, 18, 19, 20, 21],
        'generator_action': 'dq/dt=J@q; no ||J|| residual normalization',
        'no_eigenvalue_perturbation_bound': True, 'conditioning_caveat': 'Non-normality means small eigenvector residual is pseudospectral evidence, not a certified eigenvalue error bound.',
        'candidate_vectors_sha256': sha(P/'candidate-vectors.npz'),
        'matrix': str(matrix.relative_to(ROOT)), 'matrix_sha256': sha(matrix),
        'states': str(states.relative_to(ROOT)), 'states_sha256': sha(states),
        'original_metadata': str((OLD/'spatialnorm-cache0.0001-metadata.json').relative_to(ROOT)),
        'original_metadata_sha256': sha(OLD/'spatialnorm-cache0.0001-metadata.json'),
        'source_sha256': sha(Path(__file__)), 'seconds': time.monotonic()-start,
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'candidates': rows}
(P/'candidate-metadata.json').write_text(json.dumps(meta, indent=2, allow_nan=False)+'\n')
print('EXPORTED', sha(P/'candidate-vectors.npz'), sha(P/'candidate-metadata.json'))
