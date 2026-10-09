"""Read-only SVD/Ritz/DMD search in frozen C0 N16 late-state subspaces.

No new propagation, sparse factorization, source mutation or ||J|| residual
normalization. Reduced candidates need actual generator residual convergence.
"""
from pathlib import Path
import argparse
import hashlib
import json
import subprocess
import time

import numpy as np
import scipy
from scipy.linalg import svd
from scipy.sparse import load_npz

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
B = ROOT/'build-layer-research/boundary'
OLD = B/'full-tensor-propagator/full22-v2'
LONG = B/'full-tensor-C0-long-window-20261009'
FROZEN = LONG/'immutable-C0-long-window-20261009'
MATRIX = OLD/'spatialnorm-projected-J20.npz'
STATES = LONG/'spatialnorm-projected-krylov-m50-80-h0.1-t6.0.npz'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--case', default='late2-combined-s2')
parser.add_argument('--max-rank', type=int, default=96)
args = parser.parse_args()
cases = {
    'all-combined-s2': (0., 6., [0, 1], 2, True),
    'late2-combined-s2': (2., 6., [0, 1], 2, True),
    'late4-combined-s1': (4., 6., [0, 1], 1, True),
    'late4-combined-s2': (4., 6., [0, 1], 2, True),
    'late4-combined-s4': (4., 6., [0, 1], 4, True),
    'late4-combined-raw-s2': (4., 6., [0, 1], 2, False),
    'late4-gauge-s2': (4., 6., [0], 2, True),
    'late4-shell-s2': (4., 6., [1], 2, True),
    'late5-combined-s1': (5., 6., [0, 1], 1, True),
}
assert args.case in cases
assert sha(FROZEN/'index.json') == 'd1efcbea11e730eec2e784ef7981e64f5e67d64e51bc963c6cdd7ed22834d896'
catalog = json.loads((FROZEN/'index.json').read_text())
for name, entry in catalog['files'].items():
    assert sha(FROZEN/name) == entry['sha256']
metadata = json.loads((FROZEN/'large-output-metadata.json').read_text())
for p, wanted in ((MATRIX, '767b7c998e27db4d598e260f80e2181afe31d427c3f58c9a7bb60e30c35dede6'),
                  (STATES, 'b02bf12087248a96d1b217d164e6d960cfc506791892b1058d7c2df018ed82cf')):
    assert sha(p) == metadata[str(p)]['sha256'] == wanted
    assert p.stat().st_size == metadata[str(p)]['bytes']
original = B/'full-tensor-global-final/manifest.json'
assert sha(original) == '4f62819e2337c0cb20571071fe10d0e80d8b43895a928ccd4418faec0bc590d2'
pins = {str(p.relative_to(ROOT)): sha(p) for p in
        (MATRIX, STATES, FROZEN/'index.json', FROZEN/'source-identity-verification.json',
         FROZEN/'large-output-metadata.json', original, OLD/'spatialnorm-cache0.0001-metadata.json',
         Path(__file__))}
started = time.monotonic()
J = load_npz(MATRIX)
data = np.load(STATES)
times, values = data['times'], data['values']
assert J.shape == (32800, 32800) and values.shape == (241, 32800, 2)
lo, hi, seeds, stride, scale = cases[args.case]
ids = np.flatnonzero((times >= lo-1e-12) & (times <= hi+1e-12))[::stride]
delta = float(times[ids[1]]-times[ids[0]])
assert np.max(np.abs(np.diff(times[ids])-delta)) < 1e-12
# Same positive column scale in X/Y preserves each sampled propagator pair.
X = np.concatenate([values[ids[:-1], :, s].T for s in seeds], axis=1)
Y = np.concatenate([values[ids[1:], :, s].T for s in seeds], axis=1)
norms = np.linalg.norm(X, axis=0)
assert np.min(norms) > 0
if scale:
    X = X/norms
    Y = Y/norms
u, singular, vh = svd(X, full_matrices=False, lapack_driver='gesdd', check_finite=True)
max_rank = min(args.max_rank, len(singular), int(np.sum(singular/singular[0] > 1e-14)))
U = u[:, :max_rank]
JU = J@U
RR = U.T@JU
ranks = sorted({min(r, max_rank) for r in (4, 8, 12, 16, 24, 32, 48, 64, 80, 96, 128) if r <= args.max_rank} | {max_rank})
results = []
saved = {}
for rank in ranks:
    Ur, JUr = U[:, :rank], JU[:, :rank]
    ritz, vr = np.linalg.eig(RR[:rank, :rank])
    dmd_matrix = Ur.T@Y@vh[:rank, :].T/singular[:rank][None, :]
    mu, vd = np.linalg.eig(dmd_matrix)
    dmd = np.log(mu.astype(complex))/delta
    for method, eig, vectors in (('rayleigh-ritz', ritz, vr), ('projected-DMD', dmd, vd)):
        norms_v = np.linalg.norm(vectors, axis=0)
        # Actual cached-generator action, not reduced reconstruction error.
        residual_vectors = JUr@vectors-(Ur@vectors)*eig[None, :]
        absolute = np.linalg.norm(residual_vectors, axis=0)/norms_v
        relative = absolute/np.maximum(1, np.abs(eig))
        order = np.argsort(-eig.real)
        # Keep all roots/residuals; no rightmost-global certification follows.
        row = {'rank': rank, 'method': method,
               'singular_value_fraction_at_rank': float(singular[rank-1]/singular[0]),
               'history_noise_floor_caveat': bool(singular[rank-1]/singular[0] < 1e-10),
               'eigenvalues': [[float(z.real), float(z.imag)] for z in eig],
               'actual_J_residual_generator_units': absolute.tolist(),
               'actual_J_residual_relative_to_max1_lambda': relative.tolist(),
               'rightmost_indices': order[:min(8, rank)].tolist()}
        positive = np.flatnonzero(eig.real > 0)
        best = int(positive[np.argmin(relative[positive])]) if len(positive) else int(np.argmin(relative))
        row['best_positive_relative_residual_index'] = best
        selected = set(order[:min(3, rank)].tolist()+[best])
        checks = []
        for k in sorted(selected):
            v = Ur@vectors[:, k]
            v /= np.linalg.norm(v)
            actual = J@v
            direct = float(np.linalg.norm(actual-eig[k]*v))
            assert abs(direct-absolute[k]) < 1e-9*(1+direct)
            checks.append({'index': k, 'actual_fresh_J_matvec_residual': direct})
            if rank == max_rank:
                saved[method+'-mode'+str(k)] = v
        row['fresh_J_matvec_verification'] = checks
        results.append(row)
        print(args.case, method, rank, 'sigma', singular[rank-1]/singular[0],
              'rightmost', eig[order[0]], absolute[order[0]],
              'best-positive', eig[best], absolute[best], relative[best], flush=True)
report = {'status': 'COMPLETED_MODE_SEARCH_NOT_ADMISSION', 'scope': 'Frozen C0 N16 discrete generator/history subspaces; no new evolution or continuum spectrum claim',
          'case': args.case, 'window': [float(times[ids[0]]), float(times[ids[-1]])],
          'stride': stride, 'sample_dt': delta, 'samples_per_seed': len(ids),
          'seed_names': [str(data['names'][s]) for s in seeds], 'pair_column_normalization': 'unit X columns, same scale for paired Y' if scale else 'none',
          'component_units': 'Raw free20 point-major Euclidean coordinates; not a physical or proved energy norm',
          'matrix_shape': list(J.shape), 'matrix_nnz': J.nnz,
          'singular_values': singular.tolist(), 'ranks': ranks,
          'max_rank_condition_number': float(singular[0]/singular[max_rank-1]),
          'DMD_log_branch': 'principal; sampled temporal aliases not resolved',
          'results': results, 'input_sha256': pins,
          'claim_gate': 'Classification deferred: require small generator-unit residuals, rank/window/sampling convergence and source/history controls; reduced spaces cannot certify globally rightmost spectrum.',
          'versions': {'numpy': np.__version__, 'scipy': scipy.__version__},
          'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
          'seconds': time.monotonic()-started}
(P/(args.case+'.json')).write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
np.savez_compressed(P/(args.case+'-vectors.npz'), **saved)
for name, digest in pins.items():
    assert sha(ROOT/name) == digest
print('DONE', args.case, report['seconds'], flush=True)
