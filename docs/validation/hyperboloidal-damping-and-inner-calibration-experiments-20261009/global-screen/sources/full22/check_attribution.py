"""Compare actual sparse profile/C0 operators to the exact Theta-only change."""
from pathlib import Path
import hashlib, json
import numpy as np
from scipy.sparse import coo_matrix, load_npz

w = Path(__file__).resolve().parent
old = w.parents[1] / 'full-tensor-propagator/full22-v2'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
maximum = lambda A: float(abs(A.data).max()) if A.nnz else 0.
res = {'formula': 'Delta P_t=Delta Theta_t=-kinput*kappa2(Omega)*Theta/Omega; other native RHS fields unchanged',
       'semantics': 'Actual sampled matrix attribution only; no boundary/global stability theorem',
       'gauges': {}}
for g in ['production', 'spatialnorm']:
 meta = json.loads((w / f'{g}-cache0.0001-metadata.json').read_text())
 n = meta['points']
 omega = np.asarray(meta['xyz_omega_volume_ginv_chi'])[:, 3]
 k2 = .2 * (omega - 1)
 c = -10 * k2 / omega
 row = np.ravel(np.array([20*np.arange(n)+6, 20*np.arange(n)+15]).T)
 col = np.repeat(20*np.arange(n)+15, 2)
 value = np.repeat(c, 2)
 f = w / f'{g}-projected-J20.npz'
 f0 = old / f'{g}-projected-J20.npz'
 J, J0 = load_npz(f), load_npz(f0)
 E = coo_matrix((value, (row, col)), shape=J.shape).tocsr()
 delta = J-J0
 err = delta-E
 vectors = dict(np.load(w.parent / f'{g}-validation-vectors.npz'))
 pulse = vectors['gauge_pulse']
 outside = delta.copy().tolil()
 outside[row, col] = 0.
 outside = outside.tocsr(); outside.eliminate_zeros()
 record = {'profile_matrix_sha256': sha(f), 'C0_matrix_sha256': sha(f0),
           'expected_nonzero_entries': int(E.nnz), 'max_expected_Theta_coupling': float(c.max()),
           'max_actual_changed_entry': maximum(delta),
           'max_absolute_error_vs_exact_Theta_only_change': maximum(err),
           'max_absolute_unexpected_entry': maximum(outside),
           'initial_pure_gauge_action_change_l2': float(np.linalg.norm(delta@pulse)),
           'initial_pure_gauge_action_change_linf': float(abs(delta@pulse).max())}
 record['passed_matrix_attribution_1e-8'] = (record['max_absolute_error_vs_exact_Theta_only_change'] < 1e-8 and record['initial_pure_gauge_action_change_linf'] < 1e-8)
 res['gauges'][g] = record
 assert record['passed_matrix_attribution_1e-8'], record
(w / 'actual-matrix-profile-attribution.json').write_text(json.dumps(res, indent=2)+'\n')
print(json.dumps(res, indent=2))
