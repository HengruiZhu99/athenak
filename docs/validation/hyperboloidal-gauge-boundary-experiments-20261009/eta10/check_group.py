"""Local frozen Fourier group derivatives, using actual full20 jet Jacobians.

These are pointwise generators, not global PDE eigenmodes. The physical
coordinate group velocity is minus d Im(lambda)/dk for exp(lambda*t+i*k*x).
The derivative uses the independently checked second-order polynomial in k.
"""
from pathlib import Path
import hashlib
import json
import numpy as np

p = Path(__file__).resolve().parent
rows = {}
for name in ('fourier-matrices.json', 'fourier-mode-matrices.json'):
    for s in json.loads((p.parent/name).read_text()):
        if s['candidate'] == 0 and s['eps'] == 5e-7:
            rows[(s['wide'], s['r'], s['kappa'], 0, s['oblique'], s['k'])] = s
for s in json.loads((p/'shift-fourier.json').read_text()):
    if s['eps'] == 5e-7:
        rows[(s['wide'], s['r'], s['kappa'], s['eta'], s['oblique'], s['k'])] = s

report = {
    'scope': __doc__,
    'constraint_order': ['H_phys', 'M_cov_x', 'M_cov_y', 'M_cov_z',
                         'Z_cov_x', 'Z_cov_y', 'Z_cov_z', 'Theta_phys', 'null_residue'],
    'normalization': 'largest absolute independent stored-field component equals one',
    'cases': [],
}
max_fit = 0.
for wide, r in ((0, .95), (1, .85), (1, .9), (1, .95), (1, .98)):
    for kappa in (5, 10):
        for eta in (0, 10, 20):
            for oblique in (0, 1):
                tag = (wide, r, kappa, eta, oblique)
                z = rows[tag+(0,)]
                f = rows[tag+(64,)]
                b0 = np.asarray(z['B'])
                b2 = (np.asarray(f['B'])-b0)/(64.**2)
                c1 = np.asarray(f['C'])/64.
                for k in (32, 64, 128, 256):
                    s = rows[tag+(k,)]
                    m = np.asarray(s['B'])+1j*np.asarray(s['C'])
                    fit = b0+k*k*b2+1j*k*c1
                    err = float(np.max(abs(fit-m)/(1+abs(m))))
                    max_fit = max(max_fit, err)
                    assert err < 2e-6, (tag, k, err)
                    dmdk = 2*k*b2+1j*c1
                    lam, v = np.linalg.eig(m)
                    vi = np.linalg.inv(v)
                    speeds = np.array([s['beta_n']-s['light_speed'],
                                       s['beta_n'], s['beta_n']+s['light_speed']])
                    ix = int(np.argmax(lam.real))
                    # Also retain the least-damped fast outgoing root, even
                    # when eta changes which principal branch dominates.
                    phase = lam.imag/k
                    nearest = np.argmin(abs(phase[:, None]-speeds), axis=1)
                    fast = np.flatnonzero(nearest == 0)
                    fast_ix = int(fast[np.argmax(lam[fast].real)])
                    q = np.asarray(s['HB'])+1j*np.asarray(s['HC'])
                    for selection, i in (('max_real', ix), ('fast_outgoing', fast_ix)):
                        vector = v[:, i].copy()
                        derivative = np.einsum('i,ij,j->', vi[i, :], dmdk, vector)
                        # Independent centered spectral derivative with
                        # eigenvalue matching checks the left/right formula.
                        dk = 1e-3
                        roots = []
                        for kk in (k-dk, k+dk):
                            ee = np.linalg.eigvals(b0+kk*kk*b2+1j*kk*c1)
                            roots.append(ee[np.argmin(abs(ee-lam[i]))])
                        fd = (roots[1]-roots[0])/(2*dk)
                        centered_scaled_error = float(abs(fd-derivative)/(1+abs(derivative)))
                        group_verified = centered_scaled_error < 2e-4
                        vector /= np.max(abs(vector))
                        residue = np.einsum('ij,j->i', q, vector)
                        report['cases'].append({
                            'wide': wide, 'r': r, 'kappa': kappa, 'eta': eta,
                            'oblique': oblique, 'k': k, 'selection': selection,
                            'root_real': float(lam[i].real), 'root_imag': float(lam[i].imag),
                            'phase_generator_speed': float(phase[i]),
                            'group_generator_speed': float(derivative.imag) if group_verified else None,
                            'simple_branch_group_verified': group_verified,
                            'group_caveat': None if group_verified else 'Near-degenerate or branch-matching ambiguity: do not use a scalar group derivative.',
                            'physical_coordinate_group_speed': float(-derivative.imag) if group_verified else None,
                            'nearest_principal_branch': ['fast_outgoing', 'advective', 'slow_incoming'][nearest[i]],
                            'principal_phase_speeds': speeds.tolist(),
                            'group_derivative_centered_error': float(abs(fd-derivative)),
                            'constraint_abs': abs(residue).tolist(),
                            'mode_real': vector.real.tolist(), 'mode_imag': vector.imag.tolist(),
                        })
report['max_second_order_polynomial_fit_scaled_error'] = max_fit
report['unverified_group_count'] = sum(not c['simple_branch_group_verified'] for c in report['cases'])
report['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
(p/'group-report.json').write_text(json.dumps(report, indent=2)+'\n')
print('PASS local polynomial fit', max_fit, 'group verified', sum(c['simple_branch_group_verified'] for c in report['cases']), 'ambiguous', report['unverified_group_count'])
for c in report['cases']:
    if c['wide'] == 1 and c['r'] == .95 and c['kappa'] == 10 and not c['oblique'] and c['k'] == 256:
        print(c['eta'], c['selection'], 'Re/phase/group', c['root_real'],
              c['phase_generator_speed'], c['group_generator_speed'],
              c['nearest_principal_branch'], 'constraints', c['constraint_abs'])
