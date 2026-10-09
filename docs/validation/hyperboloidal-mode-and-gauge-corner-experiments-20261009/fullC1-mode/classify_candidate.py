"""Read-only native diagnostic content of a converged approximate N16 pair.

Only candidate0 is characterized, with native centered-amplitude controls.
This does not label a continuum characteristic branch or certify an eigenvalue.
"""
from pathlib import Path
import hashlib
import json
import struct
import subprocess
import time
import numpy as np
from scipy.sparse import load_npz

P = Path(__file__).resolve().parent
ROOT = P.parents[2]
B = ROOT/'build-layer-research/boundary/full-tensor-covariant-c1'
OLD = B/'full22-candidate'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(P/'candidate-vectors.npz') == '5c1fd006aeb183947b256f20c7becee4238b342ce64053947be2f4a15b9ba866'
assert sha(P/'candidate-metadata.json') == 'f1b137cf9e57b81f7e26b3ce5f6b63ce70c5e22dc36912b27c3e469316d3fc39'
assert sha(B/'server-spatialnorm') == 'b088aaa25f865a9a21754f3ee05cf1ed1f86c7717d5ff6d3cf4253a073eb3450'
assert sha(OLD/'spatialnorm-projected-J20.npz') == '1093d7c07a71019cd69e578d52ca0c47635dc190b32ccb371683fd162132241e'
meta = json.loads((OLD/'spatialnorm-cache0.0001-metadata.json').read_text())
N = meta['points']
coords = np.array(meta['xyz_omega_volume_ginv_chi'])
xyz, spacing = coords[:, :3], meta['spacing']
r = np.linalg.norm(xyz, axis=1)
weights = spacing**3*coords[:, 4]
G = np.zeros((N, 3, 3))
for f, (i, j) in enumerate(((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))):
    G[:, i, j] = G[:, j, i] = coords[:, 5]*coords[:, 6+f]
v = np.load(P/'candidate-vectors.npz')['candidate0']
lam = complex(*json.loads((P/'candidate-metadata.json').read_text())['candidates'][0]['lambda'])
J = load_npz(OLD/'spatialnorm-projected-J20.npz')
L = np.fromfile(OLD/'spatialnorm-cache0.0001-lift.bin', dtype='<f8').reshape(N, 22, 20)
full = np.einsum('pij,pj->pi', L, v.reshape(N, 20), optimize=False)
norm2 = np.sum(abs(full)**2, axis=1)
groups = {'chi': [0], 'metric': [1, 2, 3, 4, 5, 6], 'P': [7],
          'A': [8, 9, 10, 11, 12, 13], 'Lambda': [14, 15, 16],
          'Theta': [17], 'alpha': [18], 'beta': [19, 20, 21]}
stderr = (P/'native-diagnostic.stderr').open('w')
command = [str(B/'server-spatialnorm'), '16', '2.2', '0.0001']
proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=stderr)
native_meta = json.loads(proc.stdout.readline())
assert native_meta['points'] == N
assert np.array_equal(np.array(native_meta['xyz_omega_volume_ginv_chi']), coords)


def apply(real_vector, mode, eps):
    proc.stdin.write(mode.encode()+struct.pack('d', eps)+np.asarray(real_vector, dtype='<f8').tobytes())
    proc.stdin.flush()
    count = N*(7 if mode == 'd' else 20)
    raw = bytearray()
    while len(raw) < count*8:
        q = proc.stdout.read(count*8-len(raw))
        if not q:
            raise RuntimeError('Pinned native diagnostic stopped unexpectedly')
        raw.extend(q)
    return np.frombuffer(raw, dtype='<f8').copy()


started = time.monotonic()
native = []
constraints = []
for eps in (1e-3, 1e-4, 3e-5, 1e-5, 3e-6):
    jv = apply(v.real, 'f', eps)+1j*apply(v.imag, 'f', eps)
    native.append({'amplitude_scale': eps,
                   'native_Jv_minus_cached_Jv_generator_units': float(np.linalg.norm(jv-J@v)),
                   'native_centered_RHS_minus_lambda_v_generator_units': float(np.linalg.norm(jv-lam*v)),
                   'native_centered_RHS_minus_lambda_v_relative_to_max1_lambda': float(np.linalg.norm(jv-lam*v)/max(1, abs(lam)))})
    if eps in (1e-4, 3e-5, 1e-5):
        q = (apply(v.real, 'd', eps)+1j*apply(v.imag, 'd', eps)).reshape(N, 7)
        constraints.append((eps, q))
proc.stdin.close()
proc.wait()
stderr.close()
assert proc.returncode == 0
q = constraints[-1][1]
parts = np.empty((N, 3))
parts[:, 0] = abs(q[:, 0])**2
for j, start in enumerate((1, 4), 1):
    parts[:, j] = np.einsum('pi,pij,pj->p', q[:, start:start+3].conj(), G,
                            q[:, start:start+3], optimize=False).real
assert np.min(parts) >= -1e-12
parts = np.maximum(parts, 0)
bins = [0., .2, .4, .6, .8, .9, .95, 1.]
local = []
for lower, upper in zip(bins[:-1], bins[1:]):
    inside = (r >= lower) & (r < upper)
    local.append({'r_lower': lower, 'r_upper': upper, 'cells': int(inside.sum()),
                  'weighted_component_squared_fraction': float(np.sum(weights[inside]*norm2[inside])/np.sum(weights*norm2)),
                  'H_M_Z_squared_fraction': [float(np.sum(parts[inside, j])/np.sum(parts[:, j])) for j in range(3)]})
# Actual active-neighbor differences: a descriptive roughness, not a Fourier
# wavenumber or continuum branch classification on the spherical masked grid.
indices = np.rint((xyz-xyz.min(axis=0))/spacing).astype(int)
lookup = {tuple(index): at for at, index in enumerate(indices)}
roughness = []
for axis in range(3):
    edges = []
    for at, index in enumerate(indices):
        other = index.copy()
        other[axis] += 1
        if tuple(other) in lookup:
            edges.append((at, lookup[tuple(other)]))
    pairs = np.array(edges)
    difference = full[pairs[:, 1]]-full[pairs[:, 0]]
    total = float(np.sum(abs(difference)**2))
    roughness.append({'axis': axis, 'active_neighbor_pairs': len(edges),
                      'difference_over_h_full_component_ratio': float(np.sqrt(total/np.sum(norm2))/spacing),
                      'difference_over_full_component_ratio': float(np.sqrt(total/np.sum(norm2)))})
history_path = ROOT/'build-layer-research/boundary/full-tensor-C1-long-window-20261009/spatialnorm-projected-krylov-m50-80-h0.1-t6.0.npz'
hist = np.load(history_path)
pair = np.column_stack([v.real, v.imag])
orth, _ = np.linalg.qr(pair)
projection = []
for seed, name in enumerate(hist['names']):
    rows = []
    for t in (0., 2., 4., 5., 6.):
        it = int(np.argmin(abs(hist['times']-t)))
        y = hist['values'][it, :, seed]
        coefficient = np.einsum('ij,i->j', orth, y, optimize=False)
        rows.append({'time': float(hist['times'][it]),
                     'orthogonal_pair_squared_fraction_raw_free20': float(np.sum(coefficient**2)/np.sum(y*y))})
    projection.append({'seed': str(name), 'samples': rows})
record = {'status': 'APPROXIMATE_CANDIDATE_DIAGNOSTICS_COMPLETED',
          'scope': 'Only a finite-grid approximate/pseudospectral vector is characterized; no eigenvalue error bound, continuum mode, or pure gauge/subsidiary/stencil branch label.',
          'native_command': command, 'native_exit': proc.returncode,
          'native_generator_amplitude_controls': native,
          'actual_constraint_amplitude_controls': [
              {'amplitude_scale': eps, 'difference_from_eps1e-5_absolute_l2': float(np.linalg.norm(c-q)),
               'difference_from_eps1e-5_relative_l2': float(np.linalg.norm(c-q)/np.linalg.norm(q))}
              for eps, c in constraints],
          'native_H_M_Z_rms_for_unit_free20_candidate': np.sqrt(np.mean(parts, axis=0)).tolist(),
          'Theta_rms_for_unit_free20_candidate': float(np.sqrt(np.mean(abs(full[:, 17])**2))),
          'raw22_component_squared_fractions': {key: float(np.sum(weights[:, None]*abs(full[:, ids])**2)/np.sum(weights*norm2)) for key, ids in groups.items()},
          'radial_localization': local, 'peak_component_radius': float(r[np.argmax(norm2)]),
          'peak_H_M_Z_radius': [float(r[np.argmax(parts[:, j])]) for j in range(3)],
          'active_neighbor_roughness': roughness,
          'pi_over_h_reference': float(np.pi/spacing),
          'late_state_orthogonal_pair_projection': projection,
          'norm_caveat': 'Component fractions and pair overlaps use raw component units; no invariant energy or biorthogonal dynamical decomposition.',
          'source_sha256': sha(Path(__file__)), 'seconds_after_server_start': time.monotonic()-started,
          'input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in
              (P/'candidate-vectors.npz', P/'candidate-metadata.json', P/'input-pins.json', B/'server-spatialnorm',
               B/'tangent_server.cpp', OLD/'spatialnorm-cache0.0001-lift.bin',
               OLD/'spatialnorm-cache0.0001-metadata.json', OLD/'spatialnorm-projected-J20.npz', history_path)},
          'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()}
(P/'candidate-diagnostics.json').write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
print('DONE candidate0 native diagnostics', record['native_H_M_Z_rms_for_unit_free20_candidate'])
