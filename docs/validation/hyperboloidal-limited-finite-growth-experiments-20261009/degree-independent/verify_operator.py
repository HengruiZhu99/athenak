"""Independent algebraic readback of pinned finite-rb Galerkin matrix outputs.

Never assembles an operator, calls a point kernel, computes generator spectra,
or propagates a state. Energy eigenvalues and trace singular values are used
only for positivity/conditioning and algebraic rank respectively.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import warnings

import numpy as np

REQUIRED = (
    'E', 'Kweak', 'Kstrong', 'Gvolume', 'Fboundary', 'SATload',
    'Jbulk', 'Jsat', 'B', 'Hb', 'Knb', 'Pplus', 'nodal_from_modal',
    'E_nodal', 'manufactured_X', 'manufactured_load',
    'manufactured_boundary_load',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def record(path):
    p = Path(path).resolve()
    return {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size}


def fro(a):
    return float(np.sqrt(np.sum(np.abs(a) ** 2)))


def product(a, b):
    return np.einsum('ik,kj->ij', a, b, optimize=False)


def residual(a, b):
    d = a - b
    return {'scaled_frobenius': fro(d) / max(1., fro(a), fro(b)),
            'max_absolute': float(np.max(np.abs(d))) if d.size else 0.}


def columns(a, n):
    if a.ndim == 1:
        a = a[:, None]
    assert a.ndim == 2 and a.shape[0] == n, a.shape
    return a


def boundary_action(a, b):
    """Apply common/angle blocks or a full boundary map to stacked columns."""
    count = b.shape[0] // 20
    assert b.shape[0] == 20 * count
    if a.ndim == 2 and a.shape[1] == b.shape[0] and a.shape[1] != 20:
        return product(a, b)
    if a.ndim == 2:
        assert a.shape[1] == 20, a.shape
        return np.einsum('ab,qbn->qan', a, b.reshape(count, 20, -1),
                         optimize=False).reshape(count * a.shape[0], -1)
    assert a.ndim == 3 and a.shape[0] == count and a.shape[2] == 20, a.shape
    return np.einsum('qab,qbn->qan', a, b.reshape(count, 20, -1),
                     optimize=False).reshape(count * a.shape[1], -1)


def block_checks(a):
    """Return 20x20 blocks; full boundary matrices require separate treatment."""
    if a.ndim == 2 and a.shape == (20, 20):
        return [a]
    if a.ndim == 3 and a.shape[1:] == (20, 20):
        return list(a)
    assert a.ndim == 2 and a.shape[0] == a.shape[1]
    return [a]


def rank_record(a, rtol, atol):
    s = np.linalg.svd(a, compute_uv=False)
    threshold = max(atol, rtol * (float(s[0]) if s.size else 0.))
    return {'rank': int(np.count_nonzero(s > threshold)),
            'threshold': threshold, 'singular_values': s.tolist(),
            'matrix_shape': list(a.shape)}


def cholesky_solve(lower, rhs):
    intermediate = np.linalg.solve(lower, rhs)
    return np.linalg.solve(lower.T, intermediate)


def positive_record(a):
    skew = residual(a, a.T)
    # Never average an asymmetric input to hide its skew error.
    lower = np.linalg.cholesky(a)
    e = np.linalg.eigvalsh(a, UPLO='L')
    assert float(e[0]) > 0
    return {'symmetry': skew, 'minimum_energy_eigenvalue': float(e[0]),
            'maximum_energy_eigenvalue': float(e[-1]),
            'spectral_condition': float(e[-1] / e[0]),
            'minimum_Cholesky_diagonal': float(np.min(np.diag(lower))),
            'Cholesky_reconstruction': residual(product(lower, lower.T), a)}, lower


def verify(args):
    warnings.simplefilter('error')
    np.seterr(all='raise')
    assert sha(args.npz) == args.expect_sha256, 'NPZ output pin mismatch'
    assert sha(args.metadata) == args.expect_metadata_sha256, 'Metadata pin mismatch'
    meta = json.loads(Path(args.metadata).read_text())
    assert meta['matrix_sha256'] == args.expect_sha256
    assert meta['J'] in (0, 1, 2)
    assert meta['N'] >= 1 and 0 < meta['rb'] < 1
    assert meta['degree_order'] == 'channel-major modal degree'
    assert meta['nodal_map'] == 'X_nodal=T X_modal'
    assert meta['boundary_measure'] == 'B includes rb*sqrt(w_angle)'
    assert meta['k_in'] > 0 and meta['k_out'] < 0
    assert meta['expected_incoming_rank'] == {0: 4, 1: 8, 2: 10}[meta['J']]
    pin_rows = meta['input_pins']
    assert pin_rows, 'Exact input/source/build pins are required'
    checked_pins = []
    for row in pin_rows:
        path = Path(row['path'])
        if not path.is_absolute():
            path = Path(args.metadata).resolve().parent / path
        assert sha(path) == row['sha256'], str(path)
        if 'bytes' in row:
            assert path.stat().st_size == row['bytes']
        checked_pins.append(record(path))
    with np.load(args.npz, allow_pickle=False) as saved:
        assert set(REQUIRED) <= set(saved.files)
        arrays = {k: saved[k] for k in saved.files}
    for name, a in arrays.items():
        assert a.dtype == np.float64 and np.isfinite(a).all(), name
    n = meta['N'] * {0: 8, 1: 16, 2: 20}[meta['J']]
    for name in ('E', 'Kweak', 'Kstrong', 'Gvolume', 'Fboundary',
                 'SATload', 'Jbulk', 'Jsat', 'nodal_from_modal', 'E_nodal'):
        assert arrays[name].shape == (n, n), (name, arrays[name].shape, n)
    E, Kw, Ks, G, F, S = [arrays[k] for k in
                           ('E', 'Kweak', 'Kstrong', 'Gvolume', 'Fboundary', 'SATload')]
    B, H, Kn, P = [arrays[k] for k in ('B', 'Hb', 'Knb', 'Pplus')]
    assert B.ndim == 2 and B.shape[1] == n and B.shape[0] % 20 == 0
    mass, chol = positive_record(E)
    nodal_mass, _ = positive_record(arrays['E_nodal'])
    hb_records = [positive_record(h)[0] for h in block_checks(H)]
    kin = float(meta['k_in'])
    kout = float(meta['k_out'])
    alpha, beta = (kin - kout) / 2, (kin + kout) / 2
    normal_checks = []
    hblocks, kblocks, pblocks = [block_checks(a) for a in (H, Kn, P)]
    count = max(len(hblocks), len(kblocks), len(pblocks))
    assert all(len(blocks) in (1, count) for blocks in (hblocks, kblocks, pblocks))
    for i in range(count):
        h, k, p = [blocks[i if len(blocks) > 1 else 0]
                   for blocks in (hblocks, kblocks, pblocks)]
        assert h.shape == k.shape == p.shape
        eye = np.eye(k.shape[0])
        A = (k - beta * eye) / alpha
        normal_checks.append({
            'A_involution': residual(product(A, A), eye),
            'P_definition': residual(p, (eye + A) / 2),
            'P_idempotence': residual(product(p, p), p),
            'H_A_symmetry': residual(product(h, A), product(h, A).T),
            'H_P_symmetry': residual(product(h, p), product(h, p).T),
            'H_D_definition': residual(h, eye + product(A.T, A)),
        })
    boundary_F = product(B.T, boundary_action(H, boundary_action(Kn, B)))
    incoming = boundary_action(P, B)
    adjoint_load = -kin * product(B.T, boundary_action(H, incoming))
    # With energy 1/2 X^T E X, its symmetric principal boundary production
    # after SAT equals the sum of negative incoming and outgoing forms.
    outgoing = B - incoming
    negative_flux = (-kin * product(incoming.T, boundary_action(H, incoming))
                     + kout * product(outgoing.T, boundary_action(H, outgoing)))
    T = arrays['nodal_from_modal']
    checks = {
        'weak_strong': residual(Kw, Ks),
        'G_symmetry': residual(G, G.T),
        'F_symmetry': residual(F, F.T),
        'bulk_energy_identity': residual(Kw + Kw.T, F + G),
        'strong_energy_identity': residual(Ks + Ks.T, F + G),
        'boundary_flux': residual(F, boundary_F),
        'adjoint_SAT_load': residual(S, adjoint_load),
        'principal_flux_after_SAT': residual(F + S + S.T, negative_flux),
        'modal_congruence': residual(E, product(T.T, product(arrays['E_nodal'], T))),
        'bulk_solve_backward': residual(product(E, arrays['Jbulk']), Kw),
        'SAT_solve_backward': residual(product(E, arrays['Jsat']), S),
    }
    independent_bulk = cholesky_solve(chol, Kw)
    independent_sat = cholesky_solve(chol, S)
    checks['independent_bulk_solve_backward'] = residual(product(E, independent_bulk), Kw)
    checks['independent_SAT_solve_backward'] = residual(product(E, independent_sat), S)
    forward_differences = {
        'bulk': residual(independent_bulk, arrays['Jbulk']),
        'SAT': residual(independent_sat, arrays['Jsat']),
        'scope': 'Reported, not a forward-error certificate from backward stability alone.',
    }
    X = columns(arrays['manufactured_X'], n)
    load = columns(arrays['manufactured_load'], n)
    boundary_load = columns(arrays['manufactured_boundary_load'], n)
    assert X.shape == load.shape == boundary_load.shape
    rhs = product(Kw + S, X) + load + boundary_load
    exact_rate = cholesky_solve(chol, rhs)
    energy_error = fro(product(chol.T, exact_rate - X)) / max(
        1., fro(product(chol.T, X)))
    manufactured = {
        'cases': X.shape[1],
        'forced_solve_backward': residual(product(E, exact_rate), rhs),
        'exact_Xdot_E_norm_error': energy_error,
        'exact_Xdot_coefficient_error': residual(exact_rate, X),
        'direct_forcing_load_consistency': residual(load, product(E - Kw, X)),
        'exact_incoming_data_load': residual(boundary_load, -product(S, X)),
        'scope': 'Algebraic readback only; independent pointwise forcing construction must also be source-reviewed.',
    }
    rtol = float(meta.get('rank_rtol', 1e-10))
    atol = float(meta.get('rank_atol', 1e-11))
    rank = rank_record(incoming, rtol, atol)
    rank_checks = {'incoming': rank['rank'] == meta['expected_incoming_rank']}
    sectors = {}
    expected = {0: (2, 2, 0), 1: (4, 4, 0), 2: (4, 4, 2)}[meta['J']]
    sector_maps = []
    for sector, wanted in zip(('constraint', 'gauge', 'TT'), expected):
        key = sector + '_left'
        if key in arrays:
            action = boundary_action(arrays[key], B)
            sectors[sector] = rank_record(action, rtol, atol)
            rank_checks[sector] = sectors[sector]['rank'] == wanted
            sector_maps.append(action)
    if len(sector_maps) == 3:
        sectors['combined'] = rank_record(np.vstack(sector_maps), rtol, atol)
        rank_checks['combined'] = sectors['combined']['rank'] == rank['rank']
    transform_rank = rank_record(T, rtol, atol)
    rank_checks['nodal_modal_invertible'] = transform_rank['rank'] == n
    if 'incoming_left' in arrays:
        left = boundary_action(arrays['incoming_left'], B)
        sectors['incoming_left'] = rank_record(left, rtol, atol)
        rank_checks['incoming_left'] = sectors['incoming_left']['rank'] == rank['rank']
    thresholds = {'mass_solve_SAT_forcing': 2e-9, 'weak_strong_volume': 2e-8,
                  'point_normal_adapter': 5e-11, 'modal_condition': 1e12}
    assert meta['thresholds'] == thresholds, 'Predeclared thresholds changed'
    admitted = {}
    broad = {'weak_strong', 'bulk_energy_identity', 'strong_energy_identity'}
    for name, row in checks.items():
        admitted[name] = row['scaled_frobenius'] <= thresholds[
            'weak_strong_volume' if name in broad else 'mass_solve_SAT_forcing']
    admitted['mass_symmetry'] = mass['symmetry']['scaled_frobenius'] <= 2e-9
    admitted['nodal_mass_symmetry'] = nodal_mass['symmetry']['scaled_frobenius'] <= 2e-9
    admitted['modal_condition'] = mass['spectral_condition'] <= 1e12
    admitted['Cholesky_reconstruction'] = mass['Cholesky_reconstruction']['scaled_frobenius'] <= 2e-9
    admitted['boundary_H_checks'] = all(
        h['symmetry']['scaled_frobenius'] <= 5e-11 for h in hb_records)
    admitted['normal_principal_checks'] = all(
        v['scaled_frobenius'] <= 5e-11 for row in normal_checks for v in row.values())
    admitted['manufactured_E_norm'] = energy_error <= 2e-9
    admitted['manufactured_algebra'] = all(
        manufactured[k]['scaled_frobenius'] <= 2e-9 for k in (
            'forced_solve_backward', 'direct_forcing_load_consistency', 'exact_incoming_data_load'))
    admitted.update({'rank_' + k: v for k, v in rank_checks.items()})
    return {
        'status': 'PASS' if all(admitted.values()) else 'FAIL_preserved',
        'method': 'Independent saved-matrix algebra/Cholesky/energy/trace readback only.',
        'scientific_operator_assembled': False, 'generator_eigenvalues_computed': False,
        'evolution_or_propagation': False, 'source': record(__file__),
        'outputs': {'npz': record(args.npz), 'metadata': record(args.metadata)},
        'input_pins': checked_pins, 'parameters': meta,
        'array_shapes': {k: list(v.shape) for k, v in arrays.items()},
        'versions': {'python': sys.version, 'numpy': np.__version__},
        'mass': mass, 'nodal_mass': nodal_mass, 'boundary_H': hb_records,
        'normal_principal': normal_checks, 'checks': checks,
        'independent_solve_forward_comparison': forward_differences,
        'manufactured': manufactured, 'incoming_trace_rank': rank,
        'sector_ranks': sectors, 'sector_classification_complete': len(sector_maps) == 3,
        'nodal_modal_transform_rank': transform_rank,
        'thresholds': thresholds, 'passed_checks': admitted,
        'scope': 'Finite-dimensional output verification, not independent continuum-action binding, CPBC, a uniform energy bound, exact-scri closure or pulse/BH acceptance.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--npz', required=True)
    parser.add_argument('--expect-sha256', required=True)
    parser.add_argument('--metadata', required=True)
    parser.add_argument('--expect-metadata-sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = Path(args.output)
    assert not output.exists(), 'Preserve prior receipts; choose a new output path'
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        report = verify(args)
    except Exception as error:
        report = {'status': 'FAIL_preserved_exception', 'exception_type': type(error).__name__,
                  'message': str(error), 'source': record(__file__),
                  'npz': str(Path(args.npz).resolve()),
                  'metadata': str(Path(args.metadata).resolve()),
                  'scientific_operator_assembled': False,
                  'generator_eigenvalues_computed': False, 'evolution_or_propagation': False}
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'receipt': record(output)}, indent=2))
    return 0 if report['status'] == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
