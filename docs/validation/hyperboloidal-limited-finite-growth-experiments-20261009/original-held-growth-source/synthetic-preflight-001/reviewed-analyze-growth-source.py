#!/usr/bin/env python3
"""Held saved-matrix finite-radius linear growth control; no PDE assembly."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np
from scipy.linalg import expm, solve_triangular
from scipy.special import roots_jacobi, eval_jacobi

warnings.filterwarnings('error', category=RuntimeWarning)
np.seterr(all='raise', under='ignore')
HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def mm(a, b):
    return np.einsum('ik,kj->ij', a, b, optimize=False)


def error(a, b):
    delta = a-b
    return {'absolute': float(np.linalg.norm(delta)),
            'absolute_peak': float(np.max(np.abs(delta))),
            'scaled': float(np.linalg.norm(delta)/max(1., np.linalg.norm(a), np.linalg.norm(b)))}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def energy_transform(L, J):
    # A_E=L^T J L^{-T}; the right solve uses its transposed equation.
    return solve_triangular(L, mm(L.T, J).T, lower=True).T


def energy_form(L, F):
    a = solve_triangular(L, F, lower=True)
    return solve_triangular(L, a.T, lower=True).T


def make_seeds(data, admission):
    J, N, rb = admission['J'], admission['N'], admission['rb']
    channels = json.loads(Path(admission['basis_layout']).read_text())['channel_layouts'][str(J)]
    z, _ = roots_jacobi(N, 0, .5)
    rho = (z+1)*rb*rb/2
    nc = len(channels)
    nodal = np.zeros((nc*N, nc))
    rows = []
    for ch, item in enumerate(channels):
        name = item['name']
        if name in ('alpha', 'beta'):
            amplitude = .1 if name == 'alpha' else .02
            values = amplitude*(1-rho)**4*np.exp(-4*rho)
            kind = 'gauge'
        else:
            amplitude = .01
            values = amplitude*np.exp(-((rho-.49)/.16)**2)
            kind = 'constraint_shell'
        nodal[ch*N:(ch+1)*N, ch] = values
        rows.append({'channel': ch, 'field': name, 'L': item['L'],
                     'amplitude': amplitude, 'kind': kind,
                     'envelope': '(1-rho)^4 exp(-4rho)' if kind == 'gauge'
                     else 'exp(-((rho-.49)/.16)^2)',
                     'angular_scope': 'm=0 real phase for this rotationally invariant J block'})
    modal = np.linalg.solve(data['nodal_from_modal'], nodal)
    check = error(mm(data['nodal_from_modal'], modal), nodal)
    if check['scaled'] > 2e-9:
        raise RuntimeError('physical seed modal/nodal solve failed')
    held = np.linspace(0., rb*rb, 2*N+3)
    for ch, item in enumerate(channels):
        degrees = np.arange(N)
        norm = np.sqrt(2*(2*degrees+item['L']+1.5)/(rb*rb)**(item['L']+1.5))
        evaluation = np.array([norm[k]*eval_jacobi(k, 0, item['L']+.5, 2*held/(rb*rb)-1) for k in degrees]).T
        interpolated = mm(evaluation, modal[ch*N:(ch+1)*N, ch:ch+1])[:, 0]
        amplitude = rows[ch]['amplitude']
        analytic = amplitude*(1-held)**4*np.exp(-4*held) if rows[ch]['kind'] == 'gauge' else amplitude*np.exp(-((held-.49)/.16)**2)
        rows[ch]['held_envelope_interpolation_error'] = error(interpolated, analytic)
        rows[ch]['interpolation_scope'] = 'Scalar channel envelope at2N+3 fixed rho samples; no small-error admission or initial Einstein-constraint claim.'
    return modal, rows, rho, check


def rk3(A, initial, T, steps):
    h = T/steps
    state = initial.copy()
    for _ in range(steps):
        first = state+h*mm(A, state)
        second = .75*state+.25*(first+h*mm(A, first))
        state = state/3+(2/3)*(second+h*mm(A, second))
    return state


def analyze(matrix, authorization, output):
    adm = json.loads(authorization.read_text())
    if adm.get('finite_linear_growth_admitted') is not True:
        raise ValueError('finite matrix growth stage has not been admitted')
    if sha(matrix) != adm['matrix_sha256']:
        raise ValueError('matrix hash mismatch')
    required = ('source_configuration', 'mass_bulk_trace_forcing', 'paired_quadrature',
                'angular', 'incoming_sectors', 'continuum_rates', 'radial_constraint_defects_recorded')
    if not all(adm.get('gates', {}).get(k) is True for k in required):
        raise ValueError('a required source/matrix/rate gate is missing')
    if adm['J'] not in (0, 1, 2) or adm['N'] not in (8, 12, 16) or adm['rb'] not in (.98, .995):
        raise ValueError('undeclared finite-domain control')
    pins = [matrix, authorization, Path(__file__), HERE/'PLAN.md', HERE/'FINAL-ADDENDUM.md',
            Path(adm['basis_layout'])]
    pins += [Path(v['path']) for v in adm.get('review_inputs', [])]
    reviewed = {str(Path(v['path']).resolve()) for v in adm.get('review_inputs', [])}
    mandatory = {str(p.resolve()) for p in (Path(__file__), HERE/'PLAN.md', HERE/'FINAL-ADDENDUM.md', Path(adm['basis_layout']))}
    if not mandatory.issubset(reviewed):
        raise ValueError('source/plan/addendum/basis review pins must be explicitly admitted')
    for item in adm.get('review_inputs', []):
        if sha(item['path']) != item['sha256']:
            raise ValueError('independent review pin mismatch')
    before = {str(p.resolve()): sha(p) for p in pins}
    output.mkdir(parents=True, exist_ok=False)
    begin = time.monotonic()
    receipt = {'launch_HEAD': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               'command': [sys.executable, *sys.argv], 'inputs_before': before,
               'J': adm['J'], 'N': adm['N'], 'rb': adm['rb'],
               'scope': 'Saved finite energy-Galerkin matrix only; no continuum, nonlinear, exact-scri or native stability admission',
               'physical_constraint_readback': 'Separate post-propagation gate required; modal seeds/modes/Jv/states are retained here.',
               'error': None}
    dump(output/'launch.json', receipt)
    arrays = {}
    try:
        with np.load(matrix, allow_pickle=False) as saved:
            data = {k: saved[k] for k in saved.files}
        if not all(np.isfinite(a).all() for a in data.values()):
            raise ValueError('nonfinite saved matrix array')
        E, L = data['E'], data['energy_cholesky']
        J = data['Jbulk']+data['Jsat']
        A = energy_transform(L, J)
        normal = float(np.linalg.norm(A, 2))
        symmetric = (A+A.T)/2
        work = data['Fboundary']+data['Gvolume']+data['SATload']+data['SATload'].T
        independent = energy_form(L, work)/2
        checks = {'Cholesky': error(mm(L, L.T), E), 'SAT_skew': error(data['SATload'], data['SATload'].T),
                  'generator_transform': error(mm(A, L.T), mm(L.T, J)),
                  'symmetric_energy_identity': error(symmetric, independent)}
        receipt['checks'] = checks
        if any(v['scaled'] > 2e-9 for v in checks.values()):
            raise RuntimeError('finite energy-coordinate algebra gate failed')
        rates = np.linalg.eigvalsh(symmetric)
        eigenvalues, eigenvectors = np.linalg.eig(A)
        actions = mm(A, eigenvectors)
        residual = actions-eigenvectors*eigenvalues[None, :]
        absolute = np.linalg.norm(residual, axis=0)
        backward = absolute/((normal+np.abs(eigenvalues))*np.linalg.norm(eigenvectors, axis=0)) if normal else absolute
        order = np.lexsort((-eigenvalues.imag, -eigenvalues.real))
        selected = [int(i) for i in order if eigenvalues[i].imag >= 0][:4]
        modal_modes = solve_triangular(L.T, eigenvectors[:, selected], lower=False)
        actual_Jv = mm(J, modal_modes)
        arrays.update({'eigenvalues': eigenvalues, 'energy_eigenvectors': eigenvectors,
                       'eigenvector_residuals': residual, 'selected_indices': np.array(selected),
                       'selected_modal_modes': modal_modes, 'selected_actual_Jv': actual_Jv})
        receipt.update({'generator_energy_norm_2': normal,
                        'finite_matrix_logarithmic_norm': float(rates[-1]),
                        'finite_matrix_spectral_abscissa': float(np.max(eigenvalues.real)),
                        'max_eigenvector_absolute_residual': float(np.max(absolute)),
                        'max_eigenvector_relative_backward_residual': float(np.max(backward))})
        X, seeds, rho, seed_check = make_seeds(data, adm)
        Z = mm(L.T, X)
        initial_norms = np.linalg.norm(Z, axis=0)
        Bshape = data['B'].reshape(-1, 20, E.shape[0])
        traces = np.einsum('afd,dc->afc', Bshape, X, optimize=False)
        incoming = np.einsum('ef,afc->aec', data['Pplus'], traces, optimize=False)
        for k, seed in enumerate(seeds):
            seed['initial_energy_norm'] = float(initial_norms[k])
            seed['initial_incoming_surface_H_norm'] = float(np.sqrt(np.einsum('af,fg,ag->', incoming[:, :, k], data['Hb'], incoming[:, :, k], optimize=False)))
            seed['initial_sector_surface_diagnostic_norms'] = {}
            for key in ('constraint_left', 'gauge_left', 'TT_left'):
                if key in data:
                    values = np.einsum('ef,af->ae', data[key], traces[:, :, k], optimize=False)
                    seed['initial_sector_surface_diagnostic_norms'][key] = float(np.linalg.norm(values))
        arrays.update({'physical_seed_modal': X, 'seed_common_rho': rho,
                       'initial_incoming_trace': incoming})
        receipt['seeds'] = seeds
        propagation = []
        seed_states = []
        for t in (0., .25, .5, 1., 2., 4., 6.):
            receipt['active_stage'] = {'stage': 'matrix_exponential', 'time': t}
            dump(output/'active-stage.json', receipt['active_stage'])
            operator = np.eye(len(A)) if normal == 0 else expm(t*A)
            if not np.isfinite(operator).all() or np.linalg.norm(operator) > 1e100:
                dump(output/'failed-time.json', {'time': t, 'reason': 'nonfinite or Frobenius norm >1e100'})
                raise RuntimeError('finite exponential guard reached')
            receipt['active_stage'] = {'stage': 'half_time_exponential_and_SVD', 'time': t}
            dump(output/'active-stage.json', receipt['active_stage'])
            half = np.eye(len(A)) if normal == 0 else expm((t/2)*A)
            consistency = error(operator, mm(half, half))
            if consistency['scaled'] > 2e-7:
                raise RuntimeError('half-time exponential consistency check failed')
            state = mm(operator, Z)
            singular = np.linalg.svd(operator, compute_uv=False)
            propagation.append({'time': t, 'energy_operator_norm': float(singular[0]),
                                'half_time_product_check': consistency,
                                'seed_energy_norms': np.linalg.norm(state, axis=0).tolist(),
                                'seed_energy_amplifications': (np.linalg.norm(state, axis=0)/initial_norms).tolist()})
            seed_states.append(solve_triangular(L.T, state, lower=False))
            arrays['seed_states_modal'] = np.array(seed_states)
            arrays['propagation_times'] = np.array([v['time'] for v in propagation])
            dump(output/'propagation-partial.json', propagation)
        Tcheck = min(.1, 4/normal) if normal else .1
        first_steps = max(1, math.ceil(Tcheck*normal/.01))
        receipt['active_stage'] = {'stage': 'short_time_RK3', 'time': Tcheck}
        dump(output/'active-stage.json', receipt['active_stage'])
        exact = mm(expm(Tcheck*A), Z) if normal else Z.copy()
        rk_states = [rk3(A, Z, Tcheck, first_steps*factor) for factor in (1, 2, 4)] if normal else [Z.copy() for _ in range(3)]
        rk_checks = [error(v, exact) for v in rk_states]
        increments = [error(a, b) for a, b in zip(rk_states, rk_states[1:])]
        if rk_checks[-1]['scaled'] > 2e-7:
            dump(output/'failed-RK3.json', {'checks': rk_checks, 'increments': increments})
            raise RuntimeError('short-time RK3 final error unresolved')
        arrays.update({'eigenvalues': eigenvalues, 'energy_eigenvectors': eigenvectors,
                  'eigenvector_residuals': residual, 'selected_indices': np.array(selected),
                  'selected_modal_modes': modal_modes, 'selected_actual_Jv': actual_Jv,
                  'physical_seed_modal': X, 'seed_states_modal': np.array(seed_states),
                  'seed_common_rho': rho, 'initial_incoming_trace': incoming})
        np.savez_compressed(output/'growth-payload.npz', **arrays)
        receipt.update({'checks': checks, 'seed_modal_nodal_solve': seed_check,
                        'generator_energy_norm_2': normal, 'finite_matrix_logarithmic_norm': float(rates[-1]),
                        'finite_matrix_spectral_abscissa': float(np.max(eigenvalues.real)),
                        'spectrum': [[float(v.real), float(v.imag)] for v in eigenvalues],
                        'max_eigenvector_absolute_residual': float(np.max(absolute)),
                        'max_eigenvector_relative_backward_residual': float(np.max(backward)),
                        'eigenvector_condition_2': float(np.linalg.cond(eigenvectors)) if np.isfinite(np.linalg.cond(eigenvectors)) else None,
                        'eigenvalue_limit': 'Backward residuals do not bound forward errors of nonnormal eigenvalues or identify continuum eigenmodes.',
                        'selected_candidate_indices': selected, 'seeds': seeds, 'propagation': propagation,
                        'RK3': {'Tcheck': Tcheck, 'steps': [first_steps*f for f in (1, 2, 4)],
                                'checks_to_exponential': rk_checks, 'interlevel_changes': increments,
                                'scope': 'Short-time finite ODE SSPRK3 stages; no native Cartesian projection/ghost test'},
                        'payload_sha256': sha(output/'growth-payload.npz')})
    except Exception as exc:
        receipt['error'] = {'type': type(exc).__name__, 'message': str(exc)}
        if arrays:
            np.savez_compressed(output/'failed-partial-payload.npz', **arrays)
            receipt['failed_partial_payload_sha256'] = sha(output/'failed-partial-payload.npz')
    receipt['inputs_after'] = {str(p.resolve()): sha(p) for p in pins}
    receipt['input_pins_unchanged'] = receipt['inputs_before'] == receipt['inputs_after']
    receipt['seconds'] = time.monotonic()-begin
    receipt['passed_finite_ODE_numerical_checks'] = receipt['error'] is None and receipt['input_pins_unchanged']
    dump(output/'receipt.json', receipt)
    print(json.dumps({k: receipt[k] for k in ('J', 'N', 'rb', 'seconds', 'error', 'passed_finite_ODE_numerical_checks')}, indent=2))
    return 0 if receipt['passed_finite_ODE_numerical_checks'] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true')
    p.add_argument('--matrix', type=Path)
    p.add_argument('--admission', type=Path)
    p.add_argument('--output', type=Path)
    a = p.parse_args()
    if not a.execute:
        print('HELD source preparation; saved-matrix and independent-rate admission required.')
        return 0
    if any(v is None for v in (a.matrix, a.admission, a.output)):
        p.error('--execute requires --matrix, --admission and fresh --output')
    return analyze(a.matrix.resolve(), a.admission.resolve(), a.output.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
