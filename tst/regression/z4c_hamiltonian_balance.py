#!/usr/bin/env python3
"""Stage-level tests of vacuum-background Hamiltonian defect correction.

Use a z4c_tov_ks binary. These are operator tests around the direct Schwarzschild
trumpet, not a long-time perturbation-stability claim. NumPy is required.
"""
import argparse
import json
import os
import re
from pathlib import Path
import subprocess

import numpy as np


BASE = '''<job>
basename = hamiltonian_balance
<mesh>
nghost = 4
nx1 = 16
x1min = -2
x1max = 2
ix1_bc = outflow
ox1_bc = outflow
nx2 = 16
x2min = -2
x2max = 2
ix2_bc = outflow
ox2_bc = outflow
nx3 = 16
x3min = -2
x3max = 2
ix3_bc = outflow
ox3_bc = outflow
<meshblock>
nx1 = 8
nx2 = 8
nx3 = 8
<mesh_refinement>
refinement = none
<time>
evolution = dynamic
integrator = rk3
cfl_number = 0.3
nlim = 1
tlim = 1
ndiag = 1
<mhd>
evolution = dynamic
eos = ideal
dyn_eos = ideal
dyn_error = reset_floor
reconstruct = wenoz
rsolver = hlle
gamma = 1.3333333333333333
dfloor = 1e-8
pfloor = 1e-20
tfloor = 1e-12
dthreshold = 1
gamma_max = 10
fofc = true
fofc_method = llf
zero_tmunu_feedback = {zero_matter}
<adm>
<coord>
general_rel = true
is_dynamical = true
a = 0
excise = false
<z4c>
boundary_rhs = characteristic_cpbc
characteristic_bc_source = zero_rate
characteristic_bc_max_energy_density = 1e-6
extrap_order = 4
use_analytic_background = true
evolve_gauge_residual = true
residual_gauge = background_adapted
residual_hamiltonian_balance = {balance}
diss = 0.5
damp_kappa1 = 0.1
damp_kappa2 = 0
lapse_harmonic = 0
lapse_oplog = 2
lapse_advect = 1
shift_Gamma = 1
shift_advect = 1
shift_eta = 2
debug_balance = true
debug_reduction_stride = 1
rhs_term_debug = true
rhs_term_debug_stride = 1
debug_snapshot_operations = pre_rhs_state,rhs_full_vs_bg,volume_rhs,post_recast
<problem>
pgen_name = z4c_tov_ks
pure_background = true
zero_tmunu = {zero_matter}
bh_background = schwarzschild_trumpet
bh_mass = 1
bh_spin = 0
use_direct_z4c_background = true
outer_sponge_enabled = false
excision_damp_rate = 0
excision_project_state = false
excision_freeze_radius = 0
excision_ramp_radius = 0
vacuum_gauge_pulse_amplitude = {lapse_amplitude}
vacuum_gauge_pulse_x1 = 0.75
vacuum_gauge_pulse_width = 0.5
outer_sponge_test_theta_pulse_amplitude = {theta_amplitude}
outer_sponge_test_theta_pulse_radius = 1.25
outer_sponge_test_theta_pulse_width = 0.5
'''


def snapshot(run, operation, stage=1, background=False):
    stem = run / f'z4c_snapshot_{operation}_rank0_cycle0_stage{stage}'
    meta = json.loads(stem.with_suffix('.json').read_text())
    assert meta['scalar_bytes'] == 8 and meta['byte_order'] == 'little'
    path = Path(str(stem) + ('.background.bin' if background else '.bin'))
    value = np.fromfile(path, dtype='<f8').reshape(meta['shape'])
    return meta, value


def active(meta, value):
    i,j,k = meta['active_start']; ni,nj,nk = meta['active_count']
    return value[:, :, k:k+nk, j:j+nj, i:i+ni]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--exe', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    runs = {}; results = {}
    cases = [('zero', True, 0., 0., True),
             ('lapse_off', False, 1e-8, 0., True),
             ('lapse_on', True, 1e-8, 0., True),
             ('theta_off', False, 0., 1e-5, True),
             ('theta_on', True, 0., 1e-5, True),
             ('matter_off', False, 0., 0., False),
             ('matter_on', True, 0., 0., False)]
    for name, balance, lapse, theta, zero_matter in cases:
        run = args.output.resolve() / name
        run.mkdir(parents=True, exist_ok=False)
        (run/'input.athinput').write_text(BASE.format(
            balance=str(balance).lower(), lapse_amplitude=lapse,
            theta_amplitude=theta, zero_matter=str(zero_matter).lower()))
        with (run/'run.log').open('w') as log:
            subprocess.run([str(args.exe.resolve()), '-i', 'input.athinput'], cwd=run,
                           stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, 'OMP_NUM_THREADS': '1'})
        meta, rhs = snapshot(run, 'volume_rhs')
        assert np.isfinite(active(meta, rhs)).all(), name
        runs[name] = (meta, active(meta,rhs).copy(), run)
        results[name] = {'initial_theta_rhs_max': float(np.max(abs(active(meta,rhs)[:,17])))}
        # The forensic decomposition must follow the selected evolution operator.
        # Compare signed local terms against the pre-KO volume snapshot, rather
        # than adding maxima from different cells. Log values have six digits.
        log_text = (run/'run.log').read_text()
        local_rows = [line for line in log_text.splitlines()
                      if line.startswith('Z4C_RHS_TERM_LOC ')]
        assert len(local_rows) == 4, (name, len(local_rows))
        max_decomposition_error = 0.
        for line in local_rows:
            row = dict(re.findall(r'(\w+)=([^\s]+)', line))
            block_index = next(n for n,b in enumerate(meta['blocks'])
                               if b['gid'] == int(row['gid']))
            block = meta['blocks'][block_index]
            ijk = [round((float(row[axis])-lo)/dx-0.5)
                   for axis,lo,dx in zip('xyz',block['xmin'],block['dx'])]
            terms = [float(row[key]) for key in
                     ['Theta_adv','Theta_Ht','Theta_damp','Theta_mat']]
            actual = active(meta,rhs)[block_index,17,ijk[2],ijk[1],ijk[0]]
            error = abs(sum(terms)-actual)
            assert error <= 2e-5*sum(abs(v) for v in terms)+1e-16, (name,row,error)
            max_decomposition_error = max(max_decomposition_error,float(error))
        results[name]['theta_local_decomposition_max_error'] = max_decomposition_error
        maximum_line = next(line for line in log_text.splitlines()
                            if line.startswith('Z4C_RHS_TERM_MAX '))
        maximum_terms = dict(re.findall(r'(\w+)=([^\s]+)', maximum_line))
        if name == 'lapse_on':
            assert float(maximum_terms['Theta_Ht']) == 0., maximum_terms
        print(name, 'ran', flush=True)
    # Zero stays represented identically across active states at every RK stage.
    run = runs['zero'][2]
    for stage in range(1,4):
        for operation in ['pre_rhs_state','post_recast']:
            meta, values = snapshot(run, operation, stage)
            values = active(meta,values)
            assert not values.view(np.uint64).any(), (operation,stage)
        meta, values = snapshot(run, 'volume_rhs', stage)
        assert not np.count_nonzero(active(meta,values)), stage
    results['zero']['all_stages_bitwise_zero_state'] = True
    # Pure lapse: prove all other full/background input fields (including ghosts)
    # match bitwise, then demand exactly zero initial volume Theta source.
    for name in ['lapse_off','lapse_on']:
        meta, full = snapshot(runs[name][2], 'rhs_full_vs_bg')
        _, bg = snapshot(runs[name][2], 'rhs_full_vs_bg', background=True)
        mask = np.arange(full.shape[1]) != 18
        assert np.array_equal(full[:,mask].view(np.uint64), bg[:,mask].view(np.uint64))
        assert np.max(abs(full[:,18]-bg[:,18])) > 1e-9
    assert results['lapse_off']['initial_theta_rhs_max'] > 1e-12
    assert results['lapse_on']['initial_theta_rhs_max'] == 0.
    # Physical response: Theta changes K=Khat+2Theta, hence the extrinsic-curvature
    # part of H. With unchanged lapse the full/background source is preserved,
    # up to alternate floating evaluation order; it must not be suppressed.
    old = runs['theta_off'][1]; new = runs['theta_on'][1]
    difference = float(np.max(abs(old[:,17]-new[:,17])))
    amplitude = float(np.max(abs(old[:,17])))
    assert amplitude > 1e-6
    assert difference <= 1e-12*amplitude + 1e-15, (difference,amplitude)
    other = np.arange(old.shape[1]) != 17
    assert np.array_equal(old[:,other],new[:,other])
    results['theta_on']['theta_response_difference'] = difference
    results['theta_on']['non_theta_rhs_bitwise_unchanged'] = True
    # Initial atmosphere is at rest with B=0, E=rho+p/(gamma-1). Geometry/lapse
    # residuals are zero, so the initial Theta source must be -8*pi*alpha*E.
    for name in ['matter_off','matter_on']:
        meta, full = snapshot(runs[name][2], 'rhs_full_vs_bg')
        alpha = active(meta,full)[:,18]
        expected = -8*np.pi*alpha*(1e-8+3e-20)
        actual = runs[name][1][:,17]
        error = float(np.max(abs(actual-expected)))
        assert error < 2e-12*np.max(abs(expected)), (name,error)
        results[name]['matter_source_max_error'] = error
    assert np.array_equal(runs['matter_off'][1],runs['matter_on'][1])
    results['matter_on']['all_rhs_bitwise_unchanged'] = True
    (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print('All Hamiltonian-balance operator checks PASS')


if __name__ == '__main__':
    main()
