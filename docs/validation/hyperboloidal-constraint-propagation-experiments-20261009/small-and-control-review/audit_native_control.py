"""Audit native timestep/projection controls; compare full-precision states at equal times."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FAMILY = ROOT/'build-layer-research/continuum/preferred/native-overlay/spatial-norm-family'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


reader = module('binary_reader', ROOT/'vis/python/bin_convert.py')
budget = module('constraint_budget', ROOT/'tst/hyperboloidal/analyze_native_constraints.py')
runner = module('validation_runner', ROOT/'tst/hyperboloidal/run_layer_validation.py')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def history(path):
    values = np.atleast_2d(np.loadtxt(path))
    assert values.shape[1] == 15 and np.isfinite(values).all()
    assert (np.diff(values[:, 0]) > 0).all() and (values[:, 1] > 0).all()
    return values


def matrices(u, mask, offset):
    value = np.zeros((int(mask.sum()), 3, 3))
    for index, (i, j) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]):
        value[:, i, j] = value[:, j, i] = u[offset+index][mask]
    return value


def spherical_mask(data):
    """Recompute the unit-sphere mask using the saved cell-center geometry."""
    raw = np.asarray(data['mb_data']['z4c_active'])[0]
    assert np.isin(raw, [0, 1]).all()
    coordinates = []
    for axis in range(3):
        lower, upper = data['mb_geometry'][0, 2*axis:2*axis+2]
        step = (upper-lower)/data[f'nx{axis+1}_mb']
        start = data['mb_index'][0, 2*axis]
        first = lower+(start+.5)*step
        coordinates.append(first+np.arange(raw.shape[2-axis])*step)
    z, y, x = np.meshgrid(*coordinates[::-1], indexing='ij')
    mask = raw.astype(bool)
    assert np.array_equal(mask, x*x+y*y+z*z < 1)
    return mask


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('kind', choices=['half', 'stage', 'stage-reference'])
    args = parser.parse_args()
    rst = module('double_restart_reader', HERE/'rst-reader-gate/restart_reader.py')
    names = {'half': 'N24-half-step-t2', 'stage': 'stage-projection-t2',
             'stage-reference': 'stage-projection-reference'}
    run = HERE/names[args.kind]
    reference = args.kind == 'stage-reference'
    target_time = .05 if reference else 2.
    base_run = FAMILY/('native-reference' if reference else 'native-long')
    result = json.loads((run/'results.json').read_text())
    baseline_result = json.loads((base_run/'results.json').read_text())
    case, = result['cases']
    base_case, = baseline_result['cases']
    assert case['exit_status'] == 0 and case['diagnostics']['time'] == target_time
    assert base_case['exit_status'] == 0
    directory, base = run/case['name'], base_run/base_case['name']
    assert sha(directory/'layer.athinput') == case['input_sha256']
    assert sha(base/'layer.athinput') == base_case['input_sha256']
    assert sha(run/'athena-validation') == result['sha256']
    assert sha(base_run/'athena-validation') == baseline_result['sha256']
    assert baseline_result['sha256'] == (
        'dd1d189210abd4e094da339dd73e3014357924b343c9f08f658eaf8cd4ae172d')
    build_path = FAMILY/'native-build-receipt.json'
    build = json.loads(build_path.read_text())
    assert all(sha(ROOT/k) == v for k, v in build['source_sha256'].items())
    assert all(sha(FAMILY/k) == v for k, v in build['overlay_sha256'].items())
    if args.kind == 'half':
        launch_path = HERE/'N24-half-step-launch.json'
    else:
        private_path = HERE/'stage-projection-build/build-receipt.json'
        private = json.loads(private_path.read_text())
        assert all(sha(ROOT/k) == v['sha256']
                   for k, v in private['reused_base_link_inputs_sha256'].items())
        assert sha(ROOT/private['executable']) == private['executable_sha256']
        assert result['sha256'] == private['executable_sha256']
        launch_name = ('stage-projection-reference-preflight.json' if reference
                       else 'stage-projection-launch.json')
        launch_path = HERE/launch_name
    launch = json.loads(launch_path.read_text())
    if not reference:
        assert result['sha256'] == launch['native_executable_sha256']
    hp, bp = directory/'hyp.z4c.user.hst', base/'hyp.z4c.user.hst'
    h, bh = history(hp), history(bp)
    assert h[0, 0] == bh[0, 0] == 0
    assert h[-1, 0] == bh[-1, 0] == target_time
    expected_ratio = .5 if args.kind == 'half' else 1.
    assert abs(h[0, 1]/bh[0, 1]-expected_ratio) < 1e-12
    settings = runner.input_parameters((directory/'layer.athinput').read_text())
    base_settings = runner.input_parameters((base/'layer.athinput').read_text())
    differences = {k: [base_settings.get(k), settings.get(k)]
                   for k in sorted(set(settings) | set(base_settings))
                   if settings.get(k) != base_settings.get(k)}
    assert set(differences) <= ({'z4c/hyperboloidal_pole_cfl'} if args.kind == 'half'
                                else set())
    paths = sorted((directory/'bin').glob('*.z4c.*.bin'))
    restarts = sorted((directory/'rst').glob('*.rst'))
    assert len(paths) == len(restarts) == (3 if reference else 81)
    snapshots, initial, first_mask, last, last_mask = [], None, None, None, None
    for path, checkpoint in zip(paths, restarts):
        b = reader.read_binary(str(path))
        d = rst.read_rst(checkpoint)
        assert d['time'] == b['time'] and d['cycle'] == b['cycle']
        assert tuple(b['var_names'][:25]) == rst.VARIABLES
        u = d['u']
        mask = spherical_mask(b)
        assert u.shape == (25,)+mask.shape
        for index, key in enumerate(b['var_names'][:25]):
            values = np.asarray(b['mb_data'][key])[0]
            assert np.array_equal(u[index][mask].astype(np.float32), values[mask])
        assert np.isfinite(u[:, mask]).all() and (u[[0, 18]][:, mask] > 0).all()
        g, a = matrices(u, mask, 1), matrices(u, mask, 8)
        physical = g/u[0][mask, None, None]
        eigen = np.linalg.eigvalsh(physical)
        assert eigen.min() > 0
        determinant = np.linalg.det(g)
        trace = np.einsum('nij,nji->n', np.linalg.inv(g), a)
        assert np.max(np.abs(determinant-1)) < 1e-12
        assert np.max(np.abs(trace)) < 1e-12
        if initial is None:
            initial = u.copy()
            first_mask = mask.copy()
        assert np.array_equal(mask, first_mask)
        snapshots.append({'time': d['time'], 'cycle': d['cycle'],
                          'restart_header_dt': d['dt'],
                          'bin_sha256': sha(path), 'restart_sha256': sha(checkpoint),
                          'active_cells': int(mask.sum()), 'all_active_fields_finite': True,
                          'alpha_min': float(u[18][mask].min()),
                          'chi_min': float(u[0][mask].min()),
                          'physical_metric_eigen_min': float(eigen.min()),
                          'physical_metric_eigen_max': float(eigen.max()),
                          'det_error_max': float(np.abs(determinant-1).max()),
                          'trace_error_max': float(np.abs(trace).max()),
                          'full_precision_drift_from_initial_max':
                          float(np.abs(u[:, mask]-initial[:, mask]).max())})
        last, last_mask = u, mask
    base_checkpoints = sorted((base/'rst').glob('*.rst'))
    base_initial = rst.read_rst(base_checkpoints[0])
    base_final = rst.read_rst(base_checkpoints[-1])
    assert base_initial['time'] == 0 and base_final['time'] == target_time
    base_bins = sorted((base/'bin').glob('*.z4c.*.bin'))
    base_mask = spherical_mask(reader.read_binary(str(base_bins[-1])))
    assert np.array_equal(last_mask, base_mask)
    assert np.array_equal(initial[:, last_mask], base_initial['u'][:, last_mask])
    field_compare = {}
    for index, key in enumerate(b['var_names'][:25]):
        delta = last[index][last_mask]-base_final['u'][index][last_mask]
        signal = base_final['u'][index][last_mask]-base_initial['u'][index][last_mask]
        field_compare[key] = {'rms_difference': float(np.sqrt(np.mean(delta*delta))),
                              'max_difference': float(np.abs(delta).max()),
                              'baseline_rms_change': float(np.sqrt(np.mean(signal*signal)))}
    times = [target_time] if reference else [.2, .5, 1., 1.5, 2.]
    comparison = {}
    for t in times:
        comparison[str(t)] = {}
        for index, key in [(2, 'H'), (3, 'M'), (4, 'Z'), (5, 'Theta')]:
            x = float(np.interp(t, h[:, 0], h[:, index]))
            y = float(np.interp(t, bh[:, 0], bh[:, index]))
            comparison[str(t)][key] = {'control': x, 'baseline': y, 'ratio': x/y}
    budgets = [budget.analyze(p, [0, .05, .45, .85, .9, .95, 1])
               for p in sorted((directory/'bin').glob('*.con.*.bin'))]
    log_dt = [float(v) for v in re.findall(
        r'\bdt=([0-9.eE+-]+)', (directory/'run.log').read_text())]
    assert log_dt and min(log_dt) > 0
    out = {'scope': ('Completed private native control; final full-precision fields at '
                     'exact equal physical time. Interior HST comparisons are linearly '
                     'interpolated between cadence outputs. Two timesteps alone do not '
                     'establish temporal order. No pulse/stability acceptance.'),
           'kind': args.kind, 'launch': launch, 'case': case,
           'result_sha256': sha(run/'results.json'), 'history_sha256': sha(hp),
           'base_history_sha256': sha(bp), 'source_overlay_hashes_verified': True,
           'reused_link_inputs_verified': args.kind != 'half',
           'comparison_input_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [
               base_checkpoints[0], base_checkpoints[-1], base_bins[-1],
               base_run/'results.json', base/'layer.athinput',
               base_run/'athena-validation', run/'athena-validation',
               HERE/'rst-reader-gate/abi.json']},
           'input_parameter_differences': differences,
           'initial_full_precision_active_arrays_identical': True,
           'first_dt_ratio': float(h[0, 1]/bh[0, 1]), 'first_dt': float(h[0, 1]),
           'history_dt_range': [float(h[:, 1].min()), float(h[:, 1].max())],
           'rounded_log_dt_range': [min(log_dt), max(log_dt)],
           'history_rows': h.tolist(), 'snapshots': snapshots,
           'matched_history_times': comparison, 'final_equal_time_fields': field_compare,
           'radial_constraint_budgets': budgets,
           'audit_source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [
               Path(__file__), HERE/'rst-reader-gate/restart_reader.py',
               ROOT/'vis/python/bin_convert.py',
               ROOT/'tst/hyperboloidal/analyze_native_constraints.py',
               ROOT/'tst/hyperboloidal/run_layer_validation.py']},
           'all_retained_files': {str(p.relative_to(ROOT)): {
               'sha256': sha(p), 'bytes': p.stat().st_size}
               for p in sorted(run.rglob('*')) if p.is_file()}}
    serialized = json.dumps(out, indent=2, allow_nan=False)+'\n'
    (HERE/(names[args.kind]+'-audit.json')).write_text(serialized)
    print('PASS', args.kind, len(snapshots), 'double-precision positive/SPD snapshots')
    print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
