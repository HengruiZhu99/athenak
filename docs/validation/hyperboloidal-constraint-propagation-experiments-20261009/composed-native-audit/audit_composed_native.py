"""Audit private composed native runs against equal-time standard-operator runs.

All state comparisons use binary64 RST fields and matching physical box indices,
independent of allocated ghost width. Existing HST diagnostics are retained.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
FAMILY = ROOT/'build-layer-research/continuum/preferred/native-overlay/spatial-norm-family'
PRIVATE = HERE/'composed-derivative-build'
BASE_EXE = 'dd1d189210abd4e094da339dd73e3014357924b343c9f08f658eaf8cd4ae172d'
PRIVATE_EXE = 'e337453cae3553d05393478204f91131c620cb84c469e5c1a6fce3db073cfa16'


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


rst = module('composed_restart_reader', HERE/'rst-reader-gate/restart_reader.py')
binary = module('composed_binary_reader', ROOT/'vis/python/bin_convert.py')
runner = module('composed_validation_runner', ROOT/'tst/hyperboloidal/run_layer_validation.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def records(paths):
    return {str(p.relative_to(ROOT)): {'sha256': sha(p), 'bytes': p.stat().st_size}
            for p in sorted(set(paths))}


def verify_builds():
    base_path = FAMILY/'native-build-receipt.json'
    base = json.loads(base_path.read_text())
    assert all(sha(ROOT/key) == value for key, value in base['source_sha256'].items())
    assert all(sha(FAMILY/key) == value for key, value in base['overlay_sha256'].items())
    private_path = PRIVATE/'build-receipt.json'
    private = json.loads(private_path.read_text())
    assert private['base_build_receipt_sha256'] == sha(base_path)
    assert private['base_executable_sha256'] == BASE_EXE
    assert private['executable_sha256'] == PRIVATE_EXE
    assert private['link_exit_status'] == 0
    assert sha(ROOT/private['executable']) == PRIVATE_EXE
    assert sha(HERE/'build_composed_derivative.py') == private['script_sha256']
    assert all(sha(ROOT/key) == value
               for key, value in private['private_source_sha256'].items())
    assert all(sha(ROOT/key) == value
               for key, value in private['all_compiled_repository_dependencies_sha256'].items())
    assert all(sha(ROOT/key) == value['sha256']
               for key, value in private['reused_base_link_inputs_sha256'].items())
    assert len(private['compile_results']) == 6
    for row in private['compile_results']:
        assert row['exit_status'] == 0
        command = row['command']
        output = Path(command[command.index('-o')+1])
        dependency = Path(command[command.index('-MF')+1])
        assert sha(output) == row['private_object_sha256']
        assert sha(dependency) == row['dependency_file_sha256']
    return {'base_build_receipt_sha256': sha(base_path),
            'private_build_receipt_sha256': sha(private_path),
            'all_six_private_objects_and_dependency_files_verified': True,
            'all_private_sources_and_repository_dependencies_verified': True,
            'all_reused_base_link_inputs_verified': True,
            'baseline_executable_sha256': BASE_EXE,
            'private_executable_sha256': PRIVATE_EXE}


def choose_case(result, name):
    if name is None:
        value, = result['cases']
        return value
    value, = [item for item in result['cases'] if item['name'] == name]
    return value


def history(path):
    data = np.atleast_2d(np.loadtxt(path))
    assert data.shape[1] == 15 and np.isfinite(data).all()
    assert (np.diff(data[:, 0]) > 0).all() and (data[:, 1] > 0).all()
    return data


def tensor(fields, offset, mask):
    result = np.zeros((int(mask.sum()), 3, 3))
    for f, (a, b) in enumerate([(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)]):
        result[:, a, b] = result[:, b, a] = fields[offset+f][mask]
    return result


def physical_coordinates(checkpoint):
    """Reproduce native grid.first + stored_index*h, including its rounding."""
    block, region = checkpoint['mb_indcs'], checkpoint['mesh_size']
    ng = block['ng']
    coordinates = []
    for axis in range(1, 4):
        lower, step = region[f'x{axis}min'], region[f'dx{axis}']
        first = lower+(.5-ng)*step
        coordinates.append(first+(ng+np.arange(block[f'nx{axis}']))*step)
    return coordinates


def snapshot(checkpoint_path, bin_path):
    checkpoint = rst.read_rst(checkpoint_path)
    data = binary.read_binary(str(bin_path))
    assert checkpoint['time'] == data['time'] and checkpoint['cycle'] == data['cycle']
    assert tuple(data['var_names']) == rst.VARIABLES+('z4c_active',)
    view = rst.align_with_bin(checkpoint, data)[0]
    raw_mask = np.asarray(data['mb_data']['z4c_active'], dtype=np.float32)[0]
    assert np.isin(raw_mask, [0, 1]).all()
    mask = raw_mask.astype(bool)
    assert view.shape == (25,)+mask.shape and mask.any()
    for field, name in enumerate(rst.VARIABLES):
        rounded = view[field][mask].astype(np.float32)
        recorded = np.asarray(data['mb_data'][name], dtype=np.float32)[0][mask]
        assert np.array_equal(rounded.view(np.uint32), recorded.view(np.uint32))
    slices = []
    for axis in range(3):
        start = -int(data['mb_index'][0, 2*axis])
        n = checkpoint['mb_indcs'][f'nx{axis+1}']
        assert 0 <= start and start+n <= mask.shape[2-axis]
        slices.append(slice(start, start+n))
    box_mask = mask[tuple(slices[::-1])]
    u = checkpoint['active_data'][0]
    assert u.shape == (25,)+box_mask.shape
    coords = physical_coordinates(checkpoint)
    z, y, x = np.meshgrid(*coords[::-1], indexing='ij')
    assert np.array_equal(box_mask, x*x+y*y+z*z < 1)
    # No physical sphere cell may lie outside the cropped physical box.
    assert int(box_mask.sum()) == int(mask.sum())
    assert np.isfinite(u[:, box_mask]).all()
    assert (u[[0, 18]][:, box_mask] > 0).all()
    g, a = tensor(u, 1, box_mask), tensor(u, 8, box_mask)
    eigen = np.linalg.eigvalsh(g/u[0][box_mask, None, None])
    determinant = np.linalg.det(g)
    trace = np.einsum('nij,nji->n', np.linalg.inv(g), a)
    assert eigen.min() > 0
    assert np.abs(determinant-1).max() < 1e-12
    assert np.abs(trace).max() < 1e-12
    metadata = {
        'time': checkpoint['time'], 'cycle': checkpoint['cycle'],
        'restart_header_dt': checkpoint['dt'], 'nghost': checkpoint['mb_indcs']['ng'],
        'physical_box_shape': list(u.shape), 'active_cells': int(box_mask.sum()),
        'all_25_active_fields_finite': True, 'all_25_BIN_quantizations_bitwise_equal': True,
        'alpha_min': float(u[18][box_mask].min()), 'chi_min': float(u[0][box_mask].min()),
        'physical_metric_eigen_min': float(eigen.min()),
        'physical_metric_eigen_max': float(eigen.max()),
        'det_error_max': float(np.abs(determinant-1).max()),
        'trace_error_max': float(np.abs(trace).max()),
        'mask_box_sha256': hashlib.sha256(box_mask.tobytes()).hexdigest(),
        'files': records([checkpoint_path, bin_path]),
    }
    return checkpoint, u, box_mask, coords, metadata


def run_data(path, case, expected_exe):
    directory = path/case['name']
    assert case['exit_status'] == 0
    assert sha(path/'athena-validation') == expected_exe
    input_path = directory/'layer.athinput'
    assert sha(input_path) == case['input_sha256']
    settings = runner.input_parameters(input_path.read_text())
    assert float(settings['time/tlim']) == case['requested_time']
    hst = directory/'hyp.z4c.user.hst'
    h = history(hst)
    assert h[0, 0] == 0 and h[-1, 0] == case['requested_time']
    assert case['diagnostics']['time'] == case['requested_time']
    paths = sorted((directory/'rst').glob('*.rst'))
    bin_paths = sorted((directory/'bin').glob('*.z4c.*.bin'))
    assert len(paths) == len(bin_paths) >= 2
    rows, initial, first_mask, final, coordinates, first_checkpoint = [], None, None, None, None, None
    for checkpoint_path, bin_path in zip(paths, bin_paths):
        assert checkpoint_path.stem.rsplit('.', 1)[-1] == bin_path.stem.rsplit('.', 1)[-1]
        checkpoint, u, mask, coords, row = snapshot(checkpoint_path, bin_path)
        if initial is None:
            initial, first_mask = u.copy(), mask.copy()
            first_checkpoint, coordinates = checkpoint, coords
        assert np.array_equal(mask, first_mask)
        assert all(np.array_equal(x, y) for x, y in zip(coords, coordinates))
        row['full_precision_drift_from_initial_max'] = float(
            np.abs(u[:, mask]-initial[:, mask]).max())
        rows.append(row)
        final = u.copy()
    assert rows[0]['time'] == 0 and rows[-1]['time'] == case['requested_time']
    assert all(y['time'] > x['time'] for x, y in zip(rows, rows[1:]))
    return {'case': case, 'settings': settings, 'history': h, 'snapshots': rows,
            'initial': initial, 'final': final, 'mask': first_mask,
            'coordinates': coordinates, 'checkpoint': first_checkpoint,
            'files': records([path/'results.json', path/'athena-validation', input_path, hst])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('--case')
    parser.add_argument('--baseline-case')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    started = time.monotonic()
    run, baseline = args.run.resolve(), args.baseline.resolve()
    private_result = json.loads((run/'results.json').read_text())
    base_result = json.loads((baseline/'results.json').read_text())
    assert private_result['sha256'] == PRIVATE_EXE and base_result['sha256'] == BASE_EXE
    builds = verify_builds()
    private_case = choose_case(private_result, args.case)
    base_case = choose_case(base_result, args.baseline_case)
    assert private_case['requested_time'] == base_case['requested_time']
    private = run_data(run, private_case, PRIVATE_EXE)
    base = run_data(baseline, base_case, BASE_EXE)
    pset, bset = private['settings'], base['settings']
    differences = {key: [bset.get(key), pset.get(key)]
                   for key in sorted(set(pset) | set(bset)) if pset.get(key) != bset.get(key)}
    allowed = {'mesh/nghost', 'time/tlim', 'time/ndiag'}
    allowed |= {f'output{i}/dt' for i in range(1, 6)}
    assert set(differences) <= allowed
    assert bset['mesh/nghost'] == '3' and pset['mesh/nghost'] == '4'
    assert all(private['checkpoint']['mesh_size'][key] == base['checkpoint']['mesh_size'][key]
               for key in private['checkpoint']['mesh_size'])
    assert private['initial'].shape == base['initial'].shape
    assert np.array_equal(private['mask'], base['mask'])
    coordinate_error = max(float(np.abs(x-y).max())
                           for x, y in zip(private['coordinates'], base['coordinates']))
    assert coordinate_error < 64*np.finfo(float).eps
    mask = private['mask']
    initial_difference = private['initial'][:, mask]-base['initial'][:, mask]
    assert np.abs(initial_difference).max() < 1e-12
    fields = {}
    for index, name in enumerate(rst.VARIABLES):
        delta0 = initial_difference[index]
        delta = private['final'][index][mask]-base['final'][index][mask]
        signal = base['final'][index][mask]-base['initial'][index][mask]
        fields[name] = {'initial_max_difference': float(np.abs(delta0).max()),
                        'initial_rms_difference': float(np.sqrt(np.mean(delta0*delta0))),
                        'final_max_difference': float(np.abs(delta).max()),
                        'final_rms_difference': float(np.sqrt(np.mean(delta*delta))),
                        'baseline_rms_change': float(np.sqrt(np.mean(signal*signal)))}
    target = private_case['requested_time']
    comparisons = {}
    for name, index in [('H', 2), ('M', 3), ('Z', 4), ('Theta', 5)]:
        p, b = float(private['history'][-1, index]), float(base['history'][-1, index])
        comparisons[name] = {'private': p, 'baseline': b, 'ratio': p/b if b else None}
    # Original HST diagnostics are compared only at this exact shared endpoint.
    # Diagnostic BIN constraints are intentionally float32; no alternate Dxx is used.
    all_files = [p for directory in [run, baseline] for p in directory.rglob('*') if p.is_file()]
    out = {
        'scope': ('Completed bounded private composed-operator preflight. All 25 '
                  'state fields use full-precision restart data; standard production '
                  'HST differential-constraint diagnostics stay unchanged. Physical '
                  'box cells are paired by uniform grid index across ng4/ng3, with '
                  'floating-coordinate differences measured. No stability acceptance, '
                  'temporal order, exact initial bitwise identity or BH claim.'),
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'seconds': time.monotonic()-started, 'build_verification': builds,
        'private_case': private_case['name'], 'baseline_case': base_case['name'],
        'input_parameter_differences': differences,
        'physical_grid_and_index_alignment_verified': True,
        'native_coordinate_construction_max_difference': coordinate_error,
        'active_physical_box_masks_identical': True,
        'initial_arrays_bitwise_equal': bool(np.array_equal(private['initial'][:, mask],
                                                           base['initial'][:, mask])),
        'initial_full_precision_active_max_difference': float(np.abs(initial_difference).max()),
        'exact_final_comparison_time': target,
        'final_original_HST_diagnostics': comparisons,
        'final_equal_time_full_precision_fields': fields,
        'private_history_rows': private['history'].tolist(),
        'baseline_history_rows': base['history'].tolist(),
        'private_snapshots': private['snapshots'], 'baseline_snapshots': base['snapshots'],
        'private_inputs': private['files'], 'baseline_inputs': base['files'],
        'audit_source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [
            Path(__file__), HERE/'rst-reader-gate/restart_reader.py', HERE/'rst-reader-gate/abi.json',
            ROOT/'vis/python/bin_convert.py', ROOT/'tst/hyperboloidal/run_layer_validation.py']},
        'all_run_files': records(all_files),
        'precision': ('RST state binary64; all 25 active BIN fields match exact float32 '
                      'rounding bitwise. HST text uses production binary64 diagnostics. '
                      'A quantized BIN difference need not equal the true field drift. '
                      'Header dt is pm->dt at write time, not always the completed step.'),
        'status': 'PASS',
    }
    output = args.output or HERE/'composed-native-audit'/(run.name+'-audit.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    print('PASS', run.name, len(private['snapshots']), 'private snapshots,',
          len(base['snapshots']), 'baseline snapshots')
    print('initial max difference', out['initial_full_precision_active_max_difference'],
          'coordinate max difference', coordinate_error)
    print(json.dumps(comparisons, indent=2))


if __name__ == '__main__':
    main()
