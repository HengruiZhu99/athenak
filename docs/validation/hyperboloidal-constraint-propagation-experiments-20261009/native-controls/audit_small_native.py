"""Independent full-precision amplitude gate for the small angular pulse."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('small_comparison_helpers',
                                             HERE/'audit_composed_native.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    started = time.monotonic()
    small_run = HERE/'small-angular-t2'
    base_run = audit.FAMILY/'native-long'
    ref_run = audit.FAMILY/'native-reference'
    source_paths = [Path(__file__), HERE/'audit_composed_native.py',
                    HERE/'rst-reader-gate/restart_reader.py', HERE/'rst-reader-gate/abi.json',
                    ROOT/'vis/python/bin_convert.py', ROOT/'tst/hyperboloidal/run_layer_validation.py']
    source_before = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    build_path = audit.FAMILY/'native-build-receipt.json'
    build = json.loads(build_path.read_text())
    assert all(sha(ROOT/key) == value for key, value in build['source_sha256'].items())
    assert all(sha(audit.FAMILY/key) == value for key, value in build['overlay_sha256'].items())
    runs = {}
    for label, path in [('small', small_run), ('baseline', base_run), ('reference', ref_run)]:
        result = json.loads((path/'results.json').read_text())
        assert result['sha256'] == audit.BASE_EXE
        case, = result['cases']
        runs[label] = audit.run_data(path, case, audit.BASE_EXE)
    small, base, ref = (runs[key] for key in ['small', 'baseline', 'reference'])
    assert len(small['snapshots']) == len(base['snapshots']) == 81
    assert small['snapshots'][-1]['time'] == base['snapshots'][-1]['time'] == 2
    for other in [base, ref]:
        assert small['initial'].shape == other['initial'].shape
        assert np.array_equal(small['mask'], other['mask'])
        assert all(np.array_equal(x, y) for x, y in zip(small['coordinates'], other['coordinates']))
    differences = {key: [base['settings'].get(key), small['settings'].get(key)]
                   for key in sorted(set(base['settings']) | set(small['settings']))
                   if base['settings'].get(key) != small['settings'].get(key)}
    assert set(differences) == {'problem/lapse_pulse', 'problem/shift_pulse'}
    scale = .01
    for name in ['lapse_pulse', 'shift_pulse']:
        assert float(small['settings']['problem/'+name]) == scale*float(base['settings']['problem/'+name])
        assert float(ref['settings']['problem/'+name]) == 0
    mask = small['mask']
    gauge = {18, 19, 20, 21}
    geometry = [index for index in range(25) if index not in gauge]
    assert np.array_equal(small['initial'][geometry][:, mask], base['initial'][geometry][:, mask])
    assert np.array_equal(small['initial'][geometry][:, mask], ref['initial'][geometry][:, mask])
    amplitude_checks = {}
    for index in sorted(gauge):
        large = base['initial'][index][mask]-ref['initial'][index][mask]
        little = small['initial'][index][mask]-ref['initial'][index][mask]
        error = little-scale*large
        assert np.abs(error).max() < 8*np.finfo(float).eps*(1+np.abs(ref['initial'][index][mask]).max())
        amplitude_checks[audit.rst.VARIABLES[index]] = {
            'baseline_perturbation_max': float(np.abs(large).max()),
            'small_perturbation_max': float(np.abs(little).max()),
            'scaled_subtraction_residual_max': float(np.abs(error).max()),
            'scaled_subtraction_relative_to_peak': float(np.abs(error).max()/(scale*np.abs(large).max())),
        }
    final = {}
    for name, index in [('H', 2), ('M', 3), ('Z', 4), ('Theta', 5)]:
        x, y = float(small['history'][-1, index]), float(base['history'][-1, index])
        final[name] = {'small': x, 'baseline': y, 'small_divided_by_amplitude_scale': x/scale,
                       'normalized_ratio_to_baseline': x/(scale*y), 'raw_ratio_to_baseline': x/y}
    exact_histories = bool(np.array_equal(small['history'][:, 0], base['history'][:, 0]))
    comparisons = []
    for row in small['history']:
        values = {'time': float(row[0])}
        for name, index in [('H', 2), ('M', 3), ('Z', 4), ('Theta', 5)]:
            y = float(np.interp(row[0], base['history'][:, 0], base['history'][:, index]))
            values[name] = {'small': float(row[index]), 'baseline': y,
                            'normalized_small': float(row[index]/scale)}
        comparisons.append(values)
    after = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    assert after == source_before
    files = [path for directory in [small_run, base_run, ref_run]
             for path in directory.rglob('*') if path.is_file()]
    output = {
        'scope': ('Small-amplitude sensitivity control, not acceptance of the required '
                  'finite pulse. Initial geometry is exactly identical; gauge amplitudes '
                  'are factor .01 within binary64 reference-subtraction roundoff. '
                  'All states use full-precision RST; original HST constraints retained.'),
        'head': subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        'seconds': time.monotonic()-started, 'status': 'PASS',
        'base_build_receipt_sha256': sha(build_path),
        'production_source_and_gauge_overlay_unchanged': True,
        'all_three_executables_sha256': audit.BASE_EXE,
        'input_parameter_differences': differences, 'amplitude_scale': scale,
        'initial_geometry_bitwise_identical_to_baseline_and_reference': True,
        'initial_physical_coordinates_and_masks_bitwise_equal': True,
        'initial_gauge_amplitude_checks': amplitude_checks,
        'exact_final_comparison_time': 2., 'final_original_HST_constraints': final,
        'interior_history_times_exactly_equal': exact_histories,
        'interior_history_method': ('Baseline linearly interpolated at small-pulse history times; '
                                    'this equals direct pairing when recorded times coincide. '
                                    'Final endpoint is exactly t2 for both.'),
        'history_comparisons': comparisons,
        'small_snapshots': small['snapshots'], 'baseline_snapshots': base['snapshots'],
        'reference_snapshots': ref['snapshots'],
        'small_history_rows': small['history'].tolist(),
        'baseline_history_rows': base['history'].tolist(),
        'source_sha256_before': source_before, 'source_sha256_after': after,
        'all_run_files': audit.records(files),
        'precision': ('State binary64; all active BIN fields equal bitwise float32 casts. '
                      'All saved active fields finite, alpha/chi positive, physical metric '
                      'SPD, determinant/trace error <1e-12. Header dt is exact at write time.'),
    }
    destination = HERE/'small-native-audit'
    destination.mkdir(exist_ok=True)
    (destination/'receipt.json').write_text(json.dumps(output, indent=2, allow_nan=False)+'\n')
    print('PASS small pulse', len(small['snapshots']), 'full-precision snapshots')
    print(json.dumps(amplitude_checks, indent=2))
    print(json.dumps(final, indent=2))


if __name__ == '__main__':
    main()
