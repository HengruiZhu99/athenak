"""Read-only independent reconstruction of the parent's completed controls."""
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
CONTROLS = HERE.parent
spec = importlib.util.spec_from_file_location('independent_control_helpers',
                                             CONTROLS/'audit_composed_native.py')
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


started = time.monotonic()
stage_build_path = CONTROLS/'stage-projection-build/build-receipt.json'
stage_build = json.loads(stage_build_path.read_text())
base_build_path = audit.FAMILY/'native-build-receipt.json'
base_build = json.loads(base_build_path.read_text())
assert sha(base_build_path) == stage_build['base_build_receipt_sha256']
assert all(sha(ROOT/key) == value for key, value in base_build['source_sha256'].items())
assert all(sha(audit.FAMILY/key) == value for key, value in base_build['overlay_sha256'].items())
assert sha(CONTROLS/'build_stage_projection.py') == stage_build['script_sha256']
assert sha(ROOT/'src/z4c/z4c_tasks.cpp') == stage_build['source_before_sha256']
private_source = CONTROLS/'stage-projection-build/z4c_tasks.cpp'
assert sha(private_source) == stage_build['private_source_sha256']
replacement = stage_build['only_source_replacement']
assert private_source.read_text().replace(replacement['after'], replacement['before']) == (
    ROOT/'src/z4c/z4c_tasks.cpp').read_text()
assert stage_build['compile_exit_status'] == stage_build['link_exit_status'] == 0
assert sha(CONTROLS/'stage-projection-build/z4c_tasks.cpp.o') == stage_build['private_object_sha256']
assert sha(ROOT/stage_build['executable']) == stage_build['executable_sha256']
assert all(sha(ROOT/key) == value['sha256']
           for key, value in stage_build['reused_base_link_inputs_sha256'].items())
rows = []
for name in ['N24-half-step-t2', 'stage-projection-t2', 'stage-projection-reference']:
    receipt_path = CONTROLS/(name+'-audit.json')
    original_bytes = receipt_path.read_bytes()
    result = json.loads(original_bytes)
    for group in ['audit_source_sha256', 'comparison_input_sha256']:
        assert all(sha(ROOT/key) == value for key, value in result[group].items())
    assert all(sha(ROOT/key) == value['sha256']
               for key, value in result['all_retained_files'].items())
    run = CONTROLS/name
    raw = json.loads((run/'results.json').read_text())
    case, = raw['cases']
    reference = name.endswith('reference')
    base_path = audit.FAMILY/('native-reference' if reference else 'native-long')
    base_raw = json.loads((base_path/'results.json').read_text())
    base_case, = base_raw['cases']
    current = audit.run_data(run, case, raw['sha256'])
    base = audit.run_data(base_path, base_case, audit.BASE_EXE)
    mask = current['mask']
    assert np.array_equal(mask, base['mask'])
    assert np.array_equal(current['initial'][:, mask], base['initial'][:, mask])
    assert all(np.array_equal(x, y) for x, y in zip(current['coordinates'], base['coordinates']))
    assert len(current['snapshots']) == len(result['snapshots'])
    for fresh, old in zip(current['snapshots'], result['snapshots']):
        for key in ['time', 'cycle', 'restart_header_dt', 'active_cells', 'alpha_min',
                    'chi_min', 'physical_metric_eigen_min', 'physical_metric_eigen_max',
                    'det_error_max', 'trace_error_max', 'full_precision_drift_from_initial_max']:
            assert fresh[key] == old[key], (name, key, fresh[key], old[key])
    fields = {}
    for index, key in enumerate(audit.rst.VARIABLES):
        difference = current['final'][index][mask]-base['final'][index][mask]
        change = base['final'][index][mask]-base['initial'][index][mask]
        values = {'rms_difference': float(np.sqrt(np.mean(difference*difference))),
                  'max_difference': float(np.abs(difference).max()),
                  'baseline_rms_change': float(np.sqrt(np.mean(change*change)))}
        assert values == result['final_equal_time_fields'][key], (name, key)
        fields[key] = values
    endpoint = str(current['snapshots'][-1]['time'])
    ratios = {}
    for key, column in [('H', 2), ('M', 3), ('Z', 4), ('Theta', 5)]:
        x, y = float(current['history'][-1, column]), float(base['history'][-1, column])
        values = {'control': x, 'baseline': y, 'ratio': x/y}
        assert values == result['matched_history_times'][endpoint][key]
        ratios[key] = values
    assert receipt_path.read_bytes() == original_bytes
    rows.append({'name': name, 'receipt_sha256': sha(receipt_path),
                 'all_captured_source_and_run_hashes_verified': True,
                 'all_saved_double_snapshots_independently_reconstructed': True,
                 'all_25_BIN_quantizations_bitwise_equal': True,
                 'initial_active_coordinates_masks_arrays_bitwise_identical': True,
                 'snapshots': len(current['snapshots']),
                 'all_final_field_comparisons_exactly_reproduced': True,
                 'final_relative_constraints': ratios,
                 'max_final_full_field_difference': max(x['max_difference'] for x in fields.values()),
                 'first_dt_ratio': current['history'][0, 1]/base['history'][0, 1]})
out = {'status': 'PASS', 'seconds': time.monotonic()-started,
       'scope': 'Independent read-only completed-control receipt/state reconstruction; no accepted file changed.',
       'stage_private_source_exact_single_replacement_verified': True,
       'stage_private_object_executable_and_all_reused_inputs_verified': True,
       'stage_build_receipt_sha256': sha(stage_build_path),
       'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [
           Path(__file__), CONTROLS/'audit_composed_native.py', CONTROLS/'audit_native_control.py',
           CONTROLS/'rst-reader-gate/restart_reader.py']}, 'cases': rows}
(HERE/'receipt.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
