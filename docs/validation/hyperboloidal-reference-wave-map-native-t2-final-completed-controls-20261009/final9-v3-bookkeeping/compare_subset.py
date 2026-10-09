"""HELD saved scalar bookkeeping only; no arrays, kernel, subprocess or evolution."""
from pathlib import Path
import hashlib
import json
import math

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
READBACK = ROOT / 'build-layer-research/reference-wave-map-t2-readback-held-20261009'
LONG = ROOT / 'build-layer-research/wave-map-native-t2-root-20261009'
NATIVE_PLAN = ROOT / 'docs/validation/hyperboloidal-reference-wave-map-native-preflights-20261009/native-preparation/PLAN.md'
NATIVE_PLAN_SHA = '7955afb3db4b227c0fe885410255452b4fcb15e249ed97b2d52c2672c4697926'
ZERO_ABS = 1e-10
STATUS = {'native_failed', 'completed_and_admitted'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    def invalid(value):
        raise ValueError('Non-finite JSON constant: ' + value)
    value = json.loads(Path(path).read_text(), parse_constant=invalid)
    def finite(item):
        if isinstance(item, float):
            require(math.isfinite(item), 'Non-finite JSON float')
        elif isinstance(item, dict):
            for child in item.values():
                finite(child)
        elif isinstance(item, list):
            for child in item:
                finite(child)
    finite(value)
    return value


def pinned(path, digest, pins):
    path = Path(path).resolve()
    require(isinstance(digest, str) and len(digest) == 64, 'Missing SHA256 pin')
    require(sha(path) == digest, 'Hash mismatch: ' + str(path))
    key = str(path)
    require(key not in pins or pins[key] == digest, 'Conflicting pin: ' + key)
    pins[key] = digest
    return path


def scalar(value, label):
    require(isinstance(value, (float, int)) and not isinstance(value, bool), label + ' is not numeric')
    require(math.isfinite(value), label + ' is not finite')
    return value


def factor_test(value, baseline, factor):
    branch = 'relative_nonzero_baseline' if baseline != 0 else 'absolute_zero_denominator_fallback'
    passed = value <= factor * baseline if baseline != 0 else abs(value) <= ZERO_ABS
    return {'passed': passed, 'value': value, 'baseline': baseline,
            'nominal_factor': factor, 'branch': branch,
            'ratio': value / baseline if baseline != 0 else None,
            'measured_reduction_fraction': (baseline - value) / baseline if baseline != 0 else None,
            'zero_baseline_absolute_tolerance': ZERO_ABS if baseline == 0 else None}


def change_test(value, baseline):
    difference = abs(value - baseline)
    branch = 'relative_nonzero_baseline' if baseline != 0 else 'absolute_zero_denominator_fallback'
    return {'passed': difference <= .1 * abs(baseline) if baseline != 0 else abs(value) <= ZERO_ABS,
            'value': value, 'baseline': baseline, 'absolute_difference': difference,
            'relative_difference': difference / abs(baseline) if baseline != 0 else None,
            'relative_tolerance': .1, 'branch': branch,
            'zero_baseline_absolute_tolerance': ZERO_ABS if baseline == 0 else None}


def native_receipt(name, entry, registry_row, pins):
    path = Path(entry['native_launch_receipt_path']).resolve()
    expected = (LONG / 'batch001' / name / 'launch-receipt.json').resolve()
    require(path == expected, 'Wrong original native receipt path for ' + name)
    path = pinned(path, entry['native_launch_receipt_sha256'], pins)
    receipt = load(path)
    require(receipt['mode'] == registry_row['mode'], 'Wrong native mode for ' + name)
    require(receipt['sources_before_after_equal'] is True, 'Original source provenance failed: ' + name)
    return receipt


def completed_case(name, entry, registry_row, pins):
    native = native_receipt(name, entry, registry_row, pins)
    require(native['passed_native_process_and_provenance'] is True,
            'Declared admitted case lacks successful native receipt: ' + name)
    require(native['returncode'] == 0 and native['error'] is None,
            'Native process did not complete successfully: ' + name)
    attempt = Path(entry['completed_readback_attempt_directory']).resolve()
    require(attempt.parent == (READBACK / 'attempts').resolve(), 'Wrong readback tree for ' + name)
    outer_path = pinned(attempt / 'receipt.json', entry['completed_outer_receipt_sha256'], pins)
    outer = load(outer_path)
    require(outer['passed_completed_t2_snapshot_gates'] is True, 'Completed readback failed: ' + name)
    require(outer['case'] == name and outer['protected_before_after_equal'] is True,
            'Completed-case identity/protection mismatch: ' + name)
    require(outer['launch_receipt_sha256'] == entry['native_launch_receipt_sha256'],
            'Outer/native launch mismatch: ' + name)
    require(outer['analyzer_receipt_sha256'] == entry['completed_analyzer_receipt_sha256'], 'Frozen analyzer pin mismatch: ' + name)
    analysis_path = pinned(attempt / 'analysis/receipt.json', entry['completed_analyzer_receipt_sha256'], pins)
    analysis = load(analysis_path)
    require(analysis['passed_saved_snapshot_finite_and_diagnostic_gates'] is True,
            'Analyzer admission failed: ' + name)
    require(analysis['target_time'] == 2. and analysis['protected_inputs_before_after_equal'] is True,
            'Analyzer target/protection mismatch: ' + name)
    require(analysis['mode'] == registry_row['mode'] and analysis['N'] == registry_row['N'],
            'Analyzer mode/grid mismatch: ' + name)
    require(analysis['reference'] is (registry_row['profile'] == 'reference'),
            'Analyzer reference-profile mismatch: ' + name)
    require(analysis['launch_receipt_sha256'] == entry['native_launch_receipt_sha256'],
            'Analyzer/native launch mismatch: ' + name)
    require(analysis['snapshots_sha256'] == entry['completed_snapshots_sha256'], 'Frozen saved-JSON pin mismatch: ' + name)
    rows_path = pinned(attempt / 'analysis/snapshots.json', entry['completed_snapshots_sha256'], pins)
    rows = load(rows_path)
    require(isinstance(rows, list) and len(rows) == analysis['saved_arrays'] and len(rows) >= 3,
            'Snapshot-result count mismatch: ' + name)
    times = [scalar(row['time'], 'time') for row in rows]
    require(abs(times[0]) < 1e-14 and abs(times[-1] - 2.) < 1e-12,
            'Missing exact original initial/target time: ' + name)
    require(all(0 <= time <= 2. + 1e-12 for time in times), 'Out-of-range saved time: ' + name)
    require(all(left < right for left, right in zip(times, times[1:])), 'Unordered saved times: ' + name)
    for row in rows:
        norms = row['rms_H_Mcon_Zcon_Theta']
        require(isinstance(norms, list) and len(norms) == 4, 'Wrong norm-vector size: ' + name)
        require(all(scalar(value, 'RMS') >= 0 for value in norms), 'Negative RMS: ' + name)
    row1 = min(rows, key=lambda row: abs(row['time'] - 1.))
    require(abs(row1['time'] - 1.) < .001, 'No original near-t1 sample: ' + name)
    one = row1['rms_H_Mcon_Zcon_Theta']
    final = rows[-1]['rms_H_Mcon_Zcon_Theta']
    window = [row for row in rows if row['time'] >= row1['time']]
    peaks = [max(row['rms_H_Mcon_Zcon_Theta'][j] for row in window) for j in range(4)]
    return {'status': 'completed_and_admitted', 'saved_arrays': len(rows),
            'target_time': 2., 'actual_t2_sample_time': times[-1],
            'actual_t1_sample_time': row1['time'],
            't1_selection': 'exact_within_1e-12' if abs(row1['time'] - 1.) <= 1e-12
                            else 'nearest_within_0.001_no_interpolation',
            'requested_window': [1., 2.], 'actual_saved_window': [row1['time'], times[-1]],
            't1_RMS4': one, 't2_RMS4': final, 'saved_window_peak_RMS4': peaks,
            'endpoint_growth_factor2_tests': [factor_test(v, b, 2.) for v, b in zip(final, one)],
            'saved_window_peak_growth_factor2_tests': [factor_test(v, b, 2.) for v, b in zip(peaks, one)],
            'scope': 'Saved completed-case scalar results only; field/kernel admission not repeated.'}


def unavailable(dependencies, cases):
    missing = {name: cases[name]['status'] for name in dependencies
               if cases[name]['status'] != 'completed_and_admitted'}
    if not missing:
        return None
    return {'status': 'not_evaluable', 'dependencies': dependencies,
            'unavailable_dependencies': missing, 'passed': None}


def comparison(dependencies, cases, operation):
    absent = unavailable(dependencies, cases)
    if absent is not None:
        return absent
    result = operation()
    result.update(status='evaluated', dependencies=dependencies)
    return result


def compare(auth, pins):
    require(auth['completed_subset_comparison_authorized'] is True, 'Source-only template is not authorized')
    expected_sources = {'PLAN.md': HERE / 'PLAN.md', 'registry.json': HERE / 'registry.json',
                        'compare_subset.py': Path(__file__).resolve(), 'run_subset.py': HERE / 'run_subset.py',
                        'original_native_PLAN': NATIVE_PLAN,
                        'completed_readback_recipe': READBACK / 'recipe.json',
                        'final_evidence_recipe': HERE / 'bookkeeping-recipe.json',
                        'original_final_batch_receipt': LONG / 'batch001/receipt.json'}
    require(set(auth['source_sha256']) == set(expected_sources), 'Incomplete source authorization pins')
    for key, path in expected_sources.items():
        pinned(path, auth['source_sha256'][key], pins)
    require(auth['source_sha256']['original_native_PLAN'] == NATIVE_PLAN_SHA, 'Changed original numerical PLAN')
    registry = load(HERE / 'registry.json')
    names = [row['name'] for row in registry['cases']]
    require(len(names) == len(set(names)) == 9 and set(auth['cases']) == set(names), 'Wrong fixed nine-case registry')
    known = set(registry['non_overridable_known_native_failures'])
    require(len(known) == 6 and registry['pending_allowed'] is False, 'Wrong fixed final-failure set')
    batch = load(LONG / 'batch001/receipt.json')
    require(batch['native_cases'] == 9 and batch['passed_native_processes_and_provenance'] is False, 'Final batch status mismatch')
    require(batch['cases'] == {name: auth['cases'][name]['native_launch_receipt_sha256'] for name in names}, 'Final batch/case pins mismatch')
    cases = {}
    for row in registry['cases']:
        name = row['name']
        entry = auth['cases'][name]
        status = entry['status']
        require(status in STATUS, 'Unknown/pending final case status: ' + name)
        require(status == row['status'] and entry == row['evidence'], 'Final frozen case cannot be relabeled: ' + name)
        require(name not in known or status == 'native_failed', 'Known original failure cannot be relabeled: ' + name)
        if status == 'completed_and_admitted':
            cases[name] = completed_case(name, entry, row, pins)
        elif status == 'native_failed':
            native = native_receipt(name, entry, row, pins)
            require(native['passed_native_process_and_provenance'] is False, 'Failure label contradicts native receipt: ' + name)
            require(native['returncode'] != 0 or native['error'] is not None,
                    'No native failure recorded: ' + name)
            cases[name] = {'status': status, 'original_native_returncode': native['returncode'],
                           'original_native_error': native['error'], 'target_time': 2.,
                           't2_values': None, 'native_receipt_sha256': entry['native_launch_receipt_sha256'],
                           'scope': 'Original process metadata only; no partial arrays/log/history inspected.'}
        else:
            raise ValueError('Final registry has no pending branch: ' + name)
    w24, c24 = 'wave-map-N24-large-t2', 'c0-N24-large-t2'
    w32, half = 'wave-map-N32-large-t2', 'wave-map-half-N24-large-t2'
    def n24():
        values, bases = cases[w24]['t2_RMS4'][:3], cases[c24]['t2_RMS4'][:3]
        no_worse = [factor_test(v, b, 1.) for v, b in zip(values, bases)]
        factor08 = [factor_test(v, b, .8) for v, b in zip(values, bases)]
        count = sum(test['passed'] for test in factor08)
        measured = sum(test['passed'] and test['branch'] == 'relative_nonzero_baseline' for test in factor08)
        return {'no_worse_tests_HMZ': no_worse, 'factor_0p8_tests_HMZ': factor08,
                'factor_0p8_pass_count_including_original_absolute_fallback': count,
                'defined_positive_baseline_twenty_percent_reduction_count': measured,
                'fixed_useful_improvement_predicate': all(test['passed'] for test in no_worse) and count >= 2,
                'zero_fallback_is_not_a_measured_percentage_reduction': True}
    comparisons = {
        'N24_wave_map_vs_matched_C0': comparison([w24, c24], cases, n24),
        'N32_vs_N24': comparison([w32, w24], cases, lambda: {
            'factor_0p8_tests_HMZ': [factor_test(v, b, .8) for v, b in
                                    zip(cases[w32]['t2_RMS4'][:3], cases[w24]['t2_RMS4'][:3])]}),
        'half_cap_vs_standard_N24': comparison([half, w24], cases, lambda: {
            'relative_change_tests_HMZ': [change_test(v, b) for v, b in
                                         zip(cases[half]['t2_RMS4'][:3], cases[w24]['t2_RMS4'][:3])]})}
    growth_names = ['wave-map-N16-large-t2', w24, w32, half]
    growth = {}
    for name in growth_names:
        growth[name] = comparison([name], cases, lambda name=name: {
            'endpoint_tests_HMZTheta': cases[name]['endpoint_growth_factor2_tests'],
            'saved_window_peak_tests_HMZTheta': cases[name]['saved_window_peak_growth_factor2_tests'],
            'actual_t1_sample_time': cases[name]['actual_t1_sample_time'],
            'actual_saved_window': cases[name]['actual_saved_window'],
            'unsaved_times_not_tested': True})
    failed = [name for name in names if cases[name]['status'] == 'native_failed']
    pending = []
    require(set(failed) == known, 'All six original failures are mandatory')
    admitted = [name for name in names if cases[name]['status'] == 'completed_and_admitted']
    require(len(admitted) == 3, 'Exactly three completed/admitted cases required')
    all_admitted = False
    reasons = ['diagnostic_report_cannot_authorize_continuation']
    if failed:
        reasons.append('original_fixed_cases_failed_native_admission')
    if pending:
        reasons.append('fixed_case_evidence_pending')
    return {'final_fixed_case_bookkeeping_only': True, 'all_original_processes_finished': True,
            'partial_matrix_diagnostic_only': True, 'fixed_registry_size': 9,
            'cases': cases, 'admitted_completed_cases': admitted,
            'native_failed_cases': failed, 'pending_cases': pending,
            'all_nine_admitted': False, 'matrix_complete_and_admitted': False,
            'matrix_status': 'finished_with_six_mandatory_failed_original_cases',
            'comparisons': comparisons, 'large_candidate_growth_diagnostics': growth,
            't6_admission': False, 't12_admission': False, 'continuation_withheld_reasons': reasons,
            'endpoint_and_saved_window_predicates_are_distinct': True,
            'scope': 'Saved scalar bookkeeping only. Missing/failed cases never acquire t2 values. '
                     'No native/PDE stability, convergence order, exact-scri or BH inner-gauge claim.'}
