#!/usr/bin/env python3
"""Saved scalar/text review only; never open an array or run a scientific helper."""
from pathlib import Path
import hashlib
import json
import math
import re
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PARTIAL = ROOT / 'docs/validation/hyperboloidal-reference-wave-map-partial-observations-20261009'
CAPSULES = [
    PARTIAL,
    ROOT / 'docs/validation/hyperboloidal-reference-wave-map-native-t2-failure-20261009',
    ROOT / 'docs/validation/hyperboloidal-reference-wave-map-native-t2-c0-N16-failure-20261009',
    ROOT / 'docs/validation/hyperboloidal-reference-wave-map-native-t2-wave-N24-failure-20261009',
]
FORBIDDEN_SUFFIX = {'.rst', '.bin', '.npy', '.npz', '.jsonl', '.o', '.a', '.so', '.dylib', '.pyc'}
FORBIDDEN_MAGIC = {b'\x7fELF', b'\xfe\xed\xfa\xce', b'\xce\xfa\xed\xfe',
                   b'\xfe\xed\xfa\xcf', b'\xcf\xfa\xed\xfe', b'\xca\xfe\xba\xbe',
                   b'\xbe\xba\xfe\xca', b'\xca\xfe\xba\xbf', b'\xbf\xba\xfe\xca'}


def finite(value):
    if isinstance(value, dict):
        for child in value.values():
            finite(child)
    elif isinstance(value, list):
        for child in value:
            finite(child)
    elif isinstance(value, float):
        assert math.isfinite(value)


def read(path):
    assert path.suffix.lower() not in FORBIDDEN_SUFFIX, str(path)
    data = path.read_bytes()
    assert data[:4] not in FORBIDDEN_MAGIC and data[:2] != b'MZ', str(path)
    return data


def sha(path):
    return hashlib.sha256(read(path)).hexdigest()


def load(path):
    value = json.loads(read(path).decode('utf-8'),
                       parse_constant=lambda v: (_ for _ in ()).throw(ValueError(v)))
    finite(value)
    return value


def pin(path):
    data = read(path)
    return {'path': str(path.relative_to(ROOT)), 'bytes': len(data),
            'sha256': hashlib.sha256(data).hexdigest()}


def main():
    before = load(HERE / 'review-inputs-before.json')
    assert before['semantic_reads_started'] is False
    report = {'scope': 'Saved source, scalar JSON and copy-catalog review only',
              'raw_RST_BIN_reads': 0, 'array_loads': 0, 'scientific_calls': 0,
              'original_doc_sha256': None, 'catalogs': [], 'cases': [],
              'concrete_corrections': [], 'limitations': []}
    # Verify every initially pinned input, every copied original, finite JSON and format limits.
    finite_json = 0
    maximum_bytes = 0
    json_per_capsule = {}
    for item in before['files']:
        path = ROOT / item['path']
        assert pin(path) == item, item['path']
        assert item['bytes'] <= 1048576, item['path']
        maximum_bytes = max(maximum_bytes, item['bytes'])
        data = read(path)
        if path.suffix.lower() not in {'.png', '.pdf'}:
            data.decode('utf-8')
        if path.suffix.lower() == '.json':
            load(path)
            finite_json += 1
    for prefix in CAPSULES:
        catalog = load(prefix / 'catalog.json')
        preserved = load(prefix / 'preservation-receipt.json')
        assert preserved['catalog_sha256'] == sha(prefix / 'catalog.json')
        for rel, item in catalog['copied'].items():
            local = prefix / rel
            original = Path(item['source'])
            # Only copied text/scalar/plot originals are opened. All omitted array/compiled
            # payloads and the external protected inventory remain metadata-only here.
            assert len(read(local)) == item['bytes'] and sha(local) == item['sha256'], rel
            assert len(read(original)) == item['bytes'] and sha(original) == item['sha256'], rel
        for item in catalog['omitted_payloads'].values():
            assert isinstance(item['bytes'], int) and item['bytes'] >= 0
            assert re.fullmatch('[0-9a-f]{64}', item['sha256'])
            assert Path(item['source']).is_absolute()
        assert preserved['copied_files'] == len(catalog['copied'])
        assert preserved['omitted_files'] == len(catalog['omitted_payloads'])
        json_per_capsule[prefix.name] = sum(p.suffix.lower() == '.json' for p in prefix.rglob('*') if p.is_file())
        report['catalogs'].append({'directory': str(prefix.relative_to(ROOT)),
                                   'catalog_sha256': sha(prefix / 'catalog.json'),
                                   'copied_originals_exact': len(catalog['copied']),
                                   'omitted_metadata_records': len(catalog['omitted_payloads']),
                                   'finite_JSON_files': json_per_capsule[prefix.name]})
    external = load(PARTIAL / 'external-protected-identities.json')
    assert len(external) == 2408
    assert all(Path(k).is_absolute() and re.fullmatch('[0-9a-f]{64}', v) for k, v in external.items())
    pres = load(PARTIAL / 'preservation-receipt.json')
    assert pres['external_identities_sha256'] == sha(PARTIAL / 'external-protected-identities.json')
    assert pres['copied_files'] == 649 and pres['omitted_files'] == 1
    assert pres['original_files'] == 650
    assert pres['external_protected_identities_rehashed_before_after'] == 2408
    assert pres['all_originals_rehashed_before_after']
    assert pres['accepted_native_run'] is False and pres['new_native_steps'] == 0
    assert pres['new_scientific_queries'] == 0
    configs = [
        ('wave-map-N16-large-t2', 'partial-N16', 'independent-fields-N16', CAPSULES[1], 55),
        ('c0-N16-large-t2', 'partial-N16', 'independent-fields-N16', CAPSULES[2], 80),
        ('wave-map-N24-large-t2', 'partial-N24', 'independent-fields-N24', CAPSULES[3], 32),
    ]
    total = 0
    history_max = 0.0
    for case, owner_label, independent_label, failure, expected_count in configs:
        owner = PARTIAL / owner_label / 'attempts' / (case + '-001')
        independent = PARTIAL / independent_label / 'attempts' / (case + '-001')
        rows = load(owner / 'snapshot-observations.json')
        field_rows = load(independent / 'rows.json')
        receipt = load(owner / 'receipt.json')
        field_receipt = load(independent / 'receipt.json')
        history = load(owner / 'history-observation.json')
        launch = load(failure / 'launch-receipt.json')
        assert len(rows) == len(field_rows) == len(history['rows']) == expected_count
        assert receipt['saved_restart_files'] == receipt['native_probe_calls'] == expected_count
        assert receipt['observer_completed'] and receipt['protected_before_after_equal']
        assert receipt['arrays_with_diagnostic_guard_failures'] == 0
        assert receipt['partial_diagnostic_only'] and receipt['accepted_native_run'] is False
        assert receipt['native_returncode'] == -6 and receipt['original_target_time'] == 2
        assert receipt['target_reached_in_available_saved_times'] is False
        assert receipt['observations_sha256'] == sha(owner / 'snapshot-observations.json')
        assert receipt['history_observation_sha256'] == sha(owner / 'history-observation.json')
        assert receipt['console_observation_sha256'] == sha(owner / 'console-observation.json')
        assert field_receipt['diagnostic_protocol_completed'] and field_receipt['before_after_equal']
        assert field_receipt['arrays'] == expected_count and field_receipt['saved_guard_failures'] == 0
        assert field_receipt['accepted_native_run'] is False and field_receipt['new_kernel_calls'] == 0
        assert field_receipt['rows_sha256'] == sha(independent / 'rows.json')
        assert launch['returncode'] == -6 and launch['passed_native_process_and_provenance'] is False
        assert launch['sources_before_after_equal']
        assert receipt['failed_launch_receipt_sha256'] == sha(failure / 'launch-receipt.json')
        assert receipt['native_stderr_sha256'] == sha(failure / 'native.stderr')
        assert (failure / 'protected-inputs-before.json').read_bytes() == (failure / 'protected-inputs-after.json').read_bytes()
        assert (owner / 'protected-inputs-before.json').read_bytes() == (owner / 'protected-inputs-after.json').read_bytes()
        stderr = read(failure / 'native.stderr').decode('utf-8')
        stage = {}
        for key in ['mesh_time', 'cycle', 'Omega', 'alpha', 'chi', 'determinant', 'P', 'Theta']:
            match = re.search(r'\b' + key + r'=([-+0-9.eE]+)', stderr)
            assert match, key
            stage[key] = int(match[1]) if key == 'cycle' else float(match[1])
        case_history_error = 0.0
        previous = -1.0
        for number, (row, field, hrow, call) in enumerate(zip(rows, field_rows, history['rows'], receipt['calls'])):
            assert row['partial_diagnostic_only'] and row['accepted_native_run'] is False
            assert row['diagnostic_guard_failures'] == [] and row['probe_called']
            assert row['structural_and_native_diagnostic_call_completed']
            assert row['time'] > previous
            previous = row['time']
            assert len(row['finite_extrema25']) == 25 and row['nonfinite_count25'] == [0] * 25
            assert row['finite_extrema25'] == field['finite_extrema25']
            assert row['rst_path'] == field['path'] and row['rst_sha256'] == field['sha256']
            assert (row['time'], row['cycle'], row['restart_header_dt']) == (field['time'], field['cycle'], field['dt'])
            assert row['alpha_min'] > 0 and row['chi_min'] > 0
            assert row['minimum_conformal_metric_eigenvalue'] > 0
            assert row['minimum_Penrose_spatial_metric_eigenvalue'] > 0
            assert field['positive_lapse_chi'] and field['SPD_leading_minors'] and field['saved_field_guards_satisfied']
            assert field['det_error_max'] <= 1e-10 and field['trace_error_max'] <= 1e-10
            diag = row['native_diagnostics']
            assert diag['det_max'] <= 1e-10 and diag['trace_max'] <= 1e-10
            assert abs(hrow[0] - row['time']) <= 1e-12
            assert field['original_native_history_H_Mcon_Zcon_Theta'] == hrow[2:6]
            assert call['returncode'] == 0 and call['partial_diagnostic_only'] and call['accepted_native_run'] is False
            probe = owner / ('%04d-probe.stdout' % number)
            assert sha(probe) == call['stdout_sha256'] and load(probe) == diag
            assert sha(owner / ('%04d-probe.stderr' % number)) == call['stderr_sha256']
            assert (owner / ('%04d-probe.stderr' % number)).stat().st_size == 0
            case_history_error = max(case_history_error, *(abs(x-y) for x, y in zip(diag['rms_H_Mcon_Zcon_Theta'], hrow[2:6])))
            assert sum(z['count'] for z in diag['radial_bins']) == diag['active_count']
            for component in range(4):
                assert abs(sum(z['squared_fraction4'][component] for z in diag['radial_bins']) - 1) < 1e-12 or diag['rms_H_Mcon_Zcon_Theta'][component] == 0
        last = rows[-1]
        last_diag = last['native_diagnostics']
        fractions = [sum(z['squared_fraction4'][i] for z in last_diag['radial_bins'] if z['rlo'] >= .9) for i in range(4)]
        for i in range(4):
            shell_fraction = last_diag['shell_r_ge_09_count'] * last_diag['shell_rms_H_Mcon_Zcon_Theta'][i] ** 2 / (last_diag['active_count'] * last_diag['rms_H_Mcon_Zcon_Theta'][i] ** 2)
            assert abs(shell_fraction - fractions[i]) < 1e-13
        assert last['time'] < stage['mesh_time'] < 2
        assert receipt['last_available_saved_time'] == last['time']
        assert history['last_history_time'] == last['time']
        total += expected_count
        history_max = max(history_max, case_history_error)
        report['cases'].append({'case': case, 'arrays': expected_count, 'last_saved_time': last['time'],
            'abort_stage': stage, 'endpoint_RMS_H_M_Z_Theta': last_diag['rms_H_Mcon_Zcon_Theta'],
            'endpoint_shell_squared_fractions_H_M_Z_Theta': fractions,
            'history_probe_max_absolute_error': case_history_error,
            'minimum_saved_alpha': min(x['alpha_min'] for x in rows),
            'minimum_saved_chi': min(x['chi_min'] for x in rows),
            'minimum_saved_gtilde_eigenvalue': min(x['minimum_conformal_metric_eigenvalue'] for x in rows),
            'maximum_saved_native_det_error': max(x['native_diagnostics']['det_max'] for x in rows),
            'maximum_saved_native_A_trace_error': max(x['native_diagnostics']['trace_max'] for x in rows),
            'all_25_extrema_and_snapshot_identities_exact': True,
            'saved_state_guards_pass': True, 'original_process_failed': True})
    assert total == 167 and history_max <= 4.440892098500626e-16
    document = ROOT / 'docs/hyperboloidal-reference-wave-map-failure-audit.md'
    doc = read(document).decode('utf-8')
    report['original_doc_sha256'] = sha(document)
    # Explicitly retain the original typo as a failed documentary fact, rather than
    # silently repairing the reviewed document or changing this saved observation.
    assert '.9835844' in doc
    report['concrete_corrections'].append({
        'kind': 'rounded_document_number',
        'location': 'Saved states before the aborts: last N24 squared Z fraction at r>=.9',
        'original_text': '.9835844', 'saved_value': report['cases'][2]['endpoint_shell_squared_fractions_H_M_Z_Theta'][2],
        'correct_seven_place_rounding': '.9835841',
        'status': 'parent notified; original pinned document unchanged'})
    for phrase in ['two native', 'not C0', 'conditional stationary obstruction',
                   'wormhole-to-trumpet', 'Minkowski hyperboloidal reference',
                   'withheld', 'other six controls are still running']:
        assert phrase in doc, phrase
    assert re.search(r'All 55\+80\+32=167', doc)
    report.update(saved_arrays=total, all_25_extrema_identity_pairs=25*total,
                  maximum_history_probe_absolute_error=history_max,
                  pinned_review_input_files=before['count'], pinned_review_input_bytes=before['bytes'],
                  finite_JSON_files=finite_json, maximum_archived_file_bytes=maximum_bytes,
                  external_identity_records=2408, external_original_payloads_rehashed_by_this_review=False)
    report['limitations'] = [
        'No raw RST/BIN, compiled payload, omitted large original or active-run output was opened.',
        '2408 protected external identities were checked as finite SHA metadata and collector evidence; their original array/binary contents were not rehashed by this restricted review.',
        'Constraint values are compared between saved native probe output and native history, not independently differentiated.',
        'No cause of failure, resolution order, all-nine acceptance, t6/t12 admission, continuum stability or later black-hole acceptance follows.']
    after = {'files': [pin(ROOT / item['path']) for item in before['files']]}
    assert after['files'] == before['files']
    (HERE / 'review-inputs-after.json').write_text(json.dumps(after, indent=2, sort_keys=True) + '\n')
    (HERE / 'reviewed-document.md').write_bytes(read(document))
    return report


if __name__ == '__main__':
    attempt = HERE / 'attempt001'
    attempt.mkdir(exist_ok=False)
    started = time.monotonic()
    receipt = {'completed': False, 'scientific_calls': 0, 'raw_RST_BIN_reads': 0,
               'source_sha256': sha(Path(__file__)), 'python': sys.version}
    try:
        result = main()
        (attempt / 'readback.json').write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
        receipt.update(completed=True, disposition='completed_with_one_documentary_correction_required',
                       readback_sha256=sha(attempt / 'readback.json'))
    except BaseException as exc:
        receipt['failure'] = repr(exc)
        (attempt / 'failure.txt').write_text(traceback.format_exc())
    finally:
        receipt['seconds'] = time.monotonic() - started
        (attempt / 'receipt.json').write_text(json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + '\n')
        print(json.dumps(receipt, sort_keys=True, allow_nan=False))
    if not receipt['completed']:
        sys.exit(1)
