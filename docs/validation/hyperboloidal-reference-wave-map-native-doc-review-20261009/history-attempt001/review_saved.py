#!/usr/bin/env python3
"""Saved-only factual review; never opens omitted arrays or executes science."""
import hashlib
import json
import math
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parents[3]
OUT = pathlib.Path(__file__).resolve().parent
ARCHIVE = ROOT / 'docs/validation/hyperboloidal-reference-wave-map-native-preflights-20261009'
CATALOG_SHA = 'fc9de674ff19ce3bd9e6cf51b4ccd489f5ff5820748aa09e0b7f1a8ccd372349'
DOCS = ['docs/hyperboloidal-reference-wave-map-native-audit.md',
        'docs/hyperboloidal-layer.md',
        'docs/hyperboloidal-wave-map-consistency-audit.md']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def strict_json(path):
    def invalid(value):
        raise ValueError('Non-finite JSON constant: ' + value)
    value = json.loads(path.read_text(), parse_constant=invalid)
    def check(item):
        if isinstance(item, float):
            assert math.isfinite(item)
        elif isinstance(item, dict):
            for child in item.values():
                check(child)
        elif isinstance(item, list):
            for child in item:
                check(child)
    check(value)
    return value


def close(left, right, tolerance=2e-14):
    assert abs(left - right) <= tolerance * max(1.0, abs(left), abs(right)), (left, right)


def main():
    assert not (OUT / 'receipt.json').exists(), 'One-shot review destination already completed'
    assert sha(ARCHIVE / 'catalog.json') == CATALOG_SHA
    catalog = strict_json(ARCHIVE / 'catalog.json')
    archive_paths = sorted(path for path in ARCHIVE.rglob('*') if path.is_file())
    before = {str(path.relative_to(ARCHIVE)): sha(path) for path in archive_paths}
    assert len(archive_paths) == 740
    assert sum(path.stat().st_size for path in archive_paths) == 24095462
    finite_json_count = 0
    for path in archive_paths:
        data = path.read_bytes()
        assert len(data) <= 1048576
        data.decode('utf-8')
        assert path.suffix.lower() not in ('.bin', '.rst', '.npz', '.npy', '.jsonl', '.o', '.a')
        assert not data.startswith((b'PK\x03\x04', b'\x93NUMPY', b'\x7fELF', b'!<arch>\n',
                                    b'\xcf\xfa\xed\xfe', b'\xfe\xed\xfa\xcf'))
        if path.suffix == '.json':
            strict_json(path)
            finite_json_count += 1
    assert finite_json_count == 304
    for relative, record in catalog['files'].items():
        path = ARCHIVE / relative
        assert sha(path) == record['sha256']
        assert path.stat().st_size == record['bytes']
    omitted = catalog['omitted_large_payloads']
    assert len(omitted) == 278
    omitted_suffix_counts = {suffix: sum(path.endswith(suffix) for path in omitted)
                             for suffix in ('.rst', '.bin')}
    assert omitted_suffix_counts == {'.rst': 64, '.bin': 192}
    collection = strict_json(ARCHIVE / 'collection-receipt.json')
    assert collection['passed'] and collection['before_after_equal']
    assert collection['source_inventory_files'] == 1013
    assert collection['external_dependency_identities'] == 1675
    dependencies = strict_json(ARCHIVE / 'dependency-metadata.json')
    assert len(dependencies['files']) == 1675
    production = strict_json(ARCHIVE / 'production-source-identity.json')
    assert production['count'] == len(production['files']) == 365
    assert production['all_current_bytes_equal_git_implementation']
    assert production['implementation'] == '27c19d20696ea6dd4704032c51dfd026218f64f2'

    summary_rel = 'native-preparation/snapshot-preflight-summary001/summary.json'
    summary = strict_json(ARCHIVE / summary_rel)
    assert summary['passed_all_eight_short_reference_snapshot_gates']
    assert len(summary['cases']) == 8
    assert sum(case['saved_arrays'] for case in summary['cases']) == 64
    rows = []
    maximum_budget_reconstruction_error = 0.0
    maximum_native_history_error = 0.0
    for case in summary['cases']:
        name = case['case']
        prefix = 'native-preparation/snapshot-attempts/' + name + '-001/'
        snapshots = strict_json(ARCHIVE / (prefix + 'analysis/snapshots.json'))
        assert len(snapshots) == case['saved_arrays']
        close(snapshots[-1]['time'], case['target_time'], 1e-12)
        assert case['t0_actual_initializer_max_error'] == 0
        for left, right in zip(snapshots[-1]['rms_H_Mcon_Zcon_Theta'],
                               case['final_H_Mcon_Zcon_Theta_physical']):
            close(left, right)
        for snapshot in snapshots:
            bins = snapshot['radial_bins']
            assert sum(item['count'] for item in bins) == snapshot['active_count']
            for component in range(4):
                reconstructed = math.fsum(item['count'] * item['rms4'][component] ** 2
                                          for item in bins) / snapshot['active_count']
                expected = snapshot['rms_H_Mcon_Zcon_Theta'][component] ** 2
                error = abs(reconstructed - expected) / max(expected, 1e-100)
                maximum_budget_reconstruction_error = max(maximum_budget_reconstruction_error,
                                                          error)
                assert error <= 2e-13
                for item in bins:
                    fraction = item['count'] * item['rms4'][component] ** 2
                    fraction /= snapshot['active_count'] * expected if expected else 1.0
                    close(fraction, item['squared_fraction4'][component], 2e-13)
        maximum_native_history_error = max(maximum_native_history_error,
                                          case['max_history_native_kernel_scaled_RMS_error'])
        assert case['min_saved_alpha'] > 0 and case['min_saved_chi'] > 0
        assert case['min_saved_conformal_metric_eigenvalue'] > 0
        root_receipt = strict_json(ARCHIVE / ('root-native-preflights/field-audits/'
                                             + name + '-001/receipt.json'))
        assert root_receipt['passed'] and root_receipt['pins_before_after_equal']
        assert root_receipt['arrays'] == len(snapshots)
        if 'reference' in name:
            assert len(snapshots) == 11
            assert case['max_saved_reference_full25_deviation'] <= 1e-10
            assert max(case['max_saved_H_Mcon_Zcon_Theta_physical']) <= 1e-9
            assert max(case['max_saved_det_error'], case['max_saved_trace_error']) < 1.10e-15
        else:
            assert len(snapshots) == 5
            assert max(case['max_saved_det_error'], case['max_saved_trace_error']) <= 1e-10
        row = {key: case[key] for key in ('case', 'N', 'saved_arrays', 'target_time',
               'final_H_Mcon_Zcon_Theta_physical', 'max_saved_H_Mcon_Zcon_Theta_physical',
               'max_saved_reference_full25_deviation', 'min_saved_alpha', 'min_saved_chi',
               'min_saved_conformal_metric_eigenvalue', 'max_saved_det_error',
               'max_saved_trace_error')}
        if 'large-short' in name:
            row['final_H_peak_radius'] = snapshots[-1]['constraint_max_radius4'][0]
            row['final_Z_squared_fraction_r_ge_095'] = snapshots[-1]['radial_bins'][-1]['squared_fraction4'][2]
        row['Omega_min'] = min(item['Omega_min'] for item in snapshots)
        row['nominal_pole_cap'] = .03 * row['Omega_min']
        row['minimum_cell_radius'] = math.sqrt(3.0) * 1.1 / case['N']
        assert row['minimum_cell_radius'] > .05
        rows.append(row)
    assert maximum_native_history_error <= 1.735e-18

    batch = strict_json(ARCHIVE / 'root-native-preflights/batch001/receipt.json')
    assert batch['passed_native_processes_and_provenance'] and batch['native_cases'] == 8
    builds = []
    for mode, executable in [
        ('wave-map', '829d590ee02e7865744ee9ea8e2bb00e036ee0dd9b7c2609da60740c15138b8c'),
        ('c0', '9297e0389dd48e65a5c650b5f7aa3ffb0d4b4c6c358fdd6e31044318ac62bade'),
        ('wave-map-half', '256527de2e7476392f15d1b7800e8a1372d1834f2e91a46d20b5a1b696869bee')]:
        receipt = strict_json(ARCHIVE / ('native-preparation/build-attempts/' + mode + '-001/receipt.json'))
        recipe = strict_json(ARCHIVE / ('native-preparation/recipes/' + mode + '/recipe.json'))
        assert receipt['passed_compile_link'] and receipt['base_private_inputs_and_dependencies_unchanged']
        assert receipt['source_before'] == receipt['source_after']
        assert receipt['launch_HEAD'] == 'a0f8fc8464665db3104fdbdaf142661259a6a399'
        assert receipt['executable_sha256'] == executable
        assert len(receipt['compiled_objects']) == len(recipe['compile_commands']) == 6
        assert len(receipt['commands']) == 7
        assert all(command['returncode'] == 0 for command in receipt['commands'])
        assert all(command['stderr_sha256'] == hashlib.sha256(b'').hexdigest()
                   for command in receipt['commands'])
        assert len(recipe['base_link_inputs']) == 186
        assert sum(path.endswith('.o') for path in recipe['base_link_inputs']) == 182
        assert recipe['old_spatialnorm_forced_include_removed']
        if mode != 'c0':
            assert recipe['exact_inverse_cartesian_patch_restores_public']
        builds.append({'mode': mode, 'executable_sha256': executable,
                       'rebuilt_translation_units': 6, 'reused_other_objects': 176,
                       'reused_libraries': 4})
    seam_rel = 'native-preparation/probe-attempts/compile-and-seam-001/receipt.json'
    seam = strict_json(ARCHIVE / seam_rel)
    assert sha(ARCHIVE / seam_rel) == '54a7f3f955c294bcc45307986ec33f4d7a2fb5e10d6626c89df94a68061a2b73'
    assert seam['passed_compile_and_fixed_t0_seam']
    assert seam['row_count'] == 18
    assert seam['max_scaled_error'] == seam['reference_rhs_abs'] == 0
    assert seam['omit_beta_negative_control_abs'] > 1e-8
    assert seam['duplicate_alpha_negative_control_abs'] > 1e-8
    inputs = list((ARCHIVE / 'native-preparation/inputs').glob('*.athinput'))
    assert len(inputs) == 17

    evidence_paths = [summary_rel, seam_rel,
        'native-preparation/PLAN.md', 'native-preparation/SEAM-AND-SNAPSHOT-PLAN.md',
        'native-preparation/analyze_snapshots.py', 'native-preparation/native_seam_and_snapshot.cpp',
        'native-preparation/recipes/wave-map/include/native_wave_map.hpp',
        'root-native-preflights/audit_saved_fields.py',
        'root-native-preflights/batch001/receipt.json', 'collection-receipt.json']
    pins = {str(ARCHIVE.relative_to(ROOT) / relative): sha(ARCHIVE / relative)
            for relative in evidence_paths}
    for relative in DOCS:
        path = ROOT / relative
        pins[relative] = sha(path)
        shutil.copyfile(path, OUT / path.name)
    doc = (ROOT / DOCS[0]).read_text()
    assert 'wormhole-to-trumpet' in doc and 'Minkowski reference' in doc
    assert 'barGammaInv=chi*gtildeInv' in doc and 'barGamma=gtilde/chi' in doc
    assert 'minimum eigenvalue of `gtilde`' in doc
    assert '10/alpha' in doc and 'no t2 evolution data' in doc
    assert '20%' in doc and '<=.8 times N24' in doc and '<=10%' in doc
    assert '1.25 times' in doc and 'absolute `1e-10`' in doc
    assert '304 finite JSON files' in doc and CATALOG_SHA in doc
    assert 't6 or t12 continuation is admitted here' in doc
    for relative in DOCS[1:]:
        assert 'hyperboloidal-reference-wave-map-native-audit.md' in (ROOT / relative).read_text()
    after = {str(path.relative_to(ARCHIVE)): sha(path) for path in archive_paths}
    assert before == after
    receipt = {'passed_saved_only_factual_review': True,
        'scope': 'Read-only source and saved compact JSON/scalar arithmetic review. No arrays, t2 outputs, '
                 'kernel queries, builds, evolution or omitted runtime payloads were opened or executed.',
        'disposition': 'No remaining factual or scientific-scope defect found after parent clarified '
                       'Penrose M/Z norms versus gtilde/Penrose eigenvalue diagnostics.',
        'archive_sha256': CATALOG_SHA, 'archive_files': 740, 'archive_bytes': 24095462,
        'finite_JSON_files': finite_json_count, 'archive_before_after_equal': before == after,
        'catalog_records_rehashed': len(catalog['files']), 'metadata_only_payloads': 278,
        'metadata_only_RST': 64, 'metadata_only_BIN': 192, 'metadata_only_compiled': 22,
        'original_inventory_entries_reported_by_collector': 1013,
        'external_dependency_identities_in_metadata': 1675,
        'external_runtime_rehash_repeated_by_this_review': False,
        'native_cases': 8, 'saved_binary64_array_results_reviewed': 64,
        'maximum_budget_reconstruction_relative_error': maximum_budget_reconstruction_error,
        'maximum_native_history_RMS_error': maximum_native_history_error,
        'builds': builds, 'case_facts': rows, 'prepared_inputs': 17,
        'pins': pins, 'review_source_sha256': sha(pathlib.Path(__file__)),
        'commands': ['python3 docs/validation/hyperboloidal-reference-wave-map-native-doc-review-20261009/review_saved.py'],
        'python_version': sys.version, 'new_scientific_queries': 0, 'native_steps': 0,
        't2_output_inspection': False, 'limitations': [
            'Array/kernel gates are reviewed as completed saved evidence, not repeated.',
            'Metadata-only runtime payloads are not distributed or replayed by this review.',
            'No finite-time acceptance, convergence order, exact-scri closure or black-hole gauge admission follows.']}
    (OUT / 'receipt.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'passed': True, 'reviewed_doc_sha256': pins[DOCS[0]],
                      'receipt_sha256': sha(OUT / 'receipt.json'),
                      'archive_files': 740, 'saved_array_result_records': 64,
                      'maximum_budget_relative_error': maximum_budget_reconstruction_error}, indent=2))


if __name__ == '__main__':
    main()
