#!/usr/bin/env python3
"""Require the actual, complete Aurora short-gate reports before common segments.

This reads archived evidence; it does not run or alter a solver or change gates.
A descriptor binds each report's path and SHA. Scientific acceptance is parsed
from its known schema/coverage/ranks, never from a descriptor's passed flag.
"""
import argparse
import hashlib
import json
import shlex
from pathlib import Path

DT = .00046944203735617445
NATIVE = 'f3a85db7528986ddec347b31675a8c5d588120a2380133aed73d5967f7aab05c'
ORIGINAL = '494ec8e38fb34806eb3981fb19bdb5e3bc75f46615b05a21d8314a0db6583753'
INITIAL = 'cfa18d09e96a52f1cf22fb724c907a097a0ca42925c623cdbbba22af2ebeae77'
OBSERVED = 'f2c2e6cd910279f0c558164466da8027f0e5b18c5bbd26bb4e36a853dd669f9d'
COUNTS = {'bulk': 229392, 'surface': 864}
LABELS = ['geometry', 'accepted0', 'accepted1', 'accepted8', 'first-rhs'] + [
    'stage'+str(stage)+'-'+phase for stage in range(1, 5) for phase in ('input', 'output')]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def fields(report, count, exact=False):
    require(report['comparisons'] == count*31 and report['failed_values'] == 0,
            'incomplete or failed31-field comparison')
    require(len(report['fields']) == 31 and all(f['comparisons'] == count and
            f['failed_values'] == 0 for f in report['fields']), 'per-field coverage differs')
    if exact:
        require(report['bitwise_equal'] is True, 'required bitwise fields differ')


def full_state(report, ranks):
    require(report['passed'] is True and report['original_binary_sha256'] == ORIGINAL,
            'failed or wrong immutable original control')
    p = report['original_plan']
    require(p['backend'] == 'SYCL' and p['ranks'] == ranks and p['nlim'] == 8 and
            p['bulk_points'] == 229392 and p['surface_points'] == 864,
            'wrong fullstate backend/ranks/workload')
    require(report['coordinates']['bulk_points'] == 229392 and
            report['coordinates']['exact_equal'] is True, 'actual bulk geometry gate failed')
    require(set(report['steps']) == {'0', '8'}, 'missing initial/eight snapshots')
    for step in ('0', '8'):
        row = report['steps'][step]
        require(row['passed'] is True and row['writer_mpi_ranks'] == ranks and row['dt'] == DT,
                'failed snapshot or wrong native writer decomposition')
        require(row['common_resolved_policy']['all_common_exact'] is True and
                row['chi_aggregates']['passed'] is True, 'policy/chi aggregate gate failed')
        for region, count in COUNTS.items():
            fields(row['fields'][region], count)


def cross_state(report, step):
    require(report['passed'] is True and report['step'] == step and report['dt'] == DT and
            report['same_backend_numerical_identity_exact'] is True, 'wrong cross-rank snapshot')
    require(report['reference']['ranks'] == 24 and report['candidate']['ranks'] == 96,
            'cross-rank test is not24→96')
    for region, count in COUNTS.items():
        fields(report['fields'][region], count, True)
        record = report['records'][region]
        require(record['comparisons'] == count*8 and record['bitwise_equal'] is True and
                record['different_values'] == 0, 'missing/different chi records')


def trace_metadata(report, ranks, mode):
    require(report['schema'] == 'native-stage-trace-v1' and report['passed'] is True and
            report['backend'] == 'SYCL' and report['ranks'] == ranks and report['mode'] == mode,
            'wrong native trace backend/ranks/mode')
    require(report['steps'] == 8 and report['dt'] == DT and report['bulk_points'] == 229392 and
            report['scri_points'] == 864 and len(report['stages']) == 32,
            'incomplete native trace workload/stages')
    for q, row in enumerate(report['stages']):
        require((row['cycle'], row['stage']) == (q//4, q%4+1), 'missing trace stage')


def trace_exact(report, ranks, mode):
    require(report['schema'] == 'native-stage-trace-audit-v1' and report['passed'] is True and
            report['backend'] == 'SYCL' and report['ranks'] == ranks and report['mode'] == mode and
            report['stage_records_exact'] is True, 'wrong/failed native trace exact comparison')
    expected = {label+'-'+region for label in LABELS for region in ('bulk', 'scri')}
    require(set(report['labels']) == expected and all(report['transition_checks'].values()) and
            len(report['transition_checks']) == 10, 'missing observer labels/transitions')
    for name, row in report['labels'].items():
        require(row['points'] == (229392 if name.endswith('-bulk') else 864) and
                row['comparison']['exact'] is True, 'incomplete/nonexact stage dump')


def matched_trace(report, ranks, legacy_exe):
    require(report['schema'] == 'legacy-native-stage-comparison-v1' and report['passed'] is True,
            'matched stage comparison failed')
    legacy = report['legacy']
    require(legacy['backend'] == 'SYCL' and legacy['ranks'] == ranks and legacy['steps'] == 8,
            'wrong matched legacy backend/ranks/steps')
    require(report['legacy_build']['executable_sha256'] == legacy_exe and
            report['legacy_build']['original_executable_sha256'] == ORIGINAL and
            report['accessor_admission']['passed'] is True, 'unadmitted old accessor')
    require(report['one_control'] is not None, 'immutable one-step control missing')
    for region, count in [('bulk', 229392), ('scri', 864)]:
        fields(report['accessor_admission']['accepted1-vs-control-final-'+region], count)
    require(len(report['comparisons']) == 2 and {x['mode'] for x in report['comparisons']} ==
            {'clean', 'poison'}, 'both clean/poison matched comparisons required')
    for row in report['comparisons']:
        require(row['passed'] is True and row['ranks'] == ranks and
                len(row['step_clocks']) == 8 and all(x['passed'] for x in row['step_clocks']),
                'matched stage clock gate failed')
        require(all(row['transitions'].values()) and len(row['transitions']) == 10,
                'matched stage transitions incomplete')
        for region, count in [('bulk', 229392), ('scri', 864)]:
            require(row['geometry'][region]['bitwise_equal'] is True, 'actual geometry differs')
            for label in ('accepted0', 'accepted1', 'accepted8', 'first-rhs'):
                item = row['fields'][label+'-'+region]
                require(item['passed'] is True, 'failed matched state/RHS label')
                fields(item['fields'], count)
                if label == 'accepted1':
                    fields(item['immutable_one_step_fields'], count)
                if label != 'first-rhs':
                    require(item['records']['failed_values'] == 0 and
                            item['records']['comparisons'] == count*8 and
                            len(item['records']['fields']) == 8 and all(
                                f['comparisons'] == count and f['failed_values'] == 0
                                for f in item['records']['fields']), 'matched chi records failed/incomplete')
    require(len(report['native_trace_admission']) == 2 and all(
        x['complete_labels'] == 26 and x['stages'] == 32 and x['all_owned_values_finite'] is True and
        x['complete_unique_integer_keys'] is True for x in report['native_trace_admission']),
        'native trace finite/unique coverage missing')


def geometry_binding(report):
    require(report['schema'] == 'native-n12-aurora-map-binding-v1' and report['passed'] is True,
            'missing admitted actual Aurora mapping')
    require(report['native_executable_sha256'] == NATIVE and
            report['helper_executable_sha256'] == 'a14333e4829718a7903efcabaf7ab523713cdf40ea0f0f5d7a772ea95004a8a7' and
            report['helper_exit_status'] == 0 and report['helper_ranks'] == 1,
            'unadmitted or failed mapper executable/run')
    require(report['template_checkpoint_sha256'] == INITIAL and report['template_cycle'] == 0 and
            report['template_time'] == 0, 'mapping bound to different initial checkpoint')
    require(report['bulk_points'] == 229392 and report['scri_points'] == 864 and
            report['bulk_coordinate_comparisons'] == 688176 and
            report['scri_coordinate_comparisons'] == 2592 and
            report['bulk_coordinate_mismatches'] == report['scri_coordinate_mismatches'] == 0,
            'actual bulk/scri exact geometry coverage differs')
    # Raw independent host constructor state precedes algebraic/scri projection.
    # Its extra comparison uses the unchanged short gate; signed-zero/roundoff
    # differences are retained explicitly and are not a geometry relaxation.
    require(report['independent_initial_fields_passed_adopted_short_gates'] is True,
            'independent constructor short gate failed')
    for region, count in COUNTS.items():
        row = report['initial_constructor_comparison'][region]
        require(row['comparisons'] == count*31 and row['failed_values'] == 0 and
                row['max_tolerance_ratio'] <= 1, 'independent constructor coverage/short gate differs')
    require(report['cpu_parent_sidecar_preserved'] is True, 'CPU sidecar overwritten')


def observed_main(report, build, build_sha):
    """Require complete actual-GPU observer/restart qualification, not a CPU stub."""
    require(build['schema'] == 'native-step-observer-build-v1' and build['status'] == 'built' and
            build['build_exit_code'] == 0 and build['runtime_inputs_unchanged'] is True and
            build['original_main_fp_flags_preserved'] is True and
            build['overlay']['inverse_byte_exact'] is True and
            build['native_executable_sha256'] == NATIVE and
            build['executable_sha256'] == OBSERVED, 'unqualified observed original main build')
    require(report['schema'] == 'native-step-observer-gpu-comparison-v1' and
            report['passed'] is True and report['native_executable_sha256'] == NATIVE and
            report['observed_executable_sha256'] == OBSERVED and
            report['observer_build_provenance_sha256'] == build_sha and
            build_sha in report['input_sha256'].values(), 'wrong or failed observed GPU comparison')
    settings = [('observed_24',24,0,8), ('observed_96',96,0,8),
                ('prefix_24',24,0,2), ('resumed_96',96,2,8)]
    values = 106*(18**3+144)*39
    def payload(row):
        require(row['bitwise_equal'] is True and row['scalar_values'] == values,
                'incomplete/nonexact observed checkpoint payload')
    for name,ranks,first,last in settings:
        meta = report['stage_metadata'][name]
        require(meta['schema'] == 'native-step-observer-v1' and meta['passed'] is True and
                meta['backend'] == 'SYCL' and meta['ranks'] == ranks and
                (meta['initial_cycle'],meta['final_cycle']) == (first,last) and
                meta['dt'] == DT and meta['stage_rows'] == 4*(last-first),
                'wrong observed backend/decomposition/stage coverage: '+name)
        checkpoints = report['checkpoints'][name]
        require({x['cycle'] for x in checkpoints} == ({0,last} if first == 0 else {last}) and
                all(x['ranks'] == ranks and x['scalar_values'] == values for x in checkpoints),
                'missing observed accepted checkpoint coverage: '+name)
    for ranks in (24,96):
        old = report['legacy_controls'][str(ranks)]
        require(old['executable_sha256'] == ORIGINAL and old['ranks'] == ranks and
                old['backend'] == 'SYCL' and old['steps'] == 8 and
                old['successful_step_limit'] is True and
                old['input_sha256'] == '3d0ce81d74e7b470272fc938802d505e9b53ec4ab1e3a011c6f3fad96bb74ac6' and
                old['configuration_sha256'] == '9c786cc8c3312ddf93389a4b3f1f6e32921811a61f4d131ee53990fb80a1d363',
                'wrong immutable original clock/CFL control')
        equivalent = report['ordinary_native_equivalence'][str(ranks)]
        require(set(equivalent['states']) == {'0','8'}, 'missing observed/ordinary snapshots')
        for row in equivalent['states'].values(): payload(row)
        diagnostics = equivalent['diagnostics']
        require(diagnostics['passed'] is True and diagnostics['files'] and
                all(x['bitwise_equal'] is True for x in diagnostics['files']) and
                any(x['name'].endswith('.hst') for x in diagnostics['files']) and
                any('.horizon' in x['name'] for x in diagnostics['files']),
                'ordinary/observed diagnostics missing or changed')
        metric = report['historical_metrics'][str(ranks)]
        require(metric['schema'] == 'native-step-metrics-v1' and metric['passed'] is True and
                metric['ranks'] == ranks and metric['gate'] == 'short' and
                metric['through_step'] == 8 and metric['stage_rows'] == 32 and
                metric['failed_values'] == 0 and metric['completed_steps'] == list(range(1,9)) and
                metric['exact_step_stage_time_dt'] is True and
                metric['maximum_compatibility'] <= 1e-7 and len(metric['steps']) == 8 and
                all(x[field]['passed'] is True for x in metric['steps'] for field in ('clock','CFL')),
                'incomplete/failed original clock or CFL comparison')
    require(set(report['cross_rank_state']) == {'0','8'}, 'missing cross-rank observer states')
    for row in report['cross_rank_state'].values(): payload(row)
    payload(report['prefix_initial_state']);payload(report['restart_state'])
    require(report['cross_rank_stage_rows_exact'] is True and
            report['restart_stage_rows_exact'] is True and
            report['restart_diagnostics']['passed'] is True and
            {x['field'] for x in report['restart_diagnostics']['comparisons']} ==
            {'Volume','H','M','Theta','Z','eVolume','eH','eM','eTheta','eZ',
             'areal_radius_M','irreducible_mass_M'} and
            all(x['passed'] is True for x in report['restart_diagnostics']['comparisons']),
            'rank-changing restart stage/state/diagnostic qualification failed')


def validate(path, reference):
    admission = json.loads(Path(path).read_text())
    require(admission['schema'] == 'native-n12-longrun-admission-v1', 'unknown admission schema')
    initial = admission['initial_reference']
    require(Path(initial['path']) == Path(reference) and initial['sha256'] == INITIAL and
            sha(reference) == INITIAL, 'initial reference path/hash differs')
    def artifact(name):
        descriptor = admission['reports'][name]
        p = Path(descriptor['path'])
        require(sha(p) == descriptor['sha256'], 'changed report: '+name)
        return p
    def read(name):
        return json.loads(artifact(name).read_text())
    def command(name, executable, ranks, mode=None):
        tokens = shlex.split(artifact(name).read_text())
        require(tokens.count(executable) == 1 and tokens.count('-n') == 1 and
                int(tokens[tokens.index('-n')+1]) == ranks, 'wrong observed executable/ranks: '+name)
        if mode is not None:
            require(tokens[-1] == mode, 'wrong observed mode: '+name)
    geometry = read('aurora_geometry_binding');geometry_binding(geometry)
    manifest = read('aurora_mapping_manifest')
    require(manifest['schema'] == 'native-n12-legacy-map-v1' and
            manifest['mapping_sha256'] == geometry['mapping_sha256'] and
            manifest['physics_identity'] == geometry['physics_identity'] and
            admission['reports']['aurora_mapping_manifest']['sha256'] == geometry['manifest_sha256'],
            'mapping manifest differs from exact Aurora binding')
    native_build = read('native_trace_build')
    require(native_build['schema'] == 'native-stage-trace-build-v1' and
            native_build['status'] == 'built' and native_build['build_exit_code'] == 0 and
            native_build['native_executable_sha256'] == NATIVE and
            native_build['runtime_inputs_unchanged'] is True, 'wrong native runtime/observer build')
    require(sha(native_build['native_executable']) == NATIVE and
            sha(native_build['executable']) == native_build['executable_sha256'],
            'native or observer executable changed')
    old_build = read('legacy_trace_build')
    require(old_build['schema'] == 'original-stage-accessor-build-v1' and old_build['status'] == 'built' and
            old_build['build_exit_code'] == 0 and old_build['original_executable_sha256'] == ORIGINAL and
            old_build['original_runtime_inputs_unchanged'] is True, 'wrong predecessor accessor build')
    require(sha(old_build['original_executable']) == ORIGINAL and
            sha(old_build['executable']) == old_build['executable_sha256'], 'old executable changed')
    observer_build = read('step_observer_build')
    observer_comparison = read('step_observer_gpu_comparison')
    observed_main(observer_comparison, observer_build,
                  admission['reports']['step_observer_build']['sha256'])
    require(sha(observer_build['executable']) == OBSERVED, 'observed original main changed')
    for name,ranks in [('observed_24',24),('observed_96',96),('prefix_24',24),('resumed_96',96)]:
        key = 'step_observer_command_'+name
        command(key, observer_build['executable'], ranks)
        tokens = shlex.split(artifact(key).read_text())
        require('-t' not in tokens and ('-r' in tokens) == (name == 'resumed_96') and
                ('-i' in tokens) == (name != 'resumed_96'), 'observed entry path/limits changed')
        require(admission['reports'][key]['sha256'] in observer_comparison['input_sha256'].values(),
                'observed command not bound to scientific comparison')
    for ranks in (24, 96):
        full_state(read('native_vs_original_'+str(ranks)), ranks)
        replay = read('saved_replay_'+str(ranks))
        require(replay['passed'] is True and replay['strict_replay_passed'] is True and
                replay['backend'] == 'SYCL' and replay['ranks'] == ranks and
                replay['step'] == 3551 and replay['readonly'] is True and
                replay['mask_count_names']['all'] == 229392 and
                replay['mask_count_names']['invalid_any'] == 0 and
                len(replay['dump_files_sha256']) == ranks, 'strict saved replay missing')
        clean = read('trace_clean_metadata_'+str(ranks));poison = read('trace_poison_metadata_'+str(ranks))
        trace_metadata(clean, ranks, 'clean');trace_metadata(poison, ranks, 'poison')
        for mode in ('clean', 'poison'):
            command('trace_'+mode+'_command_'+str(ranks), native_build['executable'], ranks, mode)
        command('legacy_trace_command_'+str(ranks), old_build['executable'], ranks)
        audit = read('trace_poison_'+str(ranks));trace_exact(audit, ranks, 'poison')
        require(audit['result_sha256'] == admission['reports']['trace_poison_metadata_'+str(ranks)]['sha256'] and
                audit['reference_result_sha256'] == admission['reports']['trace_clean_metadata_'+str(ranks)]['sha256'],
                'clean/poison report metadata binding differs')
        matched = read('matched_trace_'+str(ranks))
        matched_trace(matched, ranks, old_build['executable_sha256'])
        for mode in ('clean', 'poison'):
            require(admission['reports']['trace_'+mode+'_metadata_'+str(ranks)]['sha256'] in
                    matched['input_sha256'].values(), 'matched report bound to different native metadata')
    cross_state(read('cross_state_initial'), 0);cross_state(read('cross_state_eight'), 8)
    cross = read('cross_trace');trace_exact(cross, 96, 'clean')
    require(cross['result_sha256'] == admission['reports']['trace_clean_metadata_96']['sha256'] and
            cross['reference_result_sha256'] == admission['reports']['trace_clean_metadata_24']['sha256'],
            'cross-rank trace metadata binding differs')
    return dict(passed=True, native_executable_sha256=NATIVE, observed_executable_sha256=OBSERVED,
                observed_executable=observer_build['executable'], initial_checkpoint_sha256=INITIAL,
                source_reports=len(admission['reports']), scope='complete actual Aurora24/96 short gates')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('admission', type=Path);p.add_argument('--reference', type=Path, required=True)
    args = p.parse_args();print(json.dumps(validate(args.admission, args.reference), indent=2))
