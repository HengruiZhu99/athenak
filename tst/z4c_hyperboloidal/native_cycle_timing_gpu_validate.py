#!/usr/bin/env python3
"""Read the fixed step3551 GPU timing campaign; never build or launch work.

Four runs are required: frozen normal and timed executables at24/96ranks,
all from the same admitted checkpoint, with10 warmup plus3 windows of40.
Only complete scientific and execution success exposes accepted timings.
The existing CPU validator and qualified timing sources remain unchanged.
"""
import argparse
import json
import math
from pathlib import Path
import re
import shlex
import statistics
import struct

import numpy as np

from native_cycle_timing import reverse_transform
from native_step_observer_gpu_compare import checkpoint_set, diagnostics_exact
from n12_checkpoint import DT, expected_time, parse_native, require, sha256
from n12_long_history import LABELS, RMS, table
from restart_roundtrip import compare

SIDE = Path('/lus/flare/projects/CompactBinaryMerger/hzhu/scri-native-20261108-side')
BUILD = SIDE/'validation/cycle-timing/build'
NORMAL = 'f3a85db7528986ddec347b31675a8c5d588120a2380133aed73d5967f7aab05c'
TIMED = 'e47888006e1221d5e14bce734a97d77a45126b271f7ef39c2ceb9e9b31c0f6cf'
PROVENANCE = '89265de52cfec416139ddf4188618b10b91b4e130c3fa844ffeed473d0d492ea'
SOURCE_MANIFEST = '4706aae696919253201da4cac9dd14bff52743ee8826dc93660a382f255ec9a5'
OBJECT_MANIFEST = '462da52c2e152e9c94818a534314aec72f2a827f30decdb2d23a3559e6954547'
WRAPPER = '/opt/aurora/26.26.0/support/tools/mpi_wrapper_utils/gpu_tile_compact.sh'
WRAPPER_SHA = '4469d67ffee9450a7e09b96e291f422360c3ac04c151cd7dde22e8e129979dbd'
BINDING = '--cpu-bind=list:1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
VERIFY_TEXT = 'Verified invertible original-driver timing and unchanged native source/object/executable inputs\n'
FIRST, LAST, CONFIG = 3551, 3681, (10, 40, 3)
COMMON = [0, 356, 711, 1066, 1421, 1776, 2131, 2486, 2841, 3196, 3551]
PROFILE = ('KOKKOS_TOOLS_LIBS', 'KOKKOS_PROFILE_LIBRARY', 'KOKKOS_TOOLS_ARGS',
           'SYCL_PI_TRACE', 'SYCL_UR_TRACE', 'ZE_ENABLE_TRACING_LAYER',
           'UNITRACE_ENABLE', 'PTI_ENABLE', 'VTUNE_PROFILER_ENABLE')


def read_json(path):
    return json.loads(Path(path).read_text())


def check_files(root, records, hashes):
    """Verify collected bytes, including every checkpoint and terminal record."""
    root = root.resolve()
    for name, record in records.items():
        path = (root/name).resolve()
        require(path.is_relative_to(root) and path.is_file(), 'missing/escaping artifact: '+name)
        require(path.stat().st_size == record['size'] and sha256(path) == record['sha256'],
                'changed collected artifact: '+name)
        hashes[str(path)] = record['sha256']


def check_admission(admission, review, checkpoint_sha):
    require(admission['schema'] == 'native-n12-common-science-admission-v1' and
            admission['passed'] is True and admission['segment'] == 10 and
            admission['end_cycle'] == FIRST and admission['through_5m_complete'] is True and
            admission['missing_future_common_steps'] == [], 'full3551 science admission missing')
    require(admission['checkpoint_sha256'] == checkpoint_sha and
            admission['comparisons'][str(FIRST)]['passed'] is True and
            admission['comparisons'][str(FIRST)]['candidate_sha256'] == checkpoint_sha,
            'science admission binds another checkpoint')
    history, stages = admission['history'], admission['stage_metrics']
    require(admission['history_completed_prefix_passed'] is True and history['passed'] is True and
            history['common_steps'] == COMMON and set(history['times']) == set(map(str, COMMON)) and
            history['failures'] == [],
            'full3551 history/horizon gate missing')
    require(stages['passed'] is True and stages['through_step'] == FIRST and
            stages['full3551_complete'] is True and stages['stage_rows'] == 4*FIRST and
            stages['exact_step_stage_time_dt'] is True and stages['failed_values'] == 0,
            'full3551 stage gate missing')
    require(review['schema'] == 'native-n12-common-independent-review-v1' and
            review['passed'] is True and review['end_cycle'] == FIRST and
            review['through_5m_complete'] is True and review['checkpoint_sha256'] == checkpoint_sha and
            review['stage_rows'] == 4*FIRST and review['exact_stage_clocks'] is True and
            review['history_completed_prefix_passed'] is True and review['history_times'] == COMMON and
            review['harmonic_coefficients_each_time'] == 169 and
            review['missing_future_common_steps'] == [], 'full3551 independent review missing')


def check_source(info):
    require((info['cycle'], info['time'], info['dt']) == (FIRST, expected_time(FIRST), DT),
            'source checkpoint is not the exact3551 state')
    p = info['parameters']
    require(info['metadata'][4:] == (24, 0, 4, 0), 'source was not written by24ranks')
    require(np.array_equal(np.frombuffer(info['costs'], dtype='<f4'), np.ones(106, dtype='<f4')),
            'source block weights changed')
    clock = sum([1/6]*10)
    for block in ('output1', 'output2'):
        require(float(p[block+'/dt']) == 1/6 and float(p[block+'/last_time']) == clock and
                int(p.get(block+'/dcycle', '0')) <= 0, 'source output schedule changed')
    require(int(p['output2/file_number']) == 11 and p['output1/file_type'] == 'hst' and
            p['output2/file_type'] == 'rst' and p['output2/single_file_per_rank'] == 'false',
            'source output types/counter changed')
    require(p['time/integrator'] == 'rk4_classical' and p['time/evolution'] == 'dynamic' and
            p['mesh_refinement/refinement'] == 'static' and float(p['time/dt']) == DT and
            float(p['z4c_hyperboloidal/ko']) == .1 and float(p['z4c_hyperboloidal/chi_floor']) == 1e-10,
            'source numerical policy changed')
    # Match the actual Driver float32 predicate; final outputs are separate.
    for cycle in range(FIRST+1, LAST+1):
        for block in ('output1', 'output2'):
            due = (np.float32(expected_time(cycle)) >= np.float32(clock+1/6) and
                   np.float32(expected_time(cycle)) < np.float32(2))
            require(not due, 'prospective timing run has scheduled output')
    require(expected_time(LAST) < 2, 'insufficient time limit')


def check_build(folder, hashes):
    manifest = read_json(folder/'artifact-manifest.json')
    require(manifest['schema'] == 'native-cycle-timing-aurora-build-artifacts-v1',
            'unknown timing build collection')
    check_files(folder, manifest['files'], hashes)
    proof_path = folder/'build/provenance.json'
    require(sha256(proof_path) == PROVENANCE, 'not the qualified e478 timing build')
    proof = read_json(proof_path)
    require(proof['status'] == 'built' and proof['build_exit_code'] == 0 and
            proof['native_executable_sha256'] == NORMAL and proof['executable_sha256'] == TIMED and
            proof['runtime_inputs_unchanged'] is True and proof['original_driver_fp_flags_preserved'] is True,
            'failed/changed timing build')
    for name, digest in [('build/driver-original.cpp', proof['overlay']['original_sha256']),
                         ('build/driver-timed.cpp', proof['overlay']['transformed_sha256']),
                         ('baseline/compile_commands.json', proof['compile_database_sha256']),
                         ('baseline/driver-link.txt', proof['link_file_sha256'])]:
        require(str((folder/name).resolve()) in hashes and sha256(folder/name) == digest,
                'unbound/changed timing source or recipe: '+name)
    require(len(proof['overlay']['additions']) == 11 and
            reverse_transform((folder/'build/driver-timed.cpp').read_bytes(), proof['overlay']['additions']) ==
            (folder/'build/driver-original.cpp').read_bytes(), 'driver insertion proof differs')
    for name in ('native_cycle_timing.py', 'native_cycle_timing.hpp', 'native_cycle_timing_contract.md'):
        require((folder/'package'/name).read_bytes() == Path(__file__).with_name(name).read_bytes(),
                'qualified timing source changed: '+name)
    hashes[str((folder/'artifact-manifest.json').resolve())] = sha256(folder/'artifact-manifest.json')
    return dict(provenance_sha256=PROVENANCE, normal_sha256=NORMAL, timed_sha256=TIMED)


def check_guard(folder, suffix, checkpoint_sha, admission_sha, review_sha, source_remote):
    guard = read_json(folder/f'guard-{suffix}.json')
    expected = dict(schema='native-cycle-timing-runtime-guard-v1', passed=True,
        baseline_executable_sha256=NORMAL, timing_executable_sha256=TIMED,
        timing_build_provenance_sha256=PROVENANCE, source_checkpoint_sha256=checkpoint_sha,
        science_admission_sha256=admission_sha, independent_review_sha256=review_sha,
        source_checkpoint_remote=source_remote,
        source_manifest_sha256=SOURCE_MANIFEST, object_manifest_sha256=OBJECT_MANIFEST,
        verified_source_files=1972, verified_objects=228, gpu_wrapper_sha256=WRAPPER_SHA,
        timing_verify_exit_code=0)
    require(all(guard.get(k) == v for k, v in expected.items()), 'runtime guard differs: '+suffix)
    log = folder/f'timing-verify-{suffix}.txt'
    require(log.read_text() == VERIFY_TEXT and sha256(log) == guard['timing_verify_log_sha256'],
            'actual qualified-builder verify output missing: '+suffix)


def check_environment(environment, remote_directory):
    expected = dict(OMP_NUM_THREADS='1', ZE_FLAT_DEVICE_HIERARCHY='COMPOSITE',
        ZE_ENABLE_PCI_ID_DEVICE_ORDER='1', ONEAPI_DEVICE_SELECTOR='level_zero:gpu',
        MPIR_CVAR_ENABLE_GPU='1', ZE_AFFINITY_MASK=None,
        NEO_CACHE_DIR=str(remote_directory/'cache/neo'), SYCL_CACHE_DIR=str(remote_directory/'cache/sycl'),
        NATIVE_TIMING_WARMUP='10', NATIVE_TIMING_STEPS='40', NATIVE_TIMING_WINDOWS='3')
    expected.update({key: None for key in PROFILE})
    require(environment == expected, 'launch environment/profiling/cache binding differs')


def check_command(command, ranks, timed, source_remote, remote_directory):
    executable = BUILD/'athena-cycle-timing' if timed else SIDE/'build-sycl/src/athena'
    require(command == ['mpiexec', '-n', str(ranks), '-ppn', '12', BINDING, WRAPPER, str(executable),
            '-r', source_remote, '-d', str(remote_directory), 'time/nlim=3681', 'time/tlim=2',
            'time/ndiag=1000000000'], 'executable/restart/options/binding command differs')


def timing_windows(r, first, ranks, config, backend, restart):
    """Existing CPU window assertions, with explicit arguments for fixture reuse.

    The public GPU audit supplies only3551/24|96/(10,40,3)/SYCL/restart=True.
    This function does not grant execution or scientific admission.
    """
    warmup, steps, windows = config
    last = first+warmup+steps*windows
    require(r['schema'] == 'native-cycle-timing-v1' and r['passed'] is True and
            r['backend'] == backend and r['restart'] is restart, 'timing report/backend/entry differs')
    require((r['initial_cycle'], r['final_cycle'], r['complete_steps'], r['ranks']) ==
            (first, last, last-first, ranks), 'timing coverage differs')
    require((r['initial_time'], r['final_time'], r['dt']) ==
            (expected_time(first), expected_time(last), DT), 'timing clocks differ')
    require((r['warmup_steps'], r['steps_per_window'], r['window_count']) == config and
            len(r['windows']) == windows, 'timing windows missing/changed')
    require((r['bulk_points'], r['scri_points'], r['blocks']) == (229392, 864, 106),
            'timing workload differs')
    require(r['console_ndiag'] == 1000000000 and r['profiling_library_loaded'] is False and
            r['profiling_environment'] == [] and r['timed_output_count'] == r['timed_console_count'] == 0,
            'timing contains profiling/output')
    for index, w in enumerate(r['windows']):
        start = first+warmup+index*steps
        require((w['index'], w['first_cycle'], w['last_cycle'], w['complete_steps']) ==
                (index, start, start+steps, steps) and
                (w['first_time'], w['last_time']) == (expected_time(start), expected_time(start+steps)),
                'partial/missing/repeated complete-cycle window')
        elapsed = w['max_rank_seconds']
        require(math.isfinite(elapsed) and elapsed > 0 and
                math.isfinite(w['seconds_per_complete_step']) and w['seconds_per_complete_step'] > 0 and
                math.isfinite(w['bulk_cell_updates_per_second']) and
                w['seconds_per_complete_step'] == elapsed/steps and
                w['bulk_cell_updates'] == 229392*steps and
                w['bulk_cell_updates_per_second'] == 229392*steps/elapsed,
                'invalid window rates/counts; stages are not complete cell updates')
    for key in ('startup_max_seconds', 'warmup_max_seconds', 'execute_max_seconds', 'finalize_max_seconds'):
        require(math.isfinite(r[key]) and r[key] > 0, 'invalid separated timing: '+key)
    return r


def check_final(info, source):
    require(info['fixed'][:120] == source['fixed'][:120] and
            info['fixed'][156:-20] == source['fixed'][156:-20], 'initialized native mesh header differs')
    for key in ('atlas', 'topology', 'costs', 'identity'):
        require(info[key] == source[key], 'checkpoint metadata differs: '+key)
    p, original = info['parameters'], source['parameters']
    mutable = {'time/nlim', 'time/tlim', 'time/ndiag', 'output1/last_time', 'output2/last_time',
               'output2/file_number', 'fastflow/last_a0_0', 'fastflow/ah_found_a0_0', 'fastflow/time_first_found_0'}
    require(p.keys() == original.keys(), 'checkpoint parameter set differs')
    require(all(p[k] == original[k] for k in p if k not in mutable), 'numerical/output policy differs')
    require((int(p['time/nlim']), float(p['time/tlim']), int(p['time/ndiag'])) ==
            (LAST, 2., 1000000000), 'final stopping controls differ')
    require(int(p['output2/file_number']) == 12 and
            all(float(p[b+'/last_time']) == sum([1/6]*11) for b in ('output1', 'output2')),
            'missing/extra final output cadence')
    require(p['fastflow/ah_found_a0_0'] == '1' and math.isfinite(float(p['fastflow/last_a0_0'])) and
            float(p['fastflow/last_a0_0']) > 0 and p['fastflow/time_first_found_0'] == original['fastflow/time_first_found_0'],
            'horizon state missing/changed')


def check_diagnostics(folder, cycles):
    """Reuse ordinary table checks without the long-history reader's3551 cap."""
    stem = 'native_n12_5M'
    expected = {stem+'.z4c_hyperboloidal.hst', stem+'.horizon_summary_0.txt', stem+'.horizon_shape_0.txt'}
    actual = {p.name for pattern in ('*.hst', '*.horizon*.txt') for p in folder.glob(pattern)}
    require(actual == expected, 'missing/extra ordinary history or complete horizon files')
    history = folder/(stem+'.z4c_hyperboloidal.hst')
    headers = [line for line in history.read_text().splitlines() if line.startswith('#')]
    require(len(headers) == 2 and re.findall(r'\[\d+\]=(\S+)', headers[1]) == LABELS,
            'history column schema differs')
    rows = table(history); summary = table(folder/(stem+'.horizon_summary_0.txt'))
    lines = [line.strip() for line in (folder/(stem+'.horizon_shape_0.txt')).read_text().splitlines() if line.strip()]
    require(len(rows) == len(summary) == len(cycles) and len(lines) == 2*len(cycles),
            'missing/extra final diagnostic rows')
    for index, cycle in enumerate(cycles):
        row = rows[index]['values']; horizon = summary[index]['values']
        require(len(row) == 12 and (row[0], row[1]) == (expected_time(cycle), DT) and
                0 < row[7] <= row[2] and all(row[i] >= 0 for _, i in RMS), 'invalid history clock/volume/RMS')
        printed = float(format(expected_time(cycle), '.6g'))
        require(len(horizon) == 12 and (horizon[0], horizon[1]) == (cycle, printed) and
                all(horizon[i] > 0 for i in (2, 7, 10, 11)) and horizon[8] >= 0,
                'invalid horizon clock/properties')
        frame = re.fullmatch(r'# iter = (\d+), Time = (\S+)', lines[2*index])
        require(frame is not None and (int(frame[1]), float(frame[2])) == (cycle, printed),
                'horizon shape clock differs')
        coefficients = [float(x) for x in lines[2*index+1].split()]
        require(len(coefficients) == 169 and all(math.isfinite(x) for x in coefficients),
                'incomplete/nonfinite169-coefficient horizon shape')
    return dict(cycles=list(cycles), files=sorted(expected), harmonic_coefficients_per_row=169,
                full_harmonic_mask_scope='unchanged guarded native diagnostic objects')


def check_terminal(package, text, actual_job_id, nodes):
    job = re.search(r'^Job Id:\s*(\S+)', text, re.M)
    require(job and job[1].split('.')[0] == str(package['job_id']).split('.')[0] ==
            actual_job_id.strip().split('.')[0], 'terminal/actual/package PBS job differs')
    expected_name = 'native-timing-two' if nodes == 2 else 'native-timing-eight'
    for key, expected in (('job_state', 'F'), ('Exit_status', '0'),
                          ('Job_Name', expected_name), ('Resource_List.nodect', str(nodes))):
        match = re.search(r'^\s*'+re.escape(key)+r'\s*=\s*(\S+)\s*$', text, re.M)
        require(match is not None and match[1] == expected, 'terminal PBS field differs: '+key)


def collection(folder, nodes, source_sha, admission_sha, review_sha, hashes):
    package = read_json(folder/'package.json'); root = folder/'run-artifacts'
    require(package['schema'] == 'native-cycle-timing-collection-v1' and
            (package['nodes'], package['ranks'], package['job_state'], package['exit_status']) ==
            (nodes, 12*nodes, 'F', 0), 'not a terminal successful GPU collection')
    require(package['source_checkpoint_sha256'] == source_sha and
            package['science_admission_sha256'] == admission_sha and
            package['independent_review_sha256'] == review_sha,
            'collection admission/source bindings differ')
    manifest = read_json(root/'files.json')
    require(len(manifest) == package['files'], 'collection file count differs')
    check_files(root, manifest, hashes)
    # Every consumed case file must be covered, not just selected successful files.
    pair = root/f'runs/cycle-timing-nodes{nodes}'
    terminal = root/'terminal-qstat.txt'
    check_terminal(package, terminal.read_text(), (pair/'job-id.txt').read_text(), nodes)
    for path in [terminal, *pair.rglob('*')]:
        if path.is_file() and 'cache' not in path.relative_to(root).parts:
            require(str(path.resolve()) in hashes, 'consumed file missing from collection manifest')
    for suffix in ('before', 'after'):
        check_guard(pair, suffix, source_sha, admission_sha, review_sha, package['source_checkpoint_remote'])
    owners = {}
    for line in (pair/'binding.txt').read_text().splitlines():
        m = re.fullmatch(r'(\S+) local_rank=(\d+) tile=(\d+\.\d+) implicit_scaling=0', line)
        require(m is not None, 'invalid GPU binding line')
        host, rank, tile = m.groups(); rank = int(rank)
        require((host, rank) not in owners and tile == f'{rank//2}.{rank%2}', 'duplicate/wrong GPU tile')
        owners[host, rank] = tile
    hosts = {h for h, r in owners}
    require(len(hosts) == nodes and all({r for h, r in owners if h == host} == set(range(12)) for host in hosts),
            'GPU node/rank coverage differs')
    for path in (folder/'package.json', root/'files.json'):
        hashes[str(path.resolve())] = sha256(path)
    return package, pair


def audit(args, result):
    hashes = result['input_sha256']
    for path in (args.source_checkpoint, args.admission, args.independent_review):
        hashes[str(path.resolve())] = sha256(path)
    source_sha = sha256(args.source_checkpoint)
    admission, review = read_json(args.admission), read_json(args.independent_review)
    check_admission(admission, review, source_sha)
    require(review['science_admission_sha256'] == sha256(args.admission), 'independent review binds another science report')
    source = parse_native(args.source_checkpoint); check_source(source)
    result['unused_global_coarse_indices']['source'] = list(struct.unpack('<9i', source['fixed'][120:156]))
    result['build'] = check_build(args.build_artifacts, hashes)
    result['source_checkpoint_sha256'] = source_sha
    result['source_scientifically_admitted'] = True
    states, reports, parameters = {}, {}, {}
    remote_source = None; jobs = set()
    for nodes, folder in ((2, args.two), (8, args.eight)):
        ranks = 12*nodes
        package, pair = collection(folder, nodes, source_sha, sha256(args.admission),
                                   sha256(args.independent_review), hashes)
        job_id = str(package['job_id']).split('.')[0]
        require(job_id not in jobs, 'two/eight collections claim the same allocation')
        jobs.add(job_id)
        require(Path(package['source_checkpoint_remote']).is_absolute(), 'restart path is not absolute')
        if remote_source is None: remote_source = package['source_checkpoint_remote']
        require(remote_source == package['source_checkpoint_remote'], 'rank decompositions used different source paths')
        for kind in ('normal', 'timed'):
            run = pair/kind; remote = SIDE/f'runs/cycle-timing-nodes{nodes}'/kind
            check_command(shlex.split((run/'command.txt').read_text()), ranks, kind == 'timed', remote_source, remote)
            check_environment(read_json(run/'environment.json'), remote)
            require((run/'exit-code.txt').read_text().strip() == '0' and
                    'Terminating on cycle limit' in (run/'run.log').read_text(), 'solver did not finish successfully')
            require([p.name for p in (run/'rst').glob('*.rst')] == ['native_n12_5M.00011.rst'],
                    'restart run must produce only its final checkpoint')
            found, records = checkpoint_set(run, {LAST}, ranks, hashes)
            final = parse_native(Path(found[LAST]['path'])); check_final(final, source)
            states[kind, ranks] = found[LAST]
            parameters[kind, ranks] = final['parameters']
            result['checkpoints'][f'{kind}_{ranks}'] = records
            result['unused_global_coarse_indices'][f'{kind}_{ranks}'] = list(struct.unpack('<9i', final['fixed'][120:156]))
            result['diagnostic_coverage'][f'{kind}_{ranks}'] = check_diagnostics(run, [LAST])
            if kind == 'timed':
                reports[ranks] = timing_windows(read_json(run/'native-cycle-timing.json'), FIRST, ranks, CONFIG, 'SYCL', True)
            else:
                require(not (run/'native-cycle-timing.json').exists(), 'normal control unexpectedly instrumented')
        require(parameters['normal', ranks] == parameters['timed', ranks],
                'normal/timed final parameter maps (including retained horizon state) differ')
        item = dict(state=compare(states['normal', ranks], states['timed', ranks]),
                    final_parameters_exact=True,
                    diagnostics=diagnostics_exact(pair/'normal', pair/'timed', hashes))
        result['equivalence'][str(ranks)] = item
        require(item['diagnostics']['passed'], 'normal/timed history/horizon/shape bytes differ')
    result['cross_rank_state'] = compare(states['timed', 24], states['timed', 96])
    # Publish timing values only after every scientific and execution gate above.
    result['accepted_timings'] = {str(r): reports[r] for r in (24, 96)}
    t2 = [w['seconds_per_complete_step'] for w in reports[24]['windows']]
    t8 = [w['seconds_per_complete_step'] for w in reports[96]['windows']]
    result['scaling'] = dict(workload='fixed N12',
        paired_window_efficiency=[a/(4*b) for a, b in zip(t2, t8)],
        median_seconds_per_step={'24': statistics.median(t2), '96': statistics.median(t8)},
        min_seconds_per_step={'24': min(t2), '96': min(t8)},
        max_seconds_per_step={'24': max(t2), '96': max(t8)},
        median_speedup=statistics.median(t2)/statistics.median(t8),
        median_efficiency=statistics.median(t2)/(4*statistics.median(t8)))
    result.update(passed=True, numerical_qualification_passed=True, timings_accepted=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('two', 'eight', 'source-checkpoint', 'admission', 'independent-review', 'build-artifacts', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    result = dict(schema='native-cycle-timing-gpu-validation-v1', passed=False,
        numerical_qualification_passed=False, timings_accepted=False, accepted_timings={},
        scaling=None, input_sha256={}, checkpoints={}, equivalence={}, diagnostic_coverage={},
        unused_global_coarse_indices={},
        mesh_header_scope='All initialized header fields exact; nine unused global coarse-index integers excluded',
        scope='Offline fixed-N12 restart qualification and timing; no solver, build or job launched by this reader')
    try:
        audit(args, result)
    except Exception as exc:
        result.update(passed=False, numerical_qualification_passed=False, timings_accepted=False,
                      accepted_timings={}, scaling=None, failure=f'{type(exc).__name__}: {exc}')
    result['source_sha256'] = {name: sha256(Path(__file__).with_name(name)) for name in
        ('native_cycle_timing_gpu_validate.py', 'native_cycle_timing.py', 'native_cycle_timing.hpp',
         'native_cycle_timing_contract.md', 'native_cycle_timing_validate.py',
         'native_step_observer_gpu_compare.py', 'n12_checkpoint.py', 'n12_long_history.py', 'restart_roundtrip.py')}
    (args.output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({k: result[k] for k in ('passed', 'timings_accepted', 'failure') if k in result}))
    return 0 if result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
