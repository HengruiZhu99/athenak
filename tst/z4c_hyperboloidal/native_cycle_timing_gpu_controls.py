#!/usr/bin/env python3
"""Offline reader controls: actual CPU/build artifacts plus labeled synthetic metadata.

No solver, compiler, MPI launcher or scheduler is called. Synthetic records are
tested in memory and never written as an apparent successful GPU collection.
The sole public-reader invocation must reject the actual incomplete1066 gate.
"""
import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import native_cycle_timing_gpu_validate as gpu
from n12_checkpoint import DT, expected_time, parse_native, sha256
from restart_roundtrip import compare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[2]
    qualified = ['native_cycle_timing.py', 'native_cycle_timing.hpp',
                 'native_cycle_timing_contract.md', 'native_cycle_timing_validate.py']
    before = {n: sha256(Path(__file__).with_name(n)) for n in qualified}
    result = dict(schema='native-cycle-timing-gpu-reader-controls-v1', passed=False,
        scope='Offline parser controls only: existing CPU runs/build evidence and synthetic in-memory metadata; no GPU numerical/timing qualification',
        gpu_executed=False, solver_launched=False, performance_admission=False, controls={}, input_sha256={})
    tests = result['controls']

    def positive(name, fn, scope):
        fn(); tests[name] = dict(passed=True, scope=scope)

    def negative(name, fn, expected):
        try:
            fn()
        except (ValueError, AssertionError, KeyError) as error:
            assert expected in str(error), (name, expected, str(error))
            tests[name] = dict(passed=True, rejected=True, message=str(error))
        else:
            raise AssertionError('negative control accepted: '+name)

    try:
        timed = root/'build-native/native-cycle-timing-local-validation-v1'
        normal = root/'build-native/native-step-observer-local-validation-v1/normal8-two'
        actual = gpu.read_json(timed/'native-cycle-timing.json')
        positive('actual_cpu_windows', lambda: gpu.timing_windows(actual, 0, 2, (2, 3, 2), 'Serial', False),
                 'Existing CPU0→8 timing data; no GPU evidence')
        states = {}
        for name, folder in [('normal', normal), ('timed', timed)]:
            states[name], _ = gpu.checkpoint_set(folder, {0, 8}, 2, result['input_sha256'])
            positive('actual_cpu_diagnostics_'+name, lambda f=folder: gpu.check_diagnostics(f, [0, 8]),
                     'Existing CPU complete history/horizon/169-coefficient outputs')
        result['actual_cpu_full_state'] = {str(c): compare(states['normal'][c], states['timed'][c]) for c in (0, 8)}
        exact = gpu.diagnostics_exact(normal, timed, result['input_sha256']); assert exact['passed']
        tests['actual_cpu_output_bytes'] = dict(passed=True, scope='Existing same-rank CPU ordinary outputs', files=exact['files'])
        build = root/'build-native/aurora-native-cycle-timing-build-v1/artifacts'
        positive('actual_aurora_build_only', lambda: gpu.check_build(build, result['input_sha256']),
                 'Actual isolated SYCL compilation evidence only; no GPU execution')

        # An explicit synthetic130-step report exercises the fixed future clock
        # offsets with copied CPU elapsed values. It is never GPU evidence.
        synthetic = copy.deepcopy(actual)
        synthetic.update(backend='SYCL', restart=True, ranks=24, initial_cycle=gpu.FIRST,
            final_cycle=gpu.LAST, initial_time=expected_time(gpu.FIRST), final_time=expected_time(gpu.LAST),
            complete_steps=130, warmup_steps=10, steps_per_window=40, window_count=3, windows=[])
        for index in range(3):
            first = gpu.FIRST+10+40*index; elapsed = 20.+index
            synthetic['windows'].append(dict(index=index, first_cycle=first, last_cycle=first+40,
                first_time=expected_time(first), last_time=expected_time(first+40), complete_steps=40,
                max_rank_seconds=elapsed, seconds_per_complete_step=elapsed/40,
                bulk_cell_updates=229392*40, bulk_cell_updates_per_second=229392*40/elapsed))
        check = lambda r: gpu.timing_windows(r, gpu.FIRST, 24, gpu.CONFIG, 'SYCL', True)
        positive('synthetic_restart_windows', lambda: check(synthetic), 'Synthetic in-memory metadata; not an execution')
        for name, mutate, message in [
            ('window_gap', lambda r: r['windows'][1].update(first_cycle=3602), 'window'),
            ('rk_stages_as_updates', lambda r: r['windows'][0].update(bulk_cell_updates=4*229392*40), 'rates/counts'),
            ('wrong_dt', lambda r: r.update(dt=DT*2), 'clocks'),
            ('wrong_backend', lambda r: r.update(backend='Serial'), 'backend'),
            ('wrong_rank_count', lambda r: r.update(ranks=96), 'coverage'),
            ('partial_final_cycle', lambda r: r.update(final_cycle=3680), 'coverage'),
            ('profile_loaded', lambda r: r.update(profiling_library_loaded=True), 'profiling/output'),
            ('scheduled_output', lambda r: r.update(timed_output_count=1), 'profiling/output'),
            ('nonfinite_window', lambda r: r['windows'][0].update(max_rank_seconds=float('nan')), 'rates/counts')]:
            changed = copy.deepcopy(synthetic); mutate(changed)
            negative(name, lambda r=changed: check(r), message)

        prior = root/'build-native/n12-common-segment-03-science-v1/continuation-admission.json'
        prior_review = root/'docs/z4c_hyperboloidal/evidence/native-n12-common-03-independent-review.json'
        admission = gpu.read_json(prior); review = gpu.read_json(prior_review)
        checkpoint = Path(admission['checkpoint'])
        source = parse_native(checkpoint)
        negative('actual1066_not_full3551', lambda: gpu.check_admission(admission, review, sha256(checkpoint)),
                 'full3551 science admission missing')
        negative('actual1066_not_timing_source', lambda: gpu.check_source(source), 'exact3551 state')
        model = copy.deepcopy(source)
        model.update(cycle=gpu.FIRST, time=expected_time(gpu.FIRST))
        for block in ('output1', 'output2'):
            model['parameters'][block+'/last_time'] = str(sum([1/6]*10))
        model['parameters']['output2/file_number'] = '11'
        positive('synthetic_3551_schedule', lambda: gpu.check_source(model),
                 'Actual1066 metadata adjusted in memory to future3551 schedule; no checkpoint exported')
        bad = copy.deepcopy(model); bad['parameters']['output1/last_time'] = '0'
        negative('insufficient_output_schedule', lambda: gpu.check_source(bad), 'schedule')
        bad = copy.deepcopy(model); bad['parameters']['output2/dcycle'] = '1'
        negative('added_cycle_output', lambda: gpu.check_source(bad), 'schedule')
        final = copy.deepcopy(model)
        final['parameters'].update({'time/nlim': '3681', 'time/tlim': '2', 'time/ndiag': '1000000000',
            'output1/last_time': str(sum([1/6]*11)), 'output2/last_time': str(sum([1/6]*11)),
            'output2/file_number': '12'})
        positive('synthetic_final_metadata', lambda: gpu.check_final(final, model), 'In-memory parameter/header grammar only')
        bad = copy.deepcopy(final); bad['parameters']['z4c_hyperboloidal/ko'] = '.2'
        negative('changed_final_physics', lambda: gpu.check_final(bad, model), 'policy differs')
        bad = copy.deepcopy(final); bad['parameters']['output2/file_number'] = '13'
        negative('extra_final_output_counter', lambda: gpu.check_final(bad, model), 'output cadence')
        bad = copy.deepcopy(final); bad['fixed'] = bytes([bad['fixed'][0] ^ 1])+bad['fixed'][1:]
        negative('changed_initialized_header', lambda: gpu.check_final(bad, model), 'mesh header')

        remote = gpu.SIDE/'runs/cycle-timing-nodes2/timed'
        command = ['mpiexec', '-n', '24', '-ppn', '12', gpu.BINDING, gpu.WRAPPER,
            str(gpu.BUILD/'athena-cycle-timing'), '-r', '/synthetic/reference.rst', '-d', str(remote),
            'time/nlim=3681', 'time/tlim=2', 'time/ndiag=1000000000']
        positive('synthetic_exact_command', lambda: gpu.check_command(command, 24, True, '/synthetic/reference.rst', remote),
                 'In-memory command grammar only; not launched')
        wrong = command.copy(); wrong[7] = str(gpu.SIDE/'build-sycl/src/athena')
        negative('wrong_executable', lambda: gpu.check_command(wrong, 24, True, '/synthetic/reference.rst', remote), 'command differs')
        negative('extra_dt_override', lambda: gpu.check_command(command+['time/dt=1'], 24, True, '/synthetic/reference.rst', remote), 'command differs')
        environment = dict(OMP_NUM_THREADS='1', ZE_FLAT_DEVICE_HIERARCHY='COMPOSITE',
            ZE_ENABLE_PCI_ID_DEVICE_ORDER='1', ONEAPI_DEVICE_SELECTOR='level_zero:gpu', MPIR_CVAR_ENABLE_GPU='1',
            ZE_AFFINITY_MASK=None, NEO_CACHE_DIR=str(remote/'cache/neo'), SYCL_CACHE_DIR=str(remote/'cache/sycl'),
            NATIVE_TIMING_WARMUP='10', NATIVE_TIMING_STEPS='40', NATIVE_TIMING_WINDOWS='3')
        environment.update({k: None for k in gpu.PROFILE})
        positive('synthetic_environment', lambda: gpu.check_environment(environment, remote), 'Selected environment grammar only')
        wrong = dict(environment, KOKKOS_TOOLS_LIBS='profiler.so')
        negative('profiling_environment', lambda: gpu.check_environment(wrong, remote), 'environment')
        wrong = dict(environment, ONEAPI_DEVICE_SELECTOR='opencl:cpu')
        negative('wrong_device_environment', lambda: gpu.check_environment(wrong, remote), 'environment')
        terminal = 'Job Id: 1234.server\n    Job_Name = native-timing-two\n    job_state = F\n    Exit_status = 0\n    Resource_List.nodect = 2\n'
        positive('synthetic_terminal_record', lambda: gpu.check_terminal({'job_id': '1234'}, terminal, '1234.server', 2),
                 'Synthetic PBS grammar only; not scheduler evidence')
        negative('different_actual_job', lambda: gpu.check_terminal({'job_id': '1234'}, terminal, '5678.server', 2), 'job differs')
        negative('nonzero_job_exit', lambda: gpu.check_terminal({'job_id': '1234'}, terminal.replace('Exit_status = 0', 'Exit_status = 1'), '1234', 2), 'Exit_status')
        negative('wrong_node_count', lambda: gpu.check_terminal({'job_id': '1234'}, terminal.replace('nodect = 2', 'nodect = 8'), '1234', 2), 'nodect')
        guard_dir = args.output/'synthetic-guard'; guard_dir.mkdir()
        verify = guard_dir/'timing-verify-before.txt'; verify.write_text(gpu.VERIFY_TEXT)
        checkpoint_sha = sha256(checkpoint); science_sha = 'a'*64; review_sha = 'b'*64
        guard = dict(schema='native-cycle-timing-runtime-guard-v1', passed=True,
            baseline_executable_sha256=gpu.NORMAL, timing_executable_sha256=gpu.TIMED,
            timing_build_provenance_sha256=gpu.PROVENANCE, source_checkpoint_sha256=checkpoint_sha,
            science_admission_sha256=science_sha, independent_review_sha256=review_sha,
            source_checkpoint_remote='/synthetic/reference.rst', source_manifest_sha256=gpu.SOURCE_MANIFEST,
            object_manifest_sha256=gpu.OBJECT_MANIFEST, verified_source_files=1972, verified_objects=228,
            gpu_wrapper_sha256=gpu.WRAPPER_SHA, timing_verify_exit_code=0, timing_verify_log_sha256=sha256(verify))
        (guard_dir/'guard-before.json').write_text(json.dumps(guard))
        call_guard = lambda: gpu.check_guard(guard_dir, 'before', checkpoint_sha, science_sha,
                                             review_sha, '/synthetic/reference.rst')
        positive('synthetic_guard_binding', call_guard, 'Synthetic guard grammar only; verify was not executed')
        wrong = dict(guard, independent_review_sha256='c'*64)
        (guard_dir/'guard-before.json').write_text(json.dumps(wrong))
        negative('guard_binds_wrong_review', call_guard, 'guard differs')
        (guard_dir/'guard-before.json').write_text(json.dumps(guard))
        verify.write_text('not an actual successful verify transcript\n')
        negative('guard_missing_verify_transcript', call_guard, 'verify output missing')

        # A tiny complete31+8 packet checks a late-record mismatch without
        # copying/modifying any actual197MB checkpoint or frozen evidence.
        a, b = args.output/'synthetic-packet-a.bin', args.output/'synthetic-packet-b.bin'
        values = np.arange(39, dtype='=f8'); values.tofile(a); values[-1] += 1; values.tofile(b)
        info = dict(blocks=1, record_blocks=1, cycle=0, time=0., dt=DT, atlas=b'fixture',
                    topology=b'fixture', identity=b'fixture', stride=39*8, payload_at=0)
        negative('late_chi_record_payload', lambda: compare(dict(info, path=str(a)), dict(info, path=str(b))),
                 'payload differs at block/slot/field')
        missing = args.output/'synthetic-missing-horizon'; missing.mkdir()
        (missing/'native_n12_5M.z4c_hyperboloidal.hst').write_bytes((timed/'native_n12_5M.z4c_hyperboloidal.hst').read_bytes())
        negative('both_missing_horizon_rejected', lambda: gpu.check_diagnostics(missing, [0, 8]), 'missing/extra')
        shape_dir = args.output/'synthetic-incomplete-shape'; shape_dir.mkdir()
        for name in ['native_n12_5M.z4c_hyperboloidal.hst', 'native_n12_5M.horizon_summary_0.txt', 'native_n12_5M.horizon_shape_0.txt']:
            (shape_dir/name).write_bytes((timed/name).read_bytes())
        path = shape_dir/'native_n12_5M.horizon_shape_0.txt'
        lines = path.read_text().splitlines(); lines[-1] = ' '.join(lines[-1].split()[:-1])
        path.write_text('\n'.join(lines)+'\n')
        negative('incomplete_horizon_coefficients', lambda: gpu.check_diagnostics(shape_dir, [0, 8]), '169-coefficient')

        # Public CLI must withhold all timing/scaling output for actual partial
        # science evidence, even though that prefix itself legitimately passed.
        command = [sys.executable, str(Path(gpu.__file__)), '--two', str(args.output/'absent-two'),
            '--eight', str(args.output/'absent-eight'), '--source-checkpoint', str(checkpoint),
            '--admission', str(prior), '--independent-review', str(prior_review),
            '--build-artifacts', str(build), '--output', str(args.output/'actual1066-public-rejection')]
        run = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        (args.output/'public-rejection.log').write_text(run.stdout)
        failed = gpu.read_json(args.output/'actual1066-public-rejection/results.json')
        assert run.returncode == 2 and failed['passed'] is False and failed['timings_accepted'] is False
        assert failed['accepted_timings'] == {} and failed['scaling'] is None and 'full3551' in failed['failure']
        tests['actual1066_public_reader_withholds_all_timings'] = dict(passed=True, exit_code=run.returncode,
            failure=failed['failure'], command=command)
        result['passed'] = True
    except Exception as error:
        result['failure'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        after = {n: sha256(Path(__file__).with_name(n)) for n in qualified}
        assert before == after, 'qualified timing sources changed'
        result['qualified_sources_unchanged'] = before
        result['source_sha256'] = {n: sha256(Path(__file__).with_name(n)) for n in
            ('native_cycle_timing_gpu_controls.py', 'native_cycle_timing_gpu_validate.py')}
        prior_failure = root/'build-native/native-cycle-timing-gpu-reader-controls-v1/results.json'
        if prior_failure.exists():
            result['preserved_fixture_failure'] = dict(path=str(prior_failure), sha256=sha256(prior_failure),
                reason='Late-record mismatch correctly rejected; the first control expected plain integer formatting but NumPy prints np.int64 indices. Only the control message match changed.')
        (args.output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(passed=True, controls=len(tests), gpu_evidence=False)))


if __name__ == '__main__':
    main()
