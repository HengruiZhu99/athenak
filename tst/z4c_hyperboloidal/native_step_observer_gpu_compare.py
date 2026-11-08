#!/usr/bin/env python3
"""Qualify the observed native main on actual GPU controls and rank-changing restart.

All inputs are read-only. No solver/build/job is launched. Original native
state/output equivalence is exact; historical scalar and restart diagnostic
comparisons use the already adopted gates.
"""
import argparse
import json
from pathlib import Path
import shlex

import numpy as np

from n12_checkpoint import DT, expected_time, parse_parameters, require, sha256
from n12_checkpoint_gpu_control import ORIGINAL_CONFIG, ORIGINAL_EXE
from n12_long_history import read_outputs
from native_step_metrics import compare_step_metrics
from native_step_observer_validate import stages
from restart_roundtrip import compare, parse, payload

NATIVE = 'f3a85db7528986ddec347b31675a8c5d588120a2380133aed73d5967f7aab05c'


def checkpoint_set(folder, expected_cycles, ranks, hashes):
    result = {}; reports = []
    for path in sorted((folder/'rst').glob('*.rst')):
        info = parse(path)
        require(info['blocks'] == info['record_blocks'] == 106 and
                info['metadata'][4:] == (ranks, 0, 106//ranks, 0),
                'changed fixed-N12 shared checkpoint ownership/layout')
        require(info['cycle'] in expected_cycles and info['dt'] == DT and
                info['time'] == expected_time(info['cycle']), 'checkpoint step/time/dt differs')
        data = payload(info)
        require(data.shape == (106, 18**3+144, 39) and np.isfinite(data).all(),
                'incomplete/nonfinite31-field/eight-record payload')
        require(np.array_equal(np.frombuffer(info['costs'], dtype='<f4'), np.ones(106, dtype='<f4')),
                'native block costs changed')
        if info['cycle'] in result:
            compare(result[info['cycle']], info)
        result[info['cycle']] = info
        hashes[str(path.resolve())] = sha256(path)
        reports.append(dict(path=str(path.resolve()), cycle=info['cycle'], time=info['time'],
                            ranks=ranks, scalar_values=int(data.size), sha256=hashes[str(path.resolve())]))
    require(result.keys() == set(expected_cycles), 'missing/extra accepted checkpoint cycle')
    return result, reports


def diagnostics_exact(left, right, hashes):
    def files(folder):
        return {path.name for pattern in ('*.hst', '*.horizon*.txt') for path in folder.glob(pattern)}
    names = files(left)
    require(names == files(right) and names, 'normal/observed diagnostic file set differs')
    result = []
    for name in sorted(names):
        a, b = left/name, right/name
        hashes[str(a.resolve())] = sha256(a); hashes[str(b.resolve())] = sha256(b)
        same = a.read_bytes() == b.read_bytes()
        result.append(dict(name=name, bitwise_equal=same))
    return dict(passed=all(row['bitwise_equal'] for row in result), files=result)


def restart_diagnostics(candidate, reference, hashes):
    """Compare every final row, retaining the prefix's extra horizon-search phase."""
    collected = []
    for folder in (candidate, reference):
        _, _, files, extra = read_outputs([folder])
        for path in files:
            hashes[str(path.resolve())] = sha256(path)
        h = [item['record'] for item in extra if item['kind'] == 'history' and item['step'] == 8]
        a = [item['record'] for item in extra if item['kind'] == 'horizon' and item['step'] == 8]
        require(h and a, 'restart/continuous step8 diagnostic output missing')
        collected.append((h, a))
    report = dict(passed=True, comparisons=[],
        phase='restart has an additional prefix-final horizon search; retain every final row and apply adopted diagnostic gates')
    for actual in collected[0][0]:
        for expected in collected[1][0]:
            for name, column in [('Volume', 2), ('H', 3), ('M', 4), ('Theta', 5), ('Z', 6),
                                 ('eVolume', 7), ('eH', 8), ('eM', 9), ('eTheta', 10), ('eZ', 11)]:
                x, y = actual['values'][column], expected['values'][column]
                tolerance = 5e-10+1e-6*abs(y)
                item = dict(field=name, actual=x, reference=y, absolute_difference=abs(x-y),
                            tolerance=tolerance, passed=abs(x-y) <= tolerance,
                            actual_line=actual['line'], reference_line=expected['line'])
                report['comparisons'].append(item); report['passed'] &= item['passed']
    for actual in collected[0][1]:
        for expected in collected[1][1]:
            for name in ('areal_radius_M', 'irreducible_mass_M'):
                x, y = actual[name], expected[name]; tolerance = max(1e-10, 1e-7*abs(y))
                item = dict(field=name, actual=x, reference=y, absolute_difference=abs(x-y),
                            tolerance=tolerance, passed=abs(x-y) <= tolerance,
                            actual_line=actual['line'], reference_line=expected['line'])
                report['comparisons'].append(item); report['passed'] &= item['passed']
    return report


def executable_command(folder, executable, ranks, hashes, observed=False, restart=False):
    path = folder/'command.txt'; tokens = shlex.split(path.read_text())
    hashes[str(path.resolve())] = sha256(path)
    require(tokens.count(executable) == 1 and tokens.count('-n') == 1 and
            int(tokens[tokens.index('-n')+1]) == ranks, 'wrong recorded executable or MPI ranks')
    if observed:
        require(not any(token.startswith('-t') for token in tokens),
                'observed qualification adds a wall-clock termination override')
        status = folder/'exit-code.txt'
        require(status.read_text().strip() == '0', 'observed process did not exit successfully')
        hashes[str(status.resolve())] = sha256(status)
        require(('-r' in tokens) == restart and ('-i' in tokens) != restart,
                'observed fresh/restart entry path differs')
    return tokens


def legacy_admission(folder, staging, ranks, hashes):
    require(staging['binary_sha256'] == ORIGINAL_EXE and staging['configuration_sha256'] == ORIGINAL_CONFIG,
            'immutable historical control identity differs')
    require(sha256(folder/'run.in') == staging['input_sha256'] and
            sha256(folder/'configuration.cfg') == ORIGINAL_CONFIG, 'historical input/configuration changed')
    parameters = parse_parameters((folder/'run.in').read_text())
    for key, expected in [('time/dt', DT), ('time/nlim', 8), ('numerics/ko', .1), ('numerics/chi_floor', 1e-10)]:
        require(float(parameters[key]) == expected, 'historical numerical control differs: '+key)
    plan = json.loads((folder/'plan.json').read_text()); result = json.loads((folder/'result.json').read_text())
    require(plan['backend'] == 'SYCL' and plan['ranks'] == ranks and plan['nlim'] == 8 and
            (plan['bulk_points'], plan['surface_points']) == (229392, 864), 'wrong historical GPU control plan')
    require(result['status'] == 'step_limit' and result['steps'] == 8 and
            result['time_M'] == 3*expected_time(8) and not result['failure'], 'historical control incomplete')
    for name in ('run.in', 'configuration.cfg', 'plan.json', 'result.json', 'steps.csv'):
        hashes[str((folder/name).resolve())] = sha256(folder/name)
    return dict(path=str(folder.resolve()), ranks=ranks, steps=8, backend='SYCL',
                executable_sha256=ORIGINAL_EXE, input_sha256=staging['input_sha256'],
                configuration_sha256=ORIGINAL_CONFIG, successful_step_limit=True)


def audit(args):
    hashes = {}
    proof = json.loads(args.observer_build_provenance.read_text())
    require(proof['schema'] == 'native-step-observer-build-v1' and proof['status'] == 'built' and
            proof['build_exit_code'] == 0 and proof['runtime_inputs_unchanged'] is True and
            proof['native_executable_sha256'] == NATIVE and
            proof['original_main_fp_flags_preserved'] is True and proof['overlay']['inverse_byte_exact'] is True,
            'observer build is not the qualified frozen-main overlay')
    for name in ('native_step_observer.hpp', 'native_step_observer.py'):
        require([digest for path, digest in proof['sources'].items() if Path(path).name == name] ==
                [sha256(Path(__file__).with_name(name))], 'observed build source differs: '+name)
    hashes[str(args.observer_build_provenance.resolve())] = sha256(args.observer_build_provenance)
    staging = json.loads(args.legacy_staging.read_text())
    hashes[str(args.legacy_staging.resolve())] = sha256(args.legacy_staging)
    report = dict(schema='native-step-observer-gpu-comparison-v1', passed=True,
        native_executable_sha256=NATIVE, observed_executable_sha256=proof['executable_sha256'],
        observer_build_provenance_sha256=sha256(args.observer_build_provenance),
        ordinary_native_equivalence={}, historical_metrics={}, legacy_controls={}, checkpoints={}, stage_metadata={},
        scope='actual GPU bounded observer qualification; no long trajectory or performance claim')
    states = {}; stage_rows = {}; commands = {}
    settings = [('observed_24', 24, 0, 8, {0, 8}), ('observed_96', 96, 0, 8, {0, 8}),
                ('prefix_24', 24, 0, 2, {0, 2}), ('resumed_96', 96, 2, 8, {8})]
    for name, ranks, first, last, cycles in settings:
        folder = getattr(args, name)
        commands[name] = executable_command(folder, proof['executable'], ranks, hashes, True, first != 0)
        rows, metadata = stages(folder, first, last, ranks)
        require(metadata['backend'] == 'SYCL', 'observer scientific gate requires actual GPUs')
        stage_rows[name] = rows; report['stage_metadata'][name] = metadata
        for filename in ('native-stages.csv', 'native-step-observer.json', 'run.log'):
            hashes[str((folder/filename).resolve())] = sha256(folder/filename)
        states[name], report['checkpoints'][name] = checkpoint_set(folder, cycles, ranks, hashes)
    for ranks in (24, 96):
        normal, observed = getattr(args, 'normal_'+str(ranks)), getattr(args, 'observed_'+str(ranks))
        normal_command = executable_command(normal, proof['native_executable'], ranks, hashes)
        normal_log = normal/'run.log'
        require('Terminating on cycle limit' in normal_log.read_text(),
                'original native control did not terminate at its requested cycle limit')
        hashes[str(normal_log.resolve())] = sha256(normal_log)
        normal_states, report['checkpoints']['normal_'+str(ranks)] = checkpoint_set(normal, {0, 8}, ranks, hashes)
        limits = [token if token != '-t' else token+' '+normal_command[i+1]
                  for i, token in enumerate(normal_command) if token.startswith('-t')]
        item = dict(states={str(cycle): compare(normal_states[cycle], states['observed_'+str(ranks)][cycle])
                            for cycle in (0, 8)}, diagnostics=diagnostics_exact(observed, normal, hashes),
                    original_control_wall_limit=limits,
                    original_control_termination='cycle limit; optional original watchdog did not trigger')
        report['ordinary_native_equivalence'][str(ranks)] = item
        report['passed'] &= item['diagnostics']['passed']
        legacy = getattr(args, 'legacy_'+str(ranks))
        report['legacy_controls'][str(ranks)] = legacy_admission(legacy, staging, ranks, hashes)
        metrics = compare_step_metrics([observed], legacy/'steps.csv', 8, 'short', ranks)
        report['historical_metrics'][str(ranks)] = metrics
        report['passed'] &= metrics['passed']; hashes.update(metrics['input_sha256'])
    report['cross_rank_state'] = {str(c): compare(states['observed_24'][c], states['observed_96'][c]) for c in (0, 8)}
    report['prefix_initial_state'] = compare(states['prefix_24'][0], states['observed_24'][0])
    restart_source = Path(commands['resumed_96'][commands['resumed_96'].index('-r')+1])
    downloaded_prefix = Path(states['prefix_24'][2]['path'])
    require(restart_source.parts[-3:] == downloaded_prefix.parts[-3:],
            'recorded restart source is not the qualified prefix checkpoint')
    report['restart_source'] = dict(recorded_path=str(restart_source),
        downloaded_path=str(downloaded_prefix), sha256=sha256(downloaded_prefix))
    report['cross_rank_stage_rows_exact'] = stage_rows['observed_24'] == stage_rows['observed_96']
    report['restart_state'] = compare(states['resumed_96'][8], states['observed_96'][8])
    report['restart_stage_rows_exact'] = stage_rows['prefix_24']+stage_rows['resumed_96'] == stage_rows['observed_24']
    report['restart_diagnostics'] = restart_diagnostics(args.resumed_96, args.observed_96, hashes)
    report['passed'] &= report['cross_rank_stage_rows_exact'] and report['restart_stage_rows_exact'] and report['restart_diagnostics']['passed']
    report['input_sha256'] = hashes
    report['source_sha256'] = {str(Path(__file__).with_name(name).resolve()): sha256(Path(__file__).with_name(name)) for name in
        ('native_step_observer_gpu_compare.py', 'native_step_metrics.py', 'native_step_observer_validate.py',
         'restart_roundtrip.py', 'n12_checkpoint.py', 'n12_checkpoint_gpu_control.py', 'n12_long_history.py')}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('observer-build-provenance', 'observed-24', 'observed-96', 'prefix-24', 'resumed-96',
                 'normal-24', 'normal-96', 'legacy-24', 'legacy-96', 'legacy-staging', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(parents=True, exist_ok=False)
    try:
        result = audit(args)
    except Exception as error:
        result = dict(schema='native-step-observer-gpu-comparison-v1', passed=False,
                      validation_error=f'{type(error).__name__}: {error}')
    (args.output/'results.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({key: result[key] for key in ('passed', 'cross_rank_stage_rows_exact',
                     'restart_stage_rows_exact', 'validation_error') if key in result}, indent=2))
    raise SystemExit(0 if result['passed'] else 2)


if __name__ == '__main__':
    main()
