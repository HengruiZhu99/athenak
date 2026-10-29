#!/usr/bin/env python3
"""Verify diagnostic acquisition only; never scientifically admit a checkpoint."""
import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
AURORA = HERE.parent.parent
ROOT = HERE.parents[5]
PREFLIGHT = AURORA/'preflight-dissipative-v1'
SOURCE = '0e7967f1f33559ed79b9f97d6416dacc296f0f14663d35c682177a9ac07db60f'
PLAN = 'a950caacf4d8202c9475be2a92413025033f22abd40b6c3656ead83e415f6e25'
ADMISSION = '32cc0fed05319e204f41edea3b62f614138f07a801079e6348536a07650713a1'
BINARY = 'fc8d8074351a01ade6fd05dcf460555c935cf4ffe66c9bda31cc517679500fff'
TARGET = 'build-sycl-mpi-v11/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_mpi_dissipative'
ORDER = ['snapshot-025','snapshot-030','snapshot-035']
LIMITS = {'max_state_deviation':1e-8,'H_L2':1e-7,'M_L2':1e-7}
SPEC = importlib.util.spec_from_file_location('v4_preflight_verifier',PREFLIGHT/'verify.py')
v = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v)
require, sha, read = v.require, v.sha, v.read


def reference_stability(report):
    checks = []
    for field,limit in LIMITS.items():
        value = report.get(field)
        passed = type(value) in [int,float] and math.isfinite(value) and value <= limit
        checks.append(dict(field=field,value=value,limit=limit,passed=bool(passed)))
    return dict(all_reference_limits_passed=all(x['passed'] for x in checks), checks=checks,
                scientific_not_admitted=True,
                meaning='Reference equilibrium limits only; no scientific continuation or stability admission.')


def history_summary(path, report):
    with path.open(newline='') as f:
        rows = list(csv.DictReader(f))
    require(rows and all(None not in row and None not in row.values() for row in rows), 'Missing/partial history')
    values = [{k:float(value) for k,value in row.items()} for row in rows]
    require(all(math.isfinite(row['t']) and math.isfinite(row['steps']) for row in values), 'Invalid history clock')
    require(all(b['t'] > a['t'] and b['steps'] > a['steps'] for a,b in zip(values,values[1:])), 'History clock is not increasing')
    require(values[-1]['t'] == report['diagnostic_time'] == report['t'] and values[-1]['steps'] == report['steps'],
            'Endpoint history/report clock differs')
    for field, history_field in [('max_state_deviation','state_deviation'),('H_L2','H_L2'),('M_L2','M_L2')]:
        require(values[-1][history_field] == report[field], 'Endpoint history/report amplitude differs')
    gates = []
    for field,history_field in [('max_state_deviation','state_deviation'),('H_L2','H_L2'),('M_L2','M_L2')]:
        limit = LIMITS[field]
        failed = [row for row in values if not math.isfinite(row[history_field]) or row[history_field] > limit]
        gates.append(dict(field=history_field,limit=limit,all_recorded_samples_passed=not failed,
                          first_failed_recorded_time=failed[0]['t'] if failed else None,
                          maximum=max(row[history_field] for row in values)))
    return dict(samples=len(values),last_time=values[-1]['t'],reference_limits_on_history=gates,
                note='Checks recorded diagnostics only; no interpolation between samples.')


def verify(args, result):
    raw = args.raw_dir.resolve()
    build, manifest, result['build'] = v.verify_build(PREFLIGHT/'compile-evidence',ROOT)
    parser, control = v.frozen_parser(ROOT,manifest)
    plan,binding,worker = [read(raw/p) for p in ['plan.json','binding.json','worker-status.json']]
    require(sha(raw/'plan.json') == sha(HERE.parent/'plan.json') == PLAN == worker['plan_sha256'] == binding['plan_sha256'],
            'Reviewed plan differs')
    control.validate_plan(ROOT,plan)
    require([c['name'] for c in plan['commands']] == ORDER and sum(len(c['checks']) for c in plan['commands']) == 96,
            'Acquisition case/check set differs')
    require(all(plan['reference_stability_limits'][k] == value for k,value in LIMITS.items()), 'Reference limits changed')
    require(worker['source_manifest_sha256'] == binding['source_manifest_sha256'] == SOURCE,
            'Scientific source differs')
    require(worker['binaries'] == binding['binaries'] == {TARGET:BINARY} and build['binaries'][TARGET] == BINARY,
            'Frozen binary differs')
    require(binding['pbs_sha256'] == sha(raw/'run_debug.pbs'), 'PBS script differs')
    require(worker['resources'] == binding['resources'] == plan['resources'], 'Resource declaration differs')
    require(worker['phase'] == plan['phase'] == 'stability' and worker['n'] == 16, 'Launcher phase/resolution differs')
    for name,digest in binding['launcher'].items():
        require(digest == manifest['source-v11/tst/hyperbolic_shift_3d/aurora/'+name], 'Launcher source differs')
    require(not binding['inputs'], 'Fresh analytic snapshots must not load initial/restart inputs')
    admission = read(PREFLIGHT/'admission.json')
    require(binding['admission']['sha256'] == ADMISSION == sha(PREFLIGHT/'admission.json') and admission['passed'] is True
            and admission['phase'] == 'preflight' and admission['source_manifest_sha256'] == SOURCE
            and admission['binaries'][TARGET] == BINARY and admission['low_resolution_n'] == 16,
            'Preflight admission differs')
    require(all(control.dissipation_config(c) in admission['scri_dissipation_configs'] for c in plan['commands']),
            'Epsilon configuration not preflight-bound')
    supports = [(PREFLIGHT/'raw/worker-status.json',admission['worker_status_sha256']),
                (PREFLIGHT/'verification-v2.json',admission['verification_sha256']),
                (PREFLIGHT/'verify.py',admission['verifier_sha256'])]
    supports += [(PREFLIGHT/'raw/output'/Path(r['path']).name,r['sha256']) for r in admission['supporting_reports'].values()]
    for path,digest in supports:
        require(sha(path) == digest, 'Preflight support hash differs')
    require(read(PREFLIGHT/'verification-v2.json')['passed'] is True, 'Preflight verification did not pass')
    result['admission_chain'] = dict(sha256=ADMISSION,supporting_hashes_verified=len(supports),
                                    scope='Prior implementation/restart preflight only; no duration admission.')
    affinity = worker['affinity']
    v.placement(affinity['records'])
    require(affinity['passed'] and affinity['returncode'] == 0 and not affinity['timed_out']
            and affinity['storage_error'] is None, 'Affinity probe failed')
    records = [json.loads(line.split('=',1)[1]) for line in (raw/'output/affinity.log').read_text().splitlines()
               if line.startswith('AFFINITY_JSON=')]
    require(sorted(records,key=lambda r:r['rank']) == sorted(affinity['records'],key=lambda r:r['rank']), 'Affinity log differs')
    result['one_node_twelve_tiles_verified'] = True
    terminal,release,submission = [read(raw/p) for p in ['collected.json','allocation-released.json','submission.json']]
    require(terminal['job_id'] == release['job_id'] == worker['job_id'] == submission['job_id'], 'Job identity differs')
    v.verify_release(release)
    require(submission['plan_sha256'] == PLAN and submission['source_manifest_sha256'] == SOURCE
            and submission['resources'] == plan['resources'], 'Submission binding differs')
    result['terminal'] = dict(job_id=worker['job_id'],state=terminal['scheduler']['job_state'],
                              exit_status=terminal['scheduler'].get('Exit_status'),allocation_released=True)
    require(result['terminal']['state'] == 'F', 'Allocation not terminal')
    if result['terminal']['exit_status'] != 0:
        result['issues'].append('Scheduler exited nonzero')
    require(worker['storage_at_finish']['bytes'] < 5_000_000_000, 'Home campaign cap exceeded')
    result['storage_at_finish'] = worker['storage_at_finish']
    result['worker_elapsed_seconds'] = worker['elapsed_seconds']
    started = [case['name'] for case in worker['cases']]
    require(started == ORDER[:len(started)], 'Missing, duplicate or reordered worker case')
    for index, command in enumerate(plan['commands']):
        output = dict(name=command['name'],status='not_started',issues=[],acquisition_checks_recomputed=0,
                      scientific_not_admitted=True,scientific_restart_admitted=False)
        result['cases'].append(output)
        if index >= len(worker['cases']):
            continue
        case = worker['cases'][index]
        output['status'] = case['status']
        try:
            require(case['kind'] == command['kind'] == 'minkowski', 'Physical case differs')
            argv = case['argv']
            require(argv[:7] == ['mpiexec','-n','12','-ppn','12','--cpu-bind=list:'+control.CPU_BIND,'gpu_tile_compact.sh']
                    and argv[7] == v.REMOTE+TARGET and len(argv[8:]) == len(command['args']), 'Actual launch differs')
            for actual,expected in zip(argv[8:],command['args']):
                if expected == '@WALL_SECONDS@':
                    require(math.isfinite(float(actual)) and 0 < float(actual) <= command['timeout_seconds']-30,
                            'Invalid runtime ceiling')
                else:
                    require(actual == expected, 'Actual launch argument differs')
            require(command['args'][5:7] == ['-','-'] and not case['generated_inputs'], 'Snapshot did not start fresh')
            prefix = 'campaigns/linear-window-dissipative-v2/output/'
            require(set(case['artifacts']) <= set(command['artifacts']), 'Undeclared artifact')
            for remote,expected in case['artifacts'].items():
                require(remote.startswith(prefix) and Path(remote).parent.as_posix() == prefix.rstrip('/'), 'Artifact outside output directory')
                path = raw/'output'/Path(remote).name
                actual = dict(bytes=path.stat().st_size,sha256=sha(path))
                require(actual == expected, 'Artifact size/hash differs')
                result['artifacts'].append(dict(remote_path=remote,**actual))
            required = {p for p in command['artifacts'] if not p.endswith('.failed-stage.bin')}
            if case['status'] == 'passed':
                require(required <= set(case['artifacts']), 'Passed acquisition lacks artifact')
            require(command['report'] in case['artifacts'], 'Report not hash-bound')
            report = read(raw/'output'/Path(command['report']).name)
            require(case['report_status'] == report['status'], 'Native/worker report status differs')
            require([r['check'] for r in case['checks']] == command['checks'], 'Acquisition gates differ')
            failed = []
            for record in case['checks']:
                observed,passed = v.check_value(record['check'],report)
                require(observed == record['observed'] and passed == record['passed'], 'Acquisition gate replay differs')
                output['acquisition_checks_recomputed'] += 1
                if not passed:
                    failed.append(record)
            output['failed_acquisition_checks'] = failed
            output['reference_stability'] = reference_stability(report)
            cp = v.checkpoint(raw/'output'/(command['name']+'.checkpoint'))
            require(cp['magic'] == 'HYP3D_MPI_V4' and cp['header_bytes'] == 88 and cp['scri_dissipation_epsilon'] == 2
                    and cp['scri_dissipation_degree'] == 7 and cp['kind'] == 0 and cp['n'] == 16,
                    'Checkpoint configuration differs')
            for cp_key,report_key in [('time','t'),('steps','steps'),('dt','dt'),('admitted','checkpoint_admitted')]:
                require(cp[cp_key] == report[report_key], 'Header/report differs: '+cp_key)
            output['checkpoint'] = cp
            output['native_checkpoint_admitted_flag'] = cp['admitted']
            output['amplitudes'] = {key:report[key] for key in ['max_state_deviation','H_L2','M_L2','H_exterior_L2','M_exterior_L2',
                                      'max_null_surface','max_null_ghost','max_determinant_residual','max_tracefree_residual']}
            output['time'] = report['t']; output['steps'] = report['steps']
            output['timing'] = dict(native_seconds=report['seconds'],worker_case_seconds=case['elapsed_seconds'],
                                   native_seconds_per_step=report['seconds']/report['steps'] if report['steps'] else None,
                                   scope='Whole native run divided by accepted steps; includes setup and diagnostics, not kernel throughput.')
            if report['status'] == 'completed':
                output['history'] = history_summary(raw/'output'/(command['name']+'.history.csv'),report)
            if not (case['status'] == 'passed' and case['returncode'] == 0 and case['timed_out'] is False
                    and case['storage_error'] is None and not failed):
                output['issues'].append('Acquisition did not pass all execution/report gates')
        except (KeyError, ValueError, TypeError, OSError) as error:
            output['issues'].append(f'{type(error).__name__}: {error}')
    result['checks_recomputed'] = sum(c['acquisition_checks_recomputed'] for c in result['cases'])
    result['artifacts_verified'] = len(result['artifacts'])
    result['metadata_sha256'] = {p:sha(raw/p) for p in ['plan.json','binding.json','worker-status.json','run_debug.pbs',
                                                     'submission.json','allocation-released.json','collected.json','output/affinity.log']}
    if worker['status'] != 'passed':
        result['issues'].append('Worker status is '+str(worker['status']))
    result['acquisition_passed'] = (not result['issues'] and result['checks_recomputed'] == 96
                                   and len(result['cases']) == 3 and all(c['status'] == 'passed' and not c['issues'] for c in result['cases']))


def json_safe(value, path=(), nonfinite_paths=None):
    """Keep failed checks while making invalid observed metadata serializable."""
    if nonfinite_paths is None:
        nonfinite_paths = []
    if isinstance(value, dict):
        return {k:json_safe(v,path+(k,),nonfinite_paths) for k,v in value.items()}
    if isinstance(value, list):
        return [json_safe(v,path+(index,),nonfinite_paths) for index,v in enumerate(value)]
    if isinstance(value,float) and not math.isfinite(value):
        nonfinite_paths.append(list(path))
        return None
    return value


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw-dir',type=Path,default=ROOT/'output/hyperbolic-core-shell/aurora/linear-window-dissipative-v2')
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    require(not args.output.exists(),'Output must be new')
    result = dict(acquisition_passed=False,scientific_not_admitted=True,scientific_restart_admitted=False,
                  stability_qualification=False,issues=[],cases=[],artifacts=[],verifier_sha256=sha(__file__),
                  scope='Diagnostic acquisition only; reference stability limits retained separately. Native header admission does not imply scientific admission.',
                  surface_amplitude_caveat='max_state_deviation screens volume fields only; separate field analysis must inspect surface amplitudes.')
    try:
        verify(args,result)
    except (KeyError,ValueError,TypeError,OSError,subprocess.CalledProcessError) as error:
        result['issues'].append(f'{type(error).__name__}: {error}')
    nonfinite_paths = []
    result = json_safe(result,nonfinite_paths=nonfinite_paths)
    result['nonfinite_metadata_paths'] = nonfinite_paths
    if nonfinite_paths:
        result['acquisition_passed'] = False
        result['issues'].append('Nonfinite metadata retained as null at the listed paths; verification cannot pass')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:result.get(k) for k in ['acquisition_passed','scientific_not_admitted','issues','checks_recomputed','artifacts_verified']}))
    return 0 if result['acquisition_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
