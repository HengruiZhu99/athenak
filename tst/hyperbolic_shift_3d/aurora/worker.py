#!/usr/bin/env python3
"""Execute one bound, reviewed command plan inside one Aurora debug allocation."""
import argparse
import datetime
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

import control
import storage


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def allocation_node():
    if not os.environ.get('PBS_JOBID') or not os.environ.get('PBS_NODEFILE'):
        raise RuntimeError('Worker requires a PBS allocation and node file')
    nodes = set(Path(os.environ['PBS_NODEFILE']).read_text().split())
    if len(nodes) != 1:
        raise RuntimeError('This campaign permits exactly one allocated node')
    if os.environ.get('PBS_QUEUE', 'debug') != 'debug':
        raise RuntimeError('This campaign permits only the debug queue')
    return sorted(nodes)


def mpi_command(executable, args):
    return ['mpiexec', '-n', '12', '-ppn', '12', '--cpu-bind=list:' + control.CPU_BIND,
            'gpu_tile_compact.sh', str(executable), *args]


def run_bounded(command, root, log, seconds, plan=None):
    start = time.monotonic()
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        timed_out = False
        storage_error = None
        try:
            while True:
                remaining = seconds-(time.monotonic()-start)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command,seconds)
                try:
                    code = process.wait(timeout=min(5,remaining))
                    break
                except subprocess.TimeoutExpired:
                    if plan is not None:
                        storage.require_capacity(root,plan)
        except (subprocess.TimeoutExpired, RuntimeError) as error:
            timed_out = isinstance(error,subprocess.TimeoutExpired)
            storage_error = None if timed_out else str(error)
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=5)
    return dict(returncode=code, timed_out=timed_out, storage_error=storage_error, elapsed_seconds=time.monotonic()-start)


def probe_affinity(root, output, deadline):
    code = ("import json,os,socket; print('AFFINITY_JSON='+json.dumps(dict("
            "host=socket.gethostname(),rank=int(os.environ.get('MPI_LOCALRANKID',"
            "os.environ.get('PALS_LOCAL_RANKID','-1'))),"
            "mask=os.environ.get('ZE_AFFINITY_MASK'),"
            "hierarchy=os.environ.get('ZE_FLAT_DEVICE_HIERARCHY'))),flush=True)")
    log = output / 'affinity.log'
    result = run_bounded(mpi_command('/usr/bin/python3.11', ['-c', code]), root, log,
                         min(60, max(1, deadline-time.monotonic())))
    rows = []
    for line in log.read_text().splitlines():
        if line.startswith('AFFINITY_JSON='):
            rows.append(json.loads(line.split('=', 1)[1]))
    result['records'] = rows
    result['passed'] = (not result['timed_out'] and result['returncode'] == 0
                        and len(rows) == 12 and {r['rank'] for r in rows} == set(range(12))
                        and len({r['host'] for r in rows}) == 1
                        and all(r['hierarchy'] == 'COMPOSITE' and
                                r['mask'] == f"{r['rank']//2}.{r['rank']%2}" for r in rows))
    return result


def classify(returncode, timed_out, report, checks):
    if timed_out:
        return 'external_timeout'
    status = report.get('status')
    if status == 'checkpointed' and returncode in (0, 3):
        # The expected completion-status check may fail on a valid intermediate
        # checkpoint. Scientific diagnostic failures still reject the state.
        healthy = all(c['passed'] or c.get('check', {}).get('path') == ['status']
                      for c in checks)
        return 'checkpointed' if healthy else 'failed'
    if returncode == 0 and status == 'completed' and all(c['passed'] for c in checks):
        return 'passed'
    return 'failed'


def run(root, relative):
    path, plan, binding = control.read_bound(root, relative)
    nodes = allocation_node()
    if not root.is_relative_to(Path.home()):
        raise RuntimeError('All new campaign work must reside under the Aurora home directory')
    initial_space = storage.require_capacity(root,plan,prospective=True)
    output = control.local_path(root, plan['output_dir'])
    output.mkdir(parents=True, exist_ok=False)
    status_path = path.with_name('worker-status.json')
    if status_path.exists():
        raise RuntimeError('Worker has already attempted this plan')
    for key, value in {'OMP_NUM_THREADS':'1', 'ZE_FLAT_DEVICE_HIERARCHY':'COMPOSITE',
                       'ZE_ENABLE_PCI_ID_DEVICE_ORDER':'1', 'ONEAPI_DEVICE_SELECTOR':'level_zero:gpu',
                       'MPIR_CVAR_ENABLE_GPU':'1'}.items():
        if os.environ.get(key) != value:
            raise RuntimeError(f'Required Aurora runtime setting {key}={value}')
    os.environ['HYPERBOLIC_EXPECT_MPI_SIZE'] = '12'
    began = time.monotonic()
    deadline = began + binding['worker_seconds']
    status = dict(schema=1, status='running', started=stamp(), job_id=os.environ['PBS_JOBID'],
                  launcher_host=socket.gethostname(), allocation_nodes=nodes, resources=plan['resources'],
                  source_manifest_sha256=binding['source_manifest_sha256'], binaries=binding['binaries'],
                  plan_sha256=binding['plan_sha256'], phase=plan['phase'], n=plan['n'], cases=[],
                  storage_at_start=initial_space)
    control.atomic(status_path, status)
    try:
        status['affinity'] = probe_affinity(root, output, deadline)
        control.atomic(status_path, status)
        if not status['affinity']['passed']:
            raise RuntimeError('Twelve-rank, twelve-tile affinity probe failed')
        for command in plan['commands']:
            control.read_bound(root, relative)
            storage.require_capacity(root,plan)
            remaining = int(deadline-time.monotonic())
            timeout = min(command['timeout_seconds'], remaining-10)
            if timeout < 45:
                status['status'] = 'incomplete'
                status['reason'] = 'Insufficient worker time to start the next bounded case'
                break
            args = [str(timeout-30) if a == '@WALL_SECONDS@' else a for a in command['args']]
            executable = control.local_path(root, command['executable'])
            launched = mpi_command(executable, args)
            generated = {name:control.digest(control.local_path(root, name))
                         for name in command.get('generated_inputs', [])}
            case = dict(name=command['name'], kind=command['kind'], started=stamp(), argv=launched,
                        generated_inputs=generated)
            status['cases'].append(case)
            control.atomic(status_path, status)
            case.update(run_bounded(launched, root, output/(command['name']+'.log'), timeout, plan))
            storage.require_capacity(root,plan)
            report_path = control.local_path(root, command['report'])
            report = {}
            if report_path.exists():
                try:
                    report = json.loads(report_path.read_text())
                except (ValueError, OSError) as error:
                    case['report_error'] = str(error)
            case['checks'] = [control.evaluate_check(report, c) for c in command['checks']]
            case['report_status'] = report.get('status')
            case['status'] = classify(case['returncode'], case['timed_out'], report, case['checks'])
            if case.get('storage_error'):
                case['status'] = 'failed'
            if case['status'] == 'checkpointed':
                checkpoint = report_path.with_name(report_path.name.removesuffix('.report.json') + '.checkpoint')
                if not checkpoint.is_file() or checkpoint.stat().st_size == 0:
                    case['status'] = 'failed'
                    case['report_error'] = 'Checkpointed report lacks a nonempty checkpoint file'
                else:
                    case['checkpoint_sha256'] = control.digest(checkpoint)
            case['artifacts'] = {}
            for name in [command['report'], *command.get('artifacts', [])]:
                artifact = control.local_path(root, name)
                if artifact.is_file():
                    case['artifacts'][name] = dict(sha256=control.digest(artifact), bytes=artifact.stat().st_size)
            control.read_bound(root, relative)
            case['finished'] = stamp()
            if case['status'] != 'passed':
                status['status'] = case['status']
                break
            control.atomic(status_path, status)
        else:
            status['status'] = 'passed'
    except Exception as error:
        status['status'] = 'failed'
        status['reason'] = f'{type(error).__name__}: {error}'
    status['finished'] = stamp()
    status['elapsed_seconds'] = time.monotonic()-began
    status['storage_at_finish'] = storage.footprint(root)
    control.atomic(status_path, status)
    return 0 if status['status'] == 'passed' else 3 if status['status'] == 'checkpointed' else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--plan', required=True)
    args = parser.parse_args()
    raise SystemExit(run(args.root.resolve(), args.plan))


if __name__ == '__main__':
    main()
