#!/usr/bin/env python3
"""Execute one bound, reviewed command plan inside one Aurora debug allocation."""
import argparse
import datetime
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import struct
import subprocess
import time

import control
import storage


def stamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def allocation_node(queue='debug'):
    if not os.environ.get('PBS_JOBID') or not os.environ.get('PBS_NODEFILE'):
        raise RuntimeError('Worker requires a PBS allocation and node file')
    nodes = set(Path(os.environ['PBS_NODEFILE']).read_text().split())
    if len(nodes) != 1:
        raise RuntimeError('This campaign permits exactly one allocated node')
    if queue not in control.QUEUES or os.environ.get('PBS_QUEUE') != queue:
        raise RuntimeError('Allocation queue differs from the bound debug queue')
    return sorted(nodes)


def mpi_command(executable, args):
    return ['mpiexec', '-n', '12', '-ppn', '12', '--cpu-bind=list:' + control.CPU_BIND,
            'gpu_tile_compact.sh', str(executable), *args]


def run_bounded(command, root, log, seconds, plan=None, environment=None):
    start = time.monotonic()
    env=None
    if environment is not None:
        env=dict(os.environ)
        for name in [*control.TRACE_ENV,'HYPERBOLIC_SPARSE_POISON']:env.pop(name,None)
        env.update(environment)
    with log.open('w') as stream:
        process = subprocess.Popen(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True, env=env)
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


def continuation_evidence(root, plan, command, case, report):
    """Permit only a recorded numerical rejection, never an infrastructure error.

    A resolution control remains failed. Its unadmitted checkpoint is retained
    as evidence and cannot become an accepted restart merely by continuing.
    Source/executable hashes are checked by the caller before this decision.
    """
    if (plan['phase'] != 'resolution_control' or not command.get('continue_after_failure')
            or command['kind'] == 'check' or case['status'] != 'failed'
            or case.get('timed_out') or case.get('storage_error') or case.get('report_error')
            or case.get('returncode') != 2 or report.get('status') != 'failed'
            or report.get('checkpoint_written') is not True
            or report.get('checkpoint_admitted') is not False):
        return None
    # The native runner also catches I/O/device-allocation exceptions. A
    # generic status=failed is therefore insufficient evidence of a numerical
    # failure. Compound messages (e.g. a later checkpoint-write failure) stop.
    numerical_reason = re.fullmatch(
        r'(?:nonpositive interpolated scri geometry|invalid diagnostic halo|'
        r'invalid diagnostic volume geometry|invalid accepted volume geometry|'
        r'invalid accepted (?:volume|surface) algebraic projection|'
        r'device RHS failure at active cell -?\d+ reason \d+|'
        r'device scri (?:projection|rate) failure at row -?\d+ reason [^;\r\n]+)',
        report.get('error', ''))
    if not numerical_reason:
        return None
    if (report.get('kind') != command['kind'] or report.get('n') != command['n']
            or report.get('backend') != 'SYCL' or report.get('mpi_size') != 12):
        raise ValueError('Failure report identity differs from the bound resolution control')
    relative = command['args'][1] + '.checkpoint'
    checkpoint = control.local_path(root, relative)
    layout = struct.Struct('<16s4i3Q2d')
    with checkpoint.open('rb') as stream:
        header = stream.read(layout.size)
    if len(header) != layout.size:
        raise ValueError('Numerical-failure checkpoint header is truncated')
    magic, n, nf, kind, admitted, cells, surface, steps, accepted_time, dt = layout.unpack(header)
    if (magic != b'HYP3D_MPI_V1'.ljust(16, b'\0') or n != command['n'] or nf != 31
            or kind != {'minkowski': 0, 'pulse': 1, 'puncture': 2}[command['kind']]
            or admitted != 0 or cells <= 0 or surface <= 0
            or cells != report.get('cells') or steps != report.get('steps')
            or not math.isfinite(accepted_time) or not 0 <= accepted_time <= float(command['args'][3])
            or accepted_time != report.get('t') or dt != float(command['args'][4])
            or dt != report.get('dt')
            or checkpoint.stat().st_size != layout.size + 16 * (nf * cells + surface)):
        raise ValueError('Numerical-failure checkpoint is inconsistent or incorrectly admitted')
    artifact = case['artifacts'].get(relative)
    if not artifact:
        raise ValueError('Numerical-failure checkpoint was not recorded as an artifact')
    return dict(reason=report['error'], checkpoint=relative,
                checkpoint_sha256=artifact['sha256'], checkpoint_admitted=False,
                accepted_time=accepted_time, steps=steps)


def run(root, relative):
    path, plan, binding = control.read_bound(root, relative)
    nodes = allocation_node(plan['resources']['queue'])
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
    if plan['phase'] == 'resolution_control':
        status.update(stability_admission_eligible=False, continued_failures=[])
    had_continued_failure = False
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
                status['status'] = 'failed' if had_continued_failure else 'incomplete'
                status['stop_reason_status'] = 'incomplete'
                status['reason'] = 'Insufficient worker time to start the next bounded case'
                break
            args = [str(timeout-30) if a == '@WALL_SECONDS@' else a for a in command['args']]
            executable = control.local_path(root, command['executable'])
            launched = mpi_command(executable, args)
            generated = {name:control.digest(control.local_path(root, name))
                         for name in command.get('generated_inputs', [])}
            case = dict(name=command['name'], kind=command['kind'], started=stamp(), argv=launched,
                        generated_inputs=generated,environment=control.command_environment(command))
            if command['kind'] != 'check':
                case['n'] = command.get('n', plan['n'])
            status['cases'].append(case)
            control.atomic(status_path, status)
            case.update(run_bounded(launched, root, output/(command['name']+'.log'), timeout, plan,case['environment']))
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
                evidence = continuation_evidence(root, plan, command, case, report)
                if evidence is not None:
                    case['continued_after_failure'] = True
                    case['continuation_evidence'] = evidence
                    had_continued_failure = True
                    status['continued_failures'].append(case['name'])
                    control.atomic(status_path, status)
                    continue
                status['status'] = 'failed' if had_continued_failure else case['status']
                status['stop_reason_status'] = case['status']
                break
            control.atomic(status_path, status)
        else:
            status['status'] = 'failed' if had_continued_failure else 'passed'
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
