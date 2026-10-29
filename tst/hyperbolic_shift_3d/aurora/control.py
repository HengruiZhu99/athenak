#!/usr/bin/env python3
"""Stdlib-only source binding, explicit submission, and read-only job collection.

No import submits a job. Only the `submit` subcommand calls qsub; the parent
coordinator is the sole caller. Unknown submission outcomes retain their lock.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import storage

JOB_NAME = 'hyper-scri-3d'
ACCOUNT = 'CompactBinaryMerger'
CPU_BIND = '1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
FIXED = dict(queue='debug', account=ACCOUNT, nodes=1, ranks=12, ranks_per_node=12)
PHASES = {'preflight', 'stability', 'convergence'}


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def local_path(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError('Expected a nonempty root-relative path')
    result = (root / relative).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError('Path escapes the prepared experiment root')
    return result


def wall_seconds(value):
    if not isinstance(value, str) or not re.fullmatch(r'\d{2}:\d{2}:\d{2}', value):
        raise ValueError('walltime must be HH:MM:SS')
    hours, minutes, seconds = map(int, value.split(':'))
    if minutes >= 60 or seconds >= 60:
        raise ValueError('Invalid walltime clock fields')
    total = 3600 * hours + 60 * minutes + seconds
    if not 300 <= total <= 3600:
        raise ValueError('Aurora debug walltime must be between five minutes and one hour')
    return total


def validate_plan(root, plan):
    if plan.get('schema') != 1 or plan.get('phase') not in PHASES:
        raise ValueError('Unsupported command-plan schema or phase')
    if plan.get('reviewed') is not True:
        raise ValueError('Coordinator must review the concrete command plan before binding it')
    resources = plan.get('resources', {})
    for key, value in FIXED.items():
        if resources.get(key) != value:
            raise ValueError(f'Resource policy mismatch: {key} must be {value}')
    allocation = wall_seconds(resources.get('walltime'))
    budget = plan.get('worker_seconds', allocation - 90)
    if not isinstance(budget, int) or budget <= 0 or budget > allocation - 60:
        raise ValueError('Worker budget must leave at least sixty seconds before PBS walltime')
    n = plan.get('n')
    if not isinstance(n, int) or n < 6 or n % 2:
        raise ValueError('Expected even n >= 6')
    if not isinstance(plan.get('low_resolution_n'), int) or plan['low_resolution_n'] < 6:
        raise ValueError('Declare the original low-resolution stability grid')
    if plan['phase'] in {'preflight', 'stability'} and n != plan['low_resolution_n']:
        raise ValueError('Preflight/stability must stay on the declared low-resolution grid')
    output = local_path(root, plan['output_dir'])
    commands = plan.get('commands')
    if not isinstance(commands, list) or not commands:
        raise ValueError('A reviewed plan must contain concrete commands')
    names = set()
    preceding_artifacts = set()
    for command in commands:
        name = command.get('name', '')
        if not re.fullmatch(r'[a-z0-9][a-z0-9_-]*', name) or name in names:
            raise ValueError('Case names must be unique plain identifiers')
        names.add(name)
        if command.get('kind') not in {'check', 'minkowski', 'pulse', 'puncture'}:
            raise ValueError('Unknown physical case')
        local_path(root, command['executable'])
        argv = command.get('args')
        if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
            raise ValueError('Command arguments must be an explicit list of strings')
        timeout = command.get('timeout_seconds')
        if not isinstance(timeout, int) or not 45 <= timeout <= budget:
            raise ValueError('Each command needs a bounded timeout within the worker budget')
        report = local_path(root, command['report'])
        if not report.is_relative_to(output):
            raise ValueError('Scientific reports must be written inside the fresh output directory')
        checks = command.get('checks')
        if not isinstance(checks, list) or not checks:
            raise ValueError('Every case requires explicit scientific report checks')
        for check in checks:
            if check.get('op') not in {'eq', 'le', 'ge', 'finite'}:
                raise ValueError('Unknown report-check operator')
            if not isinstance(check.get('path'), list) or not check['path']:
                raise ValueError('Checks use a nonempty JSON key/index path')
        for path in command.get('inputs', []):
            local_path(root, path)
        for path in command.get('generated_inputs', []):
            local_path(root, path)
            if path not in preceding_artifacts:
                raise ValueError('Generated inputs must be declared artifacts of an earlier case')
        if command['kind'] != 'check':
            if len(argv) not in [8, 9] or argv[0] != str(n) or argv[2] != command['kind']:
                raise ValueError('Evolution arguments must match the declared grid and physical case')
            if command['report'] != argv[1] + '.report.json':
                raise ValueError('Evolution report must match its output prefix')
            target, dt = float(argv[3]), float(argv[4])
            if not math.isfinite(target) or target < 0 or not math.isfinite(dt) or dt <= 0:
                raise ValueError('Evolution target time and timestep must be finite and physical')
            declared = command.get('inputs', []) + command.get('generated_inputs', [])
            if any(p != '-' and p not in declared for p in argv[5:7]):
                raise ValueError('Initial and restart files must be explicitly bound inputs')
            if argv[7] != '@WALL_SECONDS@':
                raise ValueError('Evolution must use the worker-bounded @WALL_SECONDS@ deadline')
        for path in command.get('artifacts', []):
            if not local_path(root, path).is_relative_to(output):
                raise ValueError('Artifacts must be inside the fresh output directory')
            preceding_artifacts.add(path)
    storage.policy(plan)
    return allocation, budget


def verify_source(root, relative):
    manifest = local_path(root, relative)
    values = json.loads(manifest.read_text())
    if not isinstance(values, dict) or not values:
        raise ValueError('Source manifest must map root-relative files to SHA256 hashes')
    for name, expected in values.items():
        if not isinstance(expected, str) or not re.fullmatch('[0-9a-f]{64}', expected):
            raise ValueError('Invalid source digest')
        if digest(local_path(root, name)) != expected:
            raise ValueError(f'Source changed: {name}')
    return digest(manifest)


def validate_admission(root, plan, source_digest, binary_hashes):
    """Admissions are reviewed evidence, never inferred from a process exit code."""
    if plan['phase'] == 'preflight':
        return None
    admission_path = local_path(root, plan['admission'])
    admission = json.loads(admission_path.read_text())
    if admission.get('passed') is not True:
        raise ValueError('A passed scientific admission is required for this phase')
    required = 'preflight' if plan['phase'] == 'stability' else 'stability'
    if admission.get('phase') != required:
        raise ValueError(f'{required} admission required')
    if admission.get('source_manifest_sha256') != source_digest:
        raise ValueError('Admission belongs to different scientific source')
    if admission.get('low_resolution_n') != plan['low_resolution_n']:
        raise ValueError('Admission belongs to a different low-resolution grid')
    admitted = admission.get('binaries', {})
    for name, sha in binary_hashes.items():
        if admitted.get(name) != sha:
            raise ValueError('Admission executable binding does not match the requested run')
    if required == 'preflight':
        for key in ['gpu_cpu_parity', 'restart_equivalence']:
            if admission.get(key) is not True:
                raise ValueError(f'Preflight admission must establish {key}')
    else:
        for key in ['minkowski_stable', 'pulse_stable', 'puncture_stable']:
            if admission.get(key) is not True:
                raise ValueError(f'Convergence requires completed low-resolution {key}')
        reached = admission.get('puncture_completed_t_over_M', 0)
        if not isinstance(reached, (int, float)) or not math.isfinite(reached) or reached < 100:
            raise ValueError('Convergence is gated on the requested 100 M low-resolution puncture run')
    return dict(path=plan['admission'], sha256=digest(admission_path))


def read_bound(root, relative, check_source=True):
    plan_path = local_path(root, relative)
    plan = json.loads(plan_path.read_text())
    validate_plan(root, plan)
    binding = json.loads(plan_path.with_name('binding.json').read_text())
    if digest(plan_path) != binding['plan_sha256']:
        raise ValueError('Plan changed after binding')
    if check_source and verify_source(root, binding['source_manifest']) != binding['source_manifest_sha256']:
        raise ValueError('Source-manifest binding changed')
    for name, sha in binding['binaries'].items():
        if digest(local_path(root, name)) != sha:
            raise ValueError(f'Executable changed: {name}')
    for name, sha in binding['inputs'].items():
        if digest(local_path(root, name)) != sha:
            raise ValueError(f'Initial/checkpoint input changed: {name}')
    if digest(plan_path.with_name('run_debug.pbs')) != binding['pbs_sha256']:
        raise ValueError('PBS resources or worker command changed after binding')
    admission = validate_admission(root, plan, binding['source_manifest_sha256'], binding['binaries'])
    if admission != binding.get('admission'):
        raise ValueError('Admission evidence changed after binding')
    for name, sha in binding['launcher'].items():
        if digest(Path(__file__).with_name(name)) != sha:
            raise ValueError(f'Launcher changed after binding: {name}')
    return plan_path, plan, binding


def bind(root, relative, manifest):
    path = local_path(root, relative)
    for name in ['binding.json', '.submission-lock', 'submission-pending.json', 'submission.json']:
        if path.with_name(name).exists():
            raise ValueError('This plan is already bound or has a submission record; use a fresh plan directory')
    plan = json.loads(path.read_text())
    allocation, budget = validate_plan(root, plan)
    space = storage.require_capacity(root, plan, prospective=True)
    if local_path(root, plan['output_dir']).exists():
        raise ValueError('Output directory already exists; scientific runs must be fresh')
    source_sha = verify_source(root, manifest)
    binaries = {c['executable']: digest(local_path(root, c['executable'])) for c in plan['commands']}
    inputs = {p: digest(local_path(root, p)) for c in plan['commands'] for p in c.get('inputs', [])}
    admission = validate_admission(root, plan, source_sha, binaries)
    template = Path(__file__).with_name('run_debug.pbs').read_text()
    if template.count('@WALLTIME@') != 1 or template.count('@WORKER_PATH@') != 1:
        raise ValueError('Unexpected PBS template')
    worker = Path(__file__).with_name('worker.py').resolve().relative_to(root.resolve()).as_posix()
    if not re.fullmatch(r'[A-Za-z0-9_./-]+',worker):
        raise ValueError('Worker path must be a plain campaign-relative path')
    pbs = path.with_name('run_debug.pbs')
    pbs.write_text(template.replace('@WALLTIME@', plan['resources']['walltime']).replace('@WORKER_PATH@',worker))
    binding = dict(schema=1, plan_sha256=digest(path), source_manifest=manifest,
                   source_manifest_sha256=source_sha, binaries=binaries, inputs=inputs,
                   pbs_sha256=digest(pbs), admission=admission, resources=plan['resources'],
                   storage_at_bind=space, launcher={name:digest(Path(__file__).with_name(name))
                             for name in ['control.py', 'worker.py', 'run_debug.pbs', 'storage.py']},
                   allocation_seconds=allocation, worker_seconds=budget)
    atomic(path.with_name('binding.json'), binding)
    return binding


def scheduler_snapshot(user):
    selected = subprocess.run(['qselect', '-u', user, '-N', JOB_NAME], capture_output=True, text=True)
    if selected.returncode not in [0, 153]:
        raise RuntimeError('Cannot establish task allocation state: ' + selected.stderr.strip())
    ids = selected.stdout.split()
    if any(not re.fullmatch(r'\d+\.[A-Za-z0-9_.-]+', job) for job in ids):
        raise RuntimeError('Unrecognized qselect output')
    if not ids:
        return dict(queried=time.time(), jobs={}, blocking=[])
    result = subprocess.run(['qstat', '-f', '-F', 'json', *ids], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Ambiguous scheduler state; recheck before any submission: ' + result.stderr.strip())
    jobs = json.loads(result.stdout).get('Jobs', {})
    if set(jobs) != set(ids):
        raise RuntimeError('Scheduler response did not account for every selected job')
    compact = {job: {key: value.get(key) for key in ['Job_Name', 'job_state', 'queue', 'Account_Name', 'Resource_List']} for job, value in jobs.items()}
    # Exact task name only: unrelated held research allocations are untouched.
    blocking = [job for job, value in compact.items() if value['job_state'] not in ['F', 'X']]
    return dict(queried=time.time(), jobs=compact, blocking=blocking)


def submit(root, relative):
    path, plan, binding = read_bound(root, relative)
    if not re.fullmatch(r'[A-Za-z0-9_./-]+', relative):
        raise ValueError('PBS plan path must not contain commas, whitespace, or shell metacharacters')
    for name in ['.submission-lock', 'submission-pending.json', 'submission.json']:
        if path.with_name(name).exists():
            raise RuntimeError('Submission already attempted; inspect the saved outcome, never duplicate it')
    space = storage.require_capacity(root, plan, prospective=True)
    atomic(path.with_name('storage-before.json'), space)
    snapshot = scheduler_snapshot(os.environ['USER'])
    atomic(path.with_name('scheduler-before.json'), snapshot)
    if snapshot['blocking']:
        raise RuntimeError('Another task allocation exists: ' + ', '.join(snapshot['blocking']))
    # A root-wide lock also closes concurrent qselect/qsub races between plans.
    global_lock = root / '.hyperbolic-debug-allocation-lock'
    global_lock.mkdir()
    path.with_name('.submission-lock').mkdir()
    atomic(global_lock / 'owner.json', dict(plan=relative, created=time.time()))
    command = ['qsub', '-v', 'HYPERBOLIC_PLAN=' + relative, str(path.with_name('run_debug.pbs'))]
    pending = dict(started=time.time(), command=command, plan_sha256=binding['plan_sha256'])
    atomic(path.with_name('submission-pending.json'), pending)
    result = subprocess.run(command, cwd=root, capture_output=True, text=True)
    atomic(path.with_name('submission-output.json'), dict(returncode=result.returncode, stdout=result.stdout, stderr=result.stderr))
    if result.returncode:
        raise RuntimeError('qsub failed; locks retained until the coordinator establishes the outcome')
    ids = re.findall(r'(?m)^(\d+\.[A-Za-z0-9_.-]+)\s*$', result.stdout.strip())
    if len(ids) != 1:
        raise RuntimeError('Ambiguous qsub result; preserve locks and inspect scheduler')
    record = dict(job_id=ids[0], submitted=time.time(), plan=relative,
                  plan_sha256=binding['plan_sha256'], source_manifest_sha256=binding['source_manifest_sha256'],
                  resources=plan['resources'])
    atomic(path.with_name('submission.json'), record)
    return record


def collect(root, relative, release=False):
    path = local_path(root, relative)
    submission = json.loads(path.with_name('submission.json').read_text())
    job = submission['job_id']
    result = subprocess.run(['qstat', '-fx', '-F', 'json', job], capture_output=True, text=True)
    record = dict(job_id=job, collected=time.time(), returncode=result.returncode,
                  stdout=result.stdout, stderr=result.stderr)
    if not result.returncode:
        jobs = json.loads(result.stdout).get('Jobs', {})
        if job not in jobs:
            raise RuntimeError('Scheduler result lacks the requested job')
        record['scheduler'] = jobs[job]
    status_path = path.with_name('worker-status.json')
    if status_path.exists():
        record['worker'] = json.loads(status_path.read_text())
    atomic(path.with_name('collected.json'), record)
    if release:
        state = record.get('scheduler', {}).get('job_state')
        if state not in ['F', 'X']:
            raise RuntimeError('Only an explicitly terminal scheduler result permits lock release')
        snapshot = scheduler_snapshot(os.environ['USER'])
        if snapshot['blocking']:
            raise RuntimeError('A task allocation is still active')
        lock = root / '.hyperbolic-debug-allocation-lock'
        owner = json.loads((lock / 'owner.json').read_text())
        if owner['plan'] != relative:
            raise RuntimeError('Root allocation lock belongs to another plan')
        (lock / 'owner.json').unlink()
        lock.rmdir()
        atomic(path.with_name('allocation-released.json'), dict(job_id=job, released=time.time()))
    return record


def evaluate_check(report, check):
    try:
        value = report
        for key in check['path']:
            value = value[key]
        op = check['op']
        if op == 'eq':
            passed = value == check['value']
        else:
            valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
            passed = valid and (op == 'finite' or (op == 'le' and value <= check['value']) or (op == 'ge' and value >= check['value']))
        observed = str(value) if isinstance(value, float) and not math.isfinite(value) else value
        return dict(check=check, observed=observed, passed=bool(passed))
    except (KeyError, IndexError, TypeError) as error:
        return dict(check=check, passed=False, error=str(error))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['bind', 'verify', 'submit', 'collect'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--source-manifest')
    parser.add_argument('--release-terminal-lock', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    if args.action == 'bind':
        if not args.source_manifest:
            parser.error('bind requires --source-manifest')
        value = bind(root, args.plan, args.source_manifest)
    elif args.action == 'verify':
        _, _, value = read_bound(root, args.plan)
    elif args.action == 'submit':
        value = submit(root, args.plan)
    else:
        value = collect(root, args.plan, args.release_terminal_lock)
    print(json.dumps(value, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
