"""Run fixed temporal series sequentially after a verified spatial series.

The prerequisite must complete; a failed or missing prerequisite never releases
the queue. This occupies its vacated evolution slot, not an additional slot.
"""
import argparse
import datetime
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from continue_puncture import atomic, digest
from continue_series import verify


IDENTITY = ['pid', 'command', 'started_utc', 'initial_parent_sha256',
            'invariant_sha256', 'target_t', 'runner_sha256', 'continuation_sha256']


def check_gate(path, anchor):
    state = json.loads(path.read_text())
    if any(state[key] != anchor[key] for key in IDENTITY):
        raise ValueError('Prerequisite identity changed')
    if state['status'] == 'completed':
        if state['active_child'] is not None:
            raise ValueError('Completed prerequisite still has an active child')
        m = verify(Path(state['last_completed_parent']))
        if abs(m['t']-anchor['target_t']) > 1e-12:
            raise ValueError('Prerequisite did not reach target')
        return True
    if state['status'] != 'running':
        raise ValueError('Prerequisite terminated without completing: '+state['status'])
    command = subprocess.run(['ps', '-p', str(state['pid']), '-o', 'command='],
                             text=True, capture_output=True, check=False).stdout
    if ' '.join(state['command']) not in command:
        # Account for the completion/exit race; never infer completion from PID absence.
        again = json.loads(path.read_text())
        if again['status'] != state['status']:
            return check_gate(path, anchor)
        raise ValueError('Prerequisite process identity is missing')
    return False


def run_jobs(root, jobs, record, runner, hashes):
    process = None
    try:
        for job in jobs:
            if (root/'STOP').exists():
                raise ValueError('Queue stopped before launching another series')
            for name, sha in hashes.items():
                if digest(runner.parent/name) != sha:
                    raise ValueError('Frozen queue harness changed')
            parent = Path(job['parent'])
            verify(parent)
            if digest(parent/'manifest.json') != job['parent_manifest_sha256']:
                raise ValueError('Temporal parent changed while waiting')
            dest = root/job['name']
            cmd = [sys.executable, str(runner), '--parent', str(parent), '--output', str(dest),
                   '--t', str(job['t']), '--segment-duration', str(job['segment_duration']),
                   '--wall-seconds', '420']
            with (root/(job['name']+'.log')).open('x') as log:
                process = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                record.update(status='running_series', active_series=dict(path=str(dest),
                              pid=process.pid, command=cmd))
                atomic(root/'queue.json', record)
                print(json.dumps(record['active_series']), flush=True)
                code = process.wait()
            process = None
            result = json.loads((dest/'series.json').read_text())
            record['finished'].append(dict(name=job['name'], exit_code=code,
                                           status=result['status'], path=str(dest)))
            record['active_series'] = None
            if code or result['status'] != 'completed':
                raise ValueError('Temporal series did not complete; no retries or later jobs')
            final = verify(Path(result['last_completed_parent']))
            if abs(final['t']-job['t']) > 1e-12:
                raise ValueError('Temporal series missed requested target')
            atomic(root/'queue.json', record)
    finally:
        if process is not None and process.poll() is None:
            record['interrupted_series'] = record['active_series']
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    config = json.loads(a.config.read_text())
    gate = Path(config['after_series']).resolve()
    anchor = json.loads(gate.read_text())
    check_gate(gate, anchor)
    jobs = config['jobs']
    names = [j['name'] for j in jobs]
    if len(set(names)) != len(names) or any(Path(n).name != n or n in ['.', '..'] for n in names):
        raise ValueError('Need distinct simple job names')
    for job in jobs:
        parent = Path(job['parent']).resolve()
        m = verify(parent)
        if digest(parent/'manifest.json') != job['parent_manifest_sha256']:
            raise ValueError('Unexpected temporal parent')
        job['parent'] = str(parent)
        if not m['t'] < job['t'] <= anchor['target_t'] or not 0 < job['segment_duration'] <= 10:
            raise ValueError('Invalid queued duration')
        for value in [job['t'], job['segment_duration']]:
            if abs(round(value/m['dt'])*m['dt']-value) > 1e-12:
                raise ValueError('Queued target or segment is off the inherited time lattice')
    root = a.output.resolve()
    root.mkdir(parents=True, exist_ok=False)
    frozen = root/'runner'
    frozen.mkdir()
    files = ['continue_series.py', 'continue_puncture.py']
    for name in files:
        shutil.copy2(Path(__file__).with_name(name), frozen/name)
    shutil.copy2(__file__, root/'frozen_queue.py')
    hashes = {name:digest(frozen/name) for name in files}
    atomic(root/'config.json', config)
    record = dict(status='waiting', pid=os.getpid(), command=sys.argv,
                  started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  gate=str(gate), gate_identity={k:anchor[k] for k in IDENTITY},
                  frozen_hashes=hashes, active_series=None, finished=[],
                  scope='Serial temporal continuations after completed spatial prerequisite; no convergence claim.')
    atomic(root/'queue.json', record)
    try:
        while not check_gate(gate, anchor):
            if (root/'STOP').exists():
                raise ValueError('Queue stopped while waiting')
            time.sleep(15)
        record['gate_completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        run_jobs(root, jobs, record, frozen/'continue_series.py', hashes)
        record['status'] = 'completed'
    except (Exception, KeyboardInterrupt) as exc:
        record['status'] = 'stopped'
        record['error'] = str(exc) or 'Interrupted'
    finally:
        record['active_series'] = None
        record['updated_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic(root/'queue.json', record)
    print(json.dumps(record), flush=True)
    if record['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
