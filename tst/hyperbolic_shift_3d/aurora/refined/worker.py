#!/usr/bin/env python3
"""Run an immutable, gated RF1 plan. No submission, resume, or deletion."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from storage import require_capacity


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def main():
    root, plan_file = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    plan = json.loads(plan_file.read_text())
    if plan['nodes'] not in (1, 2) or plan['ranks'] != 12*plan['nodes']:
        raise ValueError('Require 12 GPU tiles/ranks per node and one or two nodes')
    if not os.environ.get('PBS_JOBID'):
        raise RuntimeError('Run only in the bound PBS allocation')
    nodes = Path(os.environ['PBS_NODEFILE']).read_text().splitlines()
    if len(set(nodes)) != plan['nodes']:
        raise RuntimeError('Allocated node count differs from immutable plan')
    for relative, expected in plan['bindings'].items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root) or digest(path) != expected:
            raise RuntimeError('Provenance mismatch: '+relative)
    out = root/plan['output']
    out.mkdir(exist_ok=False)  # A second start cannot overwrite or duplicate work.
    start = time.monotonic()
    state = dict(job_id=os.environ['PBS_JOBID'], plan_sha256=digest(plan_file),
                 source_commit=plan['source_commit'], nodes=nodes, commands=[], status='running')
    write(out/'state.json', state)
    bind = '1-8:9-16:17-24:25-32:33-40:41-48:53-60:61-68:69-76:77-84:85-92:93-100'
    prefix = ['mpiexec', '-n', str(plan['ranks']), '-ppn', '12',
              '--cpu-bind=list:'+bind, 'gpu_tile_compact.sh']
    env = dict(os.environ, HYPERBOLIC_EXPECT_MPI_SIZE=str(plan['ranks']),
               HYPERBOLIC_MAPPING_PREFIX=str(out/'mapping'))
    try:
        for case in plan['commands']:
            if time.monotonic()-start+case['timeout'] > plan['deadline_seconds']:
                state['status'] = 'allocation_budget_stop'
                break
            capacity = require_capacity(root, planned=1_500_000_000)
            args = [str(root/x) if x.startswith(('source/', 'build/')) else
                    str(out/x[7:]) if x.startswith('OUTPUT/') else x for x in case['arguments']]
            command = prefix+args
            record = dict(name=case['name'], command=command, storage=capacity)
            state['commands'].append(record)
            write(out/'state.json', state)
            began = time.monotonic()
            with (out/(case['name']+'.log')).open('w') as log:
                run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                     env=env, timeout=case['timeout'])
            record.update(returncode=run.returncode, seconds=time.monotonic()-began)
            if run.returncode:
                raise RuntimeError(case['name']+' failed; numerical failures are terminal')
            if case.get('mapping'):
                placements = {}
                for rank in range(plan['ranks']):
                    text = (out/('mapping.rank'+str(rank)+'.txt')).read_text()
                    first = dict(word.split('=',1) for word in text.splitlines()[0].split())
                    fields = dict(line.split('=',1) for line in text.splitlines()[1:] if '=' in line)
                    if fields.get('visible_gpu_devices')!='1' or fields.get('device_pointer_alltoall_passed')!='1':
                        raise RuntimeError('GPU visibility or GPU-aware MPI qualification failed')
                    placements.setdefault(first['host'], []).append(fields['ZE_AFFINITY_MASK'])
                if len(placements)!=plan['nodes'] or any(len(masks)!=12 or len(set(masks))!=12 for masks in placements.values()):
                    raise RuntimeError('GPU tile placement is not 12 distinct tiles per node')
                record['verified_gpu_tiles'] = placements
            if case.get('report'):
                report = json.loads((out/case['report']).read_text())
                record['report_status'] = report['status']
                if report['status'] not in ('completed', 'checkpointed'):
                    raise RuntimeError('Unhealthy scientific report: '+case['name'])
                if case.get('fixed') and not (report['checkpoint_written'] and report['checkpoint_admitted']):
                    raise RuntimeError('Checkpoint not admitted: '+case['name'])
                if report['status'] == 'checkpointed':
                    state['status'] = 'healthy_walltime_checkpoint_requires_verified_continuation'
                    write(out/'state.json', state)
                    break
            if case.get('checkpoint_sha256'):
                actual=digest(out/(case['name']+'.checkpoint'))
                if actual!=case['checkpoint_sha256']:
                    raise RuntimeError('Repeated baseline checkpoint differs across allocations')
                record['checkpoint_sha256']=actual
            write(out/'state.json', state)
        else:
            state['status'] = 'completed'
    except Exception as error:
        state['status'], state['error'] = 'terminal_failure', str(error)
        raise
    finally:
        state['seconds'] = time.monotonic()-start
        state['storage'] = require_capacity(root)
        state['artifacts'] = {str(p.relative_to(out)):dict(bytes=p.stat().st_size, sha256=digest(p))
                              for p in sorted(out.rglob('*')) if p.is_file() and p.name!='state.json'}
        write(out/'state.json', state)


if __name__ == '__main__':
    main()
