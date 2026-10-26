"""Current user-selected resources for newly prepared scalar allocations.

This module never submits a job. Historical prepared trees remain immutable.
"""
import json
import re
from pathlib import Path

POLICY = dict(queue='capacity', account='CompactBinaryMerger', nodes=1,
              walltime='00:30:00', wall_seconds=1800,
              worker_deadline_seconds=1500, maximum_task_allocations=1)


def prepare(root):
    root=Path(root)
    if any((root/name).exists() for name in ['submission.json','submission-pending.json','.submission-lock']):
        raise RuntimeError('Resource changes apply only before any submission attempt')
    pbs=root/'run_aurora.pbs';text=pbs.read_text()
    for pattern,replacement in [
            (r'(?m)^#PBS -q \S+\s*$', '#PBS -q capacity'),
            (r'(?m)^#PBS -A \S+\s*$', '#PBS -A CompactBinaryMerger'),
            (r'(?m)^#PBS -l select=\S+\s*$', '#PBS -l select=1'),
            (r'(?m)^#PBS -l walltime=\S+\s*$', '#PBS -l walltime=00:30:00')]:
        text,count=re.subn(pattern,replacement,text)
        if count!=1:raise RuntimeError('Ambiguous PBS resources: '+pattern)
    marker='export SCALAR_ALLOCATION_WALL_SECONDS=1800'
    if 'SCALAR_ALLOCATION_WALL_SECONDS=' in text:
        text=re.sub(r'(?m)^export SCALAR_ALLOCATION_WALL_SECONDS=.*$',marker,text)
    else:
        text=text.replace('set -euo pipefail','set -euo pipefail\n'+marker,1)
    if marker not in text:raise RuntimeError('No explicit allocation budget in PBS script')
    pbs.write_text(text)
    plan=root/'cpu-plan.json'
    if plan.exists():
        data=json.loads(plan.read_text());data['limits']={**data.get('limits',{}),**POLICY}
        plan.write_text(json.dumps(data,indent=2)+'\n')
    (root/'resource-policy.json').write_text(json.dumps(POLICY,indent=2)+'\n')
    validate(root)


def validate(root):
    root=Path(root);text=(root/'run_aurora.pbs').read_text()
    declared=json.loads((root/'resource-policy.json').read_text())
    if declared!=POLICY:raise RuntimeError('Allocation does not match current user resources')
    for exact in ['#PBS -q capacity','#PBS -A CompactBinaryMerger','#PBS -l select=1','#PBS -l walltime=00:30:00','export SCALAR_ALLOCATION_WALL_SECONDS=1800']:
        if text.splitlines().count(exact)!=1:raise RuntimeError('PBS resources differ from policy: '+exact)
    resource_lines=[line for line in text.splitlines() if line.startswith('#PBS ') and any(key in line for key in [' -q ',' -A ','select=','walltime='])]
    if len(resource_lines)!=4:raise RuntimeError('Additional conflicting PBS resources')
    return dict(POLICY)
