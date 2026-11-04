#!/usr/bin/env python3
"""One-shot submission with durable intent, live scheduler capture and storage gate."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from storage import require_capacity

root = Path(sys.argv[1]).resolve()
number = int(sys.argv[2]) if len(sys.argv)>2 else 1
if number not in (1,2):
    raise ValueError('Only the two reviewed plans are supported')
plan_name = 'plan'+str(number)+'.json'
plan = json.loads((root/plan_name).read_text())
assert plan['nodes']==1 and plan['ranks']==12
assert not (root/plan['output']).exists(), 'Output already exists; never duplicate a run'
receipt = root/('submission'+str(number)+'.json')
# An unknown qsub outcome leaves this intent in place. Reconcile qstat and the
# receipt manually; never retry an ambiguous submission automatically.
with receipt.open('x') as stream:
    json.dump(dict(status='intent', utc=time.time(), plan=plan_name), stream)
live = subprocess.run(['qstat','-u',os.environ['USER']],capture_output=True,text=True,check=True)
(root/('pre-submit'+str(number)+'-jobs.txt')).write_text(live.stdout+live.stderr)
queues = subprocess.run(['qstat','-Qf','debug'],capture_output=True,text=True,check=True)
(root/('pre-submit'+str(number)+'-queue.txt')).write_text(queues.stdout+queues.stderr)
if 'enabled = True' not in queues.stdout or 'started = True' not in queues.stdout:
    raise RuntimeError('Debug queue not available; revise the plan explicitly')
capacity = require_capacity(root, planned=5_000_000_000)
command = ['qsub','-o',str(root/('job'+str(number)+'-pbs.log')),str(root/('launcher/job.pbs' if number==1 else 'launcher/job2.pbs'))]
result = subprocess.run(command,cwd=root,capture_output=True,text=True)
if result.returncode:
    raise RuntimeError('qsub failed; inspect durable intent before any retry: '+result.stderr)
job = result.stdout.strip()
if not job or not job.split('.')[0].isdigit():
    raise RuntimeError('Ambiguous qsub response; do not resubmit')
receipt.write_text(json.dumps(dict(status='submitted',utc=time.time(),job_id=job,
                                  command=command,storage=capacity),indent=2)+'\n')
print(job)
