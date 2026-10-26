#!/usr/bin/env python3
"""Read live scalar progress without changing allocations or native data."""
import argparse,hashlib,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--resume',action='store_true');a=p.parse_args();E=a.evidence.resolve();output=E/'live-scalar-progress.jsonl';ssh=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','ControlPath=/Users/hz0693/.ssh/codex-control/%C','hzhu@aurora.alcf.anl.gov']
if output.exists()!=a.resume:raise RuntimeError('Explicit resume requires an existing progress journal; a new reader requires a fresh journal')
program='''import csv,json,subprocess
from pathlib import Path
q=subprocess.run(['qselect','-u','hzhu','-N','frozen-scalar'],capture_output=True,text=True)
if q.returncode not in [0,153]:raise RuntimeError('Cannot establish task allocation state: '+q.stderr)
jobs=q.stdout.split()
if len(jobs)>1:raise RuntimeError('More than one task allocation is queued/running')
records=[]
for job in jobs:
 raw=subprocess.run(['qstat','-xf','-F','json',job],capture_output=True,text=True,check=True);s=json.loads(raw.stdout)['Jobs'][job];root=Path(s['Output_Path'].split(':',1)[1]).parent
 if not str(root).startswith('/lus/flare/projects/CompactBinaryMerger/hzhu/test-scalar-frozen-20261016/'):raise RuntimeError('Unexpected task allocation root')
 cases=[]
 for file in sorted((root/'results'/job).glob('*/history.csv')):
  parameters=file.parent/'parameters.json'
  if not parameters.exists():continue
  meta=json.loads(parameters.read_text());rows=list(csv.DictReader(file.open()));rows=[r for r in rows if r.get('region')=='exterior' and r.get('reduction_Linf')]
  if not rows:continue
  r=rows[-1]
  try:cases.append(dict(n=meta['n'],mass=meta['M'],spin=meta['a_over_M'],time=float(r['time']),reduction_Linf=float(r['reduction_Linf']),psi_max=float(r['psi_max'])))
  except (ValueError,TypeError):continue
 records.append(dict(job=job,state=s['job_state'],cases=cases))
print(json.dumps(records))
'''
with output.open('a' if a.resume else 'x') as f:f.write(json.dumps(dict(event='resumed' if a.resume else 'started',time=time.time(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Live partial data only, no admission or simulation changes.'))+'\n')
previous=None;access_error=None
while not (E/'scalar-validation-summary.json').exists():
 r=subprocess.run([*ssh,'/usr/bin/python3.11 -'],input=program,text=True,capture_output=True)
 if r.returncode==255:
  message=r.stderr.strip()
  if message!=access_error:
   with output.open('a') as f:f.write(json.dumps(dict(event='access_unavailable',time=time.time(),stderr=message,admission=False))+'\n')
   print('PROGRESS ACCESS UNAVAILABLE; no allocation modified:',message,flush=True);access_error=message
  time.sleep(45);continue
 if r.returncode:raise RuntimeError('Live progress read failed; no task was modified: '+r.stderr)
 if access_error is not None:print('PROGRESS ACCESS RESTORED',flush=True);access_error=None
 records=json.loads(r.stdout);key=json.dumps(records,sort_keys=True)
 if key!=previous:
  record=dict(time=time.time(),allocations=records,admission=False)
  with output.open('a') as f:f.write(json.dumps(record)+'\n')
  print(json.dumps(record),flush=True);previous=key
 time.sleep(60)
