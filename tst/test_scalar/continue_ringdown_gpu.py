#!/usr/bin/env python3
"""Finish only incomplete admitted GPU resolutions, then audit their ancestry."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--stage',choices=['schwarzschild','kerr'],required=True);a=p.parse_args();E=a.evidence.resolve();A=E/'aurora';stage=a.stage;scripts=Path(__file__).resolve().parent;first=A/(stage+'-stable-gpu-v1');initial=A/(first.name+'-collected');admission=E/(stage+'-stable-sbp-cpu-admission.json');journal=A/(stage+'-gpu-continuation-controller.jsonl');completion=A/(stage+'-gpu-completed-cases.json')
 if journal.exists() or completion.exists():raise RuntimeError('Existing GPU continuation must be inspected; never duplicate')
 while not (initial/'collection-receipt.json').exists():time.sleep(15)
 proof=read(admission);worker=read(initial/'result/gpu-receipt.json');ns=proof['resolutions'];binary=sha(initial/'result/test_scalar')
 if not proof['passed'] or worker['scientific_admission_sha256']!=sha(admission):raise RuntimeError('Matching full GPU scientific CPU admission required')
 for file,digest in proof['delivery_source_sha256'].items():
  if sha(Path('src/test_scalar')/file)!=digest:raise RuntimeError('Current scalar operator differs from the preserved GPU stage')
 def verify_collection(collected,prepared):
  receipt=read(collected/'collection-receipt.json');manifest=read(collected/'remote-manifest.json');job=manifest['scheduler']['Jobs'][receipt['job']]
  if not receipt['passed'] or job['job_state']!='F' or job['Exit_status']!=0 or manifest['source_files']!=read(prepared/'portable-source-manifest.json') or manifest['admission_sha256']!=sha(admission) or sha(collected/'result/test_scalar')!=binary:raise RuntimeError('GPU continuation source/scheduler/collection/executable check failed')
 verify_collection(initial,first)
 if len(worker['records'])==3 and all(r['complete'] and r['returncode']==0 for r in worker['records']):print('No GPU continuation required; original controller owns the complete-stage audit',flush=True);return
 records={r['n']:r for r in worker['records']}
 if len(records)!=len(worker['records']) or any(n not in ns or r['returncode'] for n,r in records.items()):raise RuntimeError('Failed or unexpected GPU prefix; preserve for inspection')
 finished={};parents={};pending=[]
 def remote_run(run):
  receipt=read(run.parent.parent/'collection-receipt.json');return receipt['remote_root']+'/results/'+receipt['job']+'/'+run.name
 for case in proof['gpu_cases']:
  n=case['n'];r=records.get(n);run=initial/'result'/f'{stage}-n{n}'
  if r and r['complete']:finished[n]=run;continue
  item=dict(n=n,label=f'{stage}-n{n}-continuation1',action='resume' if r else 'run',admitted_arguments=case['arguments'])
  if r:
   if not r['checkpointed']:raise RuntimeError('Incomplete GPU prefix has no confirmed native wall checkpoint')
   parents[n]=run;item['parent']=remote_run(run)
  pending.append(item)
 host='hzhu@aurora.alcf.anl.gov';options=['-o','BatchMode=yes','-o','ControlPath=/Users/hz0693/.ssh/codex-control/%C'];ssh=['ssh',*options,host];remote_base='/lus/flare/projects/CompactBinaryMerger/hzhu/test-scalar-frozen-20261016';collections=[initial];preparations=[first]
 with journal.open('x') as f:f.write(json.dumps(dict(event='confirmed_incomplete_prefix',initial_collection_sha256=sha(initial/'collection-receipt.json'),binary_sha256=binary,pending=pending,completed={n:str(d) for n,d in finished.items()}))+'\n')
 for iteration in range(24):
  if not pending:break
  name=f'{stage}-stable-gpu-resume{iteration+1}';new=A/name
  if new.exists():raise RuntimeError('Existing GPU continuation preparation/submission requires inspection')
  new.mkdir();shutil.copytree(first/'source',new/'source');shutil.copytree(scripts,new/'source/tst/test_scalar',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'));shutil.copy2(admission,new/'cpu-admission.json');shutil.copy2(initial/'result/test_scalar',new/'preserved-gpu-executable')
  plan=dict(kind='gpu_validation',cases=pending,binary_sha256=binary,cpu_admission_sha256=sha(admission),source_sha256=proof['delivery_source_sha256'],declared_at=time.time(),requires='Only missing exact CPU-admitted GPU prefixes. Original executable, native checkpoint SHA and scalar/geometry settings; no completed resolution repeated.',limits=dict(queue='debug',account='CompactBinaryMerger',nodes=1,minutes=60,max_task_allocations=1));(new/'gpu-checkpoint-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
  pbs=(first/'run_aurora.pbs').read_text();start=pbs.index('cmake -S');end=pbs.index('export ONEAPI_DEVICE_SELECTOR',start);pbs=pbs[:start]+'cp preserved-gpu-executable "$result/test_scalar"\nchmod 755 "$result/test_scalar"\nsha256sum "$result/test_scalar" >"$result/executable.sha256"\n'+pbs[end:];old='/usr/bin/python3.11 source/tst/test_scalar/gpu_worker.py \\\n  --exe "$result/test_scalar" --output "$result" --admission cpu-admission.json';replacement='mpiexec -n 1 --cpu-bind list:1-8 /usr/bin/python3.11 source/tst/test_scalar/gpu_checkpoint_worker.py --exe "$result/test_scalar" --output "$result" --admission cpu-admission.json --plan gpu-checkpoint-plan.json'
  if old not in pbs:raise RuntimeError('Original GPU PBS worker template differs; preserve and inspect')
  pbs=pbs.replace(old,replacement);(new/'run_aurora.pbs').write_text(pbs);archive=A/(name+'.tar.gz');subprocess.run([sys.executable,str(scripts/'package_aurora.py'),'--prepared',str(new),'--output',str(archive),'--extra','gpu-checkpoint-plan.json','preserved-gpu-executable'],check=True);subprocess.run(['scp',*options,str(archive),host+':'+remote_base+'/'+archive.name],check=True);remote=remote_base+'/'+name
  program='root='+repr(remote)+'\n'+'''import os,subprocess,tarfile
from pathlib import Path
p=Path(root)
assert not p.exists(),'Existing remote GPU continuation requires inspection'
p.mkdir()
with tarfile.open(root+'.tar.gz') as archive:archive.extractall(p)
subprocess.run(['/usr/bin/python3.11',str(p/'source/tst/test_scalar/submit_aurora.py')],env={**os.environ,'SCALAR_ROOT':root},check=True)
'''
  submitted=subprocess.run([*ssh,'/usr/bin/python3.11 -'],input=program,text=True,capture_output=True);(new/'submission-access.log').write_text(submitted.stdout+submitted.stderr)
  if submitted.returncode:raise RuntimeError('GPU continuation submission failed or unknown; inspect persistent remote lock')
  record=json.loads(submitted.stdout.strip());(new/'submission.json').write_text(json.dumps(record,indent=2)+'\n');print('EXACT GPU CONTINUATION',record['job_id'],flush=True);collected=A/(name+'-collected')
  with journal.open('a') as f:f.write(json.dumps(dict(event='submitted',job=record,prepared=str(new)))+'\n')
  subprocess.run([sys.executable,str(scripts/'wait_collect_aurora.py'),'--root',remote,'--job',record['job_id'],'--output',str(collected),'--journal',str(A/(name+'-scheduler-journal.jsonl'))],check=True);verify_collection(collected,new);collections.append(collected);preparations.append(new);r=read(collected/'result/gpu-checkpoint-worker-receipt.json');count=0
  for item,case in zip(r['records'],pending):
   if item['n']!=case['n'] or item['label']!=case['label']:raise RuntimeError('Unexpected GPU continuation worker prefix')
   run=collected/'result'/case['label'];n=case['n']
   if n in parents:subprocess.run([sys.executable,str(scripts/'map_checkpoint_parent.py'),'--child',str(run),'--parent',str(parents[n])],check=True)
   if item['complete']:finished[n]=run;count+=1
   else:
    if not item['checkpointed']:raise RuntimeError('Incomplete GPU continuation lacks confirmed wall checkpoint')
    parents[n]=run;case.update(action='resume',parent=remote_run(run),label=f'{stage}-n{n}-continuation{iteration+2}');break
  pending=pending[count:]
  with journal.open('a') as f:f.write(json.dumps(dict(event='collected',receipt_sha256=sha(collected/'collection-receipt.json'),pending=[c['n'] for c in pending],finished={n:str(d) for n,d in finished.items()}))+'\n')
 if pending:raise RuntimeError('Bounded GPU continuation count exhausted; native checkpoints preserved')
 runs=[finished[n] for n in ns];completion.write_text(json.dumps(dict(runs=[str(d) for d in runs],collections=[str(d) for d in collections],prepared=[str(d) for d in preparations],cpu_admission_sha256=sha(admission),binary_sha256=binary,source_sha256=sha(Path(__file__))),indent=2)+'\n');geometry=A/(first.name+'-geometry-review.json');sources=A/(first.name+'-source-review.json');physics=A/(first.name+'-completed-physics-review.json')
 for script,output,extra in [('audit_geometry.py',geometry,['--smooth-layer']),('audit_normal_sources_3d.py',sources,[])]:
  if output.exists():
   if not read(output)['passed']:raise RuntimeError('Existing original GPU geometry/source audit failed')
  else:subprocess.run([sys.executable,str(scripts/script),str(initial/'result/backend-geometry/geometry.csv'),*extra,'--output',str(output)],check=True)
 cpu=read(Path(proof['primary_review']));oracle=Path(cpu['oracle']['waveform'])
 if stage=='schwarzschild':
  coefficients=cpu['oracle'].get('normal_coefficients') or next(name for name in cpu['oracle']['files'] if Path(name).name.startswith('normal-coefficients-'))
  command=[sys.executable,str(scripts/'validate_schwarzschild.py'),'--runs',*[str(d) for d in runs],'--oracle',str(oracle.parent),'--oracle-waveform',oracle.name,'--normal-coefficients',coefficients,'--reference-sampler',cpu.get('reference_sampler','pchip'),'--output',str(physics)]
 else:
  target=next(name for name in cpu['oracle']['files'] if Path(name).name=='spin-zero-qnm-both-branches.json');command=[sys.executable,str(scripts/'validate_kerr.py'),'--runs',*[str(d) for d in runs],'--oracle-waveform',str(oracle),'--reference-sampler',cpu.get('reference_sampler','pchip'),'--expected',target,'--output',str(physics)]
 subprocess.run(command,check=True);subprocess.run([sys.executable,str(scripts/'audit_gpu_continuations.py'),'--admission',str(admission),'--cpu-review',proof['primary_review'],'--physics-review',str(physics),'--runs',*[str(d) for d in runs],'--collections',*[str(d) for d in collections],'--prepared',*[str(d) for d in preparations],'--additional-audit',str(geometry),'--additional-audit',str(sources),'--output',str(A/(first.name+'-review.json'))],check=True)
if __name__=='__main__':main()
