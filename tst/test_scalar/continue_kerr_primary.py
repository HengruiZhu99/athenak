#!/usr/bin/env python3
"""Run the predeclared Kerr CPU primary after previous-stage and oracle gates."""
import argparse,hashlib,json,os,shlex,shutil,subprocess,sys,time
from pathlib import Path

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();E=a.evidence.resolve();A=E/'aurora';scripts=Path(__file__).resolve().parent;reference=E/'kerr-oracle/stable-reference-review.json';R=E/'cpu-kerr-stable';journal=E/'kerr-primary-controller.jsonl'
 if R.exists() or journal.exists():raise RuntimeError('Existing Kerr primary controller must be inspected; never duplicate')
 while not reference.exists():time.sleep(15)
 admission_path=E/'schwarzschild-stable-sbp-cpu-admission.json';admission=read(admission_path);GPU=E/'aurora/schwarzschild-stable-gpu-v1-review.json';scientific=read(E/'kerr-stable-sbp-plan.json')
 if not read(reference)['passed'] or not read(GPU)['passed'] or not admission['passed'] or not read(E/'kerr-oracle/stable-oracle-source-preflight.json')['passed']:raise RuntimeError('Schwarzschild CPU/GPU and independent Kerr oracle gates required')
 for file,digest in scientific['source_sha256'].items():
  if sha(Path('src/test_scalar')/file)!=digest or admission['delivery_source_sha256'][file]!=digest:raise RuntimeError('Frozen scalar operator changed')
 R.mkdir();base=A/'cpu-schwarzschild-finer-n60-v1';template=read(base/'cpu-plan.json');pending=[];finished={};parents={}
 for n in scientific['resolutions']:
  args=['--n',str(n),'--tlim','100','--mass','1','--spin','.5','--cap','1','--excision','0','--boundary','radiation','--data','ringdown',*admission['numerical_arguments']];pending.append(dict(n=n,label=f'kerr-n{n}-normal',action='run',arguments=args))
 (R/'execution-plan.json').write_text(json.dumps(dict(cases=pending,source_sha256=scientific['source_sha256'],scientific_plan_sha256=sha(E/'kerr-stable-sbp-plan.json'),previous_CPU_admission_sha256=sha(admission_path),previous_GPU_review_sha256=sha(GPU),independent_reference_sha256=sha(reference),declared_at=time.time(),scope='Only complex-independent-validated real scalar on exact frozen Kerr trumpet; native exact checkpoint continuations, one-node debug/CMB/60min, one allocation at a time, no completed trajectory duplicated.'),indent=2)+'\n')
 host='hzhu@aurora.alcf.anl.gov';options=['-o','BatchMode=yes','-o','ControlPath=/Users/hz0693/.ssh/codex-control/%C'];ssh=['ssh',*options,host];remote_base='/lus/flare/projects/CompactBinaryMerger/hzhu/test-scalar-frozen-20261016'
 def remote_run(run):
  receipt=read(run.parent.parent/'collection-receipt.json');return receipt['remote_root']+'/results/'+receipt['job']+'/'+run.name
 for iteration in range(8):
  if not pending:break
  name=f'cpu-kerr-primary-v{iteration+1}';new=A/name
  if new.exists():raise RuntimeError('Existing Kerr preparation/submission requires inspection')
  new.mkdir();shutil.copytree(base/'source',new/'source')
  for file in scripts.glob('*'):
   if file.is_file() and file.suffix in ['.py','.cpp','.hpp']:shutil.copy2(file,new/'source/tst/test_scalar'/file.name)
  shutil.copy2(base/'preserved-cpu-executable',new/'preserved-cpu-executable');shutil.copy2(base/'run_aurora.pbs',new/'run_aurora.pbs');shutil.copy2(admission_path,new/'cpu-admission.json');shutil.copy2(E/'kerr-stable-sbp-plan.json',new/'scientific-plan.json');shutil.copy2(reference,new/'independent-reference-review.json')
  plan=dict(template);plan['previous_stage']='schwarzschild';plan['cases']=[{k:v for k,v in c.items() if k!='n'} for c in pending];plan['scientific_plan_sha256']=sha(new/'scientific-plan.json');plan['primary_execution_plan_sha256']=sha(R/'execution-plan.json');plan['previous_GPU_review_sha256']=sha(GPU);plan['independent_reference_sha256']=sha(reference);plan['declared_before_submission']=time.time();(new/'cpu-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
  for obsolete in ['reuse','failed_matrix_sha256']:plan.pop(obsolete,None)
  plan['requires']='Full Schwarzschild CPU/GPU and independent Kerr scalar reference gates passed; only the predeclared frozen Kerr primary36/48/60 and exact continuations.';(new/'cpu-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
  archive=A/(name+'.tar.gz');subprocess.run([sys.executable,str(scripts/'package_aurora.py'),'--prepared',str(new),'--output',str(archive),'--extra','cpu-plan.json','scientific-plan.json','preserved-cpu-executable','independent-reference-review.json'],check=True);subprocess.run(['scp',*options,str(archive),host+':'+remote_base+'/'+archive.name],check=True);remote=remote_base+'/'+name
  program='root='+repr(remote)+'\n'+'''import os,subprocess,tarfile
from pathlib import Path
p=Path(root)
assert not p.exists(),'Existing remote Kerr job must be inspected'
p.mkdir()
with tarfile.open(root+'.tar.gz') as t:t.extractall(p)
subprocess.run(['/usr/bin/python3.11',str(p/'source/tst/test_scalar/submit_cpu_aurora.py')],env={**os.environ,'SCALAR_ROOT':root},check=True)
'''
  submitted=subprocess.run([*ssh,'/usr/bin/python3.11 -'],input=program,text=True,capture_output=True);(new/'submission-access.log').write_text(submitted.stdout+submitted.stderr)
  if submitted.returncode:raise RuntimeError('Kerr submission failed or unknown; inspect persistent remote lock')
  record=json.loads(submitted.stdout.strip());(new/'submission.json').write_text(json.dumps(record,indent=2)+'\n');print('KERR CPU PRIMARY',record['job_id'],flush=True);collected=A/(name+'-collected')
  with journal.open('a') as f:f.write(json.dumps(dict(event='submitted',job=record,prepared=str(new)))+'\n')
  subprocess.run([sys.executable,str(scripts/'wait_collect_aurora.py'),'--root',remote,'--job',record['job_id'],'--output',str(collected),'--journal',str(A/(name+'-scheduler-journal.jsonl'))],check=True)
  receipt=read(collected/'collection-receipt.json');manifest=read(collected/'remote-manifest.json');scheduler=manifest['scheduler']['Jobs'][record['job_id']]
  if not receipt['passed'] or scheduler['Exit_status']!=0 or manifest['source_files']!=read(new/'portable-source-manifest.json') or sha(collected/'result/test_scalar')!=template['binary_sha256']:raise RuntimeError('Kerr collection/source/binary/scheduler check failed')
  worker=read(collected/'result/cpu-checkpoint-worker-receipt.json');count=0
  for item,case in zip(worker['records'],pending):
   if item['label']!=case['label'] or item.get('skipped'):raise RuntimeError('Unexpected Kerr worker prefix')
   run=collected/'result'/case['label']
   if case['n'] in parents:subprocess.run([sys.executable,str(scripts/'map_checkpoint_parent.py'),'--child',str(run),'--parent',str(parents[case['n']])],check=True)
   if item['complete']:finished[case['n']]=run;count+=1
   else:
    if not item['checkpointed']:raise RuntimeError('Kerr prefix has no confirmed wall checkpoint')
    parents[case['n']]=run;case.update(action='resume',parent=remote_run(run),tlim=100,label=case['label']+'-resume');case.pop('arguments',None);break
  pending=pending[count:]
  with journal.open('a') as f:f.write(json.dumps(dict(event='collected',collection_sha256=sha(collected/'collection-receipt.json'),finished={n:str(d) for n,d in finished.items()},pending=[c['n'] for c in pending]))+'\n')
 if pending:raise RuntimeError('Bounded Kerr allocation count exhausted; all checkpoints preserved')
 subprocess.run([sys.executable,str(scripts/'validate_kerr.py'),'--runs',*[str(finished[n]) for n in scientific['resolutions']],'--oracle-waveform',read(reference)['waveform'],'--reference-sampler',read(reference)['reference_sampler'],'--expected',str(E/'spin-zero-qnm-both-branches.json'),'--output',str(R/'primary-review.json')],check=True)
 subprocess.run([sys.executable,str(scripts/'certify_kerr_reference_uncertainty.py'),'--primary-review',str(R/'primary-review.json'),'--reference-review',str(reference),'--output',str(R/'reference-uncertainty-review.json')],check=True)
if __name__=='__main__':main()
