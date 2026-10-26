#!/usr/bin/env python3
"""Complete the declared Schwarzschild CPU controls in serial bounded jobs."""
import argparse,hashlib,json,os,shlex,shutil,subprocess,sys,time
from pathlib import Path

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();E=a.evidence.resolve();A=E/'aurora';scripts=Path(__file__).resolve().parent;primary_path=E/'cpu-schwarzschild-commensurate-finer/primary-review.json';R=E/'cpu-schwarzschild-finer-controls';journal=E/'schwarzschild-finer-controls-controller.jsonl'
 if R.exists() or journal.exists():raise RuntimeError('Existing controls controller/evidence must be inspected; never duplicate')
 while not primary_path.exists():time.sleep(15)
 primary=read(primary_path);scientific=read(E/'schwarzschild-commensurate-finer-plan.json');admission=read(E/'minkowski-stable-sbp-cpu-admission.json')
 if not primary['complete'] or not primary['field_checks_passed'] or not primary['finest_accuracy_passed'] or primary['resolutions']!=[36,48,60] or not admission['passed']:raise RuntimeError('Finer source-bound CPU primary gates must pass before controls')
 for file,digest in scientific['source_sha256'].items():
  if sha(Path('src/test_scalar')/file)!=digest or admission['delivery_source_sha256'][file]!=digest:raise RuntimeError('Declared scalar source changed')
 R.mkdir();base=A/'cpu-schwarzschild-finer-n60-v1';template=read(base/'cpu-plan.json');arguments=template['cases'][0]['arguments'];pending=[];finished={};parents={};rhs_collection=None
 for label,n,changes in [('time-half',48,{'--cfl':'.09'}),('time-quarter',48,{'--cfl':'.045'}),('inner',60,{'--excision':'.20'}),('cap09',60,{'--cap':'.9'}),('cap095',60,{'--cap':'.95'}),('causal-cap06',60,{'--cap':'.6'})]:
  args=list(arguments);args[args.index('--n')+1]=str(n)
  for key,value in changes.items():
   if key in args:args[args.index(key)+1]=value
   else:args.extend([key,value])
  pending.append(dict(key=label,label='schwarzschild-'+label,action='run',arguments=args))
 policy=dict(primary_review_sha256=sha(primary_path),scientific_plan_sha256=sha(E/'schwarzschild-commensurate-finer-plan.json'),source_sha256=scientific['source_sha256'],cases=pending,execution='Serial CPU-only one-node debug/CompactBinaryMerger/60min jobs; native wall checkpoints resumed with exact executable, input SHA and saved scalar settings; no completed case duplicated. First job also evaluates saved primary RHS without time stepping. No GPU or Kerr evolution launched.',source_sha256_controller=sha(Path(__file__)),declared_at=time.time());(R/'execution-plan.json').write_text(json.dumps(policy,indent=2)+'\n')
 host='hzhu@aurora.alcf.anl.gov';options=['-o','BatchMode=yes','-o','ControlPath=/Users/hz0693/.ssh/codex-control/%C'];ssh=['ssh',*options,host];remote_base='/lus/flare/projects/CompactBinaryMerger/hzhu/test-scalar-frozen-20261016'
 def remote_run(run):
  receipt=read(run.parent.parent/'collection-receipt.json');return receipt['remote_root']+'/results/'+receipt['job']+'/'+run.name
 for iteration in range(16):
  if not pending:break
  name=f'cpu-schwarzschild-finer-controls-v{iteration+1}';new=A/name
  if new.exists():raise RuntimeError('Existing controls preparation or submission: inspect it')
  new.mkdir();shutil.copytree(base/'source',new/'source')
  for file in scripts.glob('*'):
   if file.is_file() and file.suffix in ['.py','.cpp','.hpp']:shutil.copy2(file,new/'source/tst/test_scalar'/file.name)
  for file in ['cpu-admission.json','preserved-cpu-executable','scientific-plan.json']:shutil.copy2(base/file,new/file)
  plan=dict(template);plan['cases']=[{k:v for k,v in case.items() if k!='key'} for case in pending];plan['primary_review_sha256']=sha(primary_path);plan['controls_execution_plan_sha256']=sha(R/'execution-plan.json');plan['declared_before_submission']=time.time();(new/'cpu-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
  pbs=(base/'run_aurora.pbs').read_text()
  if iteration==0:
   audit=['/usr/bin/python3.11 source/tst/test_scalar/build_audit_executable.py --build '+remote_base+'/cpu-stable-endpoint-v1/build-cpu --source source/tst/test_scalar/audit_ringdown_rhs.cpp --output "$result/audit_ringdown_rhs"']
   for run,n in zip(map(Path,primary['runs']),primary['resolutions']):
    native=remote_run(run);staged='"$PWD/audit-inputs/${PBS_JOBID}/n'+str(n)+'"';out='"$result/actual-rhs-n'+str(n)+'"'
    audit.append('/lus/flare/projects/CompactBinaryMerger/hzhu/test-scalar-frozen-20261016/validation-python/bin/python source/tst/test_scalar/stage_rhs_inputs.py --run '+shlex.quote(native)+' --output '+staged)
    audit.append('mpiexec -n 1 --cpu-bind list:1-8 env OMP_NUM_THREADS=8 OMP_PROC_BIND=false "$result/audit_ringdown_rhs" '+str(n)+' 0 .25 '+staged+' '+out)
    audit.append('cp '+staged+'/input-provenance.json '+out+'/input-provenance.json')
   marker='mpiexec -n 1 --cpu-bind list:1-51,53-103';pbs=pbs.replace(marker,'\n'.join(audit)+'\n'+marker)
  (new/'run_aurora.pbs').write_text(pbs);archive=A/(name+'.tar.gz');subprocess.run([sys.executable,str(scripts/'package_aurora.py'),'--prepared',str(new),'--output',str(archive),'--extra','cpu-plan.json','scientific-plan.json','preserved-cpu-executable'],check=True)
  subprocess.run(['scp',*options,str(archive),host+':'+remote_base+'/'+archive.name],check=True);remote=remote_base+'/'+name
  program='root='+repr(remote)+'\n'+'''import os,subprocess,tarfile
from pathlib import Path
p=Path(root)
assert not p.exists(),'Existing remote controls must be inspected'
p.mkdir()
with tarfile.open(root+'.tar.gz') as t:t.extractall(p)
subprocess.run(['/usr/bin/python3.11',str(p/'source/tst/test_scalar/submit_cpu_aurora.py')],env={**os.environ,'SCALAR_ROOT':root},check=True)
'''
  submitted=subprocess.run([*ssh,'/usr/bin/python3.11 -'],input=program,text=True,capture_output=True);(new/'submission-access.log').write_text(submitted.stdout+submitted.stderr)
  if submitted.returncode:raise RuntimeError('Controls submission failed or unknown; inspect persistent remote lock')
  record=json.loads(submitted.stdout.strip());(new/'submission.json').write_text(json.dumps(record,indent=2)+'\n');print('CPU CONTROLS',record['job_id'],flush=True);collected=A/(name+'-collected')
  with journal.open('a') as f:f.write(json.dumps(dict(event='submitted',job=record,prepared=str(new)))+'\n')
  subprocess.run([sys.executable,str(scripts/'wait_collect_aurora.py'),'--root',remote,'--job',record['job_id'],'--output',str(collected),'--journal',str(A/(name+'-scheduler-journal.jsonl'))],check=True)
  receipt=read(collected/'collection-receipt.json');manifest=read(collected/'remote-manifest.json');scheduler=manifest['scheduler']['Jobs'][record['job_id']]
  if not receipt['passed'] or scheduler['Exit_status']!=0 or manifest['source_files']!=read(new/'portable-source-manifest.json') or sha(collected/'result/test_scalar')!=template['binary_sha256']:raise RuntimeError('Controls collection/source/binary/scheduler check failed')
  if iteration==0:
   rhs_collection=collected
   runs=[Path(d) for d in primary['runs']];audits=[rhs_collection/'result'/f'actual-rhs-n{n}' for n in primary['resolutions']];oracle=E/'radial-oracle/smooth'
   subprocess.run([sys.executable,str(scripts/'assess_ringdown_rhs.py'),'--runs',*[str(d) for d in runs],'--audits',*[str(d) for d in audits],'--oracle-waveform',str(oracle/'waveform-upwind-n1200.csv'),'--source','src/test_scalar','--output',str(R/'actual-rhs-review.json')],check=True)
  worker=read(collected/'result/cpu-checkpoint-worker-receipt.json');count=0
  for item,case in zip(worker['records'],pending):
   if item['label']!=case['label'] or item.get('skipped'):raise RuntimeError('Unexpected controls worker prefix')
   run=collected/'result'/case['label']
   if case['key'] in parents:subprocess.run([sys.executable,str(scripts/'map_checkpoint_parent.py'),'--child',str(run),'--parent',str(parents[case['key']])],check=True)
   if item['complete']:
    finished[case['key']]=run;count+=1
   else:
    if not item['checkpointed']:raise RuntimeError('Incomplete control has no native checkpoint')
    parents[case['key']]=run;case.update(action='resume',parent=remote_run(run),tlim=100,label=case['label']+'-resume');case.pop('arguments',None);break
  pending=pending[count:]
  with journal.open('a') as f:f.write(json.dumps(dict(event='collected',collection_sha256=sha(collected/'collection-receipt.json'),finished={k:str(v) for k,v in finished.items()},pending=[c['key'] for c in pending]))+'\n')
 if pending:raise RuntimeError('Bounded controls allocation limit exhausted; all checkpoints preserved')
 runs=[Path(d) for d in primary['runs']];oracle=E/'radial-oracle/smooth'
 subprocess.run([sys.executable,str(scripts/'assess_ringdown_controls.py'),'--primary-review',str(primary_path),'--time-runs',str(runs[1]),str(finished['time-half']),str(finished['time-quarter']),'--inner-run',str(finished['inner']),'--outer-runs',str(finished['cap09']),str(finished['cap095']),'--causal-run',str(finished['causal-cap06']),'--output',str(R/'controls-review.json')],check=True)
 subprocess.run([sys.executable,str(scripts/'assess_causal_boundary.py'),'--primary-review',str(primary_path),'--run',str(finished['causal-cap06']),'--output',str(R/'causal-review.json')],check=True)
 command=[sys.executable,str(scripts/'assemble_ringdown_admission.py'),'--evidence',str(E),'--plan',str(E/'schwarzschild-commensurate-finer-plan.json'),'--primary-review',str(primary_path),'--controls-review',str(R/'controls-review.json'),'--rhs-review',str(R/'actual-rhs-review.json'),'--causal-review',str(R/'causal-review.json'),'--previous-admission',str(E/'minkowski-stable-sbp-cpu-admission.json'),'--backend-review',str(E/'cpu-schwarzschild-stable/cpu-backend-review.json'),'--source','src/test_scalar','--exe',str(E/'cpu-v22/test_scalar'),'--output',str(E/'schwarzschild-stable-sbp-cpu-admission.json')]
 for name in ['normal-reference-review.json','fine-profile-review.json','qnm-review.json']:command.extend(['--oracle-review',str(oracle/name)])
 subprocess.run(command,check=True)
if __name__=='__main__':main()
