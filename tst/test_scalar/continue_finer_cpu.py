#!/usr/bin/env python3
"""Collect and, only if necessary, exactly resume the declared N60 CPU case."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();E=a.evidence.resolve();A=E/'aurora';scripts=Path(__file__).resolve().parent
 name='cpu-schwarzschild-finer-n60-v1';prepared=A/name;collected=A/(name+'-collected');label='schwarzschild-n60-fixed-cube';previous=None
 remote_base='/lus/flare/projects/CompactBinaryMerger/hzhu/test-scalar-frozen-20261016';host='hzhu@aurora.alcf.anl.gov';options=['-o','BatchMode=yes','-o','ControlPath=/Users/hz0693/.ssh/codex-control/%C'];ssh=['ssh',*options,host]
 output=E/'cpu-schwarzschild-commensurate-finer';review=output/'primary-review.json'
 if output.exists():raise RuntimeError('Existing finer analysis is immutable; inspect it')
 for attempt in range(4):
  while not (collected/'collection-receipt.json').exists():time.sleep(15)
  receipt=json.loads((collected/'collection-receipt.json').read_text());manifest=json.loads((collected/'remote-manifest.json').read_text());plan=json.loads((prepared/'cpu-plan.json').read_text());expected=json.loads((prepared/'portable-source-manifest.json').read_text());run=collected/'result'/label
  if not receipt['passed'] or manifest['source_files']!=expected or sha(collected/'result/test_scalar')!=plan['binary_sha256']:raise RuntimeError('Continuation collection/source/binary mismatch')
  for file,digest in plan['source_sha256'].items():
   if sha(Path('src/test_scalar')/file)!=digest:raise RuntimeError('Current delivered scalar source changed')
  native=json.loads((run/('restart-receipt.json' if previous else 'receipt.json')).read_text())
  if native['returncode']!=0 or not json.loads((run/'frozen-integrity.json').read_text())['unchanged']:raise RuntimeError('Failed native scalar attempt')
  if previous:
   subprocess.run([sys.executable,str(scripts/'map_checkpoint_parent.py'),'--child',str(run),'--parent',str(previous)],check=True)
  if (run/'complete.txt').exists():
   if 'time=100 ' not in (run/'complete.txt').read_text():raise RuntimeError('Unexpected final scalar time')
   base=A/'cpu-schwarzschild-commensurate-v1-collected/result';runs=[base/'schwarzschild-n36-fixed-cube',base/'schwarzschild-n48-fixed-cube',run];oracle=E/'radial-oracle/smooth';output.mkdir()
   subprocess.run([sys.executable,str(scripts/'validate_schwarzschild.py'),'--runs',*[str(d) for d in runs],'--oracle',str(oracle),'--oracle-waveform','waveform-upwind-n1200.csv','--normal-coefficients',str(oracle/'normal-coefficients-n1200.txt'),'--output',str(review)],check=True)
   r=json.loads(review.read_text());print('FINER CPU MATRIX',json.dumps(dict(field_checks_passed=r['field_checks_passed'],finest_accuracy_passed=r['finest_accuracy_passed'])),flush=True);return
  if not (run/'checkpointed.txt').exists() or sha(run/'checkpoint.f64')!=native['files']['checkpoint.f64']:raise RuntimeError('Missing or altered native wall checkpoint')
  if attempt==3:raise RuntimeError('Three exact bounded continuations exhausted; checkpoint preserved for inspection')
  previous=run;parent_remote=receipt['remote_root']+'/results/'+receipt['job']+'/'+label;name=f'cpu-schwarzschild-finer-n60-resume{attempt+1}';new=A/name
  if new.exists():raise RuntimeError('Existing continuation preparation: inspect its submission lock; never duplicate')
  new.mkdir();shutil.copytree(prepared/'source',new/'source')
  for file in ['cpu-admission.json','preserved-cpu-executable','run_aurora.pbs','scientific-plan.json']:shutil.copy2(prepared/file,new/file)
  label=f'schwarzschild-n60-fixed-cube-resume{attempt+1}';plan['cases']=[dict(label=label,action='resume',parent=parent_remote,tlim=100)];plan['parent_collection_sha256']=sha(collected/'collection-receipt.json');plan['parent_checkpoint_sha256']=sha(run/'checkpoint.f64');plan['declared_before_submission']=time.time();(new/'cpu-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
  archive=A/(name+'.tar.gz');subprocess.run([sys.executable,str(scripts/'package_aurora.py'),'--prepared',str(new),'--output',str(archive),'--extra','cpu-plan.json','scientific-plan.json','preserved-cpu-executable'],check=True)
  subprocess.run(['scp',*options,str(archive),host+':'+remote_base+'/'+archive.name],check=True)
  remote=remote_base+'/'+name
  program='root='+repr(remote)+'\n'+'''import os,subprocess,tarfile
from pathlib import Path
p=Path(root)
assert not p.exists(),'Existing remote continuation must be inspected'
p.mkdir()
with tarfile.open(root+'.tar.gz') as t:t.extractall(p)
subprocess.run(['/usr/bin/python3.11',str(p/'source/tst/test_scalar/submit_cpu_aurora.py')],env={**os.environ,'SCALAR_ROOT':root},check=True)
'''
  result=subprocess.run([*ssh,'/usr/bin/python3.11 -'],input=program,text=True,capture_output=True);(new/'submission-access.log').write_text(result.stdout+result.stderr)
  if result.returncode:raise RuntimeError('Continuation submission failed or unknown; inspect preserved remote lock')
  submission=json.loads(result.stdout.strip());(new/'submission.json').write_text(json.dumps(submission,indent=2)+'\n');print('EXACT CPU CONTINUATION',submission['job_id'],flush=True);prepared=new;collected=A/(name+'-collected')
  subprocess.run([sys.executable,str(scripts/'wait_collect_aurora.py'),'--root',remote,'--job',submission['job_id'],'--output',str(collected),'--journal',str(A/(name+'-scheduler-journal.jsonl'))],check=True)
if __name__=='__main__':main()
