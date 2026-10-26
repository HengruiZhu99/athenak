#!/usr/bin/env python3
"""Exactly continue the declared N96 CPU run, then assess the dyadic family."""
import argparse,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);mode=p.add_mutually_exclusive_group();mode.add_argument('--replace-initial-waiter',action='store_true');mode.add_argument('--recover-after-access-loss',action='store_true');a=p.parse_args();E=a.evidence.resolve();A=E/'aurora';scripts=Path(__file__).resolve().parent;name='cpu-schwarzschild-dyadic-n96-v1';prepared=A/name;collected=A/(name+'-collected');label='schwarzschild-n96-fixed-cube';previous=None;R=E/'cpu-schwarzschild-dyadic';journal=E/'schwarzschild-dyadic-controller.jsonl';first_attempt=0
if a.recover_after_access_loss:
 interruption=E/'access-recovery-20261017/interruption.json'
 if not interruption.exists() or not journal.exists() or (R/'controller-v3.py').exists():raise RuntimeError('Recovery requires inspected interruption evidence and a fresh controller receipt')
 old=read(R/'execution-policy-v2.json')
 if sha(R/'controller-v2.py')!=old['controller_sha256'] or old['scientific_plan_sha256']!=sha(E/'schwarzschild-dyadic-plan.json'):raise RuntimeError('Preserved controller or scientific plan changed')
 rows=[json.loads(line) for line in journal.read_text().splitlines()]
 if not rows or len(rows)>=16 or any(row['event']!='collected' or row['complete'] for row in rows):raise RuntimeError('Recovery must start after a verified unfinished allocation, without duplicate scientific assessment')
 for index,row in enumerate(rows):
  stem='cpu-schwarzschild-dyadic-n96-'+('v1' if index==0 else f'resume{index}');expected=A/(stem+'-collected')/'result'/('schwarzschild-n96-fixed-cube'+('' if index==0 else f'-resume{index}'))
  if Path(row['run'])!=expected or sha(expected.parent.parent/'collection-receipt.json')!=row['collection_sha256']:raise RuntimeError('Collected checkpoint journal does not describe a unique exact lineage')
 previous=Path(rows[-1]['run']);first_attempt=len(rows);name=f'cpu-schwarzschild-dyadic-n96-resume{first_attempt}';prepared=A/name;collected=A/(name+'-collected');label=f'schwarzschild-n96-fixed-cube-resume{first_attempt}';submitted=read(prepared/'submission.json');plan=read(prepared/'cpu-plan.json');watch=A/(name+'-scheduler-journal.jsonl')
 if submitted['job_id']!=read(interruption)['last_submitted_job'] or submitted['plan_sha256']!=sha(prepared/'cpu-plan.json') or submitted['root']!='/lus/flare/projects/CompactBinaryMerger/hzhu/test-scalar-frozen-20261016/'+name or submitted['source_manifest_sha256']!=sha(prepared/'portable-source.sha256'):raise RuntimeError('Existing pending submission differs from its immutable receipt')
 if plan['parent_collection_sha256']!=rows[-1]['collection_sha256'] or plan['parent_checkpoint_sha256']!=sha(previous/'checkpoint.f64'):raise RuntimeError('Pending submission is not the next exact checkpoint continuation')
 if collected.exists():raise RuntimeError('Inspect any partial collection explicitly before recovery; never overwrite it')
 suffix='-v3'
elif a.replace_initial_waiter:
 if journal.exists() or (collected/'collection-receipt.json').exists() or not (R/'waiting-controller-replacement.json').exists() or (R/'controller-v2.py').exists():raise RuntimeError('Replacement is restricted to the explicitly stopped initial waiter; never duplicate continuation actions')
 old=read(R/'execution-policy.json')
 if sha(R/'controller.py')!=old['controller_sha256']:raise RuntimeError('Preserved original controller differs from its receipt')
 suffix='-v2'
else:
 if R.exists() or journal.exists():raise RuntimeError('Existing dyadic controller/evidence must be inspected; never duplicate')
 R.mkdir();suffix=''
controller_lock=(R/'primary-controller.lock').open('a');fcntl.flock(controller_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
shutil.copy2(Path(__file__),R/('controller'+suffix+'.py'));(R/('execution-policy'+suffix+'.json')).write_text(json.dumps(dict(controller_sha256=sha(Path(__file__)),scientific_plan_sha256=sha(E/'schwarzschild-dyadic-plan.json'),maximum_serial_CPU_allocations=16,first_attempt=first_attempt,one_task_allocation_at_a_time=True,time_limit_per_allocation_minutes=60,checkpoint_resume='Same binary/source, saved scalar parameters and frozen physical/reference coefficients only.',oracle_selection='Only an immutable completed independent-reference selection, without any3D change.',recovery='Inspect and collect the already-submitted allocation before considering any further exact continuation.' if a.recover_after_access_loss else None),indent=2)+'\n');host='hzhu@aurora.alcf.anl.gov';options=['-o','BatchMode=yes','-o','ControlPath=/Users/hz0693/.ssh/codex-control/%C'];ssh=['ssh',*options,host];remote_base='/lus/flare/projects/CompactBinaryMerger/hzhu/test-scalar-frozen-20261016'
if a.recover_after_access_loss:
 subprocess.run([sys.executable,str(scripts/'wait_collect_aurora.py'),'--root',submitted['root'],'--job',submitted['job_id'],'--output',str(collected),'--journal',str(watch),*(['--resume-watcher'] if watch.exists() else [])],check=True)
for attempt in range(first_attempt,16):
 while not (collected/'collection-receipt.json').exists():time.sleep(15)
 receipt=read(collected/'collection-receipt.json');manifest=read(collected/'remote-manifest.json');plan=read(prepared/'cpu-plan.json');run=collected/'result'/label;scheduler=manifest['scheduler']['Jobs'][receipt['job']]
 if not receipt['passed'] or scheduler.get('Exit_status')!=0 or manifest['source_mismatches'] or manifest['source_files']!=read(prepared/'portable-source-manifest.json') or sha(collected/'result/test_scalar')!=plan['binary_sha256']:raise RuntimeError('Exact CPU collection/source/binary/scheduler integrity failed')
 for file,digest in plan['source_sha256'].items():
  if sha(Path('src/test_scalar')/file)!=digest:raise RuntimeError('Declared scalar source changed')
 native=read(run/('restart-receipt.json' if previous else 'receipt.json'));frozen=read(run/'frozen-integrity.json')
 if native['returncode']!=0 or not frozen['unchanged'] or sha(run/'checkpoint.f64')!=native['files']['checkpoint.f64']:raise RuntimeError('Failed or altered native scalar attempt')
 if previous:subprocess.run([sys.executable,str(scripts/'map_checkpoint_parent.py'),'--child',str(run),'--parent',str(previous)],check=True)
 if attempt==0 and (R/'closed-t2-prefix/prefix-collection-receipt.json').exists():
  prefix=read(R/'closed-t2-prefix/prefix-collection-receipt.json')
  if sha(run/'fields-t2.000.f64')!=prefix['local_files']['fields-t2.000.f64']:raise RuntimeError('Early closed snapshot differs from the completed native allocation payload')
 with journal.open('a') as f:f.write(json.dumps(dict(event='collected',run=str(run),collection_sha256=sha(collected/'collection-receipt.json'),complete=(run/'complete.txt').exists()))+'\n')
 if (run/'complete.txt').exists():
  if 'time=100 ' not in (run/'complete.txt').read_text():raise RuntimeError('Unexpected final scalar time')
  base=A/'cpu-schwarzschild-commensurate-v1-collected/result';runs=[base/'schwarzschild-n24-fixed-cube',base/'schwarzschild-n48-fixed-cube',run];oracle=E/'radial-oracle/smooth';review=R/'primary-review.json'
  selection=read(R/'reference-selection.json');selected=selection['reference_n'];previous_n=selection['previous_n']
  if not selection['passed'] or any(not Path(file).is_file() or sha(Path(file))!=digest for file,digest in selection['input_sha256'].items()):raise RuntimeError('Selected independent reference is incomplete or altered')
  subprocess.run([sys.executable,str(scripts/'validate_schwarzschild.py'),'--runs',*[str(d) for d in runs],'--oracle',str(oracle),'--oracle-waveform',f'waveform-upwind-n{selected}.csv','--normal-coefficients',str(oracle/f'normal-coefficients-n{selected}.txt'),'--reference-sampler','lagrange6','--output',str(review)],check=True)
  subprocess.run([sys.executable,str(scripts/'certify_schwarzschild_reference.py'),'--evidence',str(E),'--primary-review',str(review),'--reference-n',str(selected),'--previous-n',str(previous_n),'--output',str(R/'reference-certificate.json')],check=True)
  r=read(review);print('DYADIC CPU MATRIX',json.dumps(dict(field_checks_passed=r['field_checks_passed'],finest_accuracy_passed=r['finest_accuracy_passed'])),flush=True);break
 if not (run/'checkpointed.txt').exists():raise RuntimeError('Incomplete scalar state lacks a native wall checkpoint')
 if attempt==15:raise RuntimeError('Bounded continuation count exhausted; every native checkpoint preserved')
 previous=run;parent_remote=receipt['remote_root']+'/results/'+receipt['job']+'/'+label;name=f'cpu-schwarzschild-dyadic-n96-resume{attempt+1}';new=A/name
 if new.exists():raise RuntimeError('Existing continuation preparation or submission; inspect it')
 new.mkdir();shutil.copytree(prepared/'source',new/'source')
 for file in ['cpu-admission.json','preserved-cpu-executable','run_aurora.pbs','scientific-plan.json']:shutil.copy2(prepared/file,new/file)
 label=f'schwarzschild-n96-fixed-cube-resume{attempt+1}';plan['cases']=[dict(label=label,action='resume',parent=parent_remote,tlim=100)];plan['parent_collection_sha256']=sha(collected/'collection-receipt.json');plan['parent_checkpoint_sha256']=sha(run/'checkpoint.f64');plan['declared_before_submission']=time.time();(new/'cpu-plan.json').write_text(json.dumps(plan,indent=2)+'\n')
 archive=A/(name+'.tar.gz');subprocess.run([sys.executable,str(scripts/'package_aurora.py'),'--prepared',str(new),'--output',str(archive),'--extra','cpu-plan.json','scientific-plan.json','preserved-cpu-executable'],check=True);subprocess.run(['scp',*options,str(archive),host+':'+remote_base+'/'+archive.name],check=True);remote=remote_base+'/'+name
 program='root='+repr(remote)+'\n'+'''import os,subprocess,tarfile
from pathlib import Path
p=Path(root)
assert not p.exists(),'Existing remote continuation must be inspected'
p.mkdir()
with tarfile.open(root+'.tar.gz') as t:t.extractall(p)
subprocess.run(['/usr/bin/python3.11',str(p/'source/tst/test_scalar/submit_cpu_aurora.py')],env={**os.environ,'SCALAR_ROOT':root},check=True)
'''
 submitted=subprocess.run([*ssh,'/usr/bin/python3.11 -'],input=program,text=True,capture_output=True);(new/'submission-access.log').write_text(submitted.stdout+submitted.stderr)
 if submitted.returncode:raise RuntimeError('Continuation submission failed or unknown; inspect persistent lock')
 record=json.loads(submitted.stdout.strip());(new/'submission.json').write_text(json.dumps(record,indent=2)+'\n');print('EXACT DYADIC CPU CONTINUATION',record['job_id'],flush=True);prepared=new;collected=A/(name+'-collected')
 subprocess.run([sys.executable,str(scripts/'wait_collect_aurora.py'),'--root',remote,'--job',record['job_id'],'--output',str(collected),'--journal',str(A/(name+'-scheduler-journal.jsonl'))],check=True)
