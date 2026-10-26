#!/usr/bin/env python3
"""Run and independently audit one fully CPU-admitted black-hole GPU stage."""
import argparse,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--stage',choices=['schwarzschild','kerr'],required=True);a=p.parse_args();E=a.evidence.resolve();A=E/'aurora';scripts=Path(__file__).resolve().parent;stage=a.stage;admission=E/(stage+'-stable-sbp-cpu-admission.json');name=stage+'-stable-gpu-v1';prepared=A/name;remote='/lus/flare/projects/CompactBinaryMerger/hzhu/test-scalar-frozen-20261016/'+name;host='hzhu@aurora.alcf.anl.gov';options=['-o','BatchMode=yes','-o','ControlPath=/Users/hz0693/.ssh/codex-control/%C'];ssh=['ssh',*options,host]
if prepared.exists():raise RuntimeError('Existing GPU attempt must be inspected; never duplicate')
while not admission.exists():time.sleep(15)
proof=json.loads(admission.read_text())
if not proof['passed'] or proof['stage']!=stage:raise RuntimeError('Full matching black-hole CPU admission required')
prepared.mkdir();shutil.copytree(A/'cpu-stable-endpoint-v1/source',prepared/'source');shutil.copytree(scripts,prepared/'source/tst/test_scalar',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'));shutil.copy2(admission,prepared/'cpu-admission.json');shutil.copy2(A/'minkowski-stable-gpu-v5/run_aurora.pbs',prepared/'run_aurora.pbs')
for name,digest in proof['delivery_source_sha256'].items():
 if hashlib.sha256((prepared/'source/src/test_scalar'/name).read_bytes()).hexdigest()!=digest:raise RuntimeError('GPU source differs from admitted CPU operator')
plan=dict(declared_before_submission=True,source_sha256=proof['delivery_source_sha256'],CPU_admission_sha256=hashlib.sha256(admission.read_bytes()).hexdigest(),resolutions=proof['resolutions'],cases=proof['gpu_cases'],CPU_GPU_scaled_max_tolerance=1e-9,coordinate_weight_tolerance=1e-11,relaxation=False,limits=dict(queue='debug',account='CompactBinaryMerger',nodes=1,minutes=60,max_task_allocations=1),scope='Same exact frozen physical trumpet, independent immutable Minkowski reference and fixed chart. Direct independent three-grid field/derivative/reduction and scalar QNM checks plus strict CPU agreement. No next stage or metric evolution launched.')
(prepared/'gpu-plan.json').write_text(json.dumps(plan,indent=2)+'\n');archive=prepared.with_suffix('.tar.gz');subprocess.run([sys.executable,str(scripts/'package_aurora.py'),'--prepared',str(prepared),'--output',str(archive),'--extra','gpu-plan.json'],check=True);subprocess.run(['scp',*options,str(archive),host+':'+remote+'.tar.gz'],check=True)
program='root='+repr(remote)+'\n'+'''import os,subprocess,tarfile
from pathlib import Path
p=Path(root)
assert not p.exists(),'Existing remote GPU attempt requires inspection'
p.mkdir()
with tarfile.open(root+'.tar.gz') as archive:archive.extractall(p)
subprocess.run(['/usr/bin/python3.11',str(p/'source/tst/test_scalar/submit_aurora.py')],env={**os.environ,'SCALAR_ROOT':root},check=True)
'''
r=subprocess.run([*ssh,'/usr/bin/python3.11 -'],input=program,text=True,capture_output=True);(prepared/'submission-access.log').write_text(r.stdout+r.stderr)
if r.returncode:raise RuntimeError('GPU submission failed or ambiguous; inspect persistent remote lock')
record=json.loads(r.stdout.strip());(prepared/'submission.json').write_text(json.dumps(record,indent=2)+'\n');print('RINGDOWN GPU JOB',record['job_id'],flush=True);collected=A/(prepared.name+'-collected')
subprocess.run([sys.executable,str(scripts/'wait_collect_aurora.py'),'--root',remote,'--job',record['job_id'],'--output',str(collected),'--journal',str(A/(prepared.name+'-scheduler-journal.jsonl'))],check=True)
worker=json.loads((collected/'result/gpu-receipt.json').read_text())
if len(worker['records'])!=3 or not all(r['complete'] and r['returncode']==0 for r in worker['records']):raise RuntimeError('GPU prefix incomplete; preserve checkpoint and explicitly continue with the same executable')
geometry=A/(prepared.name+'-geometry-review.json');sources=A/(prepared.name+'-source-review.json');physics=A/(prepared.name+'-physics-review.json');runs=[collected/'result'/f'{stage}-n{n}' for n in proof['resolutions']]
subprocess.run([sys.executable,str(scripts/'audit_geometry.py'),str(collected/'result/backend-geometry/geometry.csv'),'--smooth-layer','--output',str(geometry)],check=True)
subprocess.run([sys.executable,str(scripts/'audit_normal_sources_3d.py'),str(collected/'result/backend-geometry/geometry.csv'),'--output',str(sources)],check=True)
cpu=json.loads(Path(proof['primary_review']).read_text());oracle=Path(cpu['oracle']['waveform'])
if stage=='schwarzschild':
 coefficients=cpu['oracle'].get('normal_coefficients') or next(name for name in cpu['oracle']['files'] if Path(name).name.startswith('normal-coefficients-'))
 cmd=[sys.executable,str(scripts/'validate_schwarzschild.py'),'--runs',*[str(d) for d in runs],'--oracle',str(oracle.parent),'--oracle-waveform',oracle.name,'--normal-coefficients',coefficients,'--reference-sampler',cpu.get('reference_sampler','pchip'),'--output',str(physics)]
else:
 target=next(name for name in cpu['oracle']['files'] if Path(name).name=='spin-zero-qnm-both-branches.json');cmd=[sys.executable,str(scripts/'validate_kerr.py'),'--runs',*[str(d) for d in runs],'--oracle-waveform',str(oracle),'--reference-sampler',cpu.get('reference_sampler','pchip'),'--expected',target,'--output',str(physics)]
subprocess.run(cmd,check=True)
subprocess.run([sys.executable,str(scripts/'audit_gpu_stage.py'),'--gpu',str(collected/'result'),'--admission',str(admission),'--source-manifest',str(prepared/'portable-source-manifest.json'),'--cpu-review',proof['primary_review'],'--physics-review',str(physics),'--additional-audit',str(geometry),'--additional-audit',str(sources),'--output',str(A/(prepared.name+'-review.json'))],check=True)
