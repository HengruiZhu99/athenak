#!/usr/bin/env python3
"""Submit and audit one corrected GPU matrix only after full CPU admission."""
import argparse,json,shutil,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();E=a.evidence.resolve();A=E/'aurora';scripts=Path(__file__).resolve().parent;admission=E/'minkowski-stable-sbp-cpu-admission.json';prepared=A/'minkowski-stable-gpu-v5';host='hzhu@aurora.alcf.anl.gov';remote='/lus/flare/projects/CompactBinaryMerger/hzhu/test-scalar-frozen-20261016/minkowski-stable-gpu-v5';ssh=['ssh','-o','BatchMode=yes','-o','ControlPath=/Users/hz0693/.ssh/codex-control/%C',host]
if prepared.exists():raise RuntimeError('Existing corrected GPU attempt: inspect it; never duplicate')
while not admission.exists():time.sleep(15)
proof=json.loads(admission.read_text())
if not proof['passed']:raise RuntimeError('Corrected full CPU admission failed')
prepared.mkdir();shutil.copytree(A/'cpu-stable-endpoint-v1/source',prepared/'source');shutil.copytree(scripts,prepared/'source/tst/test_scalar',dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'));shutil.copy2(admission,prepared/'cpu-admission.json');shutil.copy2(A/'minkowski-undamped-gpu-v4/run_aurora.pbs',prepared/'run_aurora.pbs')
pbs=prepared/'run_aurora.pbs';text=pbs.read_text();marker='/usr/bin/python3.11 source/tst/test_scalar/gpu_worker.py';text=text.replace(marker,'mpiexec -n 1 --cpu-bind list:1-8 "$result/test_scalar" --audit --smooth-layer 1 --output "$result/backend-geometry"\n'+marker);pbs.write_text(text)
plan=dict(declared_before_submission=True,trigger='Stable physical C-infinity endpoint evaluation passed independent geometry/curvature and full source-bound CPU science/controls/actual-RHS gates.',source_sha256=proof['delivery_source_sha256'],CPU_admission=str(admission),resolutions=proof['resolutions'],CPU_GPU_scaled_max_tolerance=1e-9,coordinate_weight_tolerance=1e-11,relaxation=False,limits=dict(queue='debug',account='CompactBinaryMerger',nodes=1,minutes=60,max_task_allocations=1),scope='New corrected configuration; preserve the failed v4 GPU evidence. No black-hole or metric-evolution admission.')
(prepared/'gpu-plan.json').write_text(json.dumps(plan,indent=2)+'\n');archive=prepared.with_suffix('.tar.gz');subprocess.run([sys.executable,str(scripts/'package_aurora.py'),'--prepared',str(prepared),'--output',str(archive),'--extra','gpu-plan.json'],check=True)
subprocess.run(['scp','-o','BatchMode=yes','-o','ControlPath=/Users/hz0693/.ssh/codex-control/%C',str(archive),host+':'+remote+'.tar.gz'],check=True)
program='root='+repr(remote)+'\n'+'''from pathlib import Path
import tarfile,subprocess,os
p=Path(root)
assert not p.exists(),'Existing remote attempt: inspect its submission lock'
p.mkdir()
with tarfile.open(root+'.tar.gz') as archive:archive.extractall(p)
r=subprocess.run(['/usr/bin/python3.11',str(p/'source/tst/test_scalar/submit_aurora.py')],env={**os.environ,'SCALAR_ROOT':root},text=True,capture_output=True)
print(r.stdout);print(r.stderr)
raise SystemExit(r.returncode)
'''
r=subprocess.run([*ssh,'/usr/bin/python3.11 -'],input=program,text=True,capture_output=True);(prepared/'submission-access.log').write_text(r.stdout+r.stderr)
if r.returncode:raise RuntimeError('GPU submission failed/unknown; inspect the saved lock and log before retrying')
record=json.loads(r.stdout.strip());(prepared/'submission.json').write_text(json.dumps(record,indent=2)+'\n');job=record['job_id'];print('CORRECTED GPU JOB',job,flush=True);collected=A/'minkowski-stable-gpu-v5-collected'
subprocess.run([sys.executable,str(scripts/'wait_collect_aurora.py'),'--root',remote,'--job',job,'--output',str(collected),'--journal',str(A/'minkowski-stable-gpu-v5-scheduler-journal.jsonl')],check=True)
geometry=A/'minkowski-stable-gpu-v5-geometry-review.json';sources=A/'minkowski-stable-gpu-v5-source-review.json'
subprocess.run([sys.executable,str(scripts/'audit_geometry.py'),str(collected/'result/backend-geometry/geometry.csv'),'--smooth-layer','--output',str(geometry)],check=True)
subprocess.run([sys.executable,str(scripts/'audit_normal_sources_3d.py'),str(collected/'result/backend-geometry/geometry.csv'),'--output',str(sources)],check=True)
subprocess.run([sys.executable,str(scripts/'audit_gpu_stage.py'),'--gpu',str(collected/'result'),'--admission',str(admission),'--source-manifest',str(prepared/'portable-source-manifest.json'),'--cpu-review',str(E/'cpu-v22/stable-full-review.json'),'--additional-audit',str(geometry),'--additional-audit',str(sources),'--output',str(A/'minkowski-stable-gpu-v5-review.json')],check=True)
