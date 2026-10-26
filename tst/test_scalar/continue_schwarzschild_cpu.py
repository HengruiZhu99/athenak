#!/usr/bin/env python3
"""Run the reserved corrected black-hole CPU matrix after the GPU stage review."""
import argparse,json,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();E=a.evidence.resolve();A=E/'aurora';R=E/'cpu-schwarzschild-stable';prepared=A/'cpu-schwarzschild-stable-v1';scripts=Path(__file__).resolve().parent;host='hzhu@aurora.alcf.anl.gov';remote='/lus/flare/projects/CompactBinaryMerger/hzhu/test-scalar-frozen-20261016/cpu-schwarzschild-stable-v1';ssh=['ssh','-o','BatchMode=yes','-o','ControlPath=/Users/hz0693/.ssh/codex-control/%C',host]
if (prepared/'submission.json').exists():raise RuntimeError('Existing Schwarzschild submission: inspect it, never duplicate')
gpu=A/'minkowski-stable-gpu-v5-review.json'
while not gpu.exists():time.sleep(15)
if not json.loads(gpu.read_text())['passed']:raise RuntimeError('Corrected Minkowski GPU review failed; no next allocation launched')
program='root='+repr(remote)+'\n'+'''from pathlib import Path
import tarfile,subprocess,os
p=Path(root)
assert not p.exists(),'Existing remote attempt: inspect its submission lock'
p.mkdir()
with tarfile.open(root+'.tar.gz') as archive:archive.extractall(p)
r=subprocess.run(['/usr/bin/python3.11',str(p/'source/tst/test_scalar/submit_cpu_aurora.py')],env={**os.environ,'SCALAR_ROOT':root},text=True,capture_output=True)
print(r.stdout);print(r.stderr)
raise SystemExit(r.returncode)
'''
r=subprocess.run([*ssh,'/usr/bin/python3.11 -'],input=program,text=True,capture_output=True);(prepared/'submission-access.log').write_text(r.stdout+r.stderr)
if r.returncode:raise RuntimeError('Schwarzschild submission failed/unknown; preserve its lock and inspect')
record=json.loads(r.stdout.strip());(prepared/'submission.json').write_text(json.dumps(record,indent=2)+'\n');job=record['job_id'];print('SCHWARZSCHILD CPU JOB',job,flush=True);collected=A/'cpu-schwarzschild-stable-v1-collected'
subprocess.run([sys.executable,str(scripts/'wait_collect_aurora.py'),'--root',remote,'--job',job,'--output',str(collected),'--journal',str(A/'cpu-schwarzschild-stable-v1-scheduler-journal.jsonl')],check=True)
worker=json.loads((collected/'result/cpu-worker-receipt.json').read_text())
if not worker['all_cases_complete']:raise RuntimeError('Wall checkpoint preserved; resume only the missing prefix with the exact saved binary and geometry')
while not (R/'schwarzschild-n24-normal/complete.txt').exists():time.sleep(15)
subprocess.run([sys.executable,str(scripts/'audit_cpu_backend.py'),'--collected',str(collected),'--prepared',str(prepared),'--reference',str(R/'schwarzschild-n24-normal'),'--case-name','backend-schwarzschild-n24-t2','--output',str(R/'cpu-backend-review.json')],check=True)
runs=[R/'schwarzschild-n24-normal',collected/'result/schwarzschild-n32-normal',collected/'result/schwarzschild-n40-normal'];oracle=E/'radial-oracle/smooth'
subprocess.run([sys.executable,str(scripts/'validate_schwarzschild.py'),'--runs',*[str(d) for d in runs],'--oracle',str(oracle),'--oracle-waveform','waveform-upwind-n1200.csv','--normal-coefficients',str(oracle/'normal-coefficients-n1200.txt'),'--output',str(R/'primary-review.json')],check=True)
primary=json.loads((R/'primary-review.json').read_text());print('CORRECTED SCHWARZSCHILD',json.dumps(dict(field_checks_passed=primary['field_checks_passed'],finest_accuracy_passed=primary['finest_accuracy_passed'])),flush=True)
