#!/usr/bin/env python3
"""Collect a read-only fine-grid RHS audit, then assemble corrected admission."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();E=a.evidence.resolve();R=E/'cpu-v22';A=E/'aurora';prepared=A/'cpu-stable-rhs-v1';scripts=Path(__file__).resolve().parent;host='hzhu@aurora.alcf.anl.gov';remote='/lus/flare/projects/CompactBinaryMerger/hzhu/test-scalar-frozen-20261016/cpu-stable-rhs-v1';ssh=['ssh','-o','BatchMode=yes','-o','ControlPath=/Users/hz0693/.ssh/codex-control/%C',host]
if (prepared/'submission.json').exists():raise RuntimeError('Existing RHS submission; inspect it before continuing')
while not (R/'stable-matrix-review.json').exists():time.sleep(15)
matrix=json.loads((R/'stable-matrix-review.json').read_text())
if not matrix['convergence_passed'] or not all(c['passed'] for c in matrix['checks']):raise RuntimeError('Corrected matrix must pass before the extra RHS allocation')
program='root='+repr(remote)+'\n'+'''from pathlib import Path
import tarfile,subprocess,os
p=Path(root)
assert not p.exists(),'Existing remote tree: inspect the prior outcome'
p.mkdir()
with tarfile.open(root+'.tar.gz') as archive:archive.extractall(p)
r=subprocess.run(['/usr/bin/python3.11',str(p/'source/tst/test_scalar/submit_cpu_aurora.py')],env={**os.environ,'SCALAR_ROOT':root},text=True,capture_output=True)
print(r.stdout);print(r.stderr)
raise SystemExit(r.returncode)
'''
r=subprocess.run([*ssh,'/usr/bin/python3.11 -'],input=program,text=True,capture_output=True);(prepared/'submission-access.log').write_text(r.stdout+r.stderr)
if r.returncode:raise RuntimeError('RHS submission failed/unknown; preserve its remote lock and inspect the log')
record=json.loads(r.stdout.strip());(prepared/'submission.json').write_text(json.dumps(record,indent=2)+'\n');job=record['job_id'];print('RHS JOB',job,flush=True)
collected=A/'cpu-stable-rhs-v1-collected'
subprocess.run([sys.executable,str(scripts/'wait_collect_aurora.py'),'--root',remote,'--job',job,'--output',str(collected),'--journal',str(A/'cpu-stable-rhs-v1-scheduler-journal.jsonl')],check=True)
receipt=json.loads((collected/'collection-receipt.json').read_text());manifest=json.loads((collected/'remote-manifest.json').read_text());scheduler=next(iter(manifest['scheduler']['Jobs'].values()));expected=json.loads((prepared/'portable-source-manifest.json').read_text());build=json.loads((collected/'result/audit_time_derivative.build.json').read_text())
if not receipt['passed'] or manifest['source_files']!=expected or manifest['source_mismatches'] or scheduler.get('Exit_status')!=0 or build['source_sha256']!=hashlib.sha256((scripts/'audit_time_derivative.cpp').read_bytes()).hexdigest():raise RuntimeError('Actual-RHS source/build/collection integrity failed')
review=R/'actual-time-derivative-review.json'
subprocess.run([sys.executable,str(scripts/'assess_actual_derivative.py'),'--runs',*matrix['runs'],'--audits',str(R/'actual-time-derivative-n24'),str(R/'actual-time-derivative-n32'),str(collected/'result/actual-time-derivative-n40'),'--source',str(R/'source'),'--output',str(review)],check=True)
while not (R/'stable-full-review.json').exists() or not (R/'long-minkowski-review.json').exists():time.sleep(15)
subprocess.run([sys.executable,str(scripts/'assemble_stable_minkowski_admission.py'),'--evidence',str(E),'--actual-derivative-review',str(review),'--output',str(E/'minkowski-stable-sbp-cpu-admission.json')],check=True)
