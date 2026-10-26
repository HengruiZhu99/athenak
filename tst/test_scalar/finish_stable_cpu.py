#!/usr/bin/env python3
"""Join reserved CPU evidence, then run only the missing local time controls."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
from scalar_evidence import history

p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args()
E=a.evidence.resolve();R=E/'cpu-v22';A=E/'aurora';collected=A/'cpu-stable-endpoint-v1-collected';prepared=A/'cpu-stable-endpoint-v1';scripts=Path(__file__).resolve().parent
def execute(name,*args):
 subprocess.run([sys.executable,str(scripts/name),*[str(v) for v in args]],check=True)
def review(name,*args):
 output=R/name
 if not output.exists():execute(*args,'--output',output)
 if not json.loads(output.read_text())['passed']:raise RuntimeError('Required audit failed: '+str(output))
 return output
while not all((R/f'minkowski-n{n}-stable/complete.txt').exists() for n in [24,32]) or not (collected/'collection-receipt.json').exists():time.sleep(30)
backend=review('cpu-backend-review.json','audit_cpu_backend.py','--collected',collected,'--prepared',prepared,'--reference',R/'minkowski-n24-stable')
remote=collected/'result'
review('remote-endpoint-review.json','audit_smooth_endpoint.py',remote/'smooth-endpoint-geometry.csv')
review('remote-geometry-review.json','audit_geometry.py',remote/'backend-geometry/geometry.csv','--smooth-layer')
review('remote-full-3d-source-review.json','audit_normal_sources_3d.py',remote/'backend-geometry/geometry.csv')
runs=[R/'minkowski-n24-stable',R/'minkowski-n32-stable',remote/'minkowski-n40-stable'];matrix=R/'stable-matrix-review.json'
if not matrix.exists():execute('assess_minkowski_matrix.py','--runs',*runs,'--output',matrix)
data=json.loads(matrix.read_text())
if not data['complete'] or not data['convergence_passed'] or not all(c['passed'] for c in data['checks']):raise RuntimeError('Corrected matrix failed; preserve it and do not launch controls')
execute('run_minkowski_controls.py','--review',matrix,'--exe',R/'test_scalar','--root',R,'--label','stable','--controls','time','--backend-review',backend)
full=R/'stable-full-review.json'
if not full.exists():execute('assess_minkowski_matrix.py','--runs',*runs,'--time-runs',runs[1],R/'stable-n32-cfl0.09',R/'stable-n32-cfl0.045','--outer-runs',remote/'stable-n40-cap0.9',remote/'stable-n40-cap0.95','--output',full)
if not json.loads(full.read_text())['passed']:raise RuntimeError('Corrected full controls failed')
long=remote/'minkowski-n12-stable-long';h=history(long);initial=float(h[0,'all']['psi_max']);late=max(float(row['psi_max']) for (t,region),row in h.items() if region=='all' and t>=20);frozen=json.loads((long/'frozen-integrity.json').read_text());output=R/'long-minkowski-review.json'
record=dict(run=str(long),complete=(long/'complete.txt').exists(),initial_peak=initial,late_start=20,late_maximum=late,relative_late_maximum=late/initial,limit=.01,frozen_unchanged=frozen['unchanged']);record['passed']=record['complete'] and frozen['unchanged'] and late/initial<.01
if not output.exists():output.write_text(json.dumps(record,indent=2)+'\n')
if not record['passed']:raise RuntimeError('Corrected long stability gate failed')
print('CORRECTED CPU SCIENCE AND CONTROLS PASSED; actual-RHS diagnostic/admission assembly remain',flush=True)
