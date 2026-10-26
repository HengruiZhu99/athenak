#!/usr/bin/env python3
"""Validate the stable C-infinity evaluation under the predeclared gates."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from run_cpu import run

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--exe',type=Path,required=True);a=p.parse_args()
root=a.root.resolve();exe=a.exe.resolve();scripts=Path(__file__).resolve().parent
endpoint=json.loads((root/'smooth-endpoint-review.json').read_text())
if not endpoint['passed']:raise RuntimeError('Independent endpoint geometry must pass first')
base=['--normal-upwind','1','--outer-sbp','2','--ko','0','--reduction-damping','0','--smooth-layer','1','--radial-stretch','-2']
runs=[run(exe,root,f'minkowski-n{n}-stable',['--n',str(n),'--tlim','12',*base]) for n in [24,32,40]]
command=[sys.executable,str(scripts/'assess_minkowski_matrix.py'),'--runs',*[str(d) for d in runs]]
review=root/'stable-matrix-review.json'
if not review.exists():subprocess.run([*command,'--output',str(review)],check=True)
science=json.loads(review.read_text())
if not science['complete'] or not all(c['passed'] for c in science['checks']):
 raise RuntimeError('Stable-layer convergence failed; retain the matrix and do not launch controls')
subprocess.run([sys.executable,str(scripts/'run_minkowski_controls.py'),'--review',str(review),'--exe',str(exe),'--root',str(root),'--label','stable'],check=True)
full=root/'stable-full-review.json'
if not full.exists():subprocess.run([*command,'--time-runs',str(runs[1]),str(root/'stable-n32-cfl0.09'),str(root/'stable-n32-cfl0.045'),'--outer-runs',str(root/'stable-n40-cap0.9'),str(root/'stable-n40-cap0.95'),'--output',str(full)],check=True)
if not json.loads(full.read_text())['passed']:raise RuntimeError('Full stable-layer CPU controls failed')
run(exe,root,'minkowski-n12-stable-long',['--n','12','--tlim','100',*base])
