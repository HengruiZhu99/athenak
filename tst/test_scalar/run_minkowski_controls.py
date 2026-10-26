#!/usr/bin/env python3
"""Run the declared medium timestep and finest-cap controls of a passed matrix."""
import argparse,hashlib,json
from pathlib import Path
from run_cpu import run

p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);p.add_argument('--exe',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--controls',choices=['all','time','outer'],default='all');p.add_argument('--backend-review',type=Path);a=p.parse_args()
review=json.loads(a.review.read_text())
if not review['complete'] or not review['convergence_passed'] or not all(c['passed'] for c in review['checks']):raise RuntimeError('All matrix convergence gates must pass before these controls')
paths=[Path(d) for d in review['runs']];parameters=[json.loads((d/'parameters.json').read_text()) for d in paths];binary=hashlib.sha256(a.exe.read_bytes()).hexdigest()
allowed={binary}
if a.backend_review:
 backend=json.loads(a.backend_review.read_text())
 if not backend['passed'] or not backend['source_manifest_match'] or not backend['dependency_match']:raise RuntimeError('A passed source-bound CPU backend review is required')
 allowed.add(backend['validated_binary_sha256'])
if any(json.loads((d/'invocation.json').read_text())['binary'] not in allowed for d in paths):raise RuntimeError('Controls require the same source-bound matrix executable or a validated CPU backend')
meta=parameters[1]
if any(m['M']!=0 or m['tlim']!=12 or m['L']!=3 or m['S']!=24 or m['C']!=10 for m in parameters):raise RuntimeError('Unexpected Minkowski geometry or duration')
options={'M':'mass','a_over_M':'spin','ko':'ko','reduction_damping':'reduction-damping','radial_stretch':'radial-stretch','outer_sbp':'outer-sbp','smooth_physical_layer':'smooth-layer','characteristic_transfer':'characteristic-transfer','compatible_reduction':'compatible-reduction','normal_upwind':'normal-upwind','excision':'excision','data':'data'}
base=['--tlim','12']
for key,opt in options.items():base+=['--'+opt,str(int(meta[key])) if isinstance(meta[key],bool) else str(meta[key])]
cases=[(f'{a.label}-n{meta["n"]}-cfl{cfl}',['--n',str(meta['n']),*base,'--cap','1','--boundary','exact','--cfl',str(cfl)]) for cfl in [.09,.045]] if a.controls!='outer' else []
if a.controls!='time':cases += [(f'{a.label}-n{parameters[-1]["n"]}-cap{cap}',['--n',str(parameters[-1]['n']),*base,'--cap',str(cap),'--boundary','radiation','--cfl','.18']) for cap in [.9,.95]]
for label,args in cases:run(a.exe,a.root,label,args)
