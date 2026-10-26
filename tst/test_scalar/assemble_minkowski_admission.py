#!/usr/bin/env python3
"""Bind a passed numerical matrix, exact settings and independent audits to an admission."""
import argparse,json,hashlib
from pathlib import Path

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--delivery',default='cpu-v19');a=p.parse_args()
 if a.output.exists():raise RuntimeError('Existing admission evidence is immutable')
 E=a.evidence;matrix=json.loads(a.review.read_text());R=E/'cpu-v17';expected=dict(M=0,a_over_M=0,L=3,S=24,C=10,cap=1,cfl=.18,ko=0,reduction_damping=2,radial_stretch=-2,outer_sbp=0,smooth_physical_layer=True,reference_layer='original_quintic',normal_upwind=True,compatible_reduction=False,characteristic_transfer=False,excision=0,tlim=12,boundary='exact',data='angular');settings=[];frozen=[]
 cases=[(R/f'minkowski-n{n}-normal',n,{}) for n in [24,32,40]]+[(R/f'minkowski-n32-normal-cfl{cfl}',32,dict(cfl=cfl)) for cfl in [.09,.045]]+[(R/f'minkowski-n40-normal-cap{cap}',40,dict(cap=cap,boundary='radiation')) for cap in [.9,.95]]
 nominal=json.loads((R/'minkowski-n32-normal/frozen-integrity.json').read_text());keys=['physical_before','reference_before','physical_derivatives_before','reference_derivatives_before']
 for run,n,overrides in cases:
  parameters=json.loads((run/'parameters.json').read_text());target={**expected,'n':n,**overrides};mismatch={k:dict(expected=v,actual=parameters.get(k)) for k,v in target.items() if parameters.get(k)!=v};settings.append(dict(run=str(run),mismatches=mismatch,passed=not mismatch));g=json.loads((run/'frozen-integrity.json').read_text());time_identical=all(g[k]==nominal[k] for k in keys) if n==32 else True;frozen.append(dict(run=str(run),unchanged=g['unchanged'],time_coefficients_identical=time_identical,passed=g['unchanged'] and time_identical))
 audit_paths=['geometry-smooth/review.json','geometry-normal/derivative-review.json','geometry-normal/normal-source-review.json','geometry-difference/review.json','cpu-v18/provenance-equivalence.json','cpu-v18/inner-guard-review.json','cpu-v18/wall-time-checkpoint-review.json','baseline-and-symbol-integrity.json'];audits=[dict(path=name,sha256=sha(E/name),passed=json.loads((E/name).read_text())['passed']) for name in audit_paths]
 if a.delivery!='cpu-v18':audits.append(dict(path=a.delivery+'/exterior-observer-equivalence.json',sha256=sha(E/a.delivery/'exterior-observer-equivalence.json'),passed=json.loads((E/a.delivery/'exterior-observer-equivalence.json').read_text())['passed']))
 source_paths=[E/a.delivery/'source'/name for name in ['CMakeLists.txt','main.cpp','geometry.hpp','sbp63.hpp']];source_matches=all(sha(p)==sha(Path(__file__).resolve().parents[2]/'src/test_scalar'/p.name) for p in source_paths)
 gpu_cases=[]
 for n in [24,32,40]:gpu_cases.append(dict(n=n,arguments=['--n',str(n),'--tlim','12','--mass','0','--spin','0','--cap','1','--cfl','.18','--ko','0','--reduction-damping','2','--radial-stretch','-2','--outer-sbp','0','--smooth-layer','1','--normal-upwind','1','--compatible-reduction','0','--characteristic-transfer','0','--excision','0','--boundary','exact','--data','angular']))
 passed=matrix['passed'] and matrix['complete'] and all(c['passed'] for c in settings+frozen+audits) and source_matches
 result=dict(stage='minkowski',configuration='normal-upwind',passed=passed,complete=matrix['complete'],resolutions=[24,32,40],matrix_review=str(a.review),matrix_sha256=sha(a.review),settings=settings,frozen=frozen,audits=audits,delivery_source_matches_equivalence_control=source_matches,delivery_source_sha256={p.name:sha(p) for p in source_paths},cpu_binary_sha256=sha(R/'test_scalar'),equivalent_delivery_binary_sha256=sha(E/a.delivery/'test_scalar'),gpu_cases=gpu_cases,qualification='Global and regional analytic field/velocity-reconstruction/spatial-derivative/reduction convergence; medium half/quarter timestep controls; finest .9/.95 finite-cap sensitivity. Short early cap tests do not establish later causal-reflection immunity. Scalar admission does not complete the metric-evolution goal.')
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=passed,complete=result['complete'],settings_passed=all(c['passed'] for c in settings),audits_passed=all(c['passed'] for c in audits)),indent=2))
 if not passed:raise SystemExit(1)
if __name__=='__main__':main()
