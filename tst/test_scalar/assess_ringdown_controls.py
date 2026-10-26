#!/usr/bin/env python3
"""Assess predeclared black-hole timestep, excision and finite-cap controls."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from scalar_evidence import snapshot,history,waveform
from validate_schwarzschild import norms

def common(data):return data[np.linalg.norm(data[:,:3],axis=1)>=1.2]
def main():
 p=argparse.ArgumentParser();p.add_argument('--primary-review',type=Path,required=True);p.add_argument('--time-runs',nargs=3,type=Path,required=True);p.add_argument('--inner-run',type=Path,required=True);p.add_argument('--outer-runs',nargs=2,type=Path,required=True);p.add_argument('--causal-run',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise RuntimeError('Existing controls review is immutable')
 primary=json.loads(a.primary_review.read_text());runs=[Path(d) for d in primary['runs']];times=[];boundaries=[];integrity=[];ns=primary['resolutions'];directories=[*a.time_runs,a.inner_run,*a.outer_runs]+([a.causal_run] if a.causal_run else [])
 if not all((d/'complete.txt').exists() for d in directories):raise RuntimeError('Control lineage is incomplete; resume preserved checkpoints first')
 baseline=json.loads((runs[-1]/'parameters.json').read_text());medium=json.loads((runs[1]/'parameters.json').read_text());settings=[]
 cases=[(d,{**medium,'cfl':cfl}) for d,cfl in zip(a.time_runs,[.18,.09,.045])]+[(a.inner_run,{**baseline,'excision':.20 if primary['stage']=='schwarzschild' else .06})]+[(d,{**baseline,'cap':cap}) for d,cap in zip(a.outer_runs,[.9,.95])]+([(a.causal_run,{**baseline,'cap':.6})] if a.causal_run else [])
 for d,target in cases:
  actual=json.loads((d/'parameters.json').read_text());mismatch={k:dict(expected=v,actual=actual.get(k)) for k,v in target.items() if k not in ['tlim'] and actual.get(k)!=v};settings.append(dict(run=str(d),passed=not mismatch,mismatches=mismatch))
  frozen=json.loads((d/'frozen-integrity.json').read_text());initial=common(snapshot(d,0));scale=float(np.linalg.norm(initial[:,6:9],axis=1).max());exterior=[float(r['reduction_Linf']) for (t,region),r in history(d).items() if region=='exterior'];normalized=max(exterior)/scale if exterior else None;finest_bound=target['n']==ns[-1];integrity.append(dict(run=str(d),frozen=frozen,normalized_reduction_max=normalized,finest_bound_applies=finest_bound,passed=bool(frozen['unchanged'] and normalized is not None and np.isfinite(normalized) and (not finest_bound or normalized<=.02))))
 for record in primary['records']:
  t=record['time'];arrays=[common(snapshot(d,t)) for d in a.time_runs]
  if any(not np.array_equal(arrays[0][:,:4],x[:,:4]) for x in arrays[1:]):raise RuntimeError('Timestep control grid or weights differ')
  for field,cols in [('psi',slice(4,5)),('v',slice(5,6)),('Dpsi',slice(9,12))]:
   half=norms(arrays[0][:,cols]-arrays[1][:,cols],arrays[0][:,3]);quarter=norms(arrays[1][:,cols]-arrays[2][:,cols],arrays[0][:,3])
   for name in ['L2','Linf']:
    gap=abs(record['errors'][1][field][name]-record['errors'][2][field][name]);fraction=half[name]/gap if gap>0 else None;ratio=quarter[name]/half[name] if half[name]>0 else (0 if quarter[name]==0 else None);times.append(dict(time=t,field=field,norm=name,half_difference=half[name],quarter_difference=quarter[name],spatial_error_gap=gap,fraction=fraction,quarter_ratio=ratio,passed=fraction is not None and ratio is not None and fraction<.1 and ratio<.2))
 ref=waveform(runs[-1]);rv=ref['real22']+1j*ref['imag22']
 for d,kind in [(a.inner_run,'inner')]+[(d,'outer') for d in a.outer_runs]+([(a.causal_run,'causal_supplement')] if a.causal_run else []):
  w=waveform(d)
  if not np.array_equal(w['time'],ref['time']):raise RuntimeError('Boundary waveform times differ')
  relative=float(np.linalg.norm(w['real22']+1j*w['imag22']-rv)/np.linalg.norm(rv));chars=list(csv.DictReader((d/'characteristics.csv').open()));inner=[r for r in chars if r['face'].startswith('inner')];outer=[r for r in chars if r['face'].startswith('outer')];meta=json.loads((d/'parameters.json').read_text());outflow=all(float(r['cminus'])>=0 and float(r['cplus'])>0 for r in inner);outer_ok=all(float(r['cplus'])>0 and (abs(float(r['cminus']))<=1e-10 if meta['cap']==1 else float(r['cminus'])<0) for r in outer)
  boundaries.append(dict(run=str(d),kind=kind,relative_waveform_L2=relative,inner_min_cminus=min((float(r['cminus']) for r in inner),default=None),outer_min_cminus=min(float(r['cminus']) for r in outer),characteristics_passed=outflow and outer_ok,passed=relative<=.01 and outflow and outer_ok))
 required=[r for r in boundaries if r['kind']!='causal_supplement'];complete=len(times)==24 and len(required)==3;passed=complete and primary['field_checks_passed'] and primary['finest_accuracy_passed'] and all(c['passed'] for c in times+required+integrity+settings)
 result=dict(stage=primary['stage'],passed=passed,complete=complete,primary_review=str(a.primary_review),primary_review_sha256=hashlib.sha256(a.primary_review.read_bytes()).hexdigest(),time_runs=[str(d) for d in a.time_runs],timestep=times,boundary=boundaries,integrity=integrity,settings=settings,qualification='Original declared .9/.95 cap and excision waveform gates, exact scri, all timestep/reduction/frozen checks. The closer .6 causal supplement is reported separately and cannot establish far-cap immunity to late reflections. Independent oracle resolution and QNM records remain required for final admission.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(dict(passed=passed,complete=complete,failed_times=[c for c in times if not c['passed']],boundary=boundaries),indent=2))
if __name__=='__main__':main()
