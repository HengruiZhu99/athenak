#!/usr/bin/env python3
"""Assess a new scalar matrix without changing previous admissions or failures."""
import argparse,csv,json,math,hashlib
from pathlib import Path
import numpy as np
from scalar_evidence import snapshot,history,waveform
def norms(x,w):
 if x.ndim>1:x=np.linalg.norm(x,axis=1)
 return dict(L2=float(np.sqrt(np.sum(w*x*x)/np.sum(w))),Linf=float(np.max(abs(x))))
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs=3,type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--time-runs',nargs=3,type=Path);p.add_argument('--outer-runs',nargs=2,type=Path);p.add_argument('--compatible-floor',action='store_true');a=p.parse_args()
 if a.output.exists():raise RuntimeError('Existing review is immutable')
 meta=[json.loads((d/'parameters.json').read_text()) for d in a.runs];ns=[r['n'] for r in meta]
 h=[history(d) for d in a.runs]
 complete=all((d/'complete.txt').exists() for d in a.runs)
 scale=float(np.max(np.linalg.norm(snapshot(a.runs[-1],0)[:,6:9],axis=1)));floor=max(1e-14,1e-12*scale) if a.compatible_floor else 0
 checks=[]
 for region in ['all','bulk','shell']:
  for t in [2,6,12]:
   for field in ['psi','v','Dpsi','reduction']:
    for name in ['L2','Linf']:
     values=[float(r[t,region][field+'_'+name]) for r in h];orders=[math.log(values[k]/values[k+1])/math.log(ns[k+1]/ns[k]) if min(values)>0 else None for k in [0,1]]
     pairs=[dict(below_floor=field=='reduction' and max(values[k:k+2])<=floor,passed=(field=='reduction' and max(values[k:k+2])<=floor) or (orders[k] is not None and values[k]>values[k+1] and orders[k]>=1.5)) for k in [0,1]]
     checks.append(dict(time=t,region=region,field=field,norm=name,errors=values,orders=orders,pairs=pairs,passed=all(x['passed'] for x in pairs)))
 cmax=max(float(r['reduction_Linf']) for k,r in h[-1].items() if k[1]=='all');reduction=dict(initial_derivative_scale=scale,roundoff_floor=floor,max_reduction=cmax,normalized_max=cmax/scale,passed=cmax/scale<=.02)
 times=[]
 if a.time_runs and all((d/'complete.txt').exists() for d in a.time_runs):
  for t in [2,6,12]:
   arrays=[snapshot(d,t) for d in a.time_runs];w=arrays[0][:,3]
   for field,cols in [('psi',slice(4,5)),('v',slice(5,6)),('Dpsi',slice(9,12))]:
    half=norms(arrays[0][:,cols]-arrays[1][:,cols],w);quarter=norms(arrays[1][:,cols]-arrays[2][:,cols],w)
    for name in ['L2','Linf']:
     gap=abs(float(h[1][t,'all'][field+'_'+name])-float(h[2][t,'all'][field+'_'+name]));fraction=half[name]/gap;ratio=quarter[name]/half[name]
     times.append(dict(time=t,field=field,norm=name,half_difference=half[name],quarter_difference=quarter[name],fraction=fraction,quarter_ratio=ratio,passed=fraction<.1 and ratio<.2))
 outer=[]
 if a.outer_runs:
  ref=waveform(a.runs[-1]);rv=ref['real22']+1j*ref['imag22']
  for d in a.outer_runs:
   if not (d/'complete.txt').exists():continue
   q=waveform(d)
   if not np.array_equal(q['time'],ref['time']):raise RuntimeError('Boundary waveform times differ')
   delta=q['real22']+1j*q['imag22']-rv;relative=float(np.linalg.norm(delta)/np.linalg.norm(rv));rows=list(csv.DictReader((d/'characteristics.csv').open()))
   outer.append(dict(run=str(d),relative_L2=relative,actual_incoming_characteristic=min(float(r['cminus']) for r in rows),passed=relative<=.01))
 characters=[];integrity=[]
 for d in a.runs:
  rows=list(csv.DictReader((d/'characteristics.csv').open()));characters.append(dict(run=str(d),covector='shell logical radial coordinate',max_abs_incoming=max(abs(float(r['cminus'])) for r in rows),min_outgoing=min(float(r['cplus']) for r in rows),passed=all(abs(float(r['cminus']))<=1e-10 and float(r['cplus'])>0 for r in rows)))
  integrity.append(dict(run=str(d),**json.loads((d/'frozen-integrity.json').read_text())))
 convergence=complete and all(c['passed'] for c in checks if c['region']=='all') and reduction['passed']
 result=dict(stage='minkowski',runs=[str(d) for d in a.runs],time_runs=[str(d) for d in a.time_runs] if a.time_runs else [],outer_runs=[str(d) for d in a.outer_runs] if a.outer_runs else [],resolutions=ns,complete=complete,convergence_passed=convergence,checks=checks,reduction=reduction,timestep=times,boundary=outer,characteristics=characters,frozen=integrity,passed=convergence and len(times)==18 and all(c['passed'] for c in times) and len(outer)==2 and all(c['passed'] for c in outer+characters) and all(c['unchanged'] for c in integrity),scope='Global field, auxiliary velocity, independently differentiated spatial field, and scalar reduction convergence. Regional results retained as diagnostics. Overall admission requires completed timestep and finite-boundary controls.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 failed=[c for c in checks if c['region']=='all' and not c['passed']];print(json.dumps(dict(passed=result['passed'],convergence_passed=convergence,failed_global_checks=failed,reduction=reduction),indent=2))
if __name__=='__main__':main()
