#!/usr/bin/env python3
"""Audit the coupled principal symbol; do not authorize evolution from speed-only tests."""
import argparse,json,subprocess
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--probe',type=Path,required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
out=args.output;out.mkdir(parents=True,exist_ok=True)
records=[]
def probe(r,e,theta,phi,bg,h,begin):
 command=[str(args.probe.resolve()),str(r),str(e),str(theta),str(phi),bg,str(h),str(begin)]
 return json.loads(subprocess.check_output(command,text=True))
for bg in ['flat','wormhole']:
 for begin in [.95,.98]:
  radii=sorted(set([begin-.001,begin,(3*begin+1)/4,(begin+1)/2,.995,.999,.9999]))
  for r in radii:
   for theta,phi in [(0,0),(.35,0),(np.pi/4,.37),(1.2,.8),(np.pi/2,.37)]:
    baseline=None
    for e in [0,.05,.1,.125,.15,.25,.5,.75,1]:
     raw=probe(r,e,theta,phi,bg,.001,begin);check=probe(r,e,theta,phi,bg,.0005,begin)
     a=np.array(raw['matrix']);b=np.array(check['matrix']);w,v=np.linalg.eig(a)
     error=float(np.max(abs(a-b)));norm=float(np.linalg.norm(a,2))
     residual=float(np.linalg.norm(a@v-v*w,2)/max(1,norm))
     real=bool(np.max(abs(w.imag))<1e-6)
     speed=float(max(abs(w.real))) if real else None
     if e==0:baseline=speed
     record={**raw,'background':bg,'collar_begin':begin,'fdstep':.001,'fd_halving_max_error':error,'eigen_real':w.real.tolist(),'eigen_imag':w.imag.tolist(),'max_imag':float(max(abs(w.imag))),'eigenvector_condition':float(np.linalg.cond(v)),'eigenvector_rank':int(np.linalg.matrix_rank(v,tol=1e-8)),'eigen_residual':residual,'real_spectrum_with_tolerance':real,'coordinate_max_speed':speed,'relative_speed_dt_bound':baseline/speed if real and speed and baseline else None,'incoming_count_if_real':int(np.count_nonzero(w.real>1e-6)) if real else None,'outgoing_count_if_real':int(np.count_nonzero(w.real< -1e-6)) if real else None,'convention':'q_t=P(n) partial_n q; physical characteristic speeds are minus matrix eigenvalues'}
     records.append(record)
   print(bg,begin,r,'done',flush=True)
(out/'symbols.json').write_text(json.dumps(records)+'\n')
a=[v for v in records if v['epsilon']==0];c=[v for v in records if v['epsilon']==1 and v['radius']>=.995]
summary={'samples':len(records),'max_fd_halving_error':max(v['fd_halving_max_error'] for v in records),'max_eigen_residual':max(v['eigen_residual'] for v in records),'A_max_imag':max(v['max_imag'] for v in a),'C_near_scri_max_imag':max(v['max_imag'] for v in c),'C_nonreal_samples':sum(not v['real_spectrum_with_tolerance'] for v in c),'C_near_scri_samples':len(c),'C_gate':'REJECT substantial evolution: nonreal coupled characteristic speeds' if any(v['max_imag']>1e-3 for v in c) else 'requires further analysis'}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
