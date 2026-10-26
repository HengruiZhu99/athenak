#!/usr/bin/env python3
"""Free-frequency, window-tested spin-zero ringdown fits. No fixed-frequency fit."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

def fit(t,z,kerr=False):
 tau=t-t[0];scale=np.linalg.norm(z)
 def basis(p):
  if kerr:return np.c_[np.exp((-p[1]-1j*p[0])*tau),np.exp((-p[3]+1j*p[2])*tau)]
  return np.c_[np.exp(-p[1]*tau)*np.cos(p[0]*tau),np.exp(-p[1]*tau)*np.sin(p[0]*tau)]
 def solve(p):
  M=basis(p);amplitude=np.linalg.lstsq(M,z,rcond=None)[0];r=(M@amplitude-z)/scale
  return np.r_[r.real,r.imag]
 starts=[[w,g] for w in [.3,.5,.7] for g in [.05,.1]] if not kerr else [[w,.1,v,.1] for w in [.5,.65] for v in [.3,.45]]
 best=None
 for start in starts:
  sol=least_squares(solve,start,bounds=([.1,.001]* (2 if kerr else 1),[1.5,.5]* (2 if kerr else 1)),xtol=1e-12,ftol=1e-12,gtol=1e-12,max_nfev=2000)
  if best is None or np.linalg.norm(sol.fun)<np.linalg.norm(best.fun):best=sol
 M=basis(best.x);amplitude=np.linalg.lstsq(M,z,rcond=None)[0]
 return dict(omega_real=float(best.x[0]),decay=float(best.x[1]),counterrotating_real=float(best.x[2]) if kerr else None,counterrotating_decay=float(best.x[3]) if kerr else None,relative_residual=float(np.linalg.norm(best.fun)),amplitudes=[[float(v.real),float(v.imag)] for v in amplitude],basis_condition=float(np.linalg.cond(M)),success=bool(best.success))
def main():
 p=argparse.ArgumentParser();p.add_argument('waveform',type=Path);p.add_argument('--expected',type=Path,required=True);p.add_argument('--spin',type=float,default=0);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 q=np.genfromtxt(a.waveform,delimiter=',',names=True);t=q['time'];z=q['real22']+1j*q['imag22'] if 'imag22' in q.dtype.names else q['radial_psi'].astype(complex)
 modes=json.loads(a.expected.read_text())['modes'];expected=next(r for r in modes if r['a_over_M']==a.spin and r['m']==2);counter=next((r for r in modes if r['a_over_M']==a.spin and r['m']==-2),None);records=[]
 if a.output.exists():raise RuntimeError('Existing fit evidence is immutable; select a new output')
 for lo,hi in [[35,65],[40,70],[45,75]]:
  mask=(t>=lo)&(t<=hi)
  if np.sum(mask)<40:continue
  f=fit(t[mask],z[mask],a.spin!=0);f.update(window=[lo,hi],real_error=abs(f['omega_real']-expected['Momega_real'])/expected['Momega_real'],decay_error=abs(f['decay']+expected['Momega_imag'])/(-expected['Momega_imag']));f['passed']=f['real_error']<=.02 and f['decay_error']<=.05 and f['success']
  if a.spin!=0:
   if counter:
    f['counterrotating_real_error']=abs(f['counterrotating_real']-counter['Momega_real'])/counter['Momega_real'];f['counterrotating_decay_error']=abs(f['counterrotating_decay']+counter['Momega_imag'])/(-counter['Momega_imag']);f['passed'] &= f['counterrotating_real_error']<=.02 and f['counterrotating_decay_error']<=.05
   else:f['passed']=False
  records.append(f)
 keys=['omega_real','decay']+(['counterrotating_real','counterrotating_decay'] if a.spin else [])
 spreads={key:(max(r[key] for r in records)-min(r[key] for r in records))/np.mean([r[key] for r in records]) for key in keys} if len(records)==3 else {}
 result=dict(expected=expected,counterrotating_expected=counter,records=records,window_spreads=spreads,passed=len(records)==3 and all(r['passed'] for r in records) and all(s<=.02 for s in spreads.values()),method='Multi-start variable projection with free complex frequencies and amplitudes; independent Leaver frequency used only for evaluation')
 a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
