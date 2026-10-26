#!/usr/bin/env python3
"""Uniform no-interface packet-resolution control for retained closure failures."""
import ast,json,math,hashlib
from pathlib import Path
import numpy as np
root=Path('docs/validation/goal1-minkowski-20261018');out=root/'dual-interface-v1';assert not (out/'uniform-control-review.json').exists()
for filename,names in [('check_constraint_incoming_lift.py',['mm','symbols']),('check_goal1_outer_port.py',['gram']),('check_goal1_cell_centered_interface.py',['bump','rms'])]:
 p=Path('tst/cubed_sphere')/filename
 for node in ast.parse(p.read_text()).body:
  if isinstance(node,ast.FunctionDef) and node.name in names:exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),globals())
s=next(symbols(Path('docs/validation/cubed-sphere-balanced-20261013/finite-radius-long/pulse-t02-controls/pulse-boundary-symbol-t02.txt')));M=s['matrix'];A=M[:10,:10];B=M[:10,10:];beta=-(s['lo']+s['hi'])/2;c=(s['hi']-s['lo'])/2;speed=c-beta;Ae=A-beta*np.eye(10);Bi=np.linalg.inv(B);G=gram(s['basis'][:11,:10],s['inverse']);basis=np.linalg.inv(np.linalg.cholesky(G).T)
records=[]
for n in [24,48,96,192]:
 x=-1+(np.arange(2*n)+.5)/n;H=np.full(2*n,1/n);theta=2*np.pi*np.fft.fftfreq(2*n);omega=n*(1.5*np.sin(theta)-.3*np.sin(2*theta)+np.sin(3*theta)/30)
 def deriv(f):return np.fft.ifft(1j*omega[:,None]*np.fft.fft(f,axis=0),axis=0).real
 f,df=bump(x);f*=.001;df*=.001
 for mode in range(10):
  v=basis[:,mode];pc=np.einsum('ij,j->i',Bi,np.einsum('ij,j->i',-speed*np.eye(10)-A,v));scale=.001*np.sqrt(v@v+.22**2*(pc@pc));q0=f[:,None]*v;pi0=mm(deriv(q0),Ae.T)+df[:,None]*np.einsum('ij,j->i',B,pc)
  qh=np.fft.fft(q0,axis=0);ph=np.fft.fft(pi0,axis=0)
  for phase in [0.,.75,1.5,2.5]:
   t=phase/speed;rot=np.exp(1j*beta*omega*t)[:,None];co=np.cos(c*omega*t)[:,None];si=np.sin(c*omega*t)[:,None]
   q=np.fft.ifft(rot*(co*qh+t*np.sinc(c*omega*t/np.pi)[:,None]*ph),axis=0).real;pi=np.fft.ifft(rot*(-c*omega[:,None]*si*qh+co*ph),axis=0).real;dq=deriv(q);p=mm(pi-mm(dq,Ae.T),Bi.T);dp=deriv(p)
   tx=(x-speed*t+1)%2-1;fe,de=bump(tx);z=(tx+.5)/.22;inside=abs(z)<1;d2=np.zeros_like(z);u=z[inside];d=1-u*u;d2[inside]=fe[inside]*(4*u*u/d**4-2/d**2-8*u*u/d**3)/.22**2
   field=np.hstack([q-.001*fe[:,None]*v,.22*(p-.001*de[:,None]*pc)])/scale;gradient=np.hstack([.22*(dq-.001*de[:,None]*v),.22**2*(dp-.001*d2[:,None]*pc)])/scale
   records.append({'nodes_per_unit':n,'mode':mode,'phase':phase,'field_rms':rms(field,H),'field_linf':float(np.max(np.linalg.norm(field,axis=1))),'derivative_rms':rms(gradient,H),'derivative_linf':float(np.max(np.linalg.norm(gradient,axis=1)))})
summary=[]
for phase in [0.,.75,1.5,2.5]:
 for key in ['field_rms','field_linf','derivative_rms','derivative_linf']:
  vals=[max(r[key] for r in records if r['nodes_per_unit']==n and r['phase']==phase) for n in [24,48,96,192]];summary.append({'phase':phase,'quantity':key,'values':vals,'orders':[math.log(a/b,2) if a>0 and b>0 else None for a,b in zip(vals,vals[1:])]})
report={'scope':__doc__,'records':records,'summary':summary,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'native_acceptance':False};(out/'uniform-control-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(summary,indent=2))
