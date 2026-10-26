#!/usr/bin/env python3
"""Dual-adjoint interface with resolved compact C15 packet; frozen native20."""
import argparse, ast, hashlib, json, math
from pathlib import Path
import numpy as np
import scipy
from scipy.sparse import bmat, csr_matrix, eye
from scipy.sparse.linalg import expm_multiply
np.seterr(all='raise')
def mul(x,y):
 if not isinstance(x,np.ndarray) or not isinstance(y,np.ndarray):return x@y
 return np.einsum({(1,1):'i,i->',(2,1):'ij,j->i',(1,2):'i,ij->j',(2,2):'ij,jk->ik'}[(x.ndim,y.ndim)],x,y)
p=argparse.ArgumentParser();p.add_argument('output',type=Path);args=p.parse_args();out=args.output
protocol=json.loads((out/'protocol-v1.json').read_text());assert not (out/'review.json').exists()
source=Path('docs/validation/cubed-sphere-balanced-20261013/outer-closure/cell-centered-sbp-sixth-exploration.json')
s=json.loads(source.read_text());weights=np.array(s['norm_boundary_weights']);rows=np.array(s['left_boundary_rows']);trace=np.array(s['left_trace'])
for filename,names in [('check_constraint_incoming_lift.py',['mm','symbols']),('check_goal1_outer_port.py',['gram']),('check_goal1_cell_centered_interface.py',['block','coupled','bump','rms'])]:
 helper=Path('tst/cubed_sphere')/filename
 for node in ast.parse(helper.read_text()).body:
  if isinstance(node,ast.FunctionDef) and node.name in names:exec(compile(ast.Module(body=[node],type_ignores=[]),str(helper),'exec'),globals())
def bump(x):
 z=(x+.5)/.45;inside=abs(z)<1;f=np.zeros_like(x);d=np.zeros_like(x);u=z[inside];den=1-u*u;f[inside]=den**16;d[inside]=-32*u*den**15/.45
 return f,d
raw=Path('docs/validation/cubed-sphere-balanced-20261013/finite-radius-long/pulse-t02-controls/pulse-boundary-symbol-t02.txt');states=list(symbols(raw));assert len(states)==5
records=[];matrices=[];norms=[]
for n in protocol['resolutions']:
 x,H,D,nl=coupled(n);N=len(x);_,_,_,ll,lr=block(n,-1);_,_,_,rl,rr=block(N-nl,0)
 J=np.zeros((2,N));J[0,:nl]=lr;J[0,nl:]=-rl;J[1,nl:]=rr;J[1,:nl]=-ll
 R=mul(.5*J.T,J);Du=D+R/H[:,None];Dd=D-R/H[:,None]
 adj=float(np.max(abs(H[:,None]*Dd+Du.T*H[None,:])));const=float(np.max(abs(Du.sum(axis=1))))
 matrices.append({'left_nodes':n,'dual_adjoint_absolute':adj,'constant_absolute':const,'coupling_norm':float(np.linalg.norm(Du[nl:,:nl])),'passed':bool(max(adj,const)<1e-10)})
 f,df=bump(x);f*=.001;df*=.001;duf=mul(Du,f);Ds=csr_matrix(Du);Dds=csr_matrix(Dd);k=1/min(H)
 Y0=np.zeros((2*N,3));Y0[:N,0]=f;Y0[N:,1]=duf/k;Y0[N:,2]=df/k
 for case,state in enumerate(states):
  M=state['matrix'];A=M[:10,:10];B=M[:10,10:];C=M[10:,:10];F=M[10:,10:];beta=-(state['lo']+state['hi'])/2;c=(state['hi']-state['lo'])/2;speed=c-beta;assert beta<0
  Ae=A-beta*np.eye(10);Binv=np.linalg.inv(B);G=gram(state['basis'][:11,:10],state['inverse']);basis=np.linalg.inv(np.linalg.cholesky(G).T)
  L=bmat([[(beta/c)*Ds,k*eye(N,format='csr')],[(mul(Dds,Ds))/k,(beta/c)*Ds]],format='csr');tr=float(L.diagonal().sum())
  pc=mul(Binv,mul(-speed*np.eye(10)-A,basis))
  def energy(pi,g):return np.einsum('n,ni,ij,nj->',H,pi,G,pi)+c*c*np.einsum('n,ni,ij,nj->',H,g,G,g)
  for phase in [0.,.75,1.5,2.5]:
   t=phase/speed;Y=Y0 if phase==0 else expm_multiply(L*(c*t),Y0,traceA=tr*(c*t))
   for mode in range(10):
    v=basis[:,mode];pv=pc[:,mode];coeff=np.stack([v,mul(Ae,v)/c,mul(-speed*np.eye(10)-A,v)/c]);q=mul(Y[:N],coeff);pi=mul(c*k*Y[N:],coeff);dq=mul(Du,q);pfield=mul(pi-mul(dq,Ae.T),Binv.T);dp=mul(Du,pfield);ddq=mul(Du,dq)
    qdot=beta*dq+pi;pidot=c*c*(mul(Dd,dq))+beta*(mul(Du,pi));pdot=mul(pidot-mul(mul(Du,qdot),Ae.T),Binv.T)
    nativeq=mul(dq,A.T)+mul(pfield,B.T);gradient_flux=mul(c*c*(mul(Dd-Du,dq)),Binv.T);nativep=mul(ddq,C.T)+mul(dp,F.T)+gradient_flux
    rhs=max(float(np.max(abs(qdot-nativeq))),float(np.max(abs(pdot-nativep))))/max(1,float(np.max(abs(nativeq))),float(np.max(abs(nativep))))
    qinit=f[:,None]*v;piinit=duf[:,None]*(mul(Ae,v))+df[:,None]*(mul(-speed*np.eye(10)-A,v));Ei=energy(piinit,mul(Du,qinit));Et=energy(pi,dq)
    gd=mul(Du,qdot);rate=2*np.einsum('n,ni,ij,nj->',H,pi,G,pidot)+2*c*c*np.einsum('n,ni,ij,nj->',H,dq,G,gd)
    predicted=2*beta*(np.einsum('ni,ij,nj->',mul(J,pi),G,mul(J,pi))+c*c*np.einsum('ni,ij,nj->',mul(J,dq),G,mul(J,dq)))*.5
    rate_def=abs(rate-predicted)/max(1,abs(rate),abs(predicted));growth=max(0,float(Et/Ei-1))
    tx=(x-speed*t+1)%2-1;fe,de=bump(tx);qe=.001*fe[:,None]*v;pe=.001*de[:,None]*pv;scale=.001*np.sqrt(mul(v,v)+.45**2*(mul(pv,pv)));field=np.hstack([q-qe,.45*(pfield-pe)])/scale
    z=(tx+.5)/.45;inside=abs(z)<1;d2=np.zeros_like(z);u=z[inside];den=1-u*u;d2[inside]=32*den**14*(31*u*u-1)/.45**2
    gradient=np.hstack([.45*(dq-.001*de[:,None]*v),.45**2*(dp-.001*d2[:,None]*pv)])/scale
    records.append({'left_nodes':n,'right_nodes':N-nl,'state':case,'mode':mode,'phase':phase,'time':t,'field_rms':rms(field,H),'field_linf':float(np.max(np.linalg.norm(field,axis=1))),'derivative_rms':rms(gradient,H),'derivative_linf':float(np.max(np.linalg.norm(gradient,axis=1))),'canonical_energy_ratio':float(Et/Ei),'energy_growth':growth,'energy_rate_relative_defect':float(rate_def),'actual_native_corrected_RHS_relative_defect':rhs,'right_domain_q_rms_over_amplitude':rms(q[nl:]/.001,H[nl:]),'identity_pass':bool(max(rhs,rate_def,growth)<1e-10)})
  print('completed',n,case,flush=True)
 for phase in [.75,1.5,2.5]:
  for case in range(5):
   runs=[r for r in records if r['left_nodes']==n and r['state']==case and r['phase']==phase];norms.append({'nodes':n,'state':case,'phase':phase,**{key:max(r[key] for r in runs) for key in ['field_rms','field_linf','derivative_rms','derivative_linf']}})
orders=[]
for case in range(5):
 for phase in [.75,1.5,2.5]:
  runs=sorted([r for r in norms if r['state']==case and r['phase']==phase],key=lambda r:r['nodes'])
  for key in ['field_rms','field_linf','derivative_rms','derivative_linf']:
   values=[r[key] for r in runs];order=math.log(values[-2]/values[-1],2);orders.append({'state':case,'phase':phase,'quantity':key,'values':values,'fine_order':order,'passed':bool(values[0]>values[1]>values[2] and order>=1.5)})
report={'scope':__doc__,'matrix_pass':all(r['passed'] for r in matrices),'native_reconstruction_and_energy_pass':all(r['identity_pass'] for r in records),'convergence_pass':all(r['passed'] for r in orders),'transmission_pass':min(r['right_domain_q_rms_over_amplitude'] for r in records if r['phase']==.75)>.01,'matrices':matrices,'records':records,'orders':orders,'native_interface_integrated':False,'full_method_adopted':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'scipy_version':scipy.__version__,'coefficient_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'native_symbol_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()}
(out/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print({k:v for k,v in report.items() if k not in ['matrices','records','orders']});print('failed orders',sum(not r['passed'] for r in orders),'minimum fine order',min(r['fine_order'] for r in orders))
