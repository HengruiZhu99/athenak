#!/usr/bin/env python3
"""Actual frozen-native20 two-domain SBP propagation, isolated spatial model."""
import argparse,ast,hashlib,json,math
from pathlib import Path
import numpy as np
np.seterr(all='raise')
p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--extend',action='store_true');args=p.parse_args();out=args.output;protocol=json.loads((out/'protocol-v1.json').read_text())
if args.extend:
 extension=json.loads((out/'extension-protocol-v1.json').read_text());base=json.loads((out/'review.json').read_text());assert hashlib.sha256((out/'review.json').read_bytes()).hexdigest()==extension['base_review_sha256'];assert not (out/'extension-review.json').exists()
else:assert not (out/'review.json').exists(),'Reuse completed results; use a new version for changed tests.'
source=Path('docs/validation/cubed-sphere-balanced-20261013/outer-closure/cell-centered-sbp-sixth-exploration.json');s=json.loads(source.read_text());weights=np.array(s['norm_boundary_weights']);rows=np.array(s['left_boundary_rows']);trace=np.array(s['left_trace'])
for filename,names in [('check_constraint_incoming_lift.py',['mm','symbols']),('check_goal1_outer_port.py',['gram'])]:
 helper=Path('tst/cubed_sphere')/filename
 for node in ast.parse(helper.read_text()).body:
  if isinstance(node,ast.FunctionDef) and node.name in names:exec(compile(ast.Module(body=[node],type_ignores=[]),str(helper),'exec'),globals())
raw=Path('docs/validation/cubed-sphere-balanced-20261013/finite-radius-long/pulse-t02-controls/pulse-boundary-symbol-t02.txt');states=list(symbols(raw));assert len(states)==5
def block(n,start):
 h=np.ones(n)/n;h[:8]=weights/n;h[-8:]=weights[::-1]/n;D=np.zeros((n,n));D[:8,:11]=rows*n;D[-8:,-11:]=-rows[::-1,::-1]*n
 for i in range(8,n-8):
  for j,c in [(-3,-1/60),(-2,3/20),(-1,-3/4),(1,3/4),(2,-3/20),(3,1/60)]:D[i,i+j]=c*n
 # Same documented constant anchor; do not project away energy directions.
 D[np.arange(n),np.arange(n)]-=D.sum(axis=1)
 l=np.zeros(n);l[:8]=trace;r=l[::-1];x=start+(np.arange(n)+.5)/n
 return x,h,D,l,r
def coupled(n):
 nl,nr=n,3*n//2;xL,hL,DL,ll,lr=block(nl,-1);xR,hR,DR,rl,rr=block(nr,0);H=np.r_[hL,hR];D=np.zeros((nl+nr,nl+nr));D[:nl,:nl]=DL;D[nl:,nl:]=DR
 for left,right,el,er in [(slice(0,nl),slice(nl,None),lr,rl),(slice(nl,None),slice(0,nl),rr,ll)]:
  hl=H[left];hr=H[right];D[left,left]-=.5*np.outer(el/hl,el);D[left,right]+=.5*np.outer(el/hl,er);D[right,left]-=.5*np.outer(er/hr,el);D[right,right]+=.5*np.outer(er/hr,er)
 return np.r_[xL,xR],H,D,nl
def bump(x):
 z=(x+.5)/.22;inside=abs(z)<1;f=np.zeros_like(x);d=np.zeros_like(x);u=z[inside];den=1-u*u;f[inside]=np.exp(1-1/den);d[inside]=f[inside]*(-2*u)/(.22*den*den)
 return f,d
def rms(z,H):return float(np.sqrt(np.einsum('n,ni,ni->',H,z,z)/sum(H)))
matrices=list(base['matrices']) if args.extend else [];records=list(base['records']) if args.extend else [];norms=[]
if args.extend:
 for n in protocol['resolutions']:
  for phase in [.75,1.5,2.5]:
   for case in range(5):
    runs=[r for r in records if r['left_nodes']==n and r['state']==case and r['phase']==phase];norms.append({'nodes':n,'state':case,'phase':phase,**{key:max(r[key] for r in runs) for key in ['field_rms','field_linf','derivative_rms','derivative_linf']}})
for n in (extension['new_left_nodes'] if args.extend else protocol['resolutions']):
 x,H,D,nl=coupled(n);sqrt=np.sqrt(H);S=sqrt[:,None]*D/sqrt[None,:];skew=float(np.linalg.norm(S+S.T)/max(1,np.linalg.norm(S)));Q=-1j*(S-S.T)/2
 omega,V=np.linalg.eigh(Q);projection=float(np.linalg.norm(Q+1j*S)/max(1,np.linalg.norm(S)));constant=float(np.max(abs(D.sum(axis=1))));coupling=float(np.linalg.norm(D[nl:,:nl]))
 matrices.append({'left_nodes':n,'right_nodes':len(x)-nl,'SBP_skew_relative':skew,'modal_roundoff_projection':projection,'constant_absolute':constant,'left_to_right_coupling_norm':coupling,'passed':bool(max(skew,projection,constant)<=1e-10 and coupling>0)})
 f,df=bump(x);f*=.001;df*=.001
 for case,state in enumerate(states):
  M=state['matrix'];A=M[:10,:10];B=M[:10,10:];C=M[10:,:10];F=M[10:,10:];beta=-(state['lo']+state['hi'])/2;c=(state['hi']-state['lo'])/2;speed=c-beta;Ae=A-beta*np.eye(10);Binv=np.linalg.inv(B)
  G=gram(state['basis'][:11,:10],state['inverse']);basis=np.linalg.inv(np.linalg.cholesky(G).T)
  for mode in range(10):
   v=basis[:,mode];pcoef=np.einsum('ij,j->i',Binv,np.einsum('ij,j->i',(-speed*np.eye(10)-A),v));scale=np.sqrt(v.dot(v)+.22**2*pcoef.dot(pcoef))*.001
   q0=f[:,None]*v;p0=df[:,None]*pcoef;pi0=mm(mm(D,q0),Ae.T)+mm(p0,B.T)
   qhat=mm(V.conj().T,sqrt[:,None]*q0);phat=mm(V.conj().T,sqrt[:,None]*pi0);dq0=mm(D,q0)
   E0=float(np.einsum('n,ni,ij,nj->',H,pi0,G,pi0)+c*c*np.einsum('n,ni,ij,nj->',H,dq0,G,dq0))
   for phase in [0.,.75,1.5,2.5]:
    t=phase/speed;co=np.cos(c*omega*t)[:,None];si=np.sin(c*omega*t)[:,None];rotation=np.exp(1j*beta*omega*t)[:,None]
    qt=mm(V,rotation*(co*qhat+t*np.sinc(c*omega*t/math.pi)[:,None]*phat))/sqrt[:,None]
    pit=mm(V,rotation*(-c*omega[:,None]*si*qhat+co*phat))/sqrt[:,None];imag=max(float(np.max(abs(qt.imag))),float(np.max(abs(pit.imag))));q=qt.real;pi=pit.real
    dq=mm(D,q);pfield=mm(pi-mm(dq,Ae.T),Binv.T);dp=mm(D,pfield);ddq=mm(D,dq)
    qdot=beta*dq+pi;pidot=c*c*ddq+beta*mm(D,pi);pdot=mm(pidot-mm(mm(D,qdot),Ae.T),Binv.T)
    nativeq=mm(dq,A.T)+mm(pfield,B.T);nativep=mm(ddq,C.T)+mm(dp,F.T);rhs=max(float(np.max(abs(qdot-nativeq))),float(np.max(abs(pdot-nativep))))/max(1,float(np.max(abs(nativeq))),float(np.max(abs(nativep))))
    targetx=(x-speed*t+1)%2-1;fe,de=bump(targetx);qe=.001*fe[:,None]*v;pe=.001*de[:,None]*pcoef
    field=np.hstack([q-qe,.22*(pfield-pe)])/scale
    # Analytic second derivative of the compact bump, no finite differences.
    z=(targetx+.5)/.22;inside=abs(z)<1;d2=np.zeros_like(z);u=z[inside];den=1-u*u
    d2[inside]=fe[inside]*((4*u*u)/(den**4)-2/(den**2)-8*u*u/(den**3))/(.22**2)
    gradient=np.hstack([.22*(dq-.001*de[:,None]*v),.22**2*(dp-.001*d2[:,None]*pcoef)])/scale
    E=float(np.einsum('n,ni,ij,nj->',H,pi,G,pi)+c*c*np.einsum('n,ni,ij,nj->',H,dq,G,dq));energy=abs(E/E0-1)
    record={'left_nodes':n,'right_nodes':len(x)-nl,'state':case,'mode':mode,'phase':phase,'time':t,'field_rms':rms(field,H),'field_linf':float(np.max(np.linalg.norm(field,axis=1))),
      'derivative_rms':rms(gradient,H),'derivative_linf':float(np.max(np.linalg.norm(gradient,axis=1))),'canonical_energy_relative_drift':energy,'actual_native_RHS_relative_defect':rhs,'modal_imaginary_residue':imag,
      'right_domain_q_rms_over_amplitude':rms(q[nl:]/.001,H[nl:]),'decoupled_zero_right_control_oracle_error':rms(qe[nl:]/.001,H[nl:]),'identity_pass':bool(max(rhs,energy,imag)<=1e-10)}
    records.append(record)
 for phase in [.75,1.5,2.5]:
  for case in range(5):
   runs=[r for r in records if r['left_nodes']==n and r['state']==case and r['phase']==phase]
   norms.append({'nodes':n,'state':case,'phase':phase,**{key:max(r[key] for r in runs) for key in ['field_rms','field_linf','derivative_rms','derivative_linf']}})
orders=[]
for case in range(5):
 for phase in [.75,1.5,2.5]:
  runs=sorted([r for r in norms if r['state']==case and r['phase']==phase],key=lambda r:r['nodes'])
  for key in ['field_rms','field_linf','derivative_rms','derivative_linf']:
   values=[r[key] for r in runs];tail=values[-3:];order=math.log(values[-2]/values[-1],2);orders.append({'state':case,'phase':phase,'quantity':key,'values':values,'fine_order':order,'tested_nodes':[r['nodes'] for r in runs[-3:]],'passed':bool(tail[0]>tail[1]>tail[2] and order>=1.5)})
report={'scope':__doc__,'matrix_pass':all(r['passed'] for r in matrices),'native_reconstruction_and_energy_pass':all(r['identity_pass'] for r in records),'convergence_pass':all(r['passed'] for r in orders),
 'transmission_pass':min(r['right_domain_q_rms_over_amplitude'] for r in records if r['phase']==.75)>0.01,'matrices':matrices,'records':records,'orders':orders,
 'native_interface_integrated':False,'full_method_adopted':False,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'coefficient_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'native_symbol_sha256':hashlib.sha256(raw.read_bytes()).hexdigest()}
(out/('extension-review.json' if args.extend else 'review.json')).write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print({k:v for k,v in report.items() if k not in ['matrices','records','orders']});print('max energy drift',max(r['canonical_energy_relative_drift'] for r in records));print('failed orders',sum(not r['passed'] for r in orders),'minimum fine order',min(r['fine_order'] for r in orders))
