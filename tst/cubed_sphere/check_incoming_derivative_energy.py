#!/usr/bin/env python3
"""Read-only scalar derivative-energy audit; no native changes."""
import ast,json,hashlib,argparse
from pathlib import Path
import numpy as np
np.seterr(over='raise',divide='raise',invalid='raise',under='ignore')
p=argparse.ArgumentParser();p.add_argument('parent',type=Path);p.add_argument('sbp',type=Path);p.add_argument('output',type=Path);args=p.parse_args();args.output.mkdir(parents=True,exist_ok=True)
source=Path(__file__).with_name('check_incoming_mapped_transport.py');helper=source.with_name('evolve_live_coefficient_sbp_sat.py');s=json.loads(args.sbp.read_text());weights=np.array(s['norm_boundary_weights']);rows=np.array(s['left_boundary_rows']);trace=np.array(s['left_trace'])
for path,names in [(helper,['mm','spatial']),(source,['bump','bump_derivative'])]:
 tree=ast.parse(path.read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),str(path),'exec'))

def operator(n,k,lo,t):
 xi=(np.arange(n)+.5)/n;J=k*np.exp(k*(1-xi))/np.expm1(k) if k else np.ones(n);x=1-np.expm1(k*(1-xi))/np.expm1(k) if k else xi;h0,d0,*_=spatial(n);h=h0*J/n;D=d0*n/J[:,None];tl=np.zeros(n);tl[:len(trace)]=trace;tr=tl[::-1];E=np.vstack([tl,tr]);B=np.diag([-1.,1.]);Q=h[:,None]*D;QN=.5*np.block([[Q-Q.T,mm(E.T,B)],[-mm(B,E),B]]);P=np.vstack([np.eye(n),E]);av=lo*(1+.03*(x-.5))*(1+.02*np.sin(t));af=lo*np.array([.985,1.015])*(1+.02*np.sin(t));ax=lo*.03*(1+.02*np.sin(t));an=np.r_[av,af];L=-.5*mm(P.T,mm(QN*an[None,:]+an[:,None]*QN,P))/h[:,None]+.5*ax*np.eye(n)+af[1]*np.outer(tr/h,tr);return x,h,D,L,tl,tr,af,ax

def dot(x,y):return float(np.einsum('i,i->',x,y))
def mv(A,x):return np.einsum('ij,j->i',A,x)
def quadratic(A,x):return dot(x,mv(A,x))

def whiten(A,G):
 R=np.linalg.cholesky(G);left=np.linalg.solve(R,A);Z=np.linalg.solve(R,left.T).T;return (Z+Z.T)/2
parent=json.loads((args.parent/'review.json').read_text());identities=[];screens=[]
for case in parent['cases']:
 if not case['mapped']:continue
 for key,run in case['runs'].items():
  n=int(key);payload=args.parent/(run['label']+'.npz');assert hashlib.sha256(payload.read_bytes()).hexdigest()==run['payload_sha256'];data=np.load(payload)
  for point,hist in enumerate(run['history']):
   t=hist['time'];x,h,D,L,tl,tr,af,ax=operator(n,run['k'],run['lo'],t);assert np.max(abs(data['x']-x))<1e-14;w=data['states'][point,0];r=mv(D,w);g=float(bump(t,run['tau']));gt=float(bump_derivative(t,run['tau']));zb=-gt/af[1];f=-af[1]*tr/h*g;fz=-af[1]*tr/h*zb;rdirect=mv(D,mv(L,w)+f);base=mv(L,r)-ax*r+fz;C=mv(D,mv(L,w))-mv(L,r)+ax*r+mv(D,f)-fz
   direct=2*dot(h*r,rdirect);volume=-ax*dot(h,r*r);rl,rr=dot(tl,r),dot(tr,r);boundary=af[0]*rl*rl+af[1]*(rr-zb)**2-af[1]*zb*zb;commutator=2*dot(h*r,C);decomposed=volume+boundary+commutator;scale=max(abs(direct),abs(volume)+abs(boundary)+abs(commutator),1e-30);rhsdef=float(max(abs(rdirect-base-C)))/max(float(max(abs(rdirect))),float(max(abs(base))),float(max(abs(C))),1e-30);energydef=abs(direct-decomposed)/scale
   identities.append(dict(name=case['name'],n=n,point=point,derivative_boundary_datum=zb,derivative_energy=float(dot(h,r*r)),direct_energy_rate=direct,volume_rate=volume,boundary_rate=boundary,commutator_rate=commutator,omitted_commutator_defect=abs(commutator)/scale,rhs_relative_defect=rhsdef,energy_relative_defect=energydef,passed=rhsdef<=1e-10 and energydef<=1e-10))
 model=case['runs']['256'];tau=model['tau'];ell=abs(model['lo'])*tau
 for mapped in [False,True]:
  rates=[]
  for n in [32,64,128]:
   x,h,D,L,tl,tr,af,ax=operator(n,model['k'] if mapped else 0.,model['lo'],.4*tau);H=np.diag(h);G=H+ell*ell*mm(D.T,h[:,None]*D);S=mm(G,L)+mm(L.T,G);lognorm=float(np.linalg.eigvalsh(whiten(S,G))[-1]);Sl2=h[:,None]*L+L.T*h[None,:];l2=float(np.linalg.eigvalsh(Sl2/np.sqrt(h[:,None]*h[None,:]))[-1]);known=abs(ax);rates.append(dict(n=n,energy_logarithmic_rate=lognorm,tau_rate=tau*lognorm,continuum_derivative_volume_rate=known,l2_energy_logarithmic_rate=l2,positive_excess=max(0.,lognorm-known),G_min_eigenvalue=float(np.linalg.eigvalsh(G)[0])))
  ratios=[rates[j+1]['positive_excess']/rates[j]['positive_excess'] if rates[j]['positive_excess']>0 else None for j in [0,1]];screens.append(dict(name=case['name'],mapped=mapped,rates=rates,excess_refinement_ratios=ratios,proposed_uniform_estimate_pass=all(r['tau_rate']<=10 for r in rates) and all(z is not None and z<=2 for z in ratios)))
r=dict(identities=identities,screens=screens,all_identities_pass=all(x['passed'] for x in identities),negative_control_detected=any(x['omitted_commutator_defect']>1e-3 for x in identities),scope='Scalar instantaneous H1 estimate screen, not evolution or native proof',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest());(args.output/'review.json').write_text(json.dumps(r,indent=2,default=lambda x:x.item())+'\n');print('identities',r['all_identities_pass'],'negative',r['negative_control_detected']);print([(x['name'],x['mapped'],x['proposed_uniform_estimate_pass'],[z['tau_rate'] for z in x['rates']],x['excess_refinement_ratios']) for x in screens])
