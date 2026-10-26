#!/usr/bin/env python3
"""Actual frozen native40-variable derivative energy with tangential BC rows."""
import argparse,ast,json,hashlib
from pathlib import Path
import numpy as np
np.seterr(all='raise')
p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args();out=a.output
assert (out/'protocol-v1.json').exists();base=Path('docs/validation/goal1-minkowski-20261018')
for filename,names in [('check_constraint_incoming_lift.py',['mm','symbols']),('check_goal1_outer_port.py',['gram']),('check_goal1_incoming_hierarchy.py',['gram_tt'])]:
 path=Path('tst/cubed_sphere')/filename
 for node in ast.parse(path.read_text()).body:
  if isinstance(node,ast.FunctionDef) and node.name in names:exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),globals())
oblique=list(symbols(base/'oblique-symbol.txt'))[:30]
old=Path('docs/validation/cubed-sphere-balanced-20261013/finite-radius-long/pulse-t02-controls/pulse-boundary-symbol-t02.txt');normals=list(symbols(old));assert len(normals)==5
requests=[list(map(float,s.split())) for s in (base/'oblique-symbol.samples').read_text().splitlines()][:30]
def whiten(F,W):
 L=np.linalg.cholesky(W);left=np.linalg.solve(L,F);Z=np.linalg.solve(L,left.T).T;return (Z+Z.T)/2,L
records=[]
for index,normalcase in enumerate(normals):
 cases=oblique[index*6:index*6+3];n=normalcase['normal'];inv=normalcase['inverse'];qb=normalcase['basis'][:11,:10];metric=np.linalg.inv(inv);G=gram(qb,inv)
 values=requests[index*6];alpha=values[25];chi=values[7];aij=alpha*alpha*chi*inv;an=aij.dot(n);c=np.sqrt(n.dot(an));conormal=an/(c*c)
 beta=np.array([-(s['lo']+s['hi'])/2 for s in cases]);bn=beta.dot(n);ain=c+bn
 B=cases[0]['matrix'][:10,10:];Binv=np.linalg.inv(B);Ae=[s['matrix'][:10,:10]-beta[i]*np.eye(10) for i,s in enumerate(cases)]
 # Constant scaling changes neither the four constraint conditions nor K^-1 U.
 physical=np.zeros((4,4));physical[0,0]=1/normalcase['omega'];physical[1:,1:]=2*inv
 Jp=mm(physical,cases[0]['constraints'][4:,10:]);Up=mm(Jp,Binv)
 Uq=[mm(physical,s['constraints'][4:,:10])-mm(Up,Ae[i]) for i,s in enumerate(cases)]
 U=np.hstack([Up,*Uq]);Jnormal=sum(n[i]*Uq[i] for i in range(3));Jin=Jnormal+c*Up;Jout=Jnormal-c*Up
 Fraw=np.zeros((6,11));Fraw[0,7]=1;Fraw[1:4,8:11]=np.eye(3)
 for row,T in enumerate(gram_tt(metric,inv,n),4):Fraw[row,1:7]=[T[0,0],2*T[0,1],2*T[0,2],T[1,1],2*T[1,2],T[2,2]]
 F=mm(Fraw,qb);K=np.vstack([Jin,F]);scale=np.linalg.norm(K,axis=1);K/=scale[:,None]
 incoming=np.hstack([np.eye(10)/c,*[conormal[i]*np.eye(10) for i in range(3)]])/2
 sensors=np.vstack([U,mm(F,incoming)]);lift=2*c*np.linalg.solve(K,sensors/scale[:,None])
 # A normal-only approximation replaces all actual tangential U terms by zero.
 normalonly=mm(np.hstack([Up,Jnormal]),np.block([[np.eye(10),np.zeros((10,30))],[np.zeros((10,10)),np.hstack([conormal[i]*np.eye(10) for i in range(3)])]]))
 omission=float(np.linalg.norm(U-normalonly)/max(1,np.linalg.norm(U)))
 W=np.zeros((40,40));W[:10,:10]=G;W[10:,10:]=np.kron(aij,G)
 flux=bn*W
 for i in range(3):flux[:10,10+10*i:20+10*i]+=an[i]*G;flux[10+10*i:20+10*i,:10]+=an[i]*G
 selector=np.hstack([np.eye(10),np.zeros((10,30))]);closed=flux-ain*(mm(mm(selector.T,G),lift)+mm(mm(lift.T,G),selector))
 Z,L=whiten(closed,W);eigen,vectors=np.linalg.eigh(Z);v=np.linalg.solve(L.T,vectors[:,-1]);direct=v.dot(closed.dot(v))/v.dot(W.dot(v))
 openZ,_=whiten(flux,W);speeds=np.linalg.eigvalsh(openZ);expected=np.sort(np.r_[np.full(10,bn-c),np.full(20,bn),np.full(10,bn+c)])
 spectrum=float(np.max(abs(speeds-expected))/max(1,c,abs(bn)));rate=float(eigen[-1]);rayleigh=abs(direct-rate)/max(1,abs(rate))
 records.append({'state':index,'radius':normalcase['radius'],'Omega':normalcase['omega'],'beta_normal':float(bn),'light_normal':float(c),
  'open_characteristic_speed_error':spectrum,'normalized_closed_maximum':rate/max(1,c,abs(bn)),'raw_signed_closed_maximum':rate,'direct_Rayleigh_rate':float(direct),'Rayleigh_relative_error':float(rayleigh),
  'tangential_constraint_omission_relative':omission,'K_condition':float(np.linalg.cond(K)),'W_condition':float(np.linalg.cond(W)),
  'positive_flux_witness':v.tolist() if rate>1e-10 else None,'passed':bool(spectrum<=1e-10 and rate/max(1,c,abs(bn))<=1e-10 and rayleigh<=1e-10)})
report={'scope':__doc__,'full_frozen_tangential_port_pass':all(r['passed'] for r in records),'states':len(records),'records':records,
 'omitted_tangential_control_detected':max(r['tangential_constraint_omission_relative'] for r in records)>=1e-5,'native_SAT_installed':False,'full_method_adopted':False,'exact_scri_admitted':False,
 'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'input_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in [old,base/'oblique-symbol.txt',base/'oblique-symbol.samples']}}
(out/'review.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print(json.dumps(report,indent=2))
