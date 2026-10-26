#!/usr/bin/env python3
"""Live-coefficient canonical weakSAT algebra, no native integration."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
np.seterr(all='raise')
p=argparse.ArgumentParser();p.add_argument('speeds',type=Path);p.add_argument('sbp',type=Path);p.add_argument('output',type=Path);a=p.parse_args();cases=json.loads(a.speeds.read_text())['records'];s=json.loads(a.sbp.read_text());weights=np.array(s['norm_boundary_weights']);rows=np.array(s['left_boundary_rows']);trace=np.array(s['left_trace'])
def mm(x,y):return np.einsum("ij,jk->ik",x,y)

def spatial(n):
    m = len(weights);h = np.ones(n);h[:m] = weights;h[-m:] = weights[::-1]
    d = np.zeros((n,n));width=rows.shape[1];d[:m,:width] = rows;d[-m:,-width:] = -rows[::-1,::-1]
    for k in range(m,n-m):
        for off,c in ([(-3,-1/60),(-2,3/20),(-1,-3/4),(1,3/4),(2,-3/20),(3,1/60)] if rows.shape[1]-m==3 else [(-2,1/12),(-1,-2/3),(1,2/3),(2,-1/12)]):d[k,k+off] = c
    left = np.zeros(n);left[:m] = trace;right = left[::-1]
    bl,br = np.outer(left,left),np.outer(right,right)
    residual = np.linalg.norm(h[:,None]*d+d.T*h[None,:]-(br-bl))
    assert min(h)>0 and residual<1e-10 and abs(sum(h)-n)<1e-10
    return h,d,bl,br,residual
def coeff(x,t,lo0,hi0):
 hi=hi0*(1+.04*(x-.5)+.02*np.sin(t));lo=lo0*(1+.03*(x-.5)-.01*np.sin(t));hit=np.full_like(x,.02*hi0*np.cos(t));lot=np.full_like(x,-.01*lo0*np.cos(t));beta=-(hi+lo)/2;c=(hi-lo)/2;betat=-(hit+lot)/2;aa=-lo*hi;at=-(lot*hi+lo*hit);return dict(lo=lo,hi=hi,beta=beta,c=c,betat=betat,a=aa,at=at,hit=hit,lot=lot)
records=[]
for case in cases:
 tests=[]
 for n in [24,48]:
  h0,d0,bl,br,_=spatial(n);h=h0/n;D=d0*n;H=np.diag(h);I=np.eye(n);Z=np.zeros((n,n));tl=np.zeros(n);tl[:len(trace)]=trace;tr=tl[::-1];E=np.vstack([tl,tr]);B=np.diag([-1.,1.]);Q=h[:,None]*D
  QN=.5*np.block([[Q-Q.T,mm(E.T,B)],[-mm(B,E),B]]);P=np.vstack([I,E]);QNboundary=np.zeros_like(QN);QNboundary[-2:,-2:]=B;qn_error=float(np.linalg.norm(QN+QN.T-QNboundary));qn_constant=float(max(abs(np.einsum('ij,j->i',QN,np.ones(n+2)))))
  x=(np.arange(n)+.5)/n
  for t in [0.,.37,.81]:
   u=coeff(x,t,case['lo'],case['hi']);face=coeff(np.array([0.,1.]),t,case['lo'],case['hi']);assert min(u['a'])>0 and max(u['lo'])<0 and min(u['hi'])>0
   betax=np.einsum('ij,j->i',D,u['beta']);ax=np.einsum('ij,j->i',D,u['a']);bb=u['betat']-u['beta']*betax;lower=bb-ax;betaN=np.diag(np.r_[u['beta'],face['beta']]);shift=mm(mm(P.T,mm(QN,betaN)+mm(betaN,QN)),P)/h[:,None]-np.diag(betax)
   constant_error=float(max(abs(np.einsum('ij,j->i',shift,np.ones(n)))))
   potential=mm(mm(D.T,H*np.array(u['a'])[None,:]),D);weak=(-potential+mm(mm(mm(E.T,B),np.diag(face['a'])),mm(E,D)))/h[:,None]
   L=np.block([[Z,I],[weak+lower[:,None]*D,shift]])
   left=np.r_[np.zeros(n),face['hi'][0]*tl/h][:,None];right=np.r_[np.zeros(n),-face['lo'][1]*tr/h][:,None]
   leftrow=np.hstack([face['lo'][0]*mm(tl[None,:],D),tl[None,:]]);rightrow=np.hstack([face['hi'][1]*mm(tr[None,:],D),tr[None,:]])
   L-=mm(left,leftrow)+mm(right,rightrow)
   W=np.block([[potential,Z],[Z,H]]);Wdot=np.block([[mm(mm(D.T,H*u['at'][None,:]),D),Z],[Z,Z]])
   cross=mm(D.T,H*lower[None,:]);boundary=-2*(face['c'][0]*bl+face['c'][1]*br);expected=np.block([[Wdot[:n,:n],cross],[cross.T,boundary-2*H*betax[None,:]]]);actual=mm(W,L)+mm(L.T,W)+Wdot;scale=max(1,np.linalg.norm(expected));defect=float(np.linalg.norm(actual-expected)/scale)
   naive=L.copy();naive[n:,n:]+=2*u['beta'][:,None]*D-shift;naive_defect=float(np.linalg.norm(mm(W,naive)+mm(naive.T,W)+Wdot-expected)/scale)
   La=np.zeros((2*n+1,2*n+1));La[:2*n,:2*n]=L;La[:2*n,-1]=right[:,0];La[-1,-1]=-.4;Wa=np.zeros_like(La);Wa[:2*n,:2*n]=W;Wa[-1,-1]=1;Wat=np.zeros_like(La);Wat[:2*n,:2*n]=Wdot
   Ea=mm(Wa,La)+mm(La.T,Wa)+Wat;expected_a=np.zeros_like(La);expected_a[:2*n,:2*n]=expected;expected_a[-1,-1]=-.8;expected_a[n:2*n,-1]=-face['lo'][1]*tr;expected_a[-1,n:2*n]=-face['lo'][1]*tr
   ode_defect=float(np.linalg.norm(Ea-expected_a)/max(1,np.linalg.norm(expected_a)));volume=np.zeros_like(La);volume[:n,:n]=Wdot[:n,:n];volume[:n,n:2*n]=cross;volume[n:2*n,:n]=cross.T;volume[n:2*n,n:2*n]=-2*H*betax[None,:]
   boundary_max=float(np.linalg.eigvalsh(Ea-volume).max()/max(1,np.linalg.norm(Ea-volume)))
   normalized_volume=np.zeros((n,2,2));normalized_volume[:,0,0]=-2*betax;normalized_volume[:,0,1]=normalized_volume[:,1,0]=lower/np.sqrt(u['a']);normalized_volume[:,1,1]=u['at']/u['a'];work_bound=float(max(0.,np.linalg.eigvalsh(normalized_volume).max()))

   q=np.sin(2*x)+.2*x*x;v=np.cos(3*x)-.1*x;state=np.r_[q,v];rhs=np.einsum('ij,j->i',L,state);qgrad=np.einsum('ij,j->i',D,q);gradtime=np.einsum('ij,j->i',D,rhs[:n]);direct_wt=float(np.dot(tr,rhs[n:])+face['hi'][1]*np.dot(tr,gradtime)+face['hit'][1]*np.dot(tr,qgrad));wrong_wt=direct_wt-float(face['hit'][1]*np.dot(tr,qgrad));chain=[]
   for eps in [1e-5,5e-6]:
    vals=[]
    for sign in [-1,1]:
     f=coeff(np.array([1.]),t+sign*eps,case['lo'],case['hi']);uq=q+sign*eps*rhs[:n];uv=v+sign*eps*rhs[n:];vals.append(float(np.dot(tr,uv)+f['hi'][0]*np.dot(tr,np.einsum('ij,j->i',D,uq))))
    chain.append(abs((vals[1]-vals[0])/(2*eps)-direct_wt)/max(1,abs(direct_wt)))
   pfield=v-u['beta']*qgrad;pdot=rhs[n:]-u['betat']*qgrad-u['beta']*gradtime;compatibility=float(max(abs(rhs[:n]-(u['beta']*qgrad+pfield))));omitted_betat=float(max(abs(u['betat']*qgrad)))
   tests.append(dict(nodes=n,time=t,auxiliary_ODE_energy_defect=ode_defect,auxiliary_ODE_boundary_max_energy_eigenvalue=boundary_max,volume_work_Gronwall_rate=work_bound,energy_identity_relative_defect=defect,naive_variable_advection_relative_defect=naive_defect,constant_velocity_shift_error=constant_error,face_SBP_error=qn_error,face_constant_error=qn_constant,live_characteristic_chainrule_errors=list(map(float,chain)),omitted_characteristic_time_term=abs(direct_wt-wrong_wt),omitted_p_reconstruction_time_term=omitted_betat,reduction_compatibility_error=compatibility,passed=bool(ode_defect<=1e-10 and boundary_max<=1e-10 and defect<=1e-10 and constant_error<=1e-8 and qn_error<=1e-10 and qn_constant<=1e-10 and max(chain)<=1e-7 and compatibility<=1e-12)))
 records.append(dict(source=case['source'],sample=case['sample'],lo0=case['lo'],hi0=case['hi'],tests=tests,passed=all(x['passed'] for x in tests)))
negative=max(t['naive_variable_advection_relative_defect'] for r in records for t in r['tests'])>1e-8
result=dict(passed=all(x['passed'] for x in records) and negative,negative_control_detected=negative,records=records,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.speeds,a.sbp]},scope='Live-coefficient scalar second-order weakSAT identity with decoupled face products and computed Dq. Coefficient volume work retained; no monotonic-energy assertion. Not nonlinear native boundaryODE/evolution, full constraint-subsidiary closure, ordinary GPU/AMR regression or physical stability.')
a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(dict(passed=result['passed'],states=len(records),max_energy_defect=max(t['energy_identity_relative_defect'] for r in records for t in r['tests']),negative_control_max=max(t['naive_variable_advection_relative_defect'] for r in records for t in r['tests']),failures=[t for r in records for t in r['tests'] if not t['passed']]))
