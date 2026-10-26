#!/usr/bin/env python3
"""Smooth consistency of scalar live-coefficient weakSAT, all boundary cells."""
import argparse,hashlib,json,math
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
def mv(A,v):return np.einsum('ij,j->i',A,v)
records=[]
for case in cases:
 tests=[]
 for t in [0.,.37,.81]:
  runs=[]
  for n in [64,128,256]:
   h0,d0,_,_,_=spatial(n);h=h0/n;D=d0*n;x=(np.arange(n)+.5)/n;u=coeff(x,t,case['lo'],case['hi']);face=coeff(np.array([0.,1.]),t,case['lo'],case['hi']);tl=np.zeros(n);tl[:len(trace)]=trace;tr=tl[::-1];E=np.vstack([tl,tr]);B=np.diag([-1.,1.]);Q=h[:,None]*D;QN=.5*np.block([[Q-Q.T,mm(E.T,B)],[-mm(B,E),B]]);P=np.vstack([np.eye(n),E]);bn=np.r_[u['beta'],face['beta']]
   q=np.exp(x);v=np.sin(2*x);vx=2*np.cos(2*x);qgrad=mv(D,q);vxnum=mv(D,v);betax=mv(D,u['beta']);ax=mv(D,u['a']);bb=u['betat']-u['beta']*betax
   weak=(-mv(D.T,h*u['a']*qgrad)+mv(E.T,mv(B,face['a']*mv(E,qgrad))))/h;pv=mv(P,v);shift=mv(P.T,mv(QN,bn*pv)+bn*mv(QN,pv))/h-betax*v;vrhs=weak+shift+(bb-ax)*qgrad
   vf=np.sin(2*np.array([0.,1.]));qxf=np.exp(np.array([0.,1.]));g=np.array([vf[0]+face['lo'][0]*qxf[0],vf[1]+face['hi'][1]*qxf[1]]);wv=mv(E,v);wg=mv(E,qgrad);vrhs-=face['hi'][0]*tl/h*(wv[0]+face['lo'][0]*wg[0]-g[0]);vrhs-=-face['lo'][1]*tr/h*(wv[1]+face['hi'][1]*wg[1]-g[1])
   exact_beta_x=-(.04*case['hi']+.03*case['lo'])/2;exactb=u['betat']-u['beta']*exact_beta_x;vexact=2*u['beta']*vx+u['a']*q+exactb*q;prhs=vrhs-u['betat']*qgrad-u['beta']*vxnum;pexact=u['c']**2*q+u['beta']*(vx-exact_beta_x*q-u['beta']*q);values={}
   for field,delta in [('v_rhs',vrhs-vexact),('p_rhs',prhs-pexact)]:values[field]=dict(l2=float(np.sqrt(np.sum(h*delta*delta))),linf=float(max(abs(delta))))
   runs.append(dict(n=n,errors=values))
  for field in ['v_rhs','p_rhs']:
   for norm in ['l2','linf']:
    z=[r['errors'][field][norm] for r in runs];order=math.log(z[1]/z[2],2);tests.append(dict(time=t,field=field,norm=norm,errors=z,finest_order=order,passed=z[0]>z[1]>z[2] and order>=1.5))
 records.append(dict(source=case['source'],sample=case['sample'],tests=tests,passed=all(x['passed'] for x in tests)))
r=dict(passed=all(x['passed'] for x in records),records=records,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.speeds,a.sbp]},scope='Instantaneous live-coefficient scalar weakSAT consistency with exact boundary data, no interior forcing/subtraction. Not native propagation/reflection/constraintconvergence or stability.')
a.output.write_text(json.dumps(r,indent=2)+'\n');print(dict(passed=r['passed'],minimum_order=min(x['finest_order'] for q in records for x in q['tests']),failures=[x for q in records for x in q['tests'] if not x['passed']]))
