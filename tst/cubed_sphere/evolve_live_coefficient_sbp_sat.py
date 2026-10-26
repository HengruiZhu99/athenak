#!/usr/bin/env python3
"""Bounded scalar weakSAT evolution, not native physical validation."""
import argparse,hashlib,json,math,time
from pathlib import Path
import numpy as np
np.seterr(over='raise',invalid='raise',divide='raise',under='ignore')
p=argparse.ArgumentParser();p.add_argument('speeds',type=Path);p.add_argument('sbp',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=True);cases=json.loads(a.speeds.read_text())['records'];s=json.loads(a.sbp.read_text());weights=np.array(s['norm_boundary_weights']);rows=np.array(s['left_boundary_rows']);trace=np.array(s['left_trace']);sourcehash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
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
class Sparse:
 def __init__(self,A):
  width=max(np.count_nonzero(A,axis=1));self.indices=np.zeros((len(A),width),int);self.values=np.zeros((len(A),width))
  for i,row in enumerate(A):
   idx=np.flatnonzero(row);self.indices[i,:len(idx)]=idx;self.values[i,:len(idx)]=row[idx]
 def __call__(self,x):return np.einsum('ij,ij->i',self.values,x[self.indices])
def run(case,name,kind,n,half=False):
 label=name+'-'+kind+'-'+str(n)+('-half' if half else '');meta=a.output/(label+'.json');payload=a.output/(label+'.npz')
 if meta.exists():
  j=json.loads(meta.read_text());assert j['source_sha256']==sourcehash and j['payload_sha256']==hashlib.sha256(payload.read_bytes()).hexdigest();return j,np.load(payload)
 h0,d0,_,_,_=spatial(n);h=h0/n;D=d0*n;tl=np.zeros(n);tl[:len(trace)]=trace;tr=tl[::-1];E=np.vstack([tl,tr]);B=np.diag([-1.,1.]);Q=h[:,None]*D;QN=.5*np.block([[Q-Q.T,mm(E.T,B)],[-mm(B,E),B]]);d=Sparse(D);dt=Sparse(D.T);qn=Sparse(QN);x=(np.arange(n)+.5)/n;lo0,hi0=case['lo'],case['hi']
 def extend(v):return np.r_[v,np.dot(tl,v),np.dot(tr,v)]
 def project(v):return v[:n]+tl*v[n]+tr*v[n+1]
 def energy(t,y):
  u=coeff(x,t,lo0,hi0);qg=d(y[:n]);return float(np.sum(h*(y[n:2*n]**2+u['a']*qg*qg))+y[2*n]**2)
 def rhs(t,y):
  q,v,g=y[:n],y[n:2*n],y[2*n];u=coeff(x,t,lo0,hi0);f=coeff(np.array([0.,1.]),t,lo0,hi0);qg=d(q);betax=d(u['beta']);ax=d(u['a']);lower=u['betat']-u['beta']*betax-ax
  weak=(-dt(h*u['a']*qg)+tl*(-f['a'][0]*np.dot(tl,qg))+tr*f['a'][1]*np.dot(tr,qg))/h
  pv=extend(v);bn=np.r_[u['beta'],f['beta']];shift=project(qn(bn*pv)+bn*qn(pv))/h-betax*v;fv=weak+shift+lower*qg;vl,vr=np.dot(tl,v),np.dot(tr,v);ql,qr=np.dot(tl,qg),np.dot(tr,qg)
  fv-=f['hi'][0]*tl/h*(vl+f['lo'][0]*ql);fv-=-f['lo'][1]*tr/h*(vr+f['hi'][1]*qr-g)
  work=float(np.sum(h*(-2*betax*v*v+2*lower*v*qg+u['at']*qg*qg))-2*f['c'][0]*vl*vl-2*f['c'][1]*vr*vr+2*(-f['lo'][1])*vr*g-.8*g*g)
  aa=-2*betax;bb=lower/np.sqrt(u['a']);dd=u['at']/u['a'];rate=float(max(0.,max((aa+dd+np.sqrt((aa-dd)**2+4*bb*bb))/2)))
  return np.r_[v,fv,-.4*g,work,rate]
 q=np.zeros(n);v=np.zeros(n);g0=.01 if kind=='injection' else 0.
 if kind=='outgoing':
  z=(x-.35)/.15;inside=abs(z)<1;zz=z[inside];q[inside]=np.exp(1-1/(1-zz*zz));grad=np.zeros(n);grad[inside]=-2*zz*q[inside]/(.15*(1-zz*zz)**2);v=-coeff(x,0.,lo0,hi0)['hi']*grad
 y=np.r_[q,v,g0,0.,0.];initial=energy(0.,y);assert initial>0;maxspeed=max(abs(lo0),hi0)*1.1;base=.02/n/maxspeed/(2 if half else 1);distances=[.35,.7,1.2] if kind=='outgoing' else [.05,.1,.15];t=0.;steps=0;history=[];states=[];started=time.monotonic()
 for distance in distances:
  target=distance/hi0
  while t<target:
   step=min(base,target-t);k1=rhs(t,y);k2=rhs(t+step/2,y+step*k1/2);k3=rhs(t+step/2,y+step*k2/2);k4=rhs(t+step,y+step*k3);y+=step/6*(k1+2*k2+2*k3+k4);t+=step;steps+=1
   if steps%64==0:assert np.isfinite(y).all()
  assert np.isfinite(y).all();en=energy(t,y);bound=initial*math.exp(y[-1]);balance=abs(en-initial-y[-2])/initial;ratio=en/bound;gerror=abs(y[2*n]-g0*math.exp(-.4*t));history.append(dict(time=t,distance=distance,augmented_energy=en,energy_initial_ratio=en/initial,energy_balance_relative_defect=balance,energy_Gronwall_ratio=ratio,boundary_ODE_error=gerror,velocity_l2=float(np.sqrt(np.sum(h*y[n:2*n]**2)))));states.append(np.vstack([y[:n],y[n:2*n],d(y[:n])]))
 data=np.stack(states);np.savez_compressed(payload,x=x,states=data);j=dict(label=label,name=name,kind=kind,n=n,half_step=half,source_sha256=sourcehash,payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),lo0=lo0,hi0=hi0,steps=steps,wall_seconds=time.monotonic()-started,initial_energy=initial,history=history,stability_passed=all(r['energy_balance_relative_defect']<=1e-6 and r['energy_Gronwall_ratio']<=1+1e-6 and r['boundary_ODE_error']<=1e-10 for r in history));meta.write_text(json.dumps(j,indent=2)+'\n');print(label,j['stability_passed'],round(j['wall_seconds'],1),flush=True);return j,np.load(payload)
def interpolate(data,x):
 xf=data['x'];values=data['states'];n=len(xf);start=np.floor(x*n-.5).astype(int)-1;start=np.minimum(np.maximum(start,0),n-4);indices=start[:,None]+np.arange(4);nodes=xf[indices];w=np.ones_like(nodes)
 for i in range(4):
  for k in range(4):
   if k!=i:w[:,i]*=(x-nodes[:,k])/(nodes[:,i]-nodes[:,k])
 return np.einsum('pfnj,nj->pfn',values[:,:,indices],w)
def measure(delta,x,h):
 out={}
 for region,mask in [('all',np.ones(len(x),bool)),('interior',x<.8),('outer_collar',x>.9)]:
  for f,field in enumerate(['q','v','Dq']):
   z=delta[f,mask];hh=h[mask];out[region+'/'+field+'/l2']=float(np.sqrt(np.sum(hh*z*z)/np.sum(hh)));out[region+'/'+field+'/linf']=float(max(abs(z)))
 return out
pulse=[x for x in cases if 'pulse-' in x['source']];puncture=[x for x in cases if 'puncture-' in x['source']];selected=[('pulse',min(pulse,key=lambda x:abs(x['lo']))),('puncture',max(puncture,key=lambda x:abs(x['lo'])))];allcases=[]
for name,case in selected:
 for kind in ['outgoing','injection']:
  runs={};data={}
  for n in [96,128,192,384,512]:runs[n],data[n]=run(case,name,kind,n)
  half,halfdata=run(case,name,kind,128,True);checks=[]
  for point in range(3):
   errors={};reference={};timeerrors=None
   for n in [96,128,192]:
    xx=data[n]['x'];h0,*_=spatial(n);h=h0/n;target=interpolate(data[512],xx)[point];target2=interpolate(data[384],xx)[point];errors[n]=measure(data[n]['states'][point]-target,xx,h);reference[n]=measure(target2-target,xx,h)
    if n==128:timeerrors=measure(data[n]['states'][point]-halfdata['states'][point],xx,h)
   for key in errors[96]:
    vals=[errors[n][key] for n in [96,128,192]];gaps=[abs(vals[0]-vals[1]),abs(vals[1]-vals[2])];order=math.log(vals[1]/vals[2])/math.log(192/128) if vals[1]*vals[2]>0 else None;ref=max(reference[n][key] for n in [96,128,192]);checks.append(dict(point=point,region_field_norm=key,errors=vals,finest_order=order,reference_difference=ref,time_difference=timeerrors[key],spatial_decreases=vals[0]>vals[1]>vals[2],order_pass=order is not None and order>=1.5,reference_small=ref<.1*min(gaps),time_small=timeerrors[key]<.1*min(gaps)))
  settled=kind!='outgoing' or all(runs[n]['history'][-1]['energy_initial_ratio']<=1e-4 for n in [96,128,192]);injected=kind!='injection' or min(runs[n]['history'][-1]['velocity_l2'] for n in [96,128,192])>1e-10;allcases.append(dict(name=name,kind=kind,runs=runs,half_step=half,checks=checks,energy_stability_passed=all(r['stability_passed'] for r in runs.values()) and half['stability_passed'],settling_passed=settled,injection_detected=injected,accuracy_passed=all(x['spatial_decreases'] and x['order_pass'] and x['reference_small'] and x['time_small'] for x in checks)))
  result=dict(cases=allcases,source_sha256=sourcehash,scope='Scalar live-coefficient weakSAT time evolution only. Synthetic boundaryODE/source distinct from native constraints. Accuracy and energy stability reported separately. All primary/reference/time-control grids retained; reference interpolation is read-only, no field clipping/floors/resets.');(a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
