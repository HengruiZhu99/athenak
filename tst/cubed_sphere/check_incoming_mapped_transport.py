#!/usr/bin/env python3
"""Analytic scalar incoming transport audit; no native evolution changes."""
import ast,argparse,json,hashlib,math,time
from pathlib import Path
import numpy as np
np.seterr(over='raise',invalid='raise',divide='raise',under='ignore')
p=argparse.ArgumentParser();p.add_argument('speeds',type=Path);p.add_argument('sbp',type=Path);p.add_argument('output',type=Path);p.add_argument('--uniform-time-control',action='store_true');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
s=json.loads(a.sbp.read_text());weights=np.array(s['norm_boundary_weights']);rows=np.array(s['left_boundary_rows']);trace=np.array(s['left_trace']);sourcehash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
helper=Path(__file__).with_name('evolve_live_coefficient_sbp_sat.py');tree=ast.parse(helper.read_text());exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in ['mm','spatial','Sparse']],type_ignores=[]),str(helper),'exec'))
def bump(t,tau):
 z=(np.asarray(t)/tau-.5)/.3;out=np.zeros_like(z);mask=abs(z)<1;zz=z[mask];out[mask]=.01*np.exp(1-1/(1-zz*zz));return out

def bump_derivative(t,tau):
 z=(np.asarray(t)/tau-.5)/.3;out=np.zeros_like(z);mask=abs(z)<1;zz=z[mask];out[mask]=bump(np.asarray(t)[mask],tau)*(-2*zz)/(.3*tau*(1-zz*zz)**2);return out

def integral(t):return t+.02*(1-np.cos(t))
def exact(x,t,lo,tau):
 A=1+.03*(x-.5);travel=np.log(1.015/A)/(.03*abs(lo));ret=integral(t)-travel;active=ret>0;tb=np.zeros_like(x);tb[active]=ret[active]
 for _ in range(5):tb[active]-=(integral(tb[active])-ret[active])/(1+.02*np.sin(tb[active]))
 residual=float(max(abs(integral(tb[active])-ret[active]))) if active.any() else 0.
 q=bump(tb,tau);dq=bump_derivative(tb,tau)/(abs(lo)*A*(1+.02*np.sin(tb)));return q,dq,residual

def norms(v,h,x):
 out={}
 for region,mask in [('all',np.ones(len(x),bool)),('outer_collar',x>.9)]:
  out[region+'/rms']=float(np.sqrt(np.sum(h[mask]*v[mask]**2)/np.sum(h[mask])));out[region+'/max']=float(max(abs(v[mask])))
 return out

def run(name,case,n,mapped,half=False):
 lo,hi=case['lo'],case['hi'];tau=.15/hi;targetJ=.2*abs(lo)*tau;k=0.
 if mapped:
  left,right=1.,30.
  for _ in range(60):
   mid=(left+right)/2
   if mid/np.expm1(mid)>targetJ:left=mid
   else:right=mid
  k=(left+right)/2
 xi=(np.arange(n)+.5)/n;x=1-np.expm1(k*(1-xi))/np.expm1(k) if mapped else xi;J=k*np.exp(k*(1-xi))/np.expm1(k) if mapped else np.ones(n)
 h0,d0,_,_,_=spatial(n);h=h0/n*J;D=d0*n/J[:,None];tl=np.zeros(n);tl[:len(trace)]=trace;tr=tl[::-1];E=np.vstack([tl,tr]);B=np.diag([-1.,1.]);Q=h[:,None]*D;QN=.5*np.block([[Q-Q.T,mm(E.T,B)],[-mm(B,E),B]]);d=Sparse(D);qn=Sparse(QN)
 defect=float(np.linalg.norm(Q+Q.T-mm(E.T,mm(B,E)))/np.linalg.norm(Q));assert defect<1e-10 and min(h)>0
 def rhs(t,y):
  w=y[:n];at=lo*(1+.03*(x-.5))*(1+.02*np.sin(t));af=lo*np.array([.985,1.015])*(1+.02*np.sin(t));ax=lo*.03*(1+.02*np.sin(t));pv=np.r_[w,np.dot(tl,w),np.dot(tr,w)];an=np.r_[at,af];z=qn(an*pv)+an*qn(pv);project=z[:n]+tl*z[n]+tr*z[n+1];g=float(bump(t,tau));wl,wr=np.dot(tl,w),np.dot(tr,w);dw=-.5*project/h+.5*ax*w+af[1]*tr/h*(wr-g)
  work=float(ax*np.dot(h,w*w)+af[0]*wl*wl+af[1]*(wr-g)**2-af[1]*g*g);bound=max(0.,ax)*y[-1]-af[1]*g*g
  return np.r_[dw,work,bound]
 # Only incoming transport sets this CFL. Fast outgoing cost is reported separately.
 base=.08/(n*max(abs(lo)*(1+.03*(x-.5))*1.02/J))/(2 if half else 1);base=min(base,tau/(400 if half else 200)) if a.uniform_time_control else base;y=np.zeros(n+2);t=0.;records=[];states=[];started=time.monotonic();steps=0
 for fraction in [.4,.8,1.2]:
  target=tau*fraction
  while t<target:
   dt=min(base,target-t);k1=rhs(t,y);k2=rhs(t+dt/2,y+dt*k1/2);k3=rhs(t+dt/2,y+dt*k2/2);k4=rhs(t+dt,y+dt*k3);y+=dt/6*(k1+2*k2+2*k3+k4);t+=dt;steps+=1
  assert np.isfinite(y).all();q,dq,res=exact(x,t,lo,tau);energy=float(np.dot(h,y[:n]**2));errors={**{'w/'+k:v for k,v in norms(y[:n]-q,h,x).items()},**{'Dxw/'+k:v for k,v in norms(d(y[:n])-dq,h,x).items()}};records.append(dict(time=t,energy=energy,bound=y[-1],balance_defect=abs(energy-y[-2])/y[-1],bound_ratio=energy/y[-1],exact_inversion_residual=res,errors=errors));states.append(np.vstack([y[:n],d(y[:n])]))
 label=name+('-mapped-' if mapped else '-uniform-')+str(n)+('-half' if half else '');payload=a.output/(label+'.npz');np.savez_compressed(payload,x=x,J=J,h=h,states=np.array(states));Jface=k/np.expm1(k) if mapped else 1.;r=dict(label=label,n=n,mapped=mapped,half=half,k=k,lo=lo,hi=hi,tau=tau,minJface=Jface,coordinate_volume=float(sum(h)),SBP_relative_defect=defect,history=records,steps=steps,wall_seconds=time.monotonic()-started,energy_pass=all(z['balance_defect']<=1e-6 and z['bound_ratio']<=1+1e-6 for z in records),source_sha256=sourcehash,payload_sha256=hashlib.sha256(payload.read_bytes()).hexdigest(),incoming_layer_cell_count=int(sum(x>1-abs(lo)*tau)),fast_outgoing_to_incoming_CFL_ratio=hi/abs(lo),estimated_coupled_steps=math.ceil(1.2*tau*n*hi*1.02/Jface/.08));(a.output/(label+'.json')).write_text(json.dumps(r,indent=2)+'\n');print(label,r['energy_pass'],r['wall_seconds'],flush=True);return r,np.array(states),x,h
cases=json.loads(a.speeds.read_text())['records'];selected=[('pulse',min([c for c in cases if 'pulse-' in c['source']],key=lambda c:abs(c['lo']))),('puncture',max([c for c in cases if 'puncture-' in c['source']],key=lambda c:abs(c['lo'])))];review=[]
for name,case in selected:
 for mapped in ([False] if a.uniform_time_control else [False,True]):
  runs={};states={}
  for n in [64,128,256]:runs[n],states[n],x,h=run(name,case,n,mapped)
  half,halfstate,x,h=run(name,case,128,mapped,True);checks=[]
  for point in range(3):
   timeerr={**{'w/'+k:v for k,v in norms(states[128][point,0]-halfstate[point,0],h,x).items()},**{'Dxw/'+k:v for k,v in norms(states[128][point,1]-halfstate[point,1],h,x).items()}}
   for key in runs[64]['history'][point]['errors']:
    vals=[runs[n]['history'][point]['errors'][key] for n in [64,128,256]];order=math.log2(vals[1]/vals[2]) if vals[1]*vals[2]>0 else None;checks.append(dict(point=point,norm=key,errors=vals,order=order,decreases=vals[0]>vals[1]>vals[2],order_pass=order is not None and order>=1.5,time_difference=timeerr[key],time_small=timeerr[key]<.1*min(abs(vals[0]-vals[1]),abs(vals[1]-vals[2]))))
  review.append(dict(name=name,mapped=mapped,runs=runs,half=half,checks=checks,energy_pass=all(r['energy_pass'] for r in runs.values()) and half['energy_pass'],accuracy_pass=all(c['decreases'] and c['order_pass'] and c['time_small'] for c in checks)))
  (a.output/'review.json').write_text(json.dumps(dict(cases=review,scope='Isolated incoming scalar transport, analytic solution, not native coupling/admission',source_sha256=sourcehash,helper_sha256=hashlib.sha256(helper.read_bytes()).hexdigest()),indent=2)+'\n')
