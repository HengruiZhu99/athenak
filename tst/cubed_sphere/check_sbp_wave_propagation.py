#!/usr/bin/env python3
"""Source-free frozen principal outgoing waves; native physical targets unchanged."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
np.seterr(all="raise")
p=argparse.ArgumentParser();p.add_argument('--symbols',nargs='+',type=Path,required=True);p.add_argument('--sbp',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--resolutions',nargs=3,type=int,default=[32,64,128]);a=p.parse_args();assert a.resolutions[0]<a.resolutions[1]<a.resolutions[2];sbp=json.loads(a.sbp.read_text());weights=np.array(sbp['norm_boundary_weights']);rows=np.array(sbp['left_boundary_rows']);trace=np.array(sbp['left_trace'])
def mm(x,y):return np.einsum("ij,jk->ik",x,y)

def symbols(path):
    lines = iter(path.read_text().splitlines())
    for line in lines:
        if not line.startswith('case '):
            continue
        _, family, radius, label, omega, lo, hi = line.split()
        inverse = np.array(list(map(float,next(lines).split()[1:]))).reshape(3,3)
        normal = next(lines)
        assert normal.startswith('normal ')
        assert next(lines) == 'constraints'
        constraints = np.array([list(map(float,next(lines).split())) for _ in range(8)])
        assert next(lines) == 'matrix 1'
        matrix = np.array([list(map(float,next(lines).split())) for _ in range(22)])
        assert next(lines) == 'matrix 0.5'
        other = np.array([list(map(float,next(lines).split())) for _ in range(22)])
        assert np.linalg.norm(matrix-other)/max(1,np.linalg.norm(matrix)) < 1e-11
        tr = np.array([inverse[0,0],2*inverse[0,1],2*inverse[0,2],inverse[1,1],2*inverse[1,2],inverse[2,2]])
        cq = np.zeros((1,11));cq[0,1:7] = tr
        cp = np.zeros((1,11));cp[0,:6] = tr
        q = np.linalg.svd(cq,full_matrices=True)[2][1:].T
        v = np.linalg.svd(cp,full_matrices=True)[2][1:].T
        basis = np.zeros((22,20));basis[:11,:10] = q;basis[11:,10:] = v
        yield dict(source=str(path),family=int(family),radius=float(radius),sample=label,omega=float(omega),
                   lo=float(lo),hi=float(hi),matrix=mm(mm(basis.T,matrix),basis),constraints=mm(constraints,basis))

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
def exact(x,t,speed):
 z=(x-speed*t-.30)/.08;q=np.exp(-z*z);derivative=-2*z/.08*q;return q,-speed*derivative,derivative
def norms(x,h):return dict(l2=float(np.sqrt(np.sum(h*x*x))),linf=float(max(abs(x))))
def evolve(n,beta,c,half=False):
 h0,d0,bl,br,res=spatial(n);h=h0/n;d=d0*n;d2=mm(d,d);x=(np.arange(n)+.5)/n;speed=c-beta;speed2=c*c-beta*beta
 tl=np.zeros(n);tl[:len(trace)]=trace;tr=tl[::-1];ql=np.einsum('i,ij->j',tl,d);qr=np.einsum('i,ij->j',tr,d)
 L=np.block([[np.zeros((n,n)),np.eye(n)],[speed2*d2,2*beta*d]])
 left=np.r_[np.zeros(n),(c-beta)*tl/h];right=np.r_[np.zeros(n),(c+beta)*tr/h]
 L-=np.outer(left,np.r_[-(c+beta)*ql,tl])+np.outer(right,np.r_[(c-beta)*qr,tr])
 q,v,dq=exact(x,0,speed);y=np.r_[q,v];initial_energy=float(np.sum(h*(v*v+speed2*np.einsum('ij,j->i',d,q)**2)));time=0.;steps=0;history=[];base_dt=.02/n/(c+abs(beta))/(2 if half else 1)
 def rhs(t,u):
  _,vel,grad=exact(np.array([0.]),t,speed);data=float(vel[0]-(c+beta)*grad[0]);return np.einsum('ij,j->i',L,u)+left*data
 for distance in [.35,.7,1.2]:
  target=distance/speed
  while time<target:
   dt=min(base_dt,target-time);k1=rhs(time,y);k2=rhs(time+dt/2,y+dt*k1/2);k3=rhs(time+dt/2,y+dt*k2/2);k4=rhs(time+dt,y+dt*k3);y+=dt/6*(k1+2*k2+2*k3+k4);time+=dt;steps+=1
  q,v,grad=exact(x,time,speed);numerical_grad=np.einsum('ij,j->i',d,y[:n]);errors={k:norms(z,h) for k,z in dict(q=y[:n]-q,v=y[n:]-v,Dq=numerical_grad-grad).items()};energy=float(np.sum(h*(y[n:]**2+speed2*numerical_grad**2)));assert np.isfinite(y).all();history.append(dict(distance=distance,time=time,errors=errors,energy=energy,energy_ratio=energy/initial_energy))
 return dict(n=n,dt=base_dt,steps=steps,history=history),y,d,h
records=[]
for path in a.symbols:
 for s in symbols(path):
  s.pop('matrix');s.pop('constraints');beta=-(s['lo']+s['hi'])/2;c=(s['hi']-s['lo'])/2;runs=[];states=[]
  for n in a.resolutions:
   run,state,d,h=evolve(n,beta,c);runs.append(run);states.append(state)
  half,halfstate,d,h=evolve(a.resolutions[1],beta,c,True);checks=[]
  for point in range(3):
   for field in ['q','v','Dq']:
    for norm in ['l2','linf']:
     values=[r['history'][point]['errors'][field][norm] for r in runs];order=math.log(values[1]/values[2])/math.log(a.resolutions[2]/a.resolutions[1]);temporal=abs(values[1]-half['history'][point]['errors'][field][norm]);gaps=[abs(values[0]-values[1]),abs(values[1]-values[2])];checks.append(dict(distance=runs[0]['history'][point]['distance'],field=field,norm=norm,values=values,finest_order=order,decreases=values[0]>values[1]>values[2],time_difference=temporal,time_small=temporal<.1*min(gaps),order_pass=norm!='l2' or order>=1.5))
  mid=a.resolutions[1];delta=states[1]-halfstate;pointwise=dict(q=norms(delta[:mid],h),v=norms(delta[mid:],h),Dq=norms(np.einsum('ij,j->i',d,delta[:mid]),h));settles=all(r['history'][-1]['energy_ratio']<=1e-4 for r in runs);passed=all(x['decreases'] and x['time_small'] and x['order_pass'] for x in checks) and settles
  records.append(dict(**s,beta=beta,c=c,runs=runs,half_step=half,checks=checks,endpoint_pointwise_time_difference=pointwise,settling_pass=settles,passed=passed));print(s['sample'],passed,flush=True)
r=dict(passed=all(x['passed'] for x in records),records=records,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [a.sbp,*a.symbols]},scope='Source-free one-dimensional frozen-principal propagation and settling only, analytic incoming face data. No native equations/evolution/reference change or physical goal acceptance.')
a.output.write_text(json.dumps(r,indent=2,allow_nan=False)+'\n');print('all_passed',r['passed'])
