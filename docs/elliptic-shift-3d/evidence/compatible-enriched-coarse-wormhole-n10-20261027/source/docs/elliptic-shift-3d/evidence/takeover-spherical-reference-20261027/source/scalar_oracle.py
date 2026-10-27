#!/usr/bin/env python3
"""Independent fixed-Minkowski scalar propagation and origin-reflection control.

Uses independently selectable first/second-order upwind characteristic transport, independent of the Z4c kernel.
It is deliberately a preliminary analytic control, not coupled Einstein-scalar.
"""
import argparse,json,time,hashlib,sys,platform
from pathlib import Path
import numpy as np


def gaussian(x,center=2.,width=.2):
    z=(x-center)/width
    F=np.exp(-z*z)
    return F,-2*z/width*F


def exact(r,t):
    u=t+(1-r)/(1+r)
    v=np.full_like(r,np.inf);v[:-1]=t+(1+r[:-1])/(1-r[:-1])
    out,_=gaussian(u);inc=np.zeros_like(r);inc[:-1],_=gaussian(v[:-1])
    field=np.empty_like(r);field[1:]=(out[1:]-inc[1:])/r[1:]
    field[0]=-4*gaussian(np.array([t+1]))[1][0]
    return out,inc,field


def run(n,cfl,end=3.,order=2):
    r=np.linspace(0,1,n+1);h=1/n;dt=cfl*h
    cp=(1+r)**2/2;cm=-(1-r)**2/2
    wo,wi,_=exact(r,0.);q=np.stack([wo,wi],axis=1)
    def rhs(x):
        # Reflection at a regular spherical center is an exact wave relation.
        # At scri c_-=0 so its initial value is transported tangentially.
        x=x.copy();x[0,0]=x[0,1]
        d=np.zeros_like(x)
        d[1:,0]=-cp[1:]*(x[1:,0]-x[:-1,0])/h
        d[:-1,1]=-cm[:-1]*(x[1:,1]-x[:-1,1])/h
        if order==2:
            # Stable second-order biased upwind transport. Reflection supplies
            # the one origin ghost. The scri incoming field is tangent, with
            # a smooth zero continuation of its initially compact profile.
            d[2:,0]=-cp[2:]*(3*x[2:,0]-4*x[1:-1,0]+x[:-2,0])/(2*h)
            d[1,0]=-cp[1]*(3*x[1,0]-4*x[0,0]+x[1,1])/(2*h)
            d[:-2,1]=-cm[:-2]*(-3*x[:-2,1]+4*x[1:-1,1]-x[2:,1])/(2*h)
            d[-2,1]=-cm[-2]*(-3*x[-2,1]+4*x[-1,1])/(2*h)
        d[0,0]=d[0,1]
        return d
    t=0.;history=[];next_report=0.;start=time.perf_counter()
    maxerr=0.;scrierr=0.;sample_times=[];signals=[]
    while t<end-1e-12:
        step=min(dt,end-t)
        k1=rhs(q);k2=rhs(q+step*k1/2);k3=rhs(q+step*k2/2);k4=rhs(q+step*k3)
        q+=step*(k1+2*k2+2*k3+k4)/6;t+=step
        exact_out,exact_in,exact_f=exact(r,t)
        f=np.empty_like(r);f[1:]=(q[1:,0]-q[1:,1])/r[1:]
        f[0]=f[1] # Even scalar: f(h)=f(0)+O(h^2).
        err=np.sqrt(np.sum((r*(f-exact_f))**2)/np.sum(r*r))
        maxerr=max(maxerr,float(err));scrierr=max(scrierr,float(abs(f[-1]-exact_f[-1])))
        if t>=next_report:
            history.append(dict(t=t,weighted_field_L2=float(err),scri_error=float(abs(f[-1]-exact_f[-1]))))
            sample_times.append(t);signals.append([f[-1],exact_f[-1]])
            next_report+=.01
    return {'n':n,'h':h,'cfl':cfl,'duration':end,'max_weighted_field_L2':maxerr,'max_scri_error':scrierr,'runtime_seconds':time.perf_counter()-start,'history':history},np.array(sample_times),np.array(signals)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--order',type=int,choices=[1,2],default=2);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    records=[]
    for n in [200,400,800]:
        rec,t,s=run(n,.2,order=a.order);records.append(rec);np.savez(a.output/f'wave-n{n}.npz',time=t,signals=s)
    controls=[]
    for cfl in [.1,.05]:
        rec,_,_=run(400,cfl,order=a.order);controls.append(rec)
    orders=[float(np.log(records[i]['max_weighted_field_L2']/records[i+1]['max_weighted_field_L2'])/np.log(2)) for i in range(2)]
    scri_orders=[float(np.log(records[i]['max_scri_error']/records[i+1]['max_scri_error'])/np.log(2)) for i in range(2)]
    result={'scope':'independent fixed-background spherical scalar; not a coupled evolution',
       'analytic':'phi=[F(T-R)-F(T+R)]/R, f=phi/Omega, u=t+(1-r)/(1+r),v=t+(1+r)/(1-r); f(scri)=F(t)',
       'method':f'order-{a.order} upwind characteristic transport/RK4; no Z4c kernel dependency',
       'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'command':sys.argv,'platform':platform.platform(),
       'criteria_declared_before_runs':{'finite':True,'expected_spatial_order':a.order,'minimum_preasymptotic_order':a.order-.5,'finest_max_scri_error':.01},
       'spatial_runs':records,'temporal_controls':controls,'weighted_L2_orders':orders,'scri_orders':scri_orders}
    result['passed_preliminary_control']=min(orders+scri_orders)>a.order-.5 and records[-1]['max_scri_error']<.01
    (a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['scope','weighted_L2_orders','scri_orders','passed_preliminary_control']}))
    return 0 if result['passed_preliminary_control'] else 1
if __name__=='__main__':sys.exit(main())
