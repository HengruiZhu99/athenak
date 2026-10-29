"""Fresh bounded donor 1D control, using the unchanged hyperbolic implementation."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
from initial_data import PunctureEvolution, KW
from evolution import Evolution

def save(e,q,prefix):
    e.rhs(q,**(dict(diagnostics=True) if isinstance(e,PunctureEvolution) else {}));x=e.last
    physical=x.get('physical',q)
    arr=np.c_[e.r,x['chi'],x['g'],x['alpha'],x['beta'],physical[:,2],physical[:,3]-3,
              physical[:,4:6],q[:,8:10],x['diag'][:,[0,1,3,11]]]
    np.savetxt(str(prefix)+'.csv',arr,delimiter=',',header='r,chi,g_rr,alpha,beta_r,A_rr,P,Theta,Lambda_r,B_r,D_r,H,M_r,Z_r,expansion',comments='')

def main():
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=513)
    p.add_argument('--target',type=float,required=True);p.add_argument('--kind',choices=['puncture','pulse','minkowski'],default='puncture');p.add_argument('--library',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    begin=time.monotonic()
    if a.kind=='puncture':e=PunctureEvolution(a.n,a.library.resolve(),**KW);q=e.initial()
    else:
        kw={k:v for k,v in KW.items() if k not in ['grid_stretch','shift_inner_stencil','lapse_floor']}
        e=Evolution(a.n,a.library.resolve(),**kw);q=np.zeros((a.n,10))
        if a.kind=='pulse':
            z=(e.r-.4)/.12;mask=abs(z)<1;q[mask,6]=.01*np.exp(1-1/(1-z[mask]**2))
    save(e,q,a.output/'initial')
    t=0.;steps=0;dt=.025/(a.n-1);status='completed';error='';history=[e.report(q,0)]
    try:
        while t<a.target-1e-14:
            if time.monotonic()-begin>400:status='bounded_stop';break
            step=min(dt,a.target-t);q=e.step(q,step);t+=step;steps+=1
        save(e,q,a.output/'final');history.append(e.report(q,t))
    except Exception as ex:status='failed';error=str(ex)
    np.savez_compressed(a.output/'state.npz',q=q,r=e.r,t=t,compensation=getattr(e,'compensation',np.zeros_like(q)))
    (a.output/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    report=dict(status=status,error=error,kind=a.kind,n=a.n,t=t,t_over_M=3*t,steps=steps,dt=dt,configuration=KW,
                library_sha256=hashlib.sha256(a.library.read_bytes()).hexdigest(),seconds=time.monotonic()-begin,pid=os.getpid())
    (a.output/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
