"""Dense parity-sector spectra of the copied production zero_rate strip."""
import argparse, importlib.util, json, time
from pathlib import Path
import numpy as np
from scipy.linalg import eigvals, eig
from strip_model import full_matrix

ROOT=Path(__file__).resolve().parent
ODD=[4,5,11,12,15,19]
EVEN=[i for i in range(20) if i not in ODD]
BUDGET=np.log(2)/50000

def sectors(n):
    return [np.concatenate([np.arange(f*n,(f+1)*n) for f in fs]) for fs in [EVEN,ODD]]

def run(params, output, vectors=False):
    tick=time.monotonic()
    L,_=full_matrix(**params)
    n=params['n']; ids=sectors(n)
    cross=max(np.max(abs(L[np.ix_(ids[0],ids[1])])),np.max(abs(L[np.ix_(ids[1],ids[0])])) )
    # Analytic z-reflection parity is exact at kz=beta_z=0; the numeric
    # characteristic eigenvectors can leave sub-ulp cross-sector entries.
    assert cross<1e-15,cross
    records=[]
    for label,ix in zip(['even_scalar','odd_vector'],ids):
        A=L[np.ix_(ix,ix)]
        if vectors: w,v=eig(A,check_finite=False)
        else:w=eigvals(A,check_finite=False)
        order=np.argsort(w.real)[::-1];i=order[0]
        dt=.0375;z=dt*w;rr=1+z+z*z/2+z*z*z/6
        rec=dict(sector=label,real=float(w[i].real),imag=float(w[i].imag),positive=int(sum(w.real>1e-8)),above_budget=int(sum(w.real>BUDGET)),rk3_growth=float(np.log(max(abs(rr)))/dt),leading=[[float(w[j].real),float(w[j].imag)] for j in order[:8]])
        if vectors:
            vi=v[:,i]; rec['eigenpair_residual']=float(np.linalg.norm(A@vi-w[i]*vi)/(np.linalg.norm(A,ord=np.inf)*np.linalg.norm(vi)))
            full=np.zeros((20,n),complex);full.reshape(-1)[ix]=vi
            theta=abs(full[7])**2; full[6:16]*=params['h'];weight=np.sum(abs(full)**2,axis=0)
            distance=np.minimum(np.arange(n)+.5,n-np.arange(n)-.5)*params['h'];layer=distance<params['sponge_cells']*params['h']
            rec['theta_peak_distance']=float(distance[np.argmax(theta)]) if theta.max()>0 else None
            rec['theta_layer_fraction']=float(theta[layer].sum()/theta.sum()) if theta.sum()>0 else None
            rec['fullstate_peak_distance']=float(distance[np.argmax(weight)])
            rec['fullstate_layer_fraction']=float(weight[layer].sum()/weight.sum())
        records.append(rec)
    gamma=max(r['real'] for r in records)
    result=dict(parameters=params,domain_length=params['n']*params['h'],width=params['sponge_cells']*params['h'],undamped_length=(params['n']-2*params['sponge_cells'])*params['h'],budget=BUDGET,real=gamma,exponential_factor_50000=float(np.exp(min(700,50000*max(0,gamma)))),sector_cross_block=cross,exact_zero=bool(np.all(L@np.zeros(L.shape[0])==0)),sectors=records,wall_seconds=time.monotonic()-tick)
    Path(output).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(path=str(output),gamma=gamma,rate=params['sponge_rate'],width=result['width'],kappa=params['kappa'],kh_pi=params['angle_y']/np.pi,seconds=result['wall_seconds'])),flush=True)
    return result

def defaults(n=64,h=128.,angle=.03125,width=2048.,rate=.005,kappa=.1,order='before'):
    return dict(n=n,h=h,angle_y=angle*np.pi,angle_z=0.,mode='zero_rate',degree=1,damping=True,shift=1.,lapse_damping=.1,sponge_rate=rate,sponge_cells=width/h,kappa=kappa,sponge_order=order)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=64);p.add_argument('--h',type=float,default=128);p.add_argument('--angle',type=float,default=1/32);p.add_argument('--width',type=float,default=2048);p.add_argument('--rate',type=float,default=.005);p.add_argument('--kappa',type=float,default=.1);p.add_argument('--order',default='before');p.add_argument('--out',required=True);p.add_argument('--vectors',action='store_true');a=p.parse_args()
    run(defaults(a.n,a.h,a.angle,a.width,a.rate,a.kappa,a.order),a.out,a.vectors)
