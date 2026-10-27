"""Characteristic postprocessing of immutable snapshots; no PDE evolution."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
source=ROOT/'output/elliptic-shift-1d/wormhole-physical-log-compiled-v1'
records=[];paths=[]
def load(n):
 ts=[];vs=[];ds=[]
 for p in sorted((source/f'n{n}').glob('profile-*.npz')):
  with np.load(p,allow_pickle=False) as z:
   t=float(z['t'])
   if t<1-1e-10 or t>2.5+1e-10:continue
   r=z['r'];d=z['diagnostic'];a=d[:,7];chi=d[:,8];O=(1-r*r)/2
   cm=-z['beta']-np.sqrt((a*a+4*O*O*a)*chi)
   ts.append(t);vs.append(cm)
   paths.append(dict(path=str(p.relative_to(ROOT)),sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
 return r,np.array(ts),np.array(vs)
def trace(r,ts,vs,dt):
 seed=np.linspace(.12,.4,281);x=seed.copy();logJ=np.zeros_like(x);time=1.;hist=[x.copy()];jh=[logJ.copy()];th=[time]
 def rate(t,y):
  i=min(max(np.searchsorted(ts,t,side='right')-1,0),len(ts)-2);w=(t-ts[i])/(ts[i+1]-ts[i]);v=(1-w)*vs[i]+w*vs[i+1]
  j=np.clip(np.searchsorted(r,y)-1,0,len(r)-2);grad=(v[j+1]-v[j])/(r[j+1]-r[j]);value=v[j]+grad*(y-r[j]);return value,grad
 steps=round(1.5/dt);dt=1.5/steps
 for step in range(steps):
  t=1+step*dt;k1,g1=rate(t,x);k2,g2=rate(t+dt/2,x+dt*k1/2);k3,g3=rate(t+dt/2,x+dt*k2/2);k4,g4=rate(t+dt,x+dt*k3)
  x+=dt*(k1+2*k2+2*k3+k4)/6;logJ+=dt*(g1+2*g2+2*g3+g4)/6
  if (step+1)%max(1,round(.01/dt))==0:hist.append(x.copy());jh.append(logJ.copy());th.append(1+(step+1)*dt)
 hist=np.array(hist);jh=np.array(jh);th=np.array(th);j=int(np.argmin(logJ));sep=np.diff(hist,axis=1)
 return dict(min_J=float(np.exp(logJ[j])),seed_at_min_J=float(seed[j]),r_at_min_J=float(x[j]),minimum_pair_separation=float(sep.min()),minimum_pair_ratio=float(sep.min()/(seed[1]-seed[0])),ordered=bool((sep>0).all()),dt=dt,snapshot_dt=float(ts[1]-ts[0])),(seed,th,hist,jh)
plt.rcParams.update({'font.family':'serif','mathtext.fontset':'stix','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained');colors=['#0072B2','#D55E00','#009E73','#CC79A7']
for n,c in zip([513,1025,2049,4097],colors):
 r,ts,vs=load(n)
 for dt in [.001,.0005]:
  rec,arr=trace(r,ts,vs,dt);rec.update(n=n,scope='full saved cadence');records.append(rec)
 rec,arr2=trace(r,ts[::2],vs[::2],.0005) if len(ts)%2 else trace(r,ts[np.r_[np.arange(0,len(ts)-1,2),len(ts)-1]],vs[np.r_[np.arange(0,len(ts)-1,2),len(ts)-1]],.0005)
 rec.update(n=n,scope='decimated cadence');records.append(rec)
 seed,th,hist,jh=arr;ax[1].semilogy(th,np.exp(jh.min(axis=1)),color=c,label=f'N={n}')
 if n==4097:
  _,th2,_,jh2=arr2;ax[1].semilogy(th2,np.exp(jh2.min(axis=1)),'--',color='black',label='N=4097, cadence 0.2')
  for j in range(0,len(seed),5):ax[0].plot(th,hist[:,j],color=c,lw=.7,alpha=.75)
ax[0].set(xlabel='Time',ylabel='r',title='Ingoing gauge rays, N=4097',ylim=(.1,.32));ax[1].set(xlabel='Time',ylabel=r'$\min\,\partial r(t)/\partial r(1)$',title='Ray-map compression');ax[1].legend()
for a in ax:a.grid(alpha=.2)
fig.savefig(HERE/'characteristics.pdf');fig.savefig(HERE/'characteristics.png',dpi=180)
(HERE/'rays.json').write_text(json.dumps(dict(records=records,sources=paths,scope='Piecewise-linear space/time interpolant of saved c-minus, RK4 ray ODE and log-J variational equation, t=1 to2.5, seeds .12:.4. Interpolated finite-grid Lipschitz speeds cannot yield exact crossing. A small J diagnoses focusing, not proof of continuum crossing. Cadence decimation is sensitivity only, not a finer-snapshot convergence test.',analysis_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
print(json.dumps(records,indent=2))
