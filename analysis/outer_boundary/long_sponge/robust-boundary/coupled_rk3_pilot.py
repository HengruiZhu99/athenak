"""Same sixth-order full-source exterior elimination, now original explicit RK3.

Exact discrete-time memory: with R=I+dtL+dt²L²/2+dt³L³/6, partition I/E.
Interior update is R_II uI[n]+R_IE R_EE^n uE[0]
+sum(j=0..n-1)R_IE R_EE^(n-1-j) R_EI uI[j].
No fitted kernel, PDE source omissions, or frozen incoming characteristics.
"""
import sys,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import eig
ROOT=Path(__file__).parent
sys.path.insert(0,str(ROOT/'discrete'))
from strip_reference import full_matrix
sys.path.insert(0,str(ROOT/'discrete'))
from bulk_symbol import symbol
p=Path(__file__).parent;tick=time.monotonic();n=32;h=32.;N=int(sys.argv[1]) if len(sys.argv)>1 else 8192;angle=np.pi/8;times=[0,4999.8,10000.2,19999.8,30000,50000.4]
L,_=full_matrix(n=n,h=h,angle_y=angle,degree=1,damping=True,lapse_damping=.1)
w,v=eig(L,check_finite=False);j=np.argmax(w.real);lam=w[j];mode=v[:,j].reshape(20,n);weights=np.ones(20);weights[6:16]=h;mode/=np.linalg.norm(mode*weights[:,None]);start=(N-n)//2
u0=np.zeros((N,20),complex);u0[start:start+n]=mode.T;f0=np.fft.fft(u0,axis=0);angles=2*np.pi*np.fft.fftfreq(N);V=np.stack([symbol([a,angle,0],h=h,kappa=.1,eta=2,ldamp=.1)for a in angles]);wv=np.linalg.eigvals(V)
out={'scope':__doc__,'parameters':{'n_interior':n,'n_total':N,'h':h,'length':N*h,'angle_y':angle,'kappa':.1,'eta':2,'lapse_damping':.1,'G':1},'original_eigenvalue':[float(lam.real),float(lam.imag)],'periodic_max_real':float(wv.real.max()),'runs':[]}
for dt in [.6,.3]:
 I=np.eye(20)[None];X=dt*V;X2=X@X;R=I+X+.5*X2+(X2@X)/6;powers=[R]
 for _ in range(1,round(max(times)/dt).bit_length()):powers.append(powers[-1]@powers[-1])
 sample=np.unique(np.r_[np.linspace(0,N//2,257,dtype=int),[np.argmax(wv.real)//20]])
 records=[]
 for t in times:
  steps=round(t/dt);actual=steps*dt;uf=f0.copy();P=np.broadcast_to(np.eye(20),(len(sample),20,20)).copy();m=steps;ib=0
  while m:
   if m&1:uf=np.einsum('nij,nj->ni',powers[ib],uf);P=powers[ib][sample]@P
   m>>=1;ib+=1
  u=np.fft.ifft(uf,axis=0);ss=np.linalg.svd(P*weights[None,:,None]/weights[None,None,:],compute_uv=False)[:,0];imax=np.argmax(ss);old=abs(1+dt*lam+(dt*lam)**2/2+(dt*lam)**3/6)**steps
  rec={'time':actual,'original_zero_rate_weighted_norm':float(old),'exterior_memory_full_weighted_norm':float(np.linalg.norm(u*weights)),'retained_interior_weighted_norm':float(np.linalg.norm(u[start:start+n]*weights)),'theta_max_full':float(abs(u[:,7]).max()),'sampled_induced_norm':float(ss[imax]),'induced_normal_angle':float(angles[sample[imax]]),'finite':bool(np.isfinite(u).all())};records.append(rec);print(dt,rec,flush=True)
 eigR=1+dt*wv+(dt*wv)**2/2+(dt*wv)**3/6
 out['runs'].append({'dt':dt,'max_periodic_amplification':float(abs(eigR).max()),'records':records});out['seconds']=time.monotonic()-tick;(p/('coupled-rk3-pilot.json' if N==8192 else f'coupled-rk3-N{N}.json')).write_text(json.dumps(out,indent=2)+'\n');np.savez(p/(f'coupled-rk3-final-dt{dt:g}.npz' if N==8192 else f'coupled-rk3-N{N}-final-dt{dt:g}.npz'),u=u,initial=u0,h=h)
