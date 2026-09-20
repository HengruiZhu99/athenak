"""Closed full-source PDE pilot, exact algebraic exterior elimination reference.

A periodic large exterior is evolved with the identical bulk FD stencil, which
is exactly equivalent to eliminating it into causal BDF2 boundary memory. This
is not a finite-face AthenaK implementation. Original zero_rate comparison uses
its actual finite-strip eigenmode. No scalar boundary surrogate is substituted.
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

n=32;h=32.;angle=np.pi/8;N=int(sys.argv[1]) if len(sys.argv)>1 else 4096;times=[0,5000,10000,20000,30000,50000]
p=Path(__file__).parent;tick=time.monotonic()
L,_=full_matrix(n=n,h=h,angle_y=angle,degree=1,damping=True,alpha=1,chi=1,beta=(0,0,0),lapse_damping=.1)
w,v=eig(L,check_finite=False);j=np.argmax(w.real);lam=w[j];mode=v[:,j].reshape(20,n)
weights=np.ones(20);weights[6:16]=h
mode/=np.linalg.norm(weights[:,None]*mode)
start=(N-n)//2;u0=np.zeros((N,20),complex);u0[start:start+n]=mode.T;f0=np.fft.fft(u0,axis=0)
angles=2*np.pi*np.fft.fftfreq(N);V=np.stack([symbol([a,angle,0],h=h,kappa=.1,eta=2,ldamp=.1)for a in angles]);wv=np.linalg.eigvals(V)
# Same centered interior rows reproduce periodic V action away from faces.
rng=np.random.default_rng(671);test=rng.normal(size=(N,20));F=np.fft.fft(test,axis=0);bulk=np.fft.ifft(np.einsum('nij,nj->ni',V,F),axis=0)
small=test[start:start+n].T.reshape(-1);strip=(L@small).reshape(20,n).T
match=np.max(abs(strip[4:-4]-bulk[start+4:start+n-4]))
out={'scope':__doc__,'parameters':{'n_interior':n,'n_total':N,'h':h,'length':N*h,'tangential_angle':angle,'kappa':.1,'eta':2,'lapse_damping':.1,'G':1},'original_eigenvalue':[float(lam.real),float(lam.imag)],'original_eigen_defect':float(np.linalg.norm(L@mode.reshape(-1)-lam*mode.reshape(-1))/np.linalg.norm(lam*mode.reshape(-1))),'periodic_max_real':float(wv.real.max()),'periodic_positive_count_above1e-10':int((wv.real>1e-10).sum()),'interior_stencil_match_max':float(match),'runs':[]}
print('original mode',lam,'periodic max',wv.real.max(),'match',match,flush=True)
I=np.eye(20)[None];eye40=np.eye(40)[None]
for dt in ([2.,1.] if N==8192 else [4.,2.]):
 Ainv=np.linalg.inv(1.5*I-dt*V);BE=np.linalg.inv(I-dt*V)
 R=np.zeros((N,40,40),complex);R[:,:20,:20]=2*Ainv;R[:,:20,20:]=-.5*Ainv;R[:,20:,:20]=I
 maxstep=int(max(times)/dt);powers=[R]
 for _ in range(1,maxstep.bit_length()):powers.append(powers[-1]@powers[-1])
 init=np.concatenate([np.einsum('nij,nj->ni',BE,f0),f0],axis=1)
 # Sample all Fourier modes for spectral stability; induced transient uses
 # 129 frequencies including zero/Nyquist, not a global-in-time bound.
 sample=np.unique(np.r_[np.linspace(0,N//2,129,dtype=int),[np.argmax(wv.real)//20]])
 initmap=np.concatenate([BE[sample],np.broadcast_to(np.eye(20),(len(sample),20,20))],axis=1)
 records=[]
 for t in times:
  steps=round(t/dt);actual=steps*dt
  if steps==0:uf=f0.copy();P=np.broadcast_to(np.eye(20),(len(sample),20,20));oldfac=1.+0j
  else:
   uf=init.copy();Pm=initmap.copy();m=steps-1;ib=0
   while m:
    if m&1:
     uf=np.einsum('nij,nj->ni',powers[ib],uf);Pm=powers[ib][sample]@Pm
    m>>=1;ib+=1
   uf=uf[:,:20];P=Pm[:,:20]
   # Exact BDF2 scalar recurrence for the old eigenmode, BE startup.
   a=1.5-dt*lam;rs=np.roots([a,-2,.5]);first=1/(1-dt*lam);cs=np.linalg.solve(np.array([[1,1],rs]),[1,first]);oldfac=np.dot(cs,rs**steps)
  u=np.fft.ifft(uf,axis=0);norm=float(np.linalg.norm(u*weights));inter=float(np.linalg.norm(u[start:start+n]*weights))
  PW=P*weights[None,:,None]/weights[None,None,:];ss=np.linalg.svd(PW,compute_uv=False)[:,0];imax=np.argmax(ss)
  rec={'time':actual,'original_zero_rate_weighted_norm':float(abs(oldfac)),'exterior_memory_full_weighted_norm':norm,'retained_interior_weighted_norm':inter,'theta_max_full':float(abs(u[:,7]).max()),'theta_max_interior':float(abs(u[start:start+n,7]).max()),'sampled_induced_norm':float(ss[imax]),'induced_normal_angle':float(angles[sample[imax]]),'finite':bool(np.isfinite(u).all())}
  records.append(rec);print(dt,rec,flush=True)
 run={'dt':dt,'records':records};out['runs'].append(run);out['seconds']=time.monotonic()-tick;(p/f'coupled-pde-long-N{N}.json').write_text(json.dumps(out,indent=2)+'\n')
 np.savez(p/f'coupled-pde-long-N{N}-dt{dt:g}.npz',u=u,initial=u0,h=h,mode=mode)
print('done',time.monotonic()-tick,flush=True)
