"""Direct algebraic check of causal BDF2 elimination of the full exterior."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.linalg import lu_factor,lu_solve
sys.path.insert(0,str(Path(__file__).parent/'discrete'))
from bulk_symbol import symbol
p=Path(__file__).parent;N=32;h=32.;angle=np.pi/8;dt=4.
Vs=np.stack([symbol([a,angle,0],h=h,kappa=.1,eta=2,ldamp=.1)for a in 2*np.pi*np.fft.fftfreq(N)])
K=np.fft.ifft(Vs,axis=0);L=np.block([[K[(i-j)%N]for j in range(N)]for i in range(N)])
inside=np.arange(12*20,20*20);outside=np.setdiff1d(np.arange(N*20),inside)
def prepare(alpha):
 A=alpha*np.eye(20*N)-dt*L;II=A[np.ix_(inside,inside)];IE=A[np.ix_(inside,outside)];EI=A[np.ix_(outside,inside)];EE=A[np.ix_(outside,outside)];e=lu_factor(EE);X=lu_solve(e,EI);s=lu_factor(II-IE@X)
 return lu_factor(A),IE,X,e,s
B=prepare(1.);A=prepare(1.5)
def eliminated(rhs,cache):
 _,IE,X,e,s=cache;re=lu_solve(e,rhs[outside]);ui=lu_solve(s,rhs[inside]-IE@re);ue=re-X@ui;out=np.empty_like(rhs);out[inside]=ui;out[outside]=ue;return out
rng=np.random.default_rng(813);u0=np.zeros(20*N,complex);u0[inside]=rng.normal(size=len(inside))*1e-8
ua=u0.copy();ub=u0.copy();prevA=u0.copy();prevB=u0.copy();checks=[]
for n in range(1,17):
 cache=B if n==1 else A;ra=ua if n==1 else 2*ua-.5*prevA;rb=ub if n==1 else 2*ub-.5*prevB
 va=lu_solve(cache[0],ra);vb=eliminated(rb,cache);err=np.linalg.norm(va-vb)/np.linalg.norm(va)
 checks.append({'step':n,'relative_error':float(err)});prevA,ua=ua,va;prevB,ub=ub,vb
zero=eliminated(np.zeros(20*N,complex),A);out={'scope':'Exact algebraic identity: BDF2 full periodic exterior solve versus Schur-complement boundary memory with all20 fields and source terms. This does not implement the memory closure in AthenaK.','N':N,'interior_cells':len(inside)//20,'dt':dt,'steps':checks,'zero_max':float(abs(zero).max()),'max_relative_error':max(r['relative_error']for r in checks),'operator_finite':bool(np.isfinite(L).all())};(p/'exact-elimination-check.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
