from pathlib import Path
import json,numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.linalg import expm
from principal_symbol import operator,symbols,reduced,TF,pairs
from point_operator import point_rhs
out=Path(__file__).resolve().parent
B=np.zeros((25,20));B[0,0]=1;B[7,6]=1;B[17,15]=1;B[18,16]=1;B[14:17,12:15]=np.eye(3);B[19:22,17:20]=np.eye(3)
for q,(i,j)in enumerate(pairs):B[1+q,1:6]=TF[:,i,j];B[8+q,7:12]=TF[:,i,j]
C=np.linalg.pinv(B)

def independent_minkowski(xi,shift=2,h=.25):
 D,D2,S=symbols(xi,h);n=20;vb=np.zeros(25);vb[[0,1,4,6,18]]=1;dirs=B.T;eps=1e-20
 result=np.zeros((25,n),complex)
 for complex_part in [False,True]:
  v=np.broadcast_to(vb,(n,25)).astype(complex).copy();d=np.zeros((n,3,25),complex);dd=np.zeros((n,3,3,25),complex);adv=np.zeros((n,25),complex)
  if not complex_part:v+=1j*eps*dirs;dd+=1j*eps*S.real[None,:,:,None]*dirs[:,None,None,:]
  else:d+=1j*eps*D.imag[None,:,None]*dirs[:,None,:]
  f=point_rhs(v,d,dd,adv,kappa=0);f[...,18]=-2*(v[...,7]-vb[7]);f[...,19:22]=shift*(v[...,14:17]-vb[14:17]);result+=(1j if complex_part else 1)*f.imag.T/eps
 return C@result

checks=[]
for xi in [np.array([.23,.91,2.5]),np.array([0,0,np.pi]),np.array([np.pi]*3),np.array([.02,.03,.04])]:
 A=operator(xi,shift=2)[0];actual=independent_minkowski(xi);err=float(np.max(abs(A-actual)));assert err<1e-11;checks.append({'xi':xi.tolist(),'max_abs_matrix_error':err})
# Cao/Hilditch Eq63 scalar roots, plus vector/tensor/lapse branches, in Minkowski.
for shift in [1,2]:
 for xi in [np.array([.23,.91,2.5]),np.array([np.pi]*3),np.array([.02,.03,.04])]:
  D,D2,_=symbols(xi);O2=-D2.sum();s2=-np.sum(D*D).real;mid=(4+3*shift)*O2+(shift-1)*s2;disc=max(0,mid*mid-48*shift*O2**2);v2=[(mid-np.sqrt(disc))/6,(mid+np.sqrt(disc))/6];freqs=[np.sqrt(O2)]*5+[np.sqrt(shift*O2)]*2+[np.sqrt(2*O2)]+[np.sqrt(q)for q in v2]
  expected=np.array([x for w in freqs for x in [-1j*w,1j*w]]);e=np.linalg.eigvals(operator(xi,shift=shift,scheme='compatible')[0]);a,b=linear_sum_assignment(abs(e[:,None]-expected[None,:]));err=float(np.max(abs(e[a]-expected[b])));assert err<1e-9;checks.append({'paper_eq63_shift':shift,'xi':xi.tolist(),'max_abs_eigenvalue_error':err})
# High-frequency approach tests with a pseudo-discrete first-order norm.
condition=[]
for label,alpha,chi,beta in [('flat',1,1,[0,0,0]),('core',.21650635094610965/1.2165063509461096,(.21650635094610965/1.2165063509461096)**2,[.125/1.2165063509461096**2]*3),('far_r16',16/17,(16/17)**2,[16/np.sqrt(3)/17**2]*3)]:
 for scheme in ['standard','compatible']:
  for direction in ['axis','diagonal','asymmetric']:
   for eps in [1e-1,1e-2,1e-3,1e-4,0]:
    if direction=='axis':xi=np.array([np.pi-eps,0,0])
    elif direction=='diagonal':xi=np.repeat(np.pi-eps,3)
    else:xi=np.array([np.pi-eps,np.pi-.3,.8])
    A=operator(xi,alpha,chi,beta,shift=2,scheme=scheme);M=reduced(A,xi[None,:])[0]*.25;w,V=np.linalg.eig(M);cond=float(np.linalg.cond(V));item={'point':label,'scheme':scheme,'direction':direction,'epsilon_to_Nyquist':eps,'max_real_hlambda':float(w.real.max()),'eigenvector_condition':cond}
    # Time/h chosen to resolve near-degenerate branch splitting; diagnostics only.
    tau=min(1e7,max(10,1/max(eps*eps,1e-7)));item['tau_time_over_h']=tau;item['expm_norm_at_tau']=float(np.linalg.norm(expm(M*tau),2));condition.append(item)
report={'independent_Minkowski_and_paper_checks':checks,'high_frequency_conditioning':condition,'norm':'q=(chi,hTF,alpha,beta) multiplied by sqrt(-L); curvature/Gamma/Theta unscaled. Rates multiplied by h for condition tests.','warning':'A large condition number with repeated eigenvalues is not by itself proof of defect. expm checks are finite-dimensional diagnostics; flat G2 exact branch coincidences require care.'}
(out/'symbol-validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
