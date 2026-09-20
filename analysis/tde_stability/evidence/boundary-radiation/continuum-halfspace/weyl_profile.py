from schur_check import *
from scipy import linalg,optimize
import json
rt=1.2307116366820878;ky=2*np.pi
rec,data=assess_schur(rt,ky,'radiation_weyl',True,True);U,D,T,B,R,ss,vv,Us,norms=data
coeff=np.linalg.solve(R,vv[-1].conj());coeff/=np.max(abs(U@coeff))
np.savez_compressed(ROOT/'weyl-halfspace-mode.npz',U=U,D=D,T=T,coeff=coeff,lambda_real=rt,ky=ky)

profiles=[]
for x in np.linspace(-2,0,81):
 cc=linalg.expm(T*x)@coeff;u=U@cc;ux=D@cc;uxx=D@T@cc
 def ds(v,j):return (ux if j==0 else 1j*ky*u if j==1 else 0*u)[v]
 def dds(v,i,j):
  if i==0 and j==0:return uxx[v]
  if i==2 or j==2:return 0j
  if i==1 and j==1:return -ky*ky*u[v]
  return 1j*ky*ux[v]
 def hh(i,j):return [(1,1)] if i==j==0 else [(2,1)] if i==j==1 else [(1,-1),(2,-1)] if i==j==2 else [(3,1)] if set([i,j])=={0,1} else [(4,1)] if set([i,j])=={0,2} else [(5,1)]
 def aa(i,j):return [(f+7,w) for f,w in hh(i,j)]
 H=chi*sum(sum(w*dds(f,i,j) for f,w in hh(i,j)) for i in range(3) for j in range(3))+2*sum(dds(0,i,i) for i in range(3))
 M=np.array([sum(sum(w*ds(f,j) for f,w in aa(i,j)) for j in range(3))-2/3*(ds(6,i)+2*ds(7,i)) for i in range(3)])
 E=np.array([[chi/2*(sum(sum(w*dds(f,k,i) for f,w in hh(k,j))+sum(w*dds(f,k,j) for f,w in hh(k,i)) for k in range(3))-sum(sum(w*dds(f,k,k) for f,w in hh(i,j)) for k in range(3)))+.5*(dds(0,i,j)+(sum(dds(0,k,k) for k in range(3)) if i==j else 0)) for j in range(3)] for i in range(3)])
 B=np.zeros((3,3),complex)
 for i in range(3):
  for j in range(3):
   for k in range(3):
    for ll in range(3):
     eps=(i-k)*(k-ll)*(ll-i)/2
     B[i,j]+=np.sqrt(chi)*eps*(sum(w*ds(f,k) for f,w in aa(ll,j))+((ds(6,k)+2*ds(7,k))/3 if ll==j else 0))
 profiles.append({'x':float(x),'H':float(abs(H)),'M':float(np.linalg.norm(M)),'E':float(np.linalg.norm(E)),'B':float(np.linalg.norm(B)),'state_norm':float(np.linalg.norm(u)),'lapse':float(abs(u[16])),'shift':float(np.linalg.norm(u[17:20]))})
checks=[]
fit=optimize.OptimizeResult(x=rt,fun=rec['sigma_min'])
result={'root_assessment':rec,'lambda_real':rt,'physical_profiles':profiles,'max_H':max(x['H'] for x in profiles),'max_M':max(x['M'] for x in profiles),'max_linear_electric_Weyl':max(x['E'] for x in profiles),'max_linear_magnetic_Weyl':max(x['B'] for x in profiles),'boundary_gauge_amplitudes':{k:profiles[-1][k] for k in ['lapse','shift']},'unused_local_scan':checks,'known_root_check':{'lambda_real':float(fit.x),'sigma_min':float(fit.fun)},'scope':'Only TT rows replaced by frozen published radiation equations; known old root evaluated, no new frequency search.'}
(ROOT/'weyl-physical-validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['physical_profiles','unused_local_scan']},indent=2))
