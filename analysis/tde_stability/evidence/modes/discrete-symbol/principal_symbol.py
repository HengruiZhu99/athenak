"""Constrained20-variable frozen Z4c principal symbols, source read-only.

Principal background coefficients are alpha,chi,gtilde=I,beta. Background A/K
and derivatives are lower order in the pseudo-differential reduction and are
excluded here; this is not a global variable-coefficient stability test.
"""
from pathlib import Path
import json,numpy as np
pairs=[(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]
TF=np.zeros((5,3,3));TF[0]=np.diag([1,-1,0])/np.sqrt(2);TF[1]=np.diag([1,1,-2])/np.sqrt(6)
for n,(i,j)in enumerate([(0,1),(0,2),(1,2)],2):TF[n,i,j]=TF[n,j,i]=1/np.sqrt(2)
# Reduced order: chi;hTF5;Khat;aTF5;Gamma3;Theta;alpha;beta3.
def symbols(xi,h=.25):
 xi=np.asarray(xi);D=1j*(1.5*np.sin(xi)-.3*np.sin(2*xi)+np.sin(3*xi)/30)/h
 D2=(-49/18+3*np.cos(xi)-.3*np.cos(2*xi)+np.cos(3*xi)/45)/h**2
 S=D[..., :,None]*D[...,None,:]
 for i in range(3):S[...,i,i]=D2[...,i]
 return D,D2,S

def operator(xi,alpha=1,chi=1,beta=(0,0,0),shift=2,h=.25,scheme='standard',advection='centered',diss=0,kappa=0,eta=0):
 xi=np.atleast_2d(xi);N=len(xi);D,D2,S=symbols(xi,h);L=D2.sum(-1);S1=D[..., :,None]*D[...,None,:]
 if scheme=='compatible':H=S1+np.eye(3)*(L-np.sum(D*D,axis=-1))[:,None,None]/3;G=S1
 elif scheme=='standard':H=S;G=S
 elif scheme=='all_D1D1':H=S1;G=S1;L=np.sum(D*D,axis=-1)
 else:raise ValueError(scheme)
 A=np.zeros((N,20,20),complex);v=np.eye(20,dtype=complex)
 c=v[:,0];g=np.einsum('qv,vab->qab',v[:,1:6],TF);K=v[:,6];a=np.einsum('qv,vab->qab',v[:,7:12],TF);Gamma=v[:,12:15];Theta=v[:,15];lapse=v[:,16];b=v[:,17:20]
 div=np.einsum('na,qa->nq',D,b);outc=-2*chi*div/3+2*chi*alpha*(K+2*Theta)[None,:]/3
 outg=-2*alpha*a[None,...]+np.einsum('ni,qj->nqij',D,b)+np.einsum('nj,qi->nqij',D,b)-2*np.eye(3)*div[...,None,None]/3
 outK=-chi*L[:,None]*lapse[None,:]
 ric=-.5*L[:,None,None,None]*g[None,...]+.5*(np.einsum('ni,qj->nqij',D,Gamma)+np.einsum('nj,qi->nqij',D,Gamma))+(H[:,None,:,:]+np.eye(3)*L[:,None,None,None])*c[None,:,None,None]/(2*chi)
 ra=chi*(-H[:,None,:,:]*lapse[None,:,None,None]+alpha*ric);ra-=np.eye(3)*np.trace(ra,axis1=-2,axis2=-1)[...,None,None]/3
 outTheta=alpha*(chi*np.einsum('ni,qi->nq',D,Gamma)+2*L[:,None]*c[None,:])/2
 outGamma=L[:,None,None]*b[None,...]+np.einsum('nij,qj->nqi',G,b)/3-2*alpha/3*D[:,None,:]*(2*K+Theta)[None,:,None]
 if kappa:
  # Lower-order constraint damping retained optionally. C[g]=d_j h_ij since h trace0.
  C=np.einsum('nj,qij->nqi',D,g);outK+=kappa*alpha*Theta[None,:];outTheta-=2*kappa*alpha*Theta[None,:];outGamma-=2*kappa*alpha*(Gamma[None,...]-C)
 A[:,0,:]=outc;A[:,1:6,:]=np.einsum('nqij,vij->nvq',outg,TF);A[:,6,:]=outK;A[:,7:12,:]=np.einsum('nqij,vij->nvq',ra,TF);A[:,12:15,:]=outGamma.swapaxes(1,2);A[:,15,:]=outTheta;A[:,16,:]=-2*alpha*v[:,6][None,:];A[:,17:20,:]=shift*v[:,12:15].T[None,...]-eta*v[:,17:20].T[None,...]
 beta=np.asarray(beta)
 if advection=='centered':adv=np.sum(D*beta,axis=-1)
 elif advection=='upwind':
  dl=sum(w*np.exp(1j*xi*k)for k,w in[(-4,1/60),(-3,-2/15),(-2,.5),(-1,-4/3),(0,7/12),(1,2/5),(2,-1/30)])/h
  dr=sum(w*np.exp(1j*xi*k)for k,w in[(4,-1/60),(3,2/15),(2,-.5),(1,4/3),(0,-7/12),(-1,-2/5),(-2,1/30)])/h
  adv=np.sum(beta*np.where(beta<0,dl,dr),axis=-1)
 else:raise ValueError(advection)
 ko=-diss*np.sum(np.sin(xi/2)**8,axis=-1)/h
 A+=(adv+ko)[:,None,None]*np.eye(20)
 return A

def reduced(A,xi,h=.25):
 # Discrete energy reduction controls full gradient via sqrt(-L), including Nyquist.
 _,D2,_=symbols(xi,h);omega=np.sqrt(np.maximum(1e-30,-D2.sum(-1)));qidx=[0,*range(1,6),16,17,18,19];w=np.ones((len(A),20));w[:,qidx]=omega[:,None];return A*w[:,:,None]/w[:,None,:]

def scan():
 out=Path(__file__).resolve().parent;grid=np.linspace(0,np.pi,17);xx=np.stack(np.meshgrid(grid,grid,grid,indexing='ij'),-1).reshape(-1,3);xx=xx[np.max(xx,axis=-1)>0];high=np.max(xx,axis=-1)>=np.pi/2
 points={'Minkowski':(1.,1.,[0,0,0])}
 for r in [.10825317547305482,.21650635094610965,.414578098794425,.649519052838329,1,2,4,16]:
  alpha=r/(1+r);x=r/np.sqrt(3);points[f'r{r:.6f}']=(alpha,alpha**2,[x/(1+r)**2]*3)
 results=[]
 for label,(alpha,chi,beta)in points.items():
  for scheme in ['standard','compatible']:
   for shift in [1,2]:
    for style in ['principal_centered','upwind_KO05','upwind_KO05_damping']:
     opts={'advection':'centered','diss':0,'kappa':0,'eta':0}if style=='principal_centered'else{'advection':'upwind','diss':.5,'kappa':.1 if style.endswith('damping')else 0,'eta':2 if style.endswith('damping')else 0}
     A=operator(xx,alpha,chi,beta,shift=shift,scheme=scheme,**opts);ev=np.linalg.eigvals(A);growth=np.max(ev.real,axis=-1);idx=int(np.argmax(growth));z=ev*.075;rk=1+z+z*z/2+z*z*z/6;ixrk=np.unravel_index(np.argmax(abs(rk)),rk.shape);maxrk=float(np.max(abs(rk)))
     q={'point':label,'alpha':alpha,'chi':chi,'beta':beta,'shift_driver':shift,'scheme':scheme,'style':style,'samples':len(xx),'max_real_eigenvalue':float(growth[idx]),'high_frequency_max_real_eigenvalue':float(np.max(growth[high])),'max_growth_xi_over_pi':(xx[idx]/np.pi).tolist(),'max_RK3_abs_dt_0075':maxrk,'RK3_max_xi_over_pi':(xx[ixrk[0]]/np.pi).tolist(),'max_RK3_abs_dt_00375':float(np.max(abs(1+z/2+(z/2)**2/2+(z/2)**3/6)))}
     results.append(q);print(label,scheme,shift,style,q['max_real_eigenvalue'],maxrk,flush=True)
 (out/'principal-scan.json').write_text(json.dumps({'grid':'17^3 first-octant phases including Nyquist; origin excluded','h':.25,'state_dimensions':20,'note':'Frozen principal coefficients, with optional upwind/KO and lower-order kappa/eta only. Not full background lower-order Jacobian or global evolution.','results':results},indent=2))
 np.save(out/'xi-grid.npy',xx)
if __name__=='__main__':scan()
