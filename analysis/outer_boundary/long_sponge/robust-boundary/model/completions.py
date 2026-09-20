"""Frozen linear completion tests, not a nonlinear/production boundary recipe."""
from boundary import *

def assess_complete(lam,k,cfg,completion='old',details=False):
 _,(U,D,T,B,R,_,_)=assess_mode(lam,k,cfg,'physical_radiation_coupled',True)
 c=cfg.alpha*np.sqrt(cfg.chi);b=cfg.beta_n
 if completion=='q_dirichlet':B=U[QP].copy()
 if completion=='q_wave':B=lam*U[QP]+(c-b)*D[QP]
 if completion=='dirichlet_weyl':B[0]=U[16];B[1]=U[17];B[4]=U[18];B[6]=U[19]
 if 'mixed' in completion:B[4]=U[3];B[6]=U[4]
 if 'normal_beta' in completion:B[1]=U[17]
 if completion=='wave_weyl':
  vl=np.sqrt(2*cfg.alpha*cfg.chi);vs=np.sqrt(4*cfg.G/3);vt=np.sqrt(cfg.G)
  aa=cfg.alpha*vs*vs/(vl*vl-vs*vs);bb=.5*cfg.alpha*vs*vs/(c*c-vs*vs)
  W=D[17]+1j*k*U[18]+aa*U[6]+bb*U[7]
  B[0]=lam*U[16]+(vl-b)*D[16]
  B[1]=lam*W+(vs-b)*(W@T)
  for row,idx in [(4,18),(6,19)]:
   curl=D[idx]-(1j*k*U[17] if idx==18 else 0)
   B[row]=lam*curl+(vt-b)*(curl@T)
 if 'weyl' in completion:
  B[8]=lam*(U[9]+.5*U[8])+(c-b)*(D[9]+.5*D[8])-.5*c*1j*k*U[10]-.5*cfg.chi*k*k*U[16]
  B[9]=lam*U[12]+(c-b)*D[12]-.5*c*1j*k*U[11]
  # R_code = R_physical + D_(i Q_j), so physical Weyl requires removing
  # the tangential symmetric derivative of Q from the left-hand side.
  Qy=U[14]-D[3]-1j*k*U[2]
  Qz=U[15]-D[4]-1j*k*U[5]
  B[8]-=.5*cfg.alpha*cfg.chi*1j*k*Qy
  B[9]-=.5*cfg.alpha*cfg.chi*1j*k*Qz
 Bi=B@np.linalg.inv(R);norms=np.linalg.norm(Bi,axis=1);Bs=Bi/norms[:,None];_,sv,vh=np.linalg.svd(Bs)
 q={'sigma_min':float(sv[-1]),'sigma_ratio':float(sv[-1]/sv[0])}
 if details:return q,(U,D,T,B,R,vh,norms)
 return q
