from volume import *
from dataclasses import replace

def assess_mode(lam,k,cfg,mode='zero_rate',details=False):
 U,D,T,err=schur(lam,k,cfg);B=boundary(U,D,cfg)
 if mode in ['physical_radiation','physical_radiation_damped','physical_radiation_coupled']:
  H=np.array([[U[1],U[3],U[4]],[U[3],U[2],U[5]],[U[4],U[5],-U[1]-U[2]]]);DH=np.array([[D[1],D[3],D[4]],[D[3],D[2],D[5]],[D[4],D[5],-D[1]-D[2]]])
  Q=np.stack([U[13+i]-DH[i,0]-1j*k*H[i,1]for i in range(3)])
  c=cfg.alpha*np.sqrt(cfg.chi);sigma=cfg.alpha*cfg.kappa if mode!='physical_radiation' else 0.
  Fq=(lam+sigma)*Q+(c-cfg.beta_n)*(Q@T);Ft=(lam+sigma)*U[7]+(c-cfg.beta_n)*D[7]
  if mode=='physical_radiation_coupled':Ft-=.5*sigma*np.sqrt(cfg.chi)*Q[0]
  B[2]=Ft;B[3]=Fq[0];B[5]=Fq[1];B[7]=Fq[2]
 Us,R=np.linalg.qr(U);Bi=B@np.linalg.inv(R);norms=np.linalg.norm(Bi,axis=1);Bs=Bi/norms[:,None];_,sv,vh=np.linalg.svd(Bs)
 out={'sigma_min':float(sv[-1]),'sigma_ratio':float(sv[-1]/sv[0]),'schur_residual':float(err),'stable_trace_condition':float(np.linalg.cond(U))}
 if details:return out,(U,D,T,B,R,vh,norms)
 return out

def roots(k,cfg,mode='zero_rate'):
 xs=np.geomspace(1e-7,max(.5,2*k),240);ys=[]
 for x in xs:
  try:y=assess_mode(complex(x),k,cfg,mode)['sigma_min']
  except ValueError:y=np.nan
  ys.append(y)
 minima=[]
 for i in range(1,len(xs)-1):
  if ys[i]<ys[i-1] and ys[i]<ys[i+1]:
   fn=lambda x:assess_mode(complex(x),k,cfg,mode)['sigma_min']
   res=optimize.minimize_scalar(fn,bracket=(xs[i-1],xs[i],xs[i+1]),method='brent',options={'xtol':1e-13,'maxiter':160})
   minima.append(dict(lambda_real=float(res.x),**assess_mode(complex(res.x),k,cfg,mode)))
 return minima

def profile(lam,k,cfg,mode='zero_rate'):
 diag,(U,D,T,B,R,vh,norms)=assess_mode(lam+0j,k,cfg,mode,True)
 coeff=np.linalg.solve(R,vh[-1].conj());u=U@coeff;fac=np.max(abs(u));coeff/=fac
 rows=[];V0,_=volume(0,k,cfg);Vp,_=volume(1,k,cfg);Vm,_=volume(-1,k,cfg);V1=(Vp-Vm)/2;V2=(Vp+Vm)/2-V0
 for x in [0,-1,-8,-32,-128,-512]:
  co=linalg.expm(x*T)@coeff;u=U@co;du=D@co;ddu=U@T@T@co;rhs=V0@u+V1@du+V2@ddu
  h=np.array([[u[1],u[3],u[4]],[u[3],u[2],u[5]],[u[4],u[5],-u[1]-u[2]]]);dh=np.array([[du[1],du[3],du[4]],[du[3],du[2],du[5]],[du[4],du[5],-du[1]-du[2]]]);aa=np.array([[u[8],u[10],u[11]],[u[10],u[9],u[12]],[u[11],u[12],-u[8]-u[9]]]);da=np.array([[du[8],du[10],du[11]],[du[10],du[9],du[12]],[du[11],du[12],-du[8]-du[9]]])
  Q=u[13:16]-dh[:,0]-1j*k*h[:,1];H=cfg.chi*(ddu[1]+2j*k*du[3]-k*k*u[2])+2*(ddu[0]-k*k*u[0]);K=u[6]+2*u[7];dK=du[6]+2*du[7];M=da[:,0]+1j*k*aa[:,1]-2/3*np.array([dK,1j*k*K,0])
  rows.append(dict(x=x,state_max=float(max(abs(u))),bulk_relative=float(np.linalg.norm(lam*u-rhs)/max(np.linalg.norm(rhs),abs(lam)*np.linalg.norm(u),1e-300)),Theta=float(abs(u[7])),Q_max=float(max(abs(Q))),H=float(abs(H)),M_max=float(max(abs(M))),state=[[float(z.real),float(z.imag)]for z in u]))
 boundary_res=B@coeff
 return dict(lambda_real=lam,k=k,config=asdict(cfg),mode=mode,**diag,boundary_max_abs=float(max(abs(boundary_res))),boundary_row_scaled_max=float(max(abs(boundary_res/norms))),normal_roots=[[float(z.real),float(z.imag)]for z in np.linalg.eigvals(T)],profile=rows)
