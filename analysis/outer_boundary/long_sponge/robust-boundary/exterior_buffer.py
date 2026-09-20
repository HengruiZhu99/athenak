"""Rational full-coupled DtN realization via finite auxiliary exterior grid."""
from rational_dtn import *

def pencil(lam,k,cfg):
 V0,_=volume(0,k,cfg);Vp,_=volume(1,k,cfg);Vm,_=volume(-1,k,cfg);V1=(Vp-Vm)/2;V2=(Vp+Vm)/2-V0
 part=lambda v,a,b:v[np.ix_(a,b)]
 C=part(V0,QP,PP);Ci=np.linalg.inv(C);Q0=lam*np.eye(10)-part(V0,QP,QP);Q1=-part(V1,QP,QP);P0=lam*np.eye(10)-part(V0,PP,PP);P1=-part(V1,PP,PP)
 return P0@Ci@Q0-part(V0,PP,QP),P0@Ci@Q1+P1@Ci@Q0-part(V1,PP,QP),P1@Ci@Q1-part(V2,PP,QP)

def buffer(lam,k,cfg,hs,flux=True):
 M0,M1,M2=pencil(lam,k,cfg);eye=np.eye(10);E=0*eye
 # Nodes x0 at interface, x_(n+1) terminal q=0; hs stores n+1 spacings.
 for i in range(len(hs)-1,0,-1):
  hl,hr=hs[i-1],hs[i]
  L=2/(hl*(hl+hr))*M2-hr/(hl*(hl+hr))*M1
  B=M0-2/(hl*hr)*M2+(hr-hl)/(hl*hr)*M1
  R=2/(hr*(hl+hr))*M2+hl/(hr*(hl+hr))*M1
  if i==1:E2=E.copy()
  E=-np.linalg.solve(B+R@E,L)
 if flux:
  # Boundary half-cell integrated normal flux. This is a consistent normal
  # flux; no Z4c energy/passivity theorem is assumed for this construction.
  return (E-eye)/hs[0]+.5*np.linalg.solve(M2,M1)@(E-eye)+.5*hs[0]*np.linalg.solve(M2,M0)
 h0,h1=hs[:2]
 return -(2*h0+h1)/(h0*(h0+h1))*eye+(h0+h1)/(h0*h1)*E-h0/(h1*(h0+h1))*E2@E

if __name__=='__main__':
 import argparse
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();cfg=Config(alpha=.999665896620774,chi=.9993319048666159,beta_n=.0003157233701617295);k=np.pi/256;out={'scope':'Finite full-coupled auxiliary exterior discretization, continuum interior trace. Screening, not a stable closure proof.','cases':[]}
 old=json.load(open(Path(__file__).parent/'reference-screen-results.json'))
 roots=[complex(*r['lambda_value'])for case in old['cases']if case['name']=='baseline'and abs(case['k']-k)<1e-10 for r in case['roots']]
 roots+=[cfg.beta_n*k/np.sqrt(2)+1j*cfg.alpha*np.sqrt(cfg.chi)*k/np.sqrt(2)]
 for h,n in [(2,96),(1,192),(.5,256)]:
  hs=h*np.minimum(1.055**np.arange(n),1000)
  for flux in [False,True]:
   N=lambda z:buffer(z,k,cfg,hs,flux)
   grid=[]
   for x in np.geomspace(1e-8,.04,17):
    for y in np.unique(np.r_[np.linspace(0,2.5,36),[1/np.sqrt(2),1,np.sqrt(4/3),np.sqrt(2)]])*k:
     grid.append((sigma(x+1j*y,k,cfg,N),x,y))
   seeds=sorted(grid)[:12];ref=[]
   for _,x,y in seeds:
    def fn(w):
     try:return sigma(10**w[0]+1j*k*w[1],k,cfg,N)
     except ValueError:return 1.
    opt=optimize.minimize(fn,[np.log10(x),y/k],method='Nelder-Mead',bounds=[(-9.8,np.log10(.08)),(0,3)],options={'xatol':1e-8,'fatol':1e-12,'maxiter':180})
    z=10**opt.x[0]+1j*k*opt.x[1]
    if not any(abs(z-complex(*r['lambda']))<1e-8 for r in ref):ref.append({'lambda':[float(z.real),float(z.imag)],'sigma_min':float(opt.fun),'at_lower_bound':bool(opt.x[0]<-9.79)})
   row={'h0':h,'nodes':n,'extent':float(sum(hs)),'flux':flux,'old_roots_sigma':[sigma(z,k,cfg,N)for z in roots],'grid_min':list(min(grid)),'refined':ref};out['cases'].append(row);a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(row),flush=True)
