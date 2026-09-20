from pathlib import Path
import numpy as np,json
from frozen_symbol import Frozen,background_jets,basis_projector
from point_operator import point_rhs
rng=np.random.default_rng(9361);results=[];phase=.47;xi=np.array([.47,1.3,2.4]);q=rng.normal(size=20)
for x in [[.125]*3,[.625,.125,.125]]:
 f=Frozen(x);v,d,dd,up=background_jets(x,.25);B,CP=basis_projector(v);p=B@q;D,D2,S=__import__('principal_symbol').symbols(xi);cos=np.cos(phase);cphase=np.exp(1j*phase)
 dl=sum(w*np.exp(1j*xi*k)for k,w in[(-4,1/60),(-3,-2/15),(-2,.5),(-1,-4/3),(0,7/12),(1,2/5),(2,-1/30)])/.25
 dr=sum(w*np.exp(1j*xi*k)for k,w in[(4,-1/60),(3,2/15),(2,-.5),(1,4/3),(0,-7/12),(-1,-2/5),(-2,1/30)])/.25
 U=np.where(v[19:22]<0,dl,dr);uppert=np.real(cphase*U)[:,None]*p[None,:];dpert=np.real(cphase*D)[:,None]*p[None,:];ddpert=np.real(cphase*S)[:,:,None]*p[None,None,:];ko=-.5*np.sum(np.sin(xi/2)**8)/.25
 def F(eps):
  vv=v+eps*cos*p;adv=np.sum(vv[19:22,None]*(up+eps*uppert),axis=0);rhs=point_rhs(vv[None,:],(d+eps*dpert)[None,:,:],(dd+eps*ddpert)[None,:,:,:],adv[None,:],kappa=.1)[0];rhs[18]=eps*(np.sum(v[19:22]*uppert[:,18])-2*v[18]*cos*p[7]);rhs[19:22]=eps*(2*cos*p[14:17]+np.sum(v[19:22,None]*uppert[:,19:22],axis=0)-2*cos*p[19:22]);return CP@rhs+eps*ko*cos*q
 predicted=np.real(cphase*(f.evaluate(xi)[0]@q))
 for eps in [1e-5,1e-6,1e-7]:
  measured=(F(eps)-F(-eps))/(2*eps);err=float(np.max(abs(measured-predicted)));results.append({'x':x,'epsilon':eps,'max_abs_action_error':err,'relative_max_error':err/float(np.max(abs(predicted)))})
assert max(q['relative_max_error']for q in results if q['epsilon']==1e-6)<1e-7
Path('frozen-action-validation.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))
