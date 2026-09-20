from pathlib import Path
import json,numpy as np
from frozen_symbol import Frozen
from principal_symbol import operator
out=Path(__file__).resolve().parent;grid=np.linspace(0,np.pi,17);xi=np.stack(np.meshgrid(grid,grid,grid,indexing='ij'),-1).reshape(-1,3);high=xi.max(-1)>=np.pi/2;results=[]
for h in [.5,.25,.125,.0625]:
 x=[h/2]*3;f=Frozen(x,h=h)
 for scheme in ['standard','compatible']:
  for diss in [0,.5,1.0]:
   A=f.evaluate(xi,scheme,diss=diss);ev=np.linalg.eigvals(A);growth=ev.real.max(-1);idx=int(np.argmax(growth));ih=int(np.argmax(np.where(high,growth,-np.inf)));best=A[idx];e,V=np.linalg.eig(best);j=int(np.argmax(e.real));q=V[:,j];norm=float(np.max(abs(q)));q=q/norm
   z=ev*(.15*h);item={'h':h,'x':x,'r':float(np.linalg.norm(x)),'alpha':f.v[18],'chi':f.v[0],'scheme':scheme,'KO_diss':diss,'max_real':float(growth[idx]),'h_times_max_real':float(h*growth[idx]),'max_xi_over_pi':(xi[idx]/np.pi).tolist(),'high_frequency_max_real':float(growth[ih]),'high_max_xi_over_pi':(xi[ih]/np.pi).tolist(),'RK3_max_abs_dt_015h':float(np.max(abs(1+z+z*z/2+z*z*z/6))),'dominant_local_eigenvalue':[float(e[j].real),float(e[j].imag)],'eigenvector_abs_q_maxnormalized':abs(q).tolist()};results.append(item);print(item,flush=True)
 # Actual principal/upwind/KO at this same nearest-point h, includes kappa/eta.
 A=operator(xi,alpha=f.v[18],chi=f.v[0],beta=f.beta,shift=2,h=h,scheme='standard',advection='upwind',diss=.5,kappa=.1,eta=2);eig=np.linalg.eigvals(A);g=eig.real.max(-1);g=np.where(xi.max(-1)>0,g,-np.inf);ix=int(np.argmax(g));results.append({'h':h,'x':x,'scope':'principal_upwind_KO05_kappa_eta','max_real':float(g[ix]),'high_frequency_max_real':float(g[high].max())})
(out/'nearest-cell-scan.json').write_text(json.dumps({'scope':'Full local frozen Jacobian at nearest cell x_i=h/2, G2 eta2 kappa.1. Uses actual sixth-order FD background jets and upwind stencil. Not a global growth eigenmode or a convergence proof.','grid':'17^3 first-octant phases, including origin and Nyquist. High means max|xi_i|>=pi/2.','results':results},indent=2))
