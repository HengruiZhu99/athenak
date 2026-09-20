import json
from pathlib import Path
import numpy as np
from fd_boundary import matrix,vector_matrix
out=[]
for sector,builder,nv in [('scalar',matrix,8),('vector',vector_matrix,4)]:
 for n in [16,32,64]:
  for degree in [1,3]:
   for inner in [2,4,'volume']:
    h=4/n;p=dict(n=n,h=h,degree=degree,inner=inner,damping=True)
    L,data=builder(**p);eig,v=np.linalg.eig(L);i=np.argmax(eig.real);w=v[:,i]
    q=data['Q']@w
    state=w.reshape(nv,n);derivative_norm=np.sqrt(np.linalg.norm(state[:nv//2])**2+np.linalg.norm(np.diff(state[nv//2:],axis=1)/h)**2)
    r=dict(sector=sector,**p,gamma=float(eig[i].real),omega=float(eig[i].imag),physical_Q_l2=float(np.linalg.norm(q)),derivative_state_norm=float(derivative_norm),Q_fraction=float(np.linalg.norm(q)/max(1e-100,derivative_norm)))
    out.append(r);print(json.dumps(r),flush=True)
Path('refinement-results.json').write_text(json.dumps(out,indent=2)+'\n')
