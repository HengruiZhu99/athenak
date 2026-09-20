from pathlib import Path
import json,numpy as np
from frozen_symbol import Frozen
out=Path(__file__).resolve().parent;grid=np.linspace(-np.pi,np.pi,33);xi=np.stack(np.meshgrid(grid,grid,grid,indexing='ij'),-1).reshape(-1,3);high=abs(xi).max(-1)>=np.pi/2;result=[]
for h in [.5,.25,.125,.0625]:
 f=Frozen([h/2]*3,h=h);ko=-np.sum(np.sin(xi/2)**8,axis=-1)/h
 for scheme in ['standard','compatible']:
  # Adding KO shifts every eigenvalue by a real scalar; avoid repeated eigensolves.
  e=[]
  for chunk in np.array_split(xi,8):e.append(np.linalg.eigvals(f.evaluate(chunk,scheme,diss=0)))
  e=np.concatenate(e)
  for diss in [0,.5,1]:
   values=e+ko[:,None]*diss;growth=values.real.max(-1);ix=int(np.argmax(growth));ih=int(np.argmax(np.where(high,growth,-np.inf)));z=values*(.15*h)
   q={'h':h,'x':[h/2]*3,'r':float(np.sqrt(3)*h/2),'scheme':scheme,'KO_diss':diss,'max_real':float(growth[ix]),'max_xi_over_pi':(xi[ix]/np.pi).tolist(),'high_frequency_max_real':float(growth[ih]),'high_max_xi_over_pi':(xi[ih]/np.pi).tolist(),'RK3_max_abs_dt_015h':float(np.max(abs(1+z+z*z/2+z*z*z/6)))};result.append(q);print(q,flush=True)
(out/'nearest-cell-all-signs.json').write_text(json.dumps({'scope':'Same full frozen nearest-cell Jacobian as nearest-cell-scan.json; all wavevector sign combinations, actual upwind. No global-mode interpretation.','grid':'33^3 uniform phases in[-pi,pi]^3; high means max|xi|>=pi/2. Includes prior17^3 first-octant samples. KO is a scalar identity shift of each eigenvalue.','samples':len(xi),'results':result},indent=2))
