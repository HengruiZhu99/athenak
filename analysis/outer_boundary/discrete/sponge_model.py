"""Smooth face-sponge parameter tests, original production zero_rate only."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import eigvals
from corner_model import build,analyze

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=12);p.add_argument('--rate',type=float,default=.05);p.add_argument('--cells',type=float,default=4)
 p.add_argument('--order',default='before');p.add_argument('--sector',default='all');p.add_argument('--out',required=True);p.add_argument('--vectors',action='store_true');p.add_argument('--kappa-taper-cells',type=float,default=0.);a=p.parse_args()
 params=dict(n=a.n,h=32.,sponge_rate=a.rate,sponge_cells=a.cells,sponge_order=a.order,sponge_sector=a.sector,kappa_taper_cells=a.kappa_taper_cells)
 tick=time.monotonic();L=build(**params)
 if a.vectors:result=analyze(L,a.n,32.)
 else:
  w=eigvals(L,check_finite=False);i=np.argmax(w.real);dt=.0375;z=dt*w;rr=1+z+z*z/2+z*z*z/6
  result=dict(real=float(w[i].real),imag=float(w[i].imag),positive=int(sum(w.real>1e-8)),near_neutral=int(sum(abs(w)<1e-8)),rk3_radius=float(max(abs(rr))),rk3_growth=float(np.log(max(abs(rr)))/dt))
 result.update(parameters=params,wall_seconds=time.monotonic()-tick,exact_zero_map=bool(np.all(L@np.zeros(L.shape[0])==0)))
 Path(a.out).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
