"""Long normal strip, periodic Fourier tangent, exact original zero_rate."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.linalg import eigvals
from strip_model import full_matrix
p=argparse.ArgumentParser();p.add_argument('--n',type=int,default=64);p.add_argument('--angle',type=float,default=.25);p.add_argument('--rate',type=float,default=.02);p.add_argument('--cells',type=float,default=8);p.add_argument('--out',required=True);a=p.parse_args()
params=dict(n=a.n,h=32.,angle_y=a.angle*np.pi,mode='zero_rate',degree=1,damping=True,shift=1,lapse_damping=.1,sponge_rate=a.rate,sponge_cells=a.cells)
t=time.monotonic();L,_=full_matrix(**params);w=eigvals(L,check_finite=False);i=np.argmax(w.real)
out=dict(parameters=params,real=float(w[i].real),imag=float(w[i].imag),positive=int(sum(w.real>1e-8)),near_neutral=int(sum(abs(w)<1e-8)),wall_seconds=time.monotonic()-t)
Path(a.out).write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
