"""Audit internal stability of clamped-exterior memory realization."""
import json
from pathlib import Path
import numpy as np
from scipy.linalg import eigvals
p=Path(__file__).parent
exec((p/'exact_elimination_check.py').read_text().split('B=prepare(1.);A=prepare(1.5)')[0])
dt=.6;I=np.eye(20*N);X=dt*L;R=I+X+X@X/2+X@X@X/6
II=R[np.ix_(inside,inside)];IE=R[np.ix_(inside,outside)];EI=R[np.ix_(outside,inside)];EE=R[np.ix_(outside,outside)]
w=eigvals(EE,check_finite=False);rho=float(abs(w).max());j=np.argmax(abs(w));out={'scope':'Internal spectrum of R_EE in the actual32-cell periodic/8-cell interior memory fixture; this is separate from the stable coupled full-PDE spectrum.','n_total':N,'n_interior':len(inside)//20,'dt':dt,'spectral_radius':rho,'implied_growth_rate':float(np.log(rho)/dt),'leading_eigenvalue':[float(w[j].real),float(w[j].imag)],'above_one_count':int(sum(abs(w)>1+1e-10))}
# Propagate source-map columns with binary matrix powers; do not evaluate
# long history convolution where cancellation could invalidate accuracy.
powers=[EE]
for _ in range(1,round(50000/dt).bit_length()):powers.append(powers[-1]@powers[-1])
records=[]
for t in [0,100,1000,5000,10000,50000]:
 steps=round(t/dt);X=EI.copy();m=steps;i=0
 while m:
  if m&1:X=powers[i]@X
  m>>=1;i+=1
 kernel=IE@X;records.append({'time':steps*dt,'kernel_frobenius':float(np.linalg.norm(kernel)),'finite':bool(np.isfinite(kernel).all())})
out['kernel_samples']=records;(p/'exterior-memory-internal.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out),flush=True)
