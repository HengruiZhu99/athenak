"""Independent radial-vector wave audit of the inner gauge closure.

The leading collapsed gauge subsystem has beta_tt = (4 mu/3)
(beta_rr+2 beta_r/r-2 beta/r^2). This reduced experiment isolates the
gauge stencil, not the complete coupled puncture equations.
"""
import argparse,json
from pathlib import Path
import numpy as np
from puncture import PunctureEvolution
from numerics import RadialOperators,puncture_dissipation

p=argparse.ArgumentParser();p.add_argument('--library',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
records=[]
for n in [65,129]:
 e=PunctureEvolution(n,a.library,grid_stretch=0,shift_gauge='moving-puncture');r=e.r
 I=np.eye(n);active=slice(1,-4)
 K=puncture_dissipation(I,r,e.h,.5,.8,measure=e.jac,order=8)/e.jac[:,None];K=K[active,active]
 for kind,op in [('positive',e.op),('odd',RadialOperators(r,width=7,subtract_local=True))]:
  d,dd=op.derivatives(I,-np.ones(n))
  L=4/3*(dd[1:]+2*d[1:]/r[1:,None]-2*I[1:]/r[1:,None]**2);L=L[:-4,active]
  size=len(L);P=np.block([[K,np.eye(size)],[L,K-np.eye(size)]])
  ev,vec=np.linalg.eig(P);i=int(np.argmax(ev.real))
  record=dict(n=n,closure=kind,maximum_growth=float(ev[i].real),frequency=float(ev[i].imag),mode_peak_radius=float(r[active][np.argmax(abs(vec[:size,i]))]))
  records.append(record)
  if kind=='positive':assert ev[i].real>1
  else:assert ev[i].real<0
e=PunctureEvolution(129,a.library,grid_stretch=0,shift_gauge='moving-puncture',shift_inner_stencil='odd');r=e.r
f=.3*r+.2*r**3-.1*r**5;d,dd=e.shift_op.derivatives(f,-1)
error=max(np.max(abs(d[:8]-(.3+.6*r[:8]**2-.5*r[:8]**4))),np.max(abs(dd[:8]-(1.2*r[:8]-2*r[:8]**3))))
assert error<1e-10
q=e.initial();q[1:5,5]+=.0001;rate,state=e.rhs(q,return_state=True)
assert np.array_equal(q[1:5,5],state[1:5,5])
assert state[0,9]==0 and rate[0,9]==0 and e.stats['max_relative_null']<2e-15
result=dict(reduced_wave_spectra=records,odd_polynomial_error=error,independent_connection_retained=True,
 scope='Two-field radial-vector wave on the same inner grid, same KO8=.5 and eta=1, with four fixed outer nodes. Identifies an inner closure instability; does not prove stability of the full Z4c/telegraph system or regularity of puncture geometry.')
a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
