"""Short-exponential semigroup cross-check of long full-state transients.

A large single exp(A*t) loses accuracy for this nonnormal second-order matrix.
Compare subdivisions: mathematically identical exponentials, different rounding.
"""
from pathlib import Path
import json
import numpy as np
from scipy.linalg import expm
from p_only import build
from transients import initial,TIMES
ROOT=Path(__file__).parent

def run(n,dt):
 A,raw,r,D,C,rows,labels,replaced=build(n);N=len(r);U0=initial(r);U=U0.copy();E=expm(A*dt)
 a=r/(1+r);v=a*a-r/(1+r)**2;names=['pure_lapse','Theta_pulse','constraint_free_radial_coordinate'];hist={name:[] for name in names}
 record={round(t/dt):t for t in TIMES}
 for cycle in range(round(TIMES[-1]/dt)+1):
  if cycle in record:
   t=record[cycle]
   for j,name in enumerate(names):
    state=U[:,j];der=A@state;fields={key:C[key]@state for key in ['Theta','Q','H','M']};th=fields['Theta'];q=fields['Q']
    rad=np.array([(C['Theta']@der)[0]+v[0]*((D@th)[0]+th[0]/(1+r[0])),(C['Q']@der)[0]+v[0]*((D@q)[0]+q[0]/r[0])])
    vals={'time_M':float(t),'state_L2_grid_over_initial':float(np.linalg.norm(state)/np.linalg.norm(U0[:,j])),'constraint_L2_grid_over_initial_state':float(np.linalg.norm(np.concatenate(list(fields.values())))/np.linalg.norm(U0[:,j])),'actual_radiation_boundary_absolute':abs(rad).tolist(),'finite':bool(np.all(np.isfinite(state)))}
    for key,field in fields.items():vals['max_'+key]=float(np.max(abs(field)));vals['peak_r_'+key]=float(r[np.argmax(abs(field))])
    hist[name].append(vals)
  if cycle<round(TIMES[-1]/dt):U=E@U
 out={'degree':n,'exponential_subdivision_M':dt,'method':'Repeated short dense matrix exponential, linear state propagated without clipping/reset. Subdivision changes floating-point conditioning, not continuum time discretization.','initial':'Polynomial [((r-.2)*(4-r))/1.9^2]^8 with seven endpoint derivatives zero. Pure lapse, Theta, and analytic constraint-free radial-coordinate directions.','states':hist,'zero_state_exact':bool(np.all(A@np.zeros(8*N)==0)),'matrix_one_norm':float(np.linalg.norm(A,1)),'final_state_real':U.reshape(8,N,3).tolist(),'r_M':r.tolist()}
 (ROOT/f'semigroup-n{n}-dt{dt:g}.json').write_text(json.dumps(out,indent=2)+'\n');print(n,dt,[(name,hist[name][-1]['constraint_L2_grid_over_initial_state']) for name in names],flush=True)
 return out
if __name__=='__main__':
 for n in [32,48,64,80]:run(n,.1)
 for n in [64,80]:run(n,.05)
