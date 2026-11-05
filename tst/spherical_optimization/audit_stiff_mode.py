#!/usr/bin/env python3
"""Locate the RK-limiting frozen mode and probe local KO/penalty sensitivity."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.linalg import eig
from scipy.optimize import brentq
from kernel import Kernel
from evolution import make_evolution

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--linearization',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 info=json.loads((a.linearization/'audit.json').read_text());data=np.load(a.linearization/'linearization.npz');J=data['J'];state=data['state'];active=data['active'];scales=data['scales'];values,L,V=eig(J,left=True,right=True);speed=info['speed_over_dr'];decaying=values.real<-1e-8
 def polynomial(z):return 1+z+z*z/2+z**3/6+z**4/24
 bound=brentq(lambda c:float(max(abs(polynomial(c/speed*values[decaying])))-1),.1,2.)
 idx=np.where(decaying)[0][np.argmax(abs(polynomial(bound/speed*values[decaying])))];value=values[idx];left=L[:,idx];right=V[:,idx];den=np.vdot(left,right);mode=np.zeros_like(state,dtype=complex);mode[active]=right*scales[active]
 cfg=info['run_configuration'];k=Kernel.from_config(a.library,cfg);e=make_evolution(k,cfg)
 def response(epsilon):
  z=np.zeros(len(right),dtype=complex)
  for vector,factor in [(mode.real,1.),(mode.imag,1j)]:
   z+=factor*((e.rhs(state+epsilon*vector)-e.rhs(state-epsilon*vector))/(2*epsilon)/scales)[active]
  return z
 probes=[]
 for eps in (1e-5,3e-6):
  base=response(eps);row=dict(epsilon=eps,relative_Jv_error=float(np.linalg.norm(base-np.einsum('ij,j->i',J,right))/np.linalg.norm(base)),sensitivities=[])
  for name in ('ko','penalty'):
   original=getattr(e,name)
   for fraction in (.02,.01):
    h=original*fraction;setattr(e,name,original+h);plus=response(eps);setattr(e,name,original-h);minus=response(eps);setattr(e,name,original)
    derivative=np.vdot(left,(plus-minus)/(2*h))/den;row['sensitivities'].append(dict(parameter=name,fraction=fraction,base_value=original,lambda_derivative=[float(derivative.real),float(derivative.imag)],normalized_damping_sensitivity=float(original*derivative.real/value.real)))
  probes.append(row)
 i,j=np.unravel_index(abs(mode).argmax(),mode.shape);weight=abs(mode)**2;mask=e.r>.9
 out=dict(eigenvalue=[float(value.real),float(value.imag)],RK_C_char_bound=float(bound),peak_r=float(e.r[i]),peak_independent_field=int(j),outer_r09_squared_component_fraction=float(weight[mask].sum()/weight.sum()),eigenvalue_condition=float(np.linalg.norm(left)*np.linalg.norm(right)/abs(den)),probes=probes,input_sha256=hashlib.sha256((a.linearization/'linearization.npz').read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),binary_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),qualification='Local eigenvalue derivatives along one frozen complex mode, using the full actual RHS with each coefficient varied. Sensitivities are not additive damping budgets or proof that changing the penalty preserves stability/accuracy. Component weights are not continuum energy.')
 np.savez_compressed(a.output/'mode.npz',r=e.r,mode=mode,eigenvalue=value,left=left,right=right);(a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));k.close()
if __name__=='__main__':main()
