"""Resolve the small lower-order Minkowski branch with eigenvalue conditioning.

This uses double precision; it does not classify the sign as a principal failure.
"""
import json,numpy as np
from scipy.linalg import eig
from principal_symbol import operator
from pathlib import Path
result=[]
for scheme in ['standard','compatible']:
 for denominator in [16,32,64]:
  A=operator(np.array([np.pi/denominator,0,0]),shift=2,scheme=scheme,advection='upwind',diss=.5,kappa=.1,eta=2)[0];values,L,R=eig(A,left=True,right=True);q=int(np.argmax(values.real));condition=np.linalg.norm(L[:,q])*np.linalg.norm(R[:,q])/abs(np.vdot(L[:,q],R[:,q]));result.append({'scheme':scheme,'xi_over_pi':1/denominator,'max_real':float(values[q].real),'imag':float(values[q].imag),'eigenvalue_condition':float(condition),'matrix_norm':float(np.linalg.norm(A,2))})
Path('minkowski-small-growth.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
