#!/usr/bin/env python3
"""Full tensor regular-scri limits against the independent existing radial audit."""
import json,subprocess,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'elliptic_shift_1d'))
from numerics import RadialOperators
from scri_limits import derivatives
import g1_rhs
data=json.loads(subprocess.check_output([sys.argv[1]],text=True))
r=(np.arange(48)+.5)/47.5;op=RadialOperators(r);error=0.
for case in data['radial_cases']:
 q,d,dd=[np.array(case[k]) for k in ('q','d','dd')]
 profiles=q+(r[:,None]-1)*d+(r[:,None]-1)**2*dd/2
 first=d+(r[:,None]-1)*dd;second=np.broadcast_to(dd,profiles.shape)
 regular,*_=g1_rhs.parts(op,profiles,first,second,restoring=1.)
 pole_derivative,_=derivatives(q,d,dd,case['shift_deviation'])
 reference=regular[-1]-pole_derivative
 error=max(error,float(np.max(np.abs(reference-np.array(case['limit'])))))
assert error<3e-12,error
print(json.dumps(dict(cases=len(data['radial_cases']),radial_limit_error=error,
 native_value_error=data['native_value_error'],independent_FD_derivative_error=data['independent_FD_derivative_error'])))
