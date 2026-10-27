#!/usr/bin/env python3
"""Accepted-increment arithmetic against an exact rational accumulation."""
from fractions import Fraction
import json
import numpy as np
from compensated_update import accumulate
state=np.array([.1,-.3,1.]);direct=state.copy();initial=state.copy();correction=np.zeros_like(state)
increment=np.array([1e-18,-2e-18,3e-17]);count=100000
for _ in range(count):
 state,correction=accumulate(state,correction,increment);direct+=increment
exact=np.array([float(Fraction(float(x))+count*Fraction(float(dx))) for x,dx in zip(initial,increment)])
assert np.array_equal(state,exact),(state,exact)
assert max(abs(direct-exact))>1e-12
assert np.all(abs(correction)<=np.spacing(abs(state)))
print(json.dumps(dict(correctly_rounded_rational_constant_increment=True,steps=count,max_direct_error=float(max(abs(direct-exact))),compensated_error=float(max(abs(state-exact))),scope='Arithmetic test only; evolving truncation/roundoff isolation remains required')))
