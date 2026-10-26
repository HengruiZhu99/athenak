#!/usr/bin/env python3
"""Polynomial consistency and interior dissipative sign of biased transport."""
import numpy as np
from numerics import RadialOperators
r=(np.arange(129)+.5)/128.5;op=RadialOperators(r);h=r[1]-r[0]
for power in range(7):
    q=(r**power)[:,None];expected=np.zeros_like(r) if power==0 else power*r**(power-1)
    for beta in [-np.ones(len(r)),np.ones(len(r))]:
        got=op.advection_derivative(q,np.array([(-1)**power]),beta)[:,0]
        np.testing.assert_allclose(got,expected,atol=2e-11,rtol=2e-11)
for side,beta in [(0,-1),(1,1)]:
    i=64;nodes=op.ext[op.aindex[side,i]];coefficients=op.ad1[side,i]
    for frequency in np.linspace(0,np.pi,101):
        symbol=np.sum(coefficients*np.exp(1j*frequency*(nodes-r[i])/h))
        assert beta*symbol.real<2e-10,(side,frequency,symbol)
print('Sixth-order biased transport is polynomial-consistent and has the dissipative Fourier sign in the interior.')
puncture=RadialOperators(r,puncture=True)
assert min(puncture.index.ravel())>=puncture.width
assert min(puncture.aindex.ravel())>=puncture.width
for power in range(7):
    q=r**power
    d1,d2=puncture.derivatives(q,1);e1,e2=puncture.derivatives(q,-1)
    assert np.array_equal(d1,e1) and np.array_equal(d2,e2)
    np.testing.assert_allclose(puncture.integral(q),(r**(power+1)-r[0]**(power+1))/(power+1),atol=2e-15)
print('Puncture stencils never read reflected values; quadrature excludes the puncture.')
