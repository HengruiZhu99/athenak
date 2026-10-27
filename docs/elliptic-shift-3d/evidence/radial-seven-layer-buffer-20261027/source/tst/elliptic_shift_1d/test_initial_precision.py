#!/usr/bin/env python3
"""Initial-data residual arithmetic isolation without relaxing Newton tolerance."""
from scalar_initial_data import solve_data
import numpy as np

a=solve_data(128,.058,1e-18,precision='float')
b=solve_data(128,.058,1e-18,precision='decimal')
assert not a['converged'] and b['converged']
assert b['residual']<1e-18
np.testing.assert_allclose(a['w'],b['w'],rtol=0,atol=2e-14)
assert abs(8*a['w'][-1]-8*b['w'][-1])<1e-13
assert b['rounded_double_residual']>b['residual']
print('50-digit iterate/residual reaches the unchanged tolerance; rounded-double residual is reported separately.')
