#!/usr/bin/env python3
"""Boundary consistency and negative energy of a fixed-width KO taper."""
import json
import numpy as np
from numerics import puncture_dissipation
records=[];rng=np.random.default_rng(7324)
for n in [65,129,257]:
 r=np.linspace(.03125,1,n);h=r[1]-r[0];values=(r**3)[:,None]
 raw=puncture_dissipation(values,r,h,.5,inner_width=0.)
 tapered=puncture_dissipation(values,r,h,.5,inner_width=.1)
 # Both are -B^T W B. At the first active column the unweighted truncated
 # boundary has h^2 consistency; a sixth-power taper changes this to h^8.
 sample=rng.normal(size=(n,3));rate=puncture_dissipation(sample,r,h,.5,inner_width=.1)
 assert np.sum(sample*rate)<0 and not np.any(rate[0])
 records.append(dict(n=n,h=h,raw_first_active_rate=float(abs(raw[1,0])),tapered_first_active_rate=float(abs(tapered[1,0])),energy_rate=float(np.sum(sample*rate))))
raw_orders=np.log2([records[i]['raw_first_active_rate']/records[i+1]['raw_first_active_rate'] for i in range(2)])
orders=np.log2([records[i]['tapered_first_active_rate']/records[i+1]['tapered_first_active_rate'] for i in range(2)])
assert min(raw_orders)>1.9 and max(raw_orders)<2.1,raw_orders
assert min(orders)>7.9,orders
print(json.dumps(dict(records=records,untapered_boundary_orders=raw_orders.tolist(),tapered_boundary_orders=orders.tolist(),negative_energy_verified=True,scope='Fixed physical width, nonnegative weight in -B^T W B; boundary consistency audit, not coupled stability')))
