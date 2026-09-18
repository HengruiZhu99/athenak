#!/usr/bin/env python3
"""Compare slice density-peak positions with the prescribed test-particle orbit.

Peak positions are cell samples, not a stellar center-of-mass trajectory.
Requires SciPy. Orbit uses the stationary R0=M trumpet time coordinate.
"""
import argparse
import json
import math
from pathlib import Path
from scipy.integrate import solve_ivp
from make_controls import orbit
from slice_peaks import peaks

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('files',type=Path,nargs='+')
p.add_argument('--output',type=Path,required=True)
a=p.parse_args()
o=orbit();L=o['L'];records=[peaks(f) for f in a.files]
records.sort(key=lambda r:r['time_M'])

def rhs(t,q):
    R,phi=q;r=R-1;f=1-2/R
    ur=-math.sqrt(2/R-f*L*L/(R*R))
    ut=(1+ur/r)/f
    return [ur/ut,L/(R*R*ut)]

sol=solve_ivp(rhs,[0,max(r['time_M'] for r in records)],
              [o['areal_r0_M'],0],dense_output=True,rtol=1e-11,atol=1e-12,max_step=1)
assert sol.success,sol.message
out=[]
for row in records:
    assert row['variable']=='dens'
    R,phi=sol.sol(row['time_M']);r=R-1
    expected=[float(r*math.cos(phi)),float(r*math.sin(phi))]
    observed=row['peaks'][0]['xyz_M'][:2]
    out.append(dict(time_M=row['time_M'],density_peak_xy_M=observed,
                    geodesic_xy_M=expected,xy_distance_M=math.dist(observed,expected),
                    level=row['peaks'][0]['logical_level'],
                    cell_spacing_M=row['peaks'][0]['cell_spacing_M']))
a.output.write_text(json.dumps({'orbit':o,'scope':'Slice density peaks, not centers of mass; star self gravity omitted from reference geodesic','samples':out},indent=2)+'\n')
