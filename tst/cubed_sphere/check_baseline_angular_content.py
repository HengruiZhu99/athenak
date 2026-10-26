#!/usr/bin/env python3
"""Independent quadrature of the baseline's initial lapse angular fraction."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('parameters',type=Path);p.add_argument('output',type=Path)
a=p.parse_args()
width=None
for line in a.parameters.read_text().splitlines():
    if line.split('=')[0].strip()=='pulse_width':width=float(line.split('=')[1].split()[0])
assert width is not None
results={}
for n in [64,128,256]:
    x,w=np.polynomial.legendre.leggauss(n);r=(x+1)/2;w=w/2
    envelope=(1-r*r)**4*np.exp(-r*r/width**2)
    radial=w*r*r*envelope**2
    variance=.04*r*r/3+.09*r**4/15
    analytic=float(np.sum(radial*variance)/np.sum(radial*(1+variance)))
    mu,wm=np.polynomial.legendre.leggauss(16)
    phi=2*np.pi*np.arange(32)/32
    nx=np.sqrt(1-mu[:,None]**2)*np.cos(phi)[None,:]
    ny=np.sqrt(1-mu[:,None]**2)*np.sin(phi)[None,:]
    nz=np.broadcast_to(mu[:,None],nx.shape)
    multiplier=1+.2*r[:,None,None]*nx+.3*r[:,None,None]**2*ny*nz
    weights=wm[None,:,None]/64
    mean=np.sum(weights*multiplier,axis=(1,2))
    angular=np.sum(weights*(multiplier-mean[:,None,None])**2,axis=(1,2))
    total=np.sum(weights*multiplier**2,axis=(1,2))
    numerical=float(np.sum(radial*angular)/np.sum(radial*total))
    assert abs(numerical-analytic)<1e-14
    results[str(n)]={'analytic_angular_moments':analytic,'independent_angular_quadrature':numerical}
a.output.write_text(json.dumps({'width':width,'results':results,
    'parameter_sha256':hashlib.sha256(a.parameters.read_bytes()).hexdigest(),
    'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    'scope':'Initial lapse perturbation only, coordinate-volume integral over unit sphere. No evolved-state or stability claim.',
    'required_fraction':.5,'meets_angular_requirement':results['256']['independent_angular_quadrature']>=.5},indent=2)+'\n')
print(results)
