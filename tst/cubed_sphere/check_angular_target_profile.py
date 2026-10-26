#!/usr/bin/env python3
"""Prospective pure quadrupolar relative-lapse profile; not an evolution result."""
import json
from pathlib import Path
import numpy as np
A=.2;width=.35;Q=np.array([[1.,0,0],[0,-1.,.3],[0,.3,0]])
eigen=np.linalg.eigvalsh(Q);scale=max(abs(eigen));w2=width**2
# Maximize z/w^2 exp(-z/w^2) (1-z)^4 on 0<=z<=1.
z=2*w2/(1+5*w2+np.sqrt((1+5*w2)**2-4*w2));peak=z/w2*np.exp(-z/w2)*(1-z)**4
mu,mw=np.polynomial.legendre.leggauss(32);phi=2*np.pi*(np.arange(64)+.5)/64
xx=np.sqrt(1-mu[:,None]**2)*np.cos(phi);yy=np.sqrt(1-mu[:,None]**2)*np.sin(phi);zz=np.broadcast_to(mu[:,None],xx.shape)
n=np.stack([xx,yy,zz],axis=-1);harmonic=np.einsum('...i,ij,...j->...',n,Q,n)/scale
mean=np.sum(mw[:,None]*harmonic)/(2*len(phi));second=np.sum(mw[:,None]*harmonic**2)/(2*len(phi))
expected=2*np.trace(np.einsum('ij,jk->ik',Q,Q))/(15*scale**2)
assert abs(mean)<1e-14 and abs(second-expected)<1e-14
assert abs(np.trace(Q))<1e-15 and peak>0
# Independent dense radial check against the analytic maximizer.
r=np.linspace(0,1,100001);radial=(r/width)**2*np.exp(-(r/width)**2)*(1-r*r)**4/peak
assert abs(radial.max()-1)<1e-8
out=dict(status='prospective phase-3 profile only; no native evolution performed',formula='delta_alpha/alpha_hat = A*(x^2-y^2+0.6*y*z)/(lambda_max_abs*w^2)*exp(-r^2/w^2)*(1-r^2)^4/peak',amplitude=A,width=width,Q=Q.tolist(),angular_eigenvalues=eigen.tolist(),angular_normalization=float(scale),radial_peak_radius=float(np.sqrt(z)),radial_normalization=float(peak),maximum_absolute_relative_lapse=A,relative_lapse_range=[float(A*eigen.min()/scale),float(A*eigen.max()/scale)],minimum_alpha_over_reference=1-A,angular_mean=float(mean),angular_mean_square=float(second),analytic_angular_energy_fraction=1.,near_origin='Regular Cartesian quadratic times smooth radial function; no division by r.',near_scri='Relative perturbation is O((1-r^2)^4); no exact-scri evolution or regularity theorem inferred.',acceptance='Exceeds amplitude >=0.1 and angular energy fraction >=0.5. Must still pass ordered native propagation, late settling, refinement and boundary-sensitivity phases.')
p=Path('docs/validation/cubed-sphere-balanced-20261013/angular-target-design');p.mkdir(exist_ok=True);(p/'profile.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
