#!/usr/bin/env python3
"""Check Kerr profile reconstruction against independent value derivatives."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
import numpy as np
import mpmath as mp
from scipy.special import roots_jacobi
from validate_kerr import compare
from run_kerr_reference import sampled
from assess_ringdown_rhs import reference_velocity
mp.mp.dps=50
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path);a=p.parse_args();E=a.evidence.resolve();output=a.output or E/'kerr-reference-sampling-instrument.json'
if output.exists():raise RuntimeError('Existing sampler evidence is immutable')
def W(r,z):return (1+mp.mpc(0,.2))*(1+mp.mpf('.01')*r+mp.mpf('.00002')*r**3)*(1+mp.mpf('.1')*z+mp.mpf('.05')*z*z)
def V(r,z):return mp.mpc('-.3','.1')*(1+mp.mpf('.02')*r*r)*(1-mp.mpf('.1')*z)
def phi(x,y,z):
 r=mp.sqrt(x*x+y*y+z*z);return 2*mp.re((x+1j*y)**2/(r*r)*W(r,z/r))
rho=np.linspace(.04,24,40);z,_=roots_jacobi(8,2,2);dtype=[(name,float) for name in ['rho','z','W_real','W_imag','v_real','v_imag','Dr_real','Dr_imag']];profile=np.empty(len(rho)*len(z),dtype=dtype)
for i,r in enumerate(rho):
 for j,angle in enumerate(z):
  rr=mp.mpf(str(r));zz=mp.mpf(str(angle));w=W(rr,zz);v=V(rr,zz);dr=mp.diff(lambda q:W(q,zz),rr);profile[i*len(z)+j]=(r,angle,float(w.real),float(w.imag),float(v.real),float(v.imag),float(dr.real),float(dr.imag))
rng=np.random.default_rng(20261017);xyz=rng.normal(size=(96,3));xyz*=rng.uniform(1.3,23.9,size=96)[:,None]/np.linalg.norm(xyz,axis=1)[:,None];data=np.zeros((len(xyz),12));data[:,:3]=xyz;data[:,3]=1
for i,point in enumerate(xyz):
 x,y,z=map(lambda q:mp.mpf(str(q)),point);r=mp.sqrt(x*x+y*y+z*z);data[i,4]=float(phi(x,y,z));data[i,5]=float(2*mp.re((x+1j*y)**2/(r*r)*V(r,z/r)))
 data[i,9:12]=[float(mp.diff(lambda q:phi(q,y,z),x)),float(mp.diff(lambda q:phi(x,q,z),y)),float(mp.diff(lambda q:phi(x,y,q),z))];data[i,6:9]=data[i,9:12]
path=E/'validation-tools-before-kerr-sampler/validate_kerr.py';spec=importlib.util.spec_from_file_location('preserved_kerr_validator',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);old=module.compare(data,profile);default=compare(data,profile);higher=compare(data,profile,'lagrange6');velocity=reference_velocity(data,profile,True,'lagrange6');velocity_error=float(np.max(abs(velocity-data[:,5])));r=np.linalg.norm(xyz,axis=1);angles=xyz[:,2]/r;s=sampled(profile,r,angles);expected=np.array([complex(W(mp.mpf(str(rr)),mp.mpf(str(zz)))) for rr,zz in zip(r,angles)]);sample_error=float(np.max(abs(s['W']-expected)));maximum=max(e['Linf'] for e in higher.values());passed=default==old and maximum<=2e-13 and velocity_error<=2e-13 and sample_error<=2e-13
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();output.write_text(json.dumps(dict(passed=passed,default_PCHIP_errors_bitwise_equal=default==old,higher_order_errors=higher,actual_RHS_reference_velocity_max_error=velocity_error,independent_profile_sampler_max_error=sample_error,tolerance=2e-13,interpolator_sha256=sha(Path(__file__).with_name('reference_interpolation.py')),source_sha256=sha(Path(__file__)),validator_sha256={name:sha(Path(__file__).with_name(name)) for name in ['validate_kerr.py','run_kerr_reference.py','assess_ringdown_rhs.py']},scope='Non-QNM complex radial/angular polynomial. Cartesian gradients use independent50-digit differentiation of the real scalar value function at96 off-grid points. Preserve default sampler behavior; verify six-point field, spatial-gradient, velocity and angular reconstruction before any nonzero Kerr evolution.'),indent=2)+'\n');print(json.dumps(dict(passed=passed,maximum_error=maximum,velocity_error=velocity_error,sample_error=sample_error),indent=2))
if not passed:raise SystemExit(1)
