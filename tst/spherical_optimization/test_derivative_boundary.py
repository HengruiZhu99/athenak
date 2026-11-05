#!/usr/bin/env python3
"""Consistency tests for an experimentally unstable boundary policy.

Passing these identities does not admit the policy: v7/v8 evidence contains a
resolution-growing boundary eigenmode. Retain it as a rejected control.
"""
import argparse,json,warnings
from pathlib import Path
import numpy as np
from scipy.linalg import expm
from kernel import Kernel
from evolution import DerivativeBoundaryEvolution,weights
from audit_semidiscrete import exponential

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 k=Kernel(a.library,separated=True);e=DerivativeBoundaryEvolution(k,64,normal_policy='analytic',reference_jets=True)
 z=(e.r[-7:]-1)/e.dr[-1];errors=[]
 for degree in range(8):
  y=(z/6)**degree
  exactd=np.zeros(3) if degree==0 else degree/6*(z[-3:]/6)**(degree-1)
  exactdd=np.zeros(3) if degree<2 else degree*(degree-1)/36*(z[-3:]/6)**(degree-2)
  d=np.array([weights(z,t,1)@y for t in z[-3:]])
  dd=np.array([weights(z,t,2)@y for t in z[-3:]])
  correction=exactd[-1]-d[-1]
  errors.append(max(np.max(abs(d+e.hermite_d*correction-exactd)),np.max(abs(dd+e.hermite_dd*e.dr[-1]*correction-exactdd))))
 assert max(errors)<1e-10
 q=e.initial.copy();r=e.r
 # Smooth, off-constraint perturbations include changing geometry and D.
 for f in range(10):
  if f not in (6,7):q[:,f]+=(f+1)*1e-6*np.exp(-((r-.96)/.1)**2)
 rate=e.rhs(q);x=e.last.copy()
 assert x['q'][-1,11]==q[-1,9]
 assert abs(x['clock'])<1e-10
 assert abs(x['dependent_D']-q[-1,9])<1e-10
 assert rate[-1,9]==x['auxiliary_trace_pde']
 null_tangent=rate[-1,7]+.5*np.sqrt(x['q'][-1,0]/x['q'][-1,1])*(rate[-1,0]/x['q'][-1,0]-rate[-1,1]/x['q'][-1,1])
 assert abs(null_tangent)<1e-12
 probes=[]
 for eps in (1e-6,3e-7,1e-7):
  e.rhs(q+eps*rate);plus=e.last['dependent_D'];e.rhs(q-eps*rate);minus=e.last['dependent_D']
  probes.append(abs((plus-minus)/(2*eps)-rate[-1,9]))
 # This differentiated identity is enforced by the derivative solve; it does
 # not independently establish compatibility with outgoing continuum data.
 assert max(probes)<1e-6
 rng=np.random.default_rng(537);A=rng.normal(size=(20,20))/5
 with warnings.catch_warnings(record=True) as numerical_warnings:
  warnings.simplefilter('always');scipy_result=expm(A)
 assert np.isfinite(scipy_result).all()
 exponential_error=np.linalg.norm(exponential(A)-scipy_result)/np.linalg.norm(scipy_result)
 assert exponential_error<1e-12
 out=dict(hermite_polynomial_errors=errors,clock=float(x['clock']),D_algebraic_mismatch=float(x['dependent_D']-q[-1,9]),D_pde_rate=float(rate[-1,9]),null_rate_residual=float(null_tangent),differentiated_identity_errors=probes,exponential_vs_scipy_relative_error=float(exponential_error),scipy_warnings=[str(w.message) for w in numerical_warnings],qualification='Consistency identities only. Policy rejected by resolution-growing boundary spectrum and 1M regularity errors.')
 (a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out));k.close()
if __name__=='__main__':main()
