"""Native eight-field radial weighted chart and stateless weak endpoint stencil.

No added evolved moments or gradients. The stencil currently admits only the
Minkowski-leading weak indicial kernel and smooth separate nonreference controls;
it is not a general live-leading nonlinear/3D endpoint method.
"""
import math
from functools import lru_cache
import numpy as np
import mpmath as mp
import goal1_native_radial_scri as native
SIGMA=40/43
# Weighted order: chi,g_rr,alpha,beta_r,Omega*A_rr,P,Theta,Omega*Lambda_r.
NATIVE_ROWS=[0,6,1,4,8,2,3,5]
WEIGHTED=[4,7]

def kernel(s=SIGMA):
 return np.array([-.5,1,-1.5,3,-s,1.5*(3-s),0,-2*s])

def reference(r):
 q=native.api().reference(r);out=q[:,0,NATIVE_ROWS].copy();o=native.omega(r);out[:,WEIGHTED]*=o[:,None];return out

def from_native(r,v):
 y=v[:,NATIVE_ROWS].copy();y[:,WEIGHTED]*=native.omega(r)[:,None];return y

def native_jets(r,z,zo,zoo):
 """Exact continuum pullback of reconstructed weighted offsets; positiveOmega.

z,zo,zoo are temporary interpolation jets, never Cauchy variables. Chain rules
apply to the interpolated function, not an assumed discrete derivative identity.
 """
 r=np.asarray(r);o=native.omega(r);assert np.all(r>=.95)and np.all(o>0)
 ref=native.api().reference(r);v,d,dd=[ref[:,k,:].copy()for k in range(3)]
 for k,row in enumerate(NATIVE_ROWS):
  value=z[:,k];first=zo[:,k];second=zoo[:,k]
  if k in WEIGHTED:
   value=z[:,k]/o;first=zo[:,k]/o-z[:,k]/o**2;second=zoo[:,k]/o-2*zo[:,k]/o**2+2*z[:,k]/o**3
  v[:,row]+=value;d[:,row]+=-2*r*first;dd[:,row]+=4*r*r*second-2*first
 return algebraic_jets(v,d,dd)

def algebraic_jets(v,d,dd):
 a=v[:,6];ar=d[:,6];arr=dd[:,6];g=a**-.5;gr=-.5*a**-1.5*ar;grr=.75*a**-2.5*ar**2-.5*a**-1.5*arr;v[:,7]=g;d[:,7]=gr;dd[:,7]=grr
 v[:,9]=-.5*g/a*v[:,8];d[:,9]=-.5*((gr/a-g*ar/a**2)*v[:,8]+g/a*d[:,8]);return v,d,dd

def regular_endpoint_jets(a0,a1,a2):
 """Native jets of the regular part only atOmega=0, with live supplied a0.

A/Lambda may have a removed fractional divergent native part. No boundedness of
those native fields is inferred from this regular-part evaluation.
 """
 assert max(abs(a0[WEIGHTED]))<1e-25
 ref=native.api().reference(np.array([1.]));v,d,dd=[ref[:,k,:].copy()for k in range(3)]
 for k,row in enumerate(NATIVE_ROWS):
  if k in WEIGHTED:v[0,row]+=a1[k];d[0,row]+=-2*a2[k]
  else:v[0,row]+=a0[k];d[0,row]+=-2*a1[k];dd[0,row]+=8*a2[k]-2*a1[k]
 return algebraic_jets(v,d,dd)

@lru_cache(None)
def reconstruction_weights(nodes):
 """Coupled native8 least-squares stencil, normalizedOmega nodes in(0,1]."""
 with mp.workdps(80):
  n=len(nodes);s=mp.mpf(40)/43;g=mp.matrix([-.5,1,-1.5,3,-s,mp.mpf(3)/2*(3-s),0,-2*s]);g=g/mp.norm(g);A=mp.matrix(8*n,41)
  for i,node in enumerate(nodes):
   x=mp.mpf(node)
   for field in range(8):
    for power in range(1,5):A[8*i+field,8*(power-1)+field]=x**power
    A[8*i+field,32]=x**s*g[field];A[8*i+field,33+field]=x**(s+1)
  gram=A.T*A;B=gram**-1*A.T;res=mp.norm(B*A-mp.eye(41),p=mp.inf)
 return np.array(B.tolist(),float),float(res),np.array(A.tolist(),float)

def reconstruct(r,z,a0):
 o=native.omega(np.asarray(r));assert len(r)==8 and np.all(o>0);scale=float(max(o));nodes=tuple(float(v/scale)for v in o);W,res,A=reconstruction_weights(nodes);b=(z-a0[None,:]).reshape(-1);coef=np.einsum('ij,j->i',W,b);fit=np.einsum('ij,j->i',A,coef);return {'a0':np.array(a0),'a1':coef[:8]/scale,'a2':coef[8:16]/scale**2,'coefficients':coef,'Omega_scale':scale,'weighted_fit_residual':float(max(abs(fit-b))),'MP_left_inverse_residual':res,'double_weight_norm':float(np.linalg.norm(W)),'design_condition':float(np.linalg.cond(A))}

def endpoint_action(r,z,a0):
 fit=reconstruct(r,z,a0);v,d,dd=regular_endpoint_jets(fit['a0'],fit['a1'],fit['a2']);f,meta,end=native.api().source(np.array([1.]),v,d,dd);rates=f[:,NATIVE_ROWS].copy();rates[:,WEIGHTED]=0
 return rates[0],meta[0],fit
