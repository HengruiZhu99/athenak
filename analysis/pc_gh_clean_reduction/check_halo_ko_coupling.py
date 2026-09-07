#!/usr/bin/env python3
"""Include reconstructed auxiliary ghosts in BOTH centered D and KO consumers."""
import argparse
import json
import math
from pathlib import Path
import numpy as np
import sympy as sp
from check_short_halo_closure import weights

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
rows=[]
for order in [2,4,6]:
 for point in range(order+1):
  assert sum(sp.finite_diff_weights(1,list(range(order+1)),point)[1][-1])==0
 radius_exact=order//2
 exact=sp.finite_diff_weights(1,list(range(-radius_exact,radius_exact+1)),0)[1][-1]
 assert sum(c*(-1 if j%2 else 1) for j,c in zip(range(-radius_exact,radius_exact+1),exact))==0
 m_exact=radius_exact+1
 assert sum((-1 if j%2 else 1)*math.comb(2*m_exact,m_exact+j) for j in range(-m_exact,m_exact+1))==0
 for block_n in [8,16,32,64]:
  n=2*block_n;h=1/n;radius=order//2;identity=np.eye(n)
  offsets=list(range(-radius,radius+1));coefficients=weights(offsets,0)
  derivative=np.zeros((n,n))
  for i in range(n):
   for offset,c in zip(offsets,coefficients):derivative[i,(i+offset)%n]+=c/h
  for epsilon in [0.,.3]:
   m=radius+1
   ko_coeff=[-epsilon/h*(-1.)**j*math.comb(2*m,m+j)/4**m for j in range(-m,m+1)]
   ko=np.zeros((n,n))
   for i in range(n):
    for off,c in zip(range(-m,m+1),ko_coeff):ko[i,(i+off)%n]+=c
   kd=np.zeros((n,n));kk=np.zeros((n,n))
   for block in range(2):
    origin=block*block_n
    for i in range(block_n):
     for off in range(-m,m+1):
      ghost=i+off
      if 0<=ghost<block_n:continue
      start=max(-4,min(ghost-radius,block_n+4-order-1))
      nodes=list(range(start,start+order+1));row=np.zeros(n)
      for node,value in zip(nodes,weights(nodes,ghost)):row[(origin+node)%n]+=value/h
      correction=row-derivative[(origin+ghost)%n]
      if abs(off)<=radius:kd[origin+i]+=coefficients[off+radius]/h*correction
      kk[origin+i]+=ko_coeff[off+m]*correction
   for reconstructed in [False,True]:
    a=np.block([[ko,identity,0*identity],
                [kd if reconstructed else 0*kd,ko,derivative],
                [derivative+(kk if reconstructed else 0*kk),derivative,ko-identity]])
    eigenvalues=np.linalg.eigvals(a)
    dt=.2*h;z=dt*eigenvalues
    amplification=1+z+z*z/2+z*z*z/6
    normalized=float(eigenvalues.real.max()/(1+np.linalg.norm(a,2)))
    rk_radius=float(abs(amplification).max())
    row=dict(order=order,block_n=block_n,ko=epsilon,reconstructed=reconstructed,
             max_real_eigenvalue=float(eigenvalues.real.max()),
             normalized_positive_rate=normalized,rk_spectral_radius=rk_radius,
             tangency_defect_norm=float(np.linalg.norm(kk,2)) if reconstructed else 0,
             status='FAIL' if normalized>1e-10 or rk_radius>1+1e-9 else 'PASS',
             scope='necessary same-level periodic TT test; includes KO ghost correction')
    # Quotient exact neutral h/v modes rather than interpreting sqrt(eps)
    # splitting of their physical zero-frequency Jordan block as instability.
    # D*1=KO*1=KD*1=KK*1=0 follows from stencil moments. Without KO,
    # the global centered D also annihilates the checkerboard when KD=0.
    neutral=[np.ones(n)]
    if epsilon==0 and (order<6 or not reconstructed):
     neutral.append((-1.)**np.arange(n))
    modes=np.stack(neutral,axis=1);count=len(neutral)
    q=np.linalg.qr(modes,mode='complete')[0][:,count:]
    transform=np.zeros((3*n,3*n-2*count))
    transform[:n,:n-count]=q
    transform[n:2*n,n-count:2*(n-count)]=q
    transform[2*n:,2*(n-count):]=identity
    span=np.zeros((3*n,2*count));span[:n,:count]=modes;span[n:2*n,count:]=modes
    jordan=np.zeros((2*count,2*count));jordan[:count,count:]=np.eye(count)
    invariance=float(np.linalg.norm(np.einsum('ij,jk->ik',a,span,optimize=False)
                     -np.einsum('ij,jk->ik',span,jordan,optimize=False))/(1+np.linalg.norm(a,2)))
    assert invariance<1e-12
    reduced=np.einsum('ij,jk->ik',np.einsum('ij,jk->ik',transform.T,a,optimize=False),
                       transform,optimize=False)
    assert np.isfinite(reduced).all()
    smooth=np.sin(2*np.pi*(np.arange(n)+.5)*h)
    row['tangency_defect_smooth_max']=float(abs(np.einsum('ij,j->i',kk,smooth,optimize=False)).max()) if reconstructed else 0.0
    quotient_eigenvalues=np.linalg.eigvals(reduced)
    zz=dt*quotient_eigenvalues
    quotient_radius=float(abs(1+zz+zz*zz/2+zz*zz*zz/6).max())
    quotient_rate=float(quotient_eigenvalues.real.max()/(1+np.linalg.norm(reduced,2)))
    row.update(neutral_subspace_dimension=2*count,neutral_invariance_error=invariance,
               quotient_max_real_eigenvalue=float(quotient_eigenvalues.real.max()),
               quotient_normalized_positive_rate=quotient_rate,
               quotient_rk_spectral_radius=quotient_radius,
               quotient_status='FAIL' if quotient_rate>1e-10 or quotient_radius>1+1e-9 else 'PASS')
    rows.append(row)
    print(json.dumps(row),flush=True)
    args.output.write_text(json.dumps(rows,indent=2)+'\n')
