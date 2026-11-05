#!/usr/bin/env python3
"""Local smooth-data constraint propagation through the unchanged shared RHS.

Initial jets use the map ODE and high-precision analytic differentiation.
Only derivatives of the compiled RHS use local finite differences. There is
no scri closure, evolution grid, KO, boundary penalty or background subtraction.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import mpmath as mp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq
from kernel import Kernel
from evolution import weights
from physical_constraints import evaluate

def local_jets(k,center,offsets,perturbed):
 mp.mp.dps=50;m=mp.mpf(1)/3
 native,_,om=k.initial(np.array([center]));y0=float(om[0,0]*np.sqrt(native[0,0])/center)
 def switch(r,y):
  if k.compact_config[0]:
   a=mp.mpf(float(k.config[2]));b=mp.mpf(float(k.compact_config[0]));p=mp.mpf(float(k.compact_config[1]));z=(r-a)/(b-a)
   if z<=0:return mp.mpf(0)
   if z>=1:return mp.mpf(1)
   return z**p*sum(mp.rf(p,j)*(1-z)**j/mp.factorial(j) for j in range(7))
  z=(1/y-20*m)/(20*m)
  if z<=0:return mp.mpf(0)
  if z>=1:return mp.mpf(1)
  return z**7*(1716+z*(-9009+z*(20020+z*(-24024+z*(16380+z*(-6006+924*z))))))
 def rhs(r,y):return -mp.sqrt(y*y-2*m*y**3+switch(r,y)**2)/r
 def fields(r,y):
  S=switch(r,y);F=mp.sqrt(y*y-2*m*y**3+S*S);o=(1-r*r)/2
  Sp=mp.diff(lambda x:switch(x,y),r)+mp.diff(lambda z:switch(r,z),y)*rhs(r,y)
  ar=-2*r*Sp*y/(3*o*F);K=-3*S-r*Sp*y/F
  alpha=o*F/y;beta=-r*S
  if perturbed:alpha*=1+mp.mpf('.1')*r*r;beta+=mp.mpf('.03')*r**3
  return [(r*y/o)**2,mp.mpf(1),mp.mpf(1),ar,-ar/2,K,mp.mpf(0),mp.mpf(0),alpha,beta,mp.mpf(0),mp.mpf(0)]
 nodes=center+offsets
 def integrate(end):
  return solve_ivp(lambda r,y:[float(rhs(mp.mpf(r),mp.mpf(y[0])))],(center,end),[y0],method='DOP853',rtol=3e-14,atol=3e-16,max_step=max(abs(end-center)/8,1e-8),dense_output=True)
 left=integrate(float(nodes.min()));right=integrate(float(nodes.max()));assert left.success and right.success
 q=[];d=[];dd=[]
 for r0 in nodes:
  y=mp.mpf(float((left if r0<center else right).sol(r0)[0]));r=mp.mpf(float(r0));yp=rhs(r,y);ypp=mp.diff(lambda x:rhs(x,y),r)+mp.diff(lambda z:rhs(r,z),y)*yp
  def path(t):return fields(r+t,y+yp*t+ypp*t*t/2)
  q.append([float(x) for x in path(0)]);d.append([float(mp.diff(lambda t:path(t)[j],0)) for j in range(12)]);dd.append([float(mp.diff(lambda t:path(t)[j],0,2)) for j in range(12)])
 return np.array(q),np.array(d),np.array(dd)

def physical(r,q,d,dd,om):
 # This local probe has no scri node; append a discarded duplicate for the
 # existing physical diagnostic's explicit endpoint-exclusion convention.
 return evaluate(np.r_[r,r[-1]],np.vstack((q,q[-1])),np.vstack((d,d[-1])),np.vstack((dd,dd[-1])),np.vstack((om,om[-1])))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--library',required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--radii',nargs='+',type=float,default=[35.]);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False);rows=[]
 for family,end in [('areal',0),('compact',.97)]:
  k=Kernel(a.library,separated=True,compact_end=end)
  for R_M in a.radii:
   def radius(r):
    q,_,o=k.initial(np.array([r]));return 3*r/(o[0,0]*np.sqrt(q[0,0]))
   center=brentq(lambda r:radius(r)-R_M,k.config[2]+1e-5,.999,xtol=1e-14)
   assert center-.0015>k.config[8],'Probe must lie in the complete outer gauge and native conformal frame'
   for perturbed in [False,True]:
    for step in [.0005,.00025,.000125]:
     nodes=np.ascontiguousarray(center+step*np.arange(-3,4));q,d,dd=local_jets(k,center,step*np.arange(-3,4),perturbed);native,source,om=k.initial(nodes)
     reg,pole,_,_=k.parts(nodes,q,d,dd,source);rate=reg+pole/om[:,:1]
     drate=weights(nodes,center,1)@(rate-rate[3]);ddrate=weights(nodes,center,2)@(rate-rate[3]);epsilon=1e-25
     propagation=physical(nodes[3:4],q[3:4]+1j*epsilon*rate[3:4],d[3:4]+1j*epsilon*drate[None],dd[3:4]+1j*epsilon*ddrate[None],om[3:4]).imag[0]/epsilon
     constraint=physical(nodes,q,d,dd,om)
     row=dict(family=family,reference_R_M=R_M,r=center,perturbed_gauge=perturbed,derivative_step=step,initial_constraint_max=abs(constraint).max(axis=0).tolist(),continuum_constraint_rate=propagation.tolist(),RHS_at_center=rate[3].tolist(),native_embedding_max_difference=float(abs(q[:,:8]-native[:,:8]).max()),stationary_gauge_embedding_difference=None if perturbed else float(abs(q[:,8:10]-native[:,8:10]).max()))
     rows.append(row);print(json.dumps(row),flush=True)
  k.close()
 out=dict(rows=rows,binary_sha256=hashlib.sha256(Path(a.library).read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),qualification='Interior smooth-data local derivative audit. Constraints/rates use physical coordinate units; no boundary or discrete evolution operator is involved. RHS derivative steps expose differentiation error; this is not a complete continuum proof.')
 (a.output/'audit.json').write_text(json.dumps(out,indent=2)+'\n')
if __name__=='__main__':main()
