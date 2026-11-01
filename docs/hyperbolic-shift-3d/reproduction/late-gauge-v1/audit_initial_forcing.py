"""Compare initial lapse/shift forcing from exact stored initial values.

Initial B, D and Lambda vanish. The advection terms are identical between
variants and cancel from differences, so no estimated spatial derivative is
needed. This is not an audit of B's RHS, which differs between IT1 and DO1.
"""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
path=ROOT/'output/hyperbolic-core-shell/resolution-20m-v1/archive/retired-before-n32-ac1/inputs/puncture-continuous-n16.txt'
sha=hashlib.sha256(path.read_bytes()).hexdigest();assert sha=='35c1b65aeaaefac34552f9a63b279cabe839b09326adc9af1d2c767ecc29c8ca'
coords_path=ROOT/'output/hyperbolic-core-shell/inner-telegrapher-v1/coordinates-n16.csv';coords=np.genfromtxt(coords_path,delimiter=',',names=True);r=coords['r'];w=coords['volume'];x=np.column_stack([coords[k] for k in ['x','y','z']]);a=np.loadtxt(path,skiprows=1)
assert a.shape==(len(r),31) and np.max(abs(a[:,15:18]))==0 and np.max(abs(a[:,19:22]))==0 and np.max(abs(a[:,25:28]))==0
O=(1-r*r)/2;A=(1+r*r)/2;alpha=a[:,18]+A;K=a[:,13]-3;beta=np.sum((a[:,22:25]-x)*x,axis=1)/r
assert np.all(O>0) and np.all(alpha>0)
def smooth(t):
 t=np.clip(t,0,1);y=np.minimum(t,1-t);v=y**5*(126+y*(-420+y*(540+y*(-315+70*y))));return np.where(t>.5,1-v,v)
t=np.clip((r-.8)/.1,0,1);collar=t**8/(t**8+(1-t)**8);F=alpha**2+4*O**2*alpha;xi=1+2*(1-r*r)**4;da=alpha-A
outer_lapse=r*r+(-F*(K+3)-2*xi*alpha*da+16*O**4*r*r*da/A+r*(alpha*beta+A*r))/O
inner_lapse=-2*alpha*K+beta*alpha*r/O
outer_shift=-r-4*collar*(beta+r)/O;inner_shift=-6*beta
forcing={};fig,axes=plt.subplots(1,3,figsize=(14,4));radius,ids=np.unique(np.round(r,12),return_index=True);ids=ids[(radius>.3)&(radius<.95)];ids=ids[np.argsort(r[ids])]
axes[0].plot(r[ids],K[ids],color='k');axes[0].set_ylabel('Initial K (physical)')
for label,begin,end,color in [('IT1',.2,.56,'#777777'),('DO1/DG1',.2,.9,'#b55315'),('LG1',.4,.9,'#228844')]:
 s=smooth((end-r)/(end-begin));lapse=s*inner_lapse+(1-s)*outer_lapse;shift=s*inner_shift+(1-s)*outer_shift;forcing[label]=(lapse,shift)
 axes[1].plot(r[ids],lapse[ids],label=label,color=color);axes[2].plot(r[ids],shift[ids],label=label,color=color)
axes[1].set_ylabel('Initial lapse rate minus common advection');axes[2].set_ylabel('Initial radial shift rate minus common advection')
for ax in axes:
 ax.set_xlabel('Compactified radius');ax.grid(alpha=.2);ax.axvspan(.4,.7,color='k',alpha=.05)
axes[1].legend();axes[2].legend();fig.suptitle('Same wormhole data: extending the inner gauge changes forcing across the slice transition');fig.tight_layout();fig.savefig(HERE/'initial-forcing.png',dpi=150)
result=dict(initial_sha256=sha,coordinates_sha256=hashlib.sha256(coords_path.read_bytes()).hexdigest(),scope='Exact initial lapse/shift RHS differences; common advection cancels. Not B RHS or an evolution prediction.',coordinate_time_units=True,region=[.4,.7],comparison={})
m=(r>.4)&(r<.7)&(w>0)
for label,(lapse,shift) in forcing.items():
 result['comparison'][label]={}
 for name,values,reference in [('lapse',lapse,forcing['IT1'][0]),('radial_shift',shift,forcing['IT1'][1])]:
  delta=values-reference;result['comparison'][label][name]=dict(difference_from_IT1_rms=float(np.sqrt(np.sum(w[m]*delta[m]**2)/np.sum(w[m]))),maximum_absolute_difference=float(np.max(abs(delta[m]))))
result['profiles']=dict(radius=r[ids].tolist(),physical_K=K[ids].tolist(),forcing={k:dict(lapse=v[0][ids].tolist(),radial_shift=v[1][ids].tolist()) for k,v in forcing.items()})
(HERE/'initial-forcing.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['comparison'],indent=2))
