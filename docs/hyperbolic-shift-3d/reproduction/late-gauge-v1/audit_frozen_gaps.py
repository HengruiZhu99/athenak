"""Scalar characteristic gaps on preserved states; no evolution or global proof.

All coefficients are the direct-outer gauge with either .2 or .4 inner endpoint.
The state label identifies the checkpoint, not the gauge used in this evaluation.
Squared speed differences are divided by Q = gtilde^{ij} n_i n_j > 0.
"""
from pathlib import Path
import hashlib,json
import numpy as np
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
coords_path=ROOT/'output/hyperbolic-core-shell/inner-telegrapher-v1/coordinates-n16.csv'
coords=np.genfromtxt(coords_path,delimiter=',',names=True);r=coords['r'];omega=(1-r*r)/2
cases=[('DO1','direct-outer-v1','n16-direct-outer-fresh10m-v1-segment01','edc58b0df7ef0391567221d8b95990e82ab05aad6264bc87d7e4a49ff2449c5c'),('DG1','damping-ablation-v1','n16-damping-ablation-fresh10m-v1-segment01','b582e457f8a8ba9b143436225bb57318b7e28a241fb495b913eea4787bfa04cf')]
def smooth(t):
 t=np.clip(t,0,1);x=np.minimum(t,1-t);v=x**5*(126+x*(-420+x*(540+x*(-315+70*x))));return np.where(t>.5,1-v,v)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
result=dict(scope='Frozen DO1/DG1 10M states, direct-outer coefficients only; does not establish earlier-time behavior or global hyperbolicity',coordinates_sha256=digest(coords_path),states={})
for label,variant,campaign,sha in cases:
 cp=ROOT/'output/hyperbolic-core-shell'/variant/'aurora/campaigns'/campaign/'output/puncture.checkpoint';assert digest(cp)==sha
 a=np.memmap(cp,dtype='<f8',mode='r',offset=168,shape=(len(r),31));chi=a[:,0];alpha=a[:,18]+(1+r*r)/2
 g=np.zeros((len(r),3,3))
 for f,(i,j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):g[:,i,j]=g[:,j,i]=a[:,1+f]+(i==j)
 assert np.all(np.linalg.eigvalsh(g)>0) and np.all(chi>0) and np.all(alpha>0)
 state=dict(checkpoint_sha256=sha,placements={})
 for begin in [.2,.4]:
  s=smooth((.9-r)/(.9-begin));F=alpha*alpha+4*omega*omega*alpha
  lapse=chi*((1-s)*F+s*2*omega*alpha)+2*s*s*np.maximum(omega*omega*chi,1e-5)
  mu=s+(1-s)*1.5*chi*F
  gaps={'transverse_shift_minus_lapse':mu-lapse,'longitudinal_shift_minus_lapse':4*mu/3-lapse,'lapse_minus_physical':lapse-alpha*alpha*chi}
  regions={}
  for name,lo,hi in [('exterior_bulk',.1,.95),('overlap_neighborhood',.13,.35)]:
   ids=np.flatnonzero((r>lo)&(r<hi));region=dict(radius_interval=[lo,hi],cells=len(ids),gaps={})
   for key,gap in gaps.items():
    k=ids[np.argmin(gap[ids])];j=ids[np.argmin(np.abs(gap[ids]))]
    region['gaps'][key]=dict(minimum=float(gap[k]),minimum_radius=float(r[k]),minimum_cell=int(k),negative_cells=int(np.sum(gap[ids]<0)),zero_cells=int(np.sum(gap[ids]==0)),closest_to_zero=float(gap[j]),closest_radius=float(r[j]))
   regions[name]=region
  state['placements'][str(begin)]=dict(inner_endpoint=begin,outer_endpoint=.9,regions=regions)
 result['states'][label]=state
 assert digest(cp)==sha
(HERE/'frozen-gap-audit.json').write_text(json.dumps(result,indent=2)+'\n')
for label,state in result['states'].items():
 for begin,p in state['placements'].items():print(label,begin,{name:{k:v['minimum'] for k,v in d['gaps'].items()} for name,d in p['regions'].items()})
