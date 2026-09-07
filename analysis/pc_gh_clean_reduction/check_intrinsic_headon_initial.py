#!/usr/bin/env python3
"""Check equal-bare-mass native head-on data with complex-step gradients."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
from intrinsic_restart import read_restart
p=argparse.ArgumentParser();p.add_argument('restart',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
d=read_restart(a.restart,allow_refinement=True,allow_outflow=True);assert d['time']==0
assert d['header']['problem']['pgen_name']=='intrinsic_headon'
mass=float(d['header']['problem']['mass']);sep=float(d['header']['problem']['separation'])
ng,nx,ny,nz=d['mb'][:4];error=0.
def w(coords):
 x,y,z=coords;psi=1+mass/4/np.sqrt((x-sep/2)**2+y*y+z*z)+mass/4/np.sqrt((x+sep/2)**2+y*y+z*z)
 return psi**-2
for m,loc in enumerate(d['locations']):
 h=d['domain'][6:9]/2.**(loc[3]-d['root_level'])
 xyz=[d['domain'][axis]+(loc[axis]*[nx,ny,nz][axis]+np.arange([nx,ny,nz][axis])+.5)*h[axis] for axis in range(3)]
 z,y,x=np.meshgrid(xyz[2],xyz[1],xyz[0],indexing='ij');xyz=[x,y,z]
 u=d['state'][m,:,ng:ng+nz,ng:ng+ny,ng:ng+nx];expected=np.zeros_like(u);expected[0]=w(xyz);expected[1]=1
 for axis in range(3):
  v=[q.astype(complex) for q in xyz];v[axis]+=1e-30j;expected[20+axis]=expected[23+axis]=w(v).imag/1e-30
 error=max(error,float(np.max(abs(u-expected))))
assert error<3e-15,error
result=dict(status='PASS_INITIAL_ONLY',maximum_state_error=error,blocks=len(d['locations']),time=d['time'],restart=str(a.restart))
with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
