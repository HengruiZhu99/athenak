#!/usr/bin/env python3
"""Verify the exact sinusoidal four-cell-average bias in actual coarse ghosts."""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
from intrinsic_restart import read_restart
p=argparse.ArgumentParser();p.add_argument('--folders',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
for folder in a.folders:
 data=read_restart(sorted((folder/'fd6-2d-none').glob('rst/*.rst'))[0],allow_refinement=True);ng,nx,ny,nz=data['mb'][:4];length=data['domain'][3:6]-data['domain'][:3];wave=2*np.pi/length;maximum=0.;count=0;bias=0.;fine=[]
 assert nz==1 and np.all(data['domain'][:3]==0)
 for loc in data['locations']:
  if loc[3]==data['root_level']:continue
  h=data['domain'][6:9]/2**int(loc[3]-data['root_level']);lo=loc[:3]*np.array([nx,ny,nz])*h;fine.append((lo,lo+np.array([nx,ny,nz])*h))
 for m,loc in enumerate(data['locations']):
  if loc[3]!=data['root_level']:continue
  h=data['domain'][6:9];x,y=np.meshgrid((loc[0]*nx+np.arange(nx+2*ng)-ng+.5)*h[0],(loc[1]*ny+np.arange(ny+2*ng)-ng+.5)*h[1]);x%=length[0];y%=length[1];mask=np.zeros(x.shape,bool)
  for lo,hi in fine:mask|=(x>=lo[0])&(x<hi[0])&(y>=lo[1])&(y<hi[1])
  if not mask.any():continue
  v=np.arange(50)[:,None,None];osc=.001*np.sin(wave[0]*x+wave[1]*y+.17*v)/(1+.03*v);factor=np.cos(wave[0]*h[0]/4)*np.cos(wave[1]*h[1]/4);pred=(v<2)+factor*osc
  maximum=max(maximum,float(np.max(abs(data['state'][m,:,0][:,mask]-pred[:,mask]))));bias=max(bias,float(np.max(abs((factor-1)*osc[:,mask]))));count+=int(mask.sum())
 assert count>0 and maximum<5e-16
 rows.append(dict(block_n=int(nx),cells=count,average_prediction_max_error=maximum,analytic_bias_maximum=bias))
with a.output.open('x') as f:json.dump(rows,f,indent=2);f.write('\n')
print(json.dumps(rows,indent=2))
