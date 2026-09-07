#!/usr/bin/env python3
"""Independent smooth exact-ghost checks for the interface diagnostic oracle."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
from analyze_intrinsic_interface_budget import block_budget,qfield
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
rows=[]
for n in (16,32,64):
 h=np.array([1/n,1.3/n,1.7]);ng=4
 y,x=np.meshgrid((np.arange(n+8)-ng+.5)*h[1],(np.arange(n+8)-ng+.5)*h[0],indexing='ij');phase=(2*np.pi*(x+y/1.3))[None];v=np.arange(50)[:,None,None,None]
 u=(v<2)+.001*np.sin(phase+.17*v)/(1+.03*v);wave=np.array([2*np.pi,2*np.pi/1.3,0]);jet=np.array([.001*w*np.cos(phase+.17*v)/(1+.03*v) for w in wave])
 observed,exact=block_budget(u,u,jet,h,ng,(1,n,n),6)
 row={k:float(np.sqrt(np.mean((observed[k]-exact[k])**2))) for k in exact}
 q1=qfield(u.astype(complex)+1e-30j*jet[0]).imag/1e-30;q2=qfield(u.astype(complex)+1e-20j*jet[0]).imag/1e-20
 assert np.max(abs(q1-q2))<2e-17
 rows.append(row)
rates={k:np.log2(np.array([r[k] for r in rows[:-1]])/[r[k] for r in rows[1:]]).tolist() for k in rows[0]};assert all(min(v)>5.8 for v in rates.values())
result=dict(status='PASS',errors=rows,rates=rates,scope='smooth exact-ghost FD6 convergence and complex-step scale consistency; no mesh transfer')
(a.output/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
