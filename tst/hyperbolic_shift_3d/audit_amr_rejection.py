#!/usr/bin/env python3
"""Reconstruct the preserved N12 first-regrid chi rejection without evolving it."""
import argparse,json,pathlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--prefix',required=True);p.add_argument('--output',required=True);a=p.parse_args()
n=12
old=np.fromfile(a.prefix+'.failed-stage.bin',dtype=np.float64).reshape(-1,31)
new=np.fromfile(a.prefix+'.rejected-transfer.bin',dtype=np.float64).reshape(-1,31)
leaves=[(j&1,(j>>1)&1,j>>2,2) for j in range(8)]+[(j&1,(j>>1)&1,j>>2,1) for j in range(1,8)]
def sample(x):
    owner=None
    for m,(i,j,k,level) in enumerate(leaves):
        lo=-.18+np.array([i,j,k])*.36/(2**level);h=.36/(2**level)/n
        if np.all(x>=lo-1e-14) and np.all(x<=lo+n*h+1e-14):
            if owner is None or h<owner[2]:owner=(m,lo,h)
    m,lo,h=owner;t=(x-lo)/h-.5;indices=[];weights=[]
    for v in t:
        if abs(v-round(v))<1e-10:indices.append([round(v)]);weights.append([1.]);continue
        start=max(0,min(int(np.floor(v))-2,n-6));ix=list(range(start,start+6));indices.append(ix)
        weights.append([np.prod([(v-k)/(j-k) for k in ix if k!=j]) for j in ix])
    q=old[m*n**3:(m+1)*n**3,0].reshape(n,n,n);anchor=q[indices[2][0],indices[1][0],indices[0][0]];value=anchor
    for k,wk in zip(indices[2],weights[2]):
        for j,wj in zip(indices[1],weights[1]):
            for i,wi in zip(indices[0],weights[0]):value+=wk*wj*wi*(q[k,j,i]-anchor)
    return value
w=np.array([-.02197265625,.205078125,.9228515625,-.123046875,.01708984375])
# Parent root octant (1,0,0), first x / last y,z parent cell. The new
# child target has offsets (-1/4,+1/4,+1/4) in parent-cell units.
rows=[];value=0.
for k in range(5):
    for j in range(5):
        for i in range(5):
            x=np.array([0,-.18,-.18])+(np.array([i-2,j+9,k+9])+.5)*.015
            donor=sample(x);weight=w[i]*w[4-j]*w[4-k];value+=weight*donor
            rows.append(dict(xyz=x.tolist(),chi=float(donor),weight=float(weight)))
actual=float(new[25908,0]);report=dict(target_cell=25908,target_xyz=[.00375,-.00375,-.00375],accepted_core_min_chi=float(old[:15*n**3,0].min()),stencil_min_chi=min(r['chi'] for r in rows),reconstructed_chi=value,actual_chi=actual,absolute_difference=abs(value-actual),stencil=rows)
pathlib.Path(a.output).write_text(json.dumps(report,indent=2)+'\n')
assert abs(value-actual)<1e-13 and actual<0
print({k:v for k,v in report.items() if k!='stencil'})
