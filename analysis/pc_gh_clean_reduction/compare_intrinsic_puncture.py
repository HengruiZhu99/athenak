#!/usr/bin/env python3
"""Signed three-resolution comparisons on matching physical leaf layouts."""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
from intrinsic_restart import read_restart


def matrix(n,points):
    w=np.zeros((n,2*n))
    for i in range(n):
        x=2*i+.5;start=max(0,min(2*i-points//2+1,2*n-points));nodes=np.arange(start,start+points)
        for node in nodes:
            other=nodes[nodes!=node];w[i,node]=np.prod((x-other)/(node-other))
    return w


def down(u,points):
    assert u.shape[-1]==u.shape[-2]==u.shape[-3] and u.shape[-1]%2==0
    w=matrix(u.shape[-1]//2,points)
    # Successive tensor contractions avoid enormous interpolation weight tensors.
    out=np.einsum('ai,vkji->vkja',w,u,optimize=False)
    out=np.einsum('bj,vkja->vkba',w,out,optimize=False)
    return np.einsum('ck,vkba->vcba',w,out,optimize=False)


def compare(paths,points):
    data=[read_restart(p,allow_refinement=True,allow_outflow=True) for p in paths]
    assert max(d['time'] for d in data)-min(d['time'] for d in data)<1e-12
    assert all(np.array_equal(data[0]['locations'],d['locations']) for d in data)
    assert all(np.array_equal(data[0]['domain'][:6],d['domain'][:6]) for d in data)
    assert data[1]['mb'][1]==2*data[0]['mb'][1] and data[2]['mb'][1]==4*data[0]['mb'][1]
    regions={'full':(0,np.inf),'r_lt_0.5':(0,.5),'r_0.5_1':(.5,1),'r_1_2':(1,2),'r_2_4':(2,4),'r_ge_4':(4,np.inf)}
    sums={key:np.zeros((3,50)) for key in regions}; volumes={key:0. for key in regions}
    for m,loc in enumerate(data[0]['locations']):
        u=[]
        for d in data:
            ng,n,_,_=d['mb'][:4];u.append(d['state'][m,:,ng:ng+n,ng:ng+n,ng:ng+n])
        d1=u[0]-down(u[1],points);d2=down(u[1]-down(u[2],points),points)
        n=u[0].shape[-1];d=data[0];h=d['domain'][6:9]/2.**(loc[3]-d['root_level'])
        xyz=[d['domain'][axis]+(loc[axis]*n+np.arange(n)+.5)*h[axis] for axis in range(3)]
        radius=np.sqrt(xyz[0][None,None,:]**2+xyz[1][None,:,None]**2+xyz[2][:,None,None]**2)
        for key,(lo,hi) in regions.items():
            mask=(radius>=lo)&(radius<hi);v1=d1[:,mask];v2=d2[:,mask];dv=np.prod(h)
            sums[key]+=np.array([np.sum(v1*v1,axis=1),np.sum(v2*v2,axis=1),np.sum(v1*v2,axis=1)])*dv
            volumes[key]+=mask.sum()*dv
    result={}
    for key,s in sums.items():
        groups={}
        for group,sl in [('primary',slice(0,20)),('all50',slice(None))]:
            a,b,c=s[:,sl].sum(axis=1)
            groups[group]=dict(coarse_difference_L2=float(np.sqrt(a)),fine_difference_L2=float(np.sqrt(b)),order=float(.5*np.log2(a/b)) if a*b>0 else None,alignment=float(c/np.sqrt(a*b)) if a*b>0 else None)
        result[key]=dict(volume=float(volumes[key]),groups=groups,component_squared_norms_and_inner_product=s.tolist())
    return dict(time=data[0]['time'],points=points,regions=result,paths=[str(p) for p in paths])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('restarts',type=Path,nargs=3);p.add_argument('--points',type=int,choices=[6,8],default=6);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    with a.output.open('x') as f:json.dump(compare(a.restarts,a.points),f,indent=2);f.write('\n')
