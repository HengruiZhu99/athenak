#!/usr/bin/env python3
"""Independent physical/auxiliary diagnostics on synchronized puncture leaf ghosts.

The direct stencil reaches <=3 cells, so periodic array rolls in the analysis
helper never enter the retained active cells (ng=4). No global periodic wrapping
or interpolation is used at physical/refinement boundaries.
"""
import argparse,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
from intrinsic_restart import read_restart
from intrinsic_diagnostics import diagnostics,component_names


def analyze(path,initial=False):
    d=read_restart(path,allow_refinement=True,allow_outflow=True)
    ng,nx,ny,nz=d['mb'][:4]; names=[v for group in component_names().values() for v in group]
    regions={'full':(0,np.inf),'r_lt_0.5':(0,.5),'r_0.5_1':(.5,1),'r_1_2':(1,2),'r_2_4':(2,4),'r_ge_4':(4,np.inf)}
    sums={key:dict(volume=0.,cells=0,L1=np.zeros(89),L2sq=np.zeros(89),maximum=np.full(89,-1.),signed=np.zeros(89)) for key in regions}
    initial_error=0.; mass=float(d['header']['problem'].get('mass',1))
    for m,loc in enumerate(d['locations']):
        h=d['domain'][6:9]/2.**(loc[3]-d['root_level'])
        xyz=[d['domain'][axis]+(loc[axis]*[nx,ny,nz][axis]+np.arange([nx,ny,nz][axis])+.5)*h[axis] for axis in range(3)]
        x,y,z=xyz[0][None,None,:],xyz[1][None,:,None],xyz[2][:,None,None]
        radius=np.sqrt(x*x+y*y+z*z); u=d['state'][m];active=u[:,ng:ng+nz,ng:ng+ny,ng:ng+nx]
        if initial:
            psi=1+mass/(2*radius);expected=np.zeros_like(active);expected[0]=psi**-2;expected[1]=1
            for axis,coord in enumerate([x,y,z]):expected[20+axis]=expected[23+axis]=mass*coord*psi**-3/radius**3
            initial_error=max(initial_error,float(np.max(abs(active-expected))))
        fields=diagnostics(u,h,6,physical_operator='direct')
        values=np.concatenate(list(fields.values()))[:,ng:ng+nz,ng:ng+ny,ng:ng+nx]
        dv=float(np.prod(h))
        for key,(lo,hi) in regions.items():
            mask=(radius>=lo)&(radius<hi);n=int(mask.sum())
            if not n:continue
            a=sums[key]; v=values[:,mask];a['volume']+=dv*n;a['cells']+=n
            a['L1']+=np.sum(abs(v),axis=1)*dv;a['L2sq']+=np.sum(v*v,axis=1)*dv
            win=np.argmax(abs(v),axis=1);signed=v[np.arange(89),win];better=abs(signed)>a['maximum'];a['maximum'][better]=abs(signed[better]);a['signed'][better]=signed[better]
    for a in sums.values():
        a['RMS']=np.sqrt(a['L2sq']/a['volume']) if a['volume'] else np.zeros(89)
        for key,value in list(a.items()):
            if isinstance(value,np.ndarray):a[key]=value.tolist()
    if initial:assert initial_error<3e-15,(path,initial_error)
    assert abs(sums['full']['volume']-np.prod(d['domain'][3:6]-d['domain'][:3]))<1e-9
    return dict(path=str(path),time=d['time'],cycle=d['cycle'],components=names,regions=sums,initial_maximum_error=initial_error if initial else None)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('restart',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--initial',action='store_true');a=p.parse_args()
    result=analyze(a.restart,a.initial)
    with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
    print(json.dumps(dict(time=result['time'],cycle=result['cycle'],initial_maximum_error=result['initial_maximum_error'],H_RMS=result['regions']['full']['RMS'][0])))
