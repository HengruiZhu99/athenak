#!/usr/bin/env python3
"""Locate speed-audit violations in saved lines; not a full 3D certification."""
import argparse
import gzip
import json
import math
from pathlib import Path


def read(path):
    lines=gzip.open(str(path),'rt').read().splitlines()
    names=lines[1].lstrip('#').split()
    return [dict(zip(names,map(float,line.split()))) for line in lines[2:]
            if line and not line.startswith('#')]


def diagnose(run):
    bad=[];outermax=0;invalid_metric=0
    for p in sorted((run/'tab').glob('*.z4c.*.gz')):
        for u,v in zip(read(p.with_name(p.name.replace('.z4c.','.adm.'))),read(p)):
            assert (u['gid'],u['i'],u['x1v'])==(v['gid'],v['i'],v['x1v'])
            xx,xy,xz,yy,yz,zz=[u['adm_'+s] for s in ('gxx','gxy','gxz','gyy','gyz','gzz')]
            det=xx*(yy*zz-yz*yz)-xy*(xy*zz-xz*yz)+xz*(xy*yz-xz*yy)
            inv=[(yy*zz-yz*yz)/det,(xx*zz-xz*xz)/det,(xx*yy-xy*xy)/det]
            alpha=abs(v['z4c_alpha']);chi=v['z4c_chi'];x=v['x1v']
            speed=None
            if all(q>=0 for q in inv):
                speed=max(abs(v['z4c_beta'+axis])+max(alpha*math.sqrt(q),
                    math.sqrt((2*alpha+2)*q),math.sqrt(4/3*q/max(chi,1e-30)))
                    for axis,q in zip('xyz',inv))
            else: invalid_metric+=1
            if abs(x)>=2:
                assert speed is not None
                outermax=max(outermax,speed)
            if speed is None or speed>8:
                bad.append(dict(snapshot=p.name,x=x,chi=chi,speed=speed))
    return dict(scope='1D evidence only; cannot certify the 3D exterior',
        gauge='telegrapher tau=.1 kappa=.2',bad_line_samples=len(bad),
        invalid_metric_line_samples=invalid_metric,
        bad_x_range=[min((b['x'] for b in bad),default=None),max((b['x'] for b in bad),default=None)],
        max_line_speed_abs_x_ge_2=outermax,examples=bad[:5])

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);args=p.parse_args()
    print(json.dumps(diagnose(args.run),indent=2,allow_nan=False))
