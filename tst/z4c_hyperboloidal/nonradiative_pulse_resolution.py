#!/usr/bin/env python3
"""Radial sampling survey; exact ghost values, not a full native operator test."""
import argparse,json
from pathlib import Path
import numpy as np

def bump(r,center,width,shape=1):
    z=(np.asarray(r)-center)/width
    b=np.zeros_like(z);d=b.copy();dd=b.copy();inside=abs(z)<1
    t=z[inside];v=1-t*t
    b[inside]=np.exp(shape*(1-1/v))
    logd=-2*shape*t/(width*v*v)
    logdd=-2*shape*(1+3*t*t)/(width*width*v**3)
    d[inside]=b[inside]*logd;dd[inside]=b[inside]*(logdd+logd*logd)
    return b,d,dd

def survey(center,width,n,shape=1):
    radii=[.08,.16,.24,.32,.40,.48,.56,.64,.72,.80,.88,.94,1.]
    rows=[]
    for lo,hi in zip(radii[:-1],radii[1:]):
        h=(hi-lo)/n
        offset=0 if hi==1 else .5  # ChartAtlas outer_endpoint_grid
        for r in lo+(np.arange(n)+offset)*h:
            if r+4*h<center-width or r-4*h>center+width:continue
            values=bump(r+np.arange(-4,5)*h,center,width,shape)[0]
            d=np.dot([1,-8,0,8,-1],values[2:7])/(12*h)
            dd=np.dot([-1,16,-30,16,-1],values[2:7])/(12*h*h)
            exact=bump(np.array([r]),center,width,shape)
            rows.append((r,h,float(exact[0][0]),d-float(exact[1][0]),dd-float(exact[2][0])))
    rows=np.array(rows);dense=np.linspace(center-width,center+width,20001)
    exact=bump(dense,center,width,shape)
    return dict(N=n,radial_centers_in_support=int(np.count_nonzero(rows[:,2]>0)),
                radial_centers_above_half_peak=int(np.count_nonzero(rows[:,2]>.5)),
                relative_max_first_error=float(max(abs(rows[:,3]))/max(abs(exact[1]))),
                relative_max_second_error=float(max(abs(rows[:,4]))/max(abs(exact[2]))))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    result={'scope':'Radial centered fourth-order stencil with analytic ghost values; excludes native interpolation, angular errors and compatible-Hessian modification.',
            'legacy':{'center':.9555,'half_width':.009,'support_width':.018},
            'wide':{'center':.86,'half_width':.08,'shape':6,'support_width':.16}}
    for name in ('legacy','wide'):
        c=result[name];c['sampling']=[survey(c['center'],c['half_width'],n,c.get('shape',1)) for n in (12,16,20)]
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
