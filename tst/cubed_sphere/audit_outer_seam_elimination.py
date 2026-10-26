#!/usr/bin/env python3
"""Source-matched donor-plan model: test radial elimination at shell seams.

This is an independent reproduction of Map and donor-plan algebra, not an
instrumented native plan dump. It tests linear operator coefficients, without
claiming a native evolution or energy estimate.
"""
import argparse
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
import numpy as np
from audit_outer_radial_operators import D1,D2,DL,DR,eliminate


def experiment(na,nr,stretch,degree):
    L=.25;R=1.
    def forward(face,i,j,k):
        s=(i+.5)/nr;u=-1+2*(j+.5)/na;v=-1+2*(k+.5)/na
        t=s if stretch==0 else -np.expm1(-stretch*s)/(-np.expm1(-stretch))
        d=(1-t)*L+t*R/np.sqrt(1+u*u+v*v)
        a=face//2;b=(a+1)%3;c=(a+2)%3
        x=np.zeros(3);x[a]=(-1 if face%2 else 1)*d;x[b]=d*u;x[c]=d*v
        return x
    @lru_cache(None)
    def value(face,i,j,k):
        if 0<=i<nr and 0<=j<na and 0<=k<na:return {(face,i,j,k):1.}
        x=forward(face,i,j,k);a=int(np.argmax(abs(x)));owner=2*a+int(x[a]<0)
        b=(a+1)%3;c=(a+2)%3;d=abs(x[a]);u=x[b]/d;v=x[c]/d
        t=(d-L)/(R/np.sqrt(1+u*u+v*v)-L)
        s=t if stretch==0 else -np.log1p(-(-np.expm1(-stretch))*t)/stretch
        z=[s*nr-.5,(u+1)*na/2-.5,(v+1)*na/2-.5]
        outside=np.dot(x,x)>=R*R
        starts=[];weights=[]
        for coord,n in zip(z,[nr,na,na]):
            nearest=round(coord)
            if not outside and abs(coord-nearest)<1e-11 and 0<=nearest<n:
                starts.append(nearest);weights.append([1.]);continue
            deg=degree if outside else 5
            start=min(max(int(np.floor(coord))-deg//2,0),n-deg-1)
            w=[]
            for aa in range(deg+1):
                ww=1.
                for bb in range(deg+1):
                    if aa!=bb:ww*=(coord-start-bb)/(aa-bb)
                w.append(ww)
            starts.append(start);weights.append(w)
        result={}
        for aa,wa in enumerate(weights[0]):
            for bb,wb in enumerate(weights[1]):
                for cc,wc in enumerate(weights[2]):
                    key=(owner,starts[0]+aa,starts[1]+bb,starts[2]+cc)
                    result[key]=wa*wb*wc
        # The native anchor form makes constant preservation exact in real
        # arithmetic. Include the floating coefficient remainder at its anchor.
        anchor=(owner,*starts);result[anchor]+=1-sum(result.values())
        return result
    @lru_cache(None)
    def smooth(key):
        return float(np.exp(np.dot([3.1,-2.7,1.9],forward(*key))))
    def add(out,weights,factor):
        for key,w in weights.items():out[key]=out.get(key,0.)+factor*w
    rows=[]
    for face in range(6):
        for depth in range(3):
            for location,j,k in [('interior',na//2,na//2),('edge',na-1,na//2),('corner',na-1,na-1)]:
                for name,stencil,angular in [('D1',D1,None),('D2',D2,None),('adv_negative',DL,None),('adv_positive',DR,None),('mixed_ru',D1,1),('mixed_rv',D1,2)]:
                    actual={};collapsed={};aw={0:1.} if angular is None else D1
                    for off,w in stencil.items():
                        for ao,wa in aw.items():
                            add(actual,value(face,nr-1-depth+off,j+(ao if angular==1 else 0),k+(ao if angular==2 else 0)),float(w*wa))
                    for off,w in eliminate(stencil,depth,degree).items():
                        for ao,wa in aw.items():
                            add(collapsed,value(face,nr-1+off,j+(ao if angular==1 else 0),k+(ao if angular==2 else 0)),float(w*wa))
                    diff={key:actual.get(key,0)-collapsed.get(key,0) for key in actual.keys()|collapsed.keys()}
                    error=sum(abs(v) for v in diff.values());scale=max(1.,sum(abs(v) for v in actual.values()))
                    spacing_factor=nr*(na/2 if angular else nr if name=='D2' else 1)
                    smooth_difference=abs(math.fsum(w*smooth(key) for key,w in diff.items()))*spacing_factor
                    roundoff_bound=128*np.finfo(float).eps*spacing_factor*sum((abs(actual.get(key,0))+abs(collapsed.get(key,0)))*abs(smooth(key)) for key in diff)
                    rows.append(dict(conservative_roundoff_bound=roundoff_bound,smooth_logical_derivative_difference=smooth_difference,face=face,depth=depth,location=location,operator=name,coefficient_l1_difference=error,relative_l1_difference=error/scale,actual_support=len(actual),collapsed_support=len(collapsed)))
    return rows


def main():
    p=argparse.ArgumentParser();p.add_argument('output',type=Path);a=p.parse_args()
    cases=[]
    for stretch in [0.,2.8592697712828787]:
        for degree in range(2,6):
            rows=experiment(12,48,stretch,degree)
            pure=[r for r in rows if not r['operator'].startswith('mixed')]
            interior=[r for r in rows if r['location']=='interior']
            assert max(r['relative_l1_difference'] for r in pure+interior)<1e-10
            cases.append(dict(angular=12,radial=48,stretch=stretch,degree=degree,rows=rows))
    refinement=[]
    for degree in [2,3,4,5]:
        values=[];bounds=[]
        for na in [12,24,48]:
            rows=experiment(na,4*na,0.,degree)
            selected=[r for r in rows if r['operator'].startswith('mixed') and r['location']!='interior']
            values.append(max(r['smooth_logical_derivative_difference'] for r in selected))
            bounds.append(max(r['conservative_roundoff_bound'] for r in selected))
        refinement.append(dict(degree=degree,angular=[12,24,48],radial=[48,96,192],max_smooth_mixed_difference=values,conservative_roundoff_bounds=bounds,observed_orders=[math.log(values[i]/values[i+1],2) if values[i]>bounds[i] and values[i+1]>bounds[i+1] else None for i in range(2)]))
    sources=['src/mesh/cubed_sphere/map.hpp','src/mesh/cubed_sphere/extension.cpp','src/mesh/cubed_sphere/conformal_probe.cpp',__file__]
    result=dict(scope=__doc__,source_sha256={s:hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in sources},cases=cases,smooth_refinement=refinement,smooth_function="exp(3.1*x-2.7*y+1.9*z); null order means unresolved against conservative floating-point bound",
        interpretation='Pure radial ghost elimination agrees to floating roundoff. Mixed derivatives at seams need the full cross-patch donor operator; extrapolate-then-interpolate need not commute with interpolate-then-extrapolate. Coefficients omit logical grid spacing powers. This does not identify the cause of native instability.')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n')
    print('Smooth model refinement:',refinement)
    for c in cases:
        print(c['stretch'],c['degree'],{loc:max(r['relative_l1_difference'] for r in c['rows'] if r['location']==loc) for loc in ['interior','edge','corner']})
if __name__=='__main__':main()
