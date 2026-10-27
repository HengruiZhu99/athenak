#!/usr/bin/env python3
"""Independent inner-product checks of every adjoint, including physical scri."""
import argparse,json,time
from pathlib import Path
import numpy as np
from mapped_sbp_probe import ShellGrid
from affine_sbp_probe import AffineConsistentGrid
from cached_tensor_gauge import CachedTensorGauge
from enriched_outer_gauge import EnrichedOuterGauge,EnrichedPreconditioner
from adjoint_tensor_gauge import GaugeAdjoint,EnrichedGaugeAdjoint,EnrichedPreconditionerAdjoint
from mapped_sbp_preconditioner import MappedPreconditioner


def run(args):
    grid=AffineConsistentGrid(ShellGrid(10,radial=(0,.45,.8,1),transfer_seed='zero'))
    rng=np.random.default_rng(412731);W=grid.weights;X=grid.coordinates
    dot=lambda a,b:float(np.sum(W.reshape(W.shape+(1,)*(a.ndim-4))*a*b))
    rows=[]
    for curved in [False,True]:
        metric=np.broadcast_to(np.eye(3),W.shape+(3,3)).copy()
        if curved:
            v=np.stack([.3+X[...,0],.2+X[...,1],-.1+X[...,2]],axis=-1)
            metric+=.1*v[..., :,None]*v[...,None,:]
            metric[...,0,0]+=.1*np.exp(X[...,2])
        gauge=CachedTensorGauge(grid,metric);adjoint=GaugeAdjoint(gauge)
        enriched=EnrichedOuterGauge(gauge);eadjoint=EnrichedGaugeAdjoint(enriched)
        scalar=rng.normal(size=W.shape);other=rng.normal(size=W.shape)
        beta=rng.normal(size=W.shape+(3,));vector=rng.normal(size=beta.shape)
        tensor=rng.normal(size=W.shape+(3,3));tensor=(tensor+np.swapaxes(tensor,-1,-2))/2
        tests=[]
        def check(name,left,right):
            tests.append({'name':name,'left':left,'right':right,'relative_defect':abs(left-right)/max(1,abs(left),abs(right))})
        for axis in range(3):
            for sign in [-1,1]:
                check(f'derivative_{axis}_{sign}',dot(scalar,grid.derivative(other,axis,sign)),dot(adjoint.derivative(scalar,axis,sign),other))
        check('full_variation',dot(vector,gauge.variation(tensor)),dot(adjoint.variation(vector),tensor))
        check('full_metric_rate',dot(tensor,gauge.rate(beta)),dot(adjoint.rate(tensor),beta))
        check('enriched_metric_rate',dot(tensor,enriched.rate(beta)),dot(eadjoint.rate(tensor),beta))
        check('enriched_full_operator',dot(vector,enriched.apply(beta)),dot(eadjoint.apply(vector),beta))
        if args.preconditioner and not curved:
            def apply(value):
                out=gauge.apply(value);out[enriched.boundary]=value[enriched.boundary];return out
            base=MappedPreconditioner(grid,metric,apply)
            pre=EnrichedPreconditioner(enriched,base);padjoint=EnrichedPreconditionerAdjoint(pre,eadjoint)
            check('complete_preconditioner',dot(vector,pre(beta)),dot(padjoint(vector),beta))
        rows.append({'curved':curved,'tests':tests})
    return {'scope':__doc__,'rows':rows,'gates':{'all_adjoint_defects_below_1e-10':all(t['relative_defect']<1e-10 for r in rows for t in r['tests'])},'evolution_adopted':False}


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--preconditioner',action='store_true');args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise');start=time.monotonic()
    result=run(args);result['seconds']=time.monotonic()-start
    (args.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if all(result['gates'].values()) else 1)
