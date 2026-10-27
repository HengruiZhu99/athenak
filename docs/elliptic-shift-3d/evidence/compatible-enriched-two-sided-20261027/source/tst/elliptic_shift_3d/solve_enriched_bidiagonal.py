#!/usr/bin/env python3
"""Two-sided conditioning diagnosis on the unchanged enriched physical problem."""
import argparse,json
from pathlib import Path
import numpy as np
import solve_enriched_wormhole as experiment
from bounded_bidiagonal import investigate
from adjoint_tensor_gauge import EnrichedGaugeAdjoint,EnrichedPreconditionerAdjoint


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,choices=[10,12],default=10)
    p.add_argument('--radial',type=float,nargs='+',default=[0.,.45,.8,1.])
    p.add_argument('--max-iterations',type=int,default=120);args=p.parse_args()
    args.restart=100;args.enriched_coarse_degree=0
    args.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
    def solve(apply,rhs,pre,W,max_iterations,restart,progress):
        root=np.sqrt(W/np.sum(W))[...,None];shape=rhs.shape
        pack=lambda a:(root*a).ravel();unpack=lambda a:a.reshape(shape)/root
        adjoint=EnrichedGaugeAdjoint(pre.e);padjoint=EnrichedPreconditionerAdjoint(pre,adjoint)
        action=lambda a:pack(apply(pre(unpack(a))))
        transpose=lambda a:pack(padjoint(adjoint.apply(unpack(a))))
        y,weak,result=investigate(action,transpose,pack(rhs),max_iterations,progress=progress,
                                  point_check=lambda r:float(np.max(abs(unpack(r)))))
        coefficients=pre(unpack(y));weak_coefficients=pre(unpack(weak))
        np.savez_compressed(args.output/'weak-direction.npz',coefficients=weak_coefficients,
                            action=apply(weak_coefficients),metric_rate=pre.e.rate(weak_coefficients),
                            boundary=pre.e.boundary,coordinates=pre.e.grid.coordinates,weights=W)
        return coefficients,result
    experiment.gmres=solve
    result=experiment.run(args);result['solver_scope']=__doc__
    (args.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2));raise SystemExit(0 if all(result['gates'].values()) else 1)
