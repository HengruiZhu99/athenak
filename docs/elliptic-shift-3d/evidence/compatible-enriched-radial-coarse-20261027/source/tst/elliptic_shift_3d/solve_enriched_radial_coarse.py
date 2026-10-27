#!/usr/bin/env python3
"""Unmodified 3D enriched problem with a global radial coarse preconditioner."""
import argparse,json
from pathlib import Path
import numpy as np
import solve_enriched_wormhole as experiment
from radial_coarse_preconditioner import RadialCoarsePreconditioner


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,choices=[10,12],default=10)
    p.add_argument('--radial',type=float,nargs='+',default=[0.,.45,.8,1.])
    p.add_argument('--max-iterations',type=int,default=600);p.add_argument('--restart',type=int,default=100)
    p.add_argument('--radial-degree',type=int,default=16);args=p.parse_args();args.enriched_coarse_degree=0
    args.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
    experiment.EnrichedPreconditioner=lambda e,b:RadialCoarsePreconditioner(e,b,degree=args.radial_degree)
    result=experiment.run(args);result['preconditioner_entrypoint']=__doc__
    (args.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2));raise SystemExit(0 if all(result['gates'].values()) else 1)
