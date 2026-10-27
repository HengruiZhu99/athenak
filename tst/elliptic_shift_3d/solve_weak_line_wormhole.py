#!/usr/bin/env python3
"""Same flat-metric weak control with positive radial strain-energy blocks."""
import argparse,json
from pathlib import Path
import numpy as np
import solve_weak_wormhole as experiment
from weak_line_preconditioner import WeakLinePreconditioner


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,choices=[10,12,14],default=10)
    p.add_argument('--radial',type=float,nargs='+',default=[0.,.45,.8,1.]);p.add_argument('--max-iterations',type=int,default=800)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
    experiment.WeakBlockPreconditioner=WeakLinePreconditioner
    result=experiment.run(args);result['preconditioner_entrypoint']=__doc__
    (args.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2));raise SystemExit(0 if all(result['gates'].values()) else 1)
