#!/usr/bin/env python3
"""Full variable-metric MMS using the endpoint finite-difference shell pair."""
import argparse,json
from pathlib import Path
import numpy as np
import solve_mapped_sbp as experiment
from endpoint_sbp import shell_grid


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,choices=[10,12,14],default=10)
    p.add_argument('--max-iterations',type=int,default=800);args=p.parse_args()
    args.restart=100;args.amr_core=False;args.affine_correction=True;args.cached=True
    args.mortar_seed='zero';args.metric_amplitude=.08;args.coarse_degree=1;args.balanced=False
    args.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
    experiment.ShellGrid=lambda n,transfer_seed:shell_grid(n,radial=(0,.45,1.),transfer_seed=transfer_seed)
    result=experiment.run(args);result['shell_rule']=__doc__
    (args.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2));raise SystemExit(0 if all(result['gates'].values()) else 1)
