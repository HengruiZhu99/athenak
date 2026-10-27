#!/usr/bin/env python3
"""Unmodified full strong tensor gauge with local endpoint SBP shell rules."""
import argparse,json
from pathlib import Path
import numpy as np
import solve_wormhole_sbp as experiment
from endpoint_sbp import shell_grid


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,choices=[10,12,14],default=10)
    p.add_argument('--radial',type=float,nargs='+',default=[0.,.45,.8,1.]);p.add_argument('--max-iterations',type=int,default=800)
    p.add_argument('--restart',type=int,default=100);args=p.parse_args()
    args.amr_core=False;args.affine_correction=True;args.cached=True
    args.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
    experiment.ShellGrid=shell_grid
    result=experiment.run(args);result['shell_rule']=__doc__
    # Retain the existing bulk/boundary-value gates, and explicitly add the
    # unrelaxed strong all-node/boundary gate for this new family.
    result['gates']['strong_all_node_rate']=(result['all_node_metric_rate_rms']<1e-8 and result['raw_scri_metric_rate_max']<1e-7)
    (args.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2));raise SystemExit(0 if all(result['gates'].values()) else 1)
