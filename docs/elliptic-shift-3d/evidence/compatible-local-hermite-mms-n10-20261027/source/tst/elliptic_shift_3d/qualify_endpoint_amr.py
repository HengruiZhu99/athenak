#!/usr/bin/env python3
"""Endpoint shell-rule check with the genuine 15-leaf Cartesian AMR core."""
import argparse,json,time
from pathlib import Path
import numpy as np
import mapped_sbp_probe as mapped
import amr_shell_sbp_probe as combined
import qualify_amr_shell_sbp as experiment
from endpoint_sbp import local
from affine_sbp_probe import AffineConsistentGrid


def make_grid(n):
    previous_mapped=mapped.lobatto;previous_combined=combined.lobatto
    try:
        mapped.lobatto=local;combined.lobatto=local
        return AffineConsistentGrid(combined.AMRShellGrid(n))
    finally:
        mapped.lobatto=previous_mapped;combined.lobatto=previous_combined


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise');start=time.monotonic()
    experiment.AMRShellGrid=make_grid;result=experiment.run();result['shell_rule']=__doc__
    result['seconds']=time.monotonic()-start;result['evolution_adopted']=False
    (args.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps(result,indent=2))
    raise SystemExit(0 if all(result['gates'].values()) else 1)
