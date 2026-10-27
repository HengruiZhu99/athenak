#!/usr/bin/env python3
"""Same physical enriched equations, with coupled radial-line relaxation.

Reuses the common experiment without copying its physical setup/diagnostics.
Only the preconditioner factory is replaced, explicitly below. Both programs
are standalone research probes and neither is an evolution implementation.
"""
import argparse,json
from pathlib import Path
import numpy as np
import solve_enriched_wormhole as experiment
from enriched_line_preconditioner import EnrichedLinePreconditioner


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,choices=[10,12],default=10)
    p.add_argument('--radial',type=float,nargs='+',default=[0.,.45,.8,1.])
    p.add_argument('--max-iterations',type=int,default=800);p.add_argument('--restart',type=int,default=100)
    p.add_argument('--enriched-coarse-degree',type=int,choices=[0,1,2],default=0)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
    experiment.EnrichedPreconditioner=EnrichedLinePreconditioner
    result=experiment.run(args);result['preconditioner_entrypoint']=__doc__
    (args.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2));raise SystemExit(0 if all(result['gates'].values()) else 1)
