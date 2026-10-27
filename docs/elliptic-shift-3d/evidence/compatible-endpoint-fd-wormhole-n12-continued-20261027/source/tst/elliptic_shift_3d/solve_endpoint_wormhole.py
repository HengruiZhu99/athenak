#!/usr/bin/env python3
"""Unmodified full strong tensor gauge with local endpoint SBP shell rules."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import solve_wormhole_sbp as experiment
from endpoint_sbp import shell_grid


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--n',type=int,choices=[10,11,12,14],default=10)
    p.add_argument('--radial',type=float,nargs='+',default=[0.,.45,.8,1.]);p.add_argument('--max-iterations',type=int,default=800)
    p.add_argument('--restart',type=int,default=100)
    p.add_argument('--continue-case');args=p.parse_args()
    args.amr_core=False;args.affine_correction=True;args.cached=True
    args.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
    experiment.ShellGrid=shell_grid
    if args.continue_case:
        if Path(args.continue_case).name!=args.continue_case:raise ValueError('case name must be a basename')
        root=Path(__file__).resolve().parents[2]
        previous=root/'docs/elliptic-shift-3d/evidence'/args.continue_case
        assessment=json.loads((previous/'result/assessment.json').read_text())
        manifest=json.loads((previous/'manifest.json').read_text())
        if 'exit_code' not in manifest or assessment['time']!=0 or assessment['n']!=args.n or assessment['radial_partition_s']!=args.radial:
            raise ValueError('linear continuation must match a terminal initial-state case')
        for name in ['endpoint_sbp.py','mapped_sbp_probe.py','affine_sbp_probe.py','compatible_sbp.py','compatible_tensor_gauge.py','cached_tensor_gauge.py','solve_wormhole_sbp.py']:
            key='tst/elliptic_shift_3d/'+name
            if hashlib.sha256((root/key).read_bytes()).hexdigest()!=manifest['source_hashes'][key]:raise ValueError('operator changed since saved iterate')
        saved=np.load(previous/'result/solution.npz');standard=experiment.gmres
        def continued(apply,rhs,pre,W,max_iterations,restart,progress):
            if not np.array_equal(saved['coordinates'],pre.grid.coordinates) or not np.array_equal(saved['weights'],W):raise ValueError('saved linear grid mismatch')
            initial=saved['beta']+saved['coordinates']
            residual=rhs-apply(initial)
            start_rms=float(np.sqrt(np.sum(W[...,None]*residual**2)/np.sum(W)))
            delta,r=standard(apply,residual,pre,W,max_iterations,restart,progress=progress)
            solution=initial+delta;actual=rhs-apply(solution)
            r['original_rms']=float(np.sqrt(np.sum(W[...,None]*actual**2)/np.sum(W)))
            r['original_point']=float(np.max(abs(actual)))
            r['linear_continuation']={'case':args.continue_case,'solution_sha256':hashlib.sha256((previous/'result/solution.npz').read_bytes()).hexdigest(),
                                     'prior_iterations':assessment['iterations'],'initial_residual_rms':start_rms,'total_iterations':assessment['iterations']+r['iterations']}
            return solution,r
        experiment.gmres=continued
    result=experiment.run(args);result['shell_rule']=__doc__
    # Retain the existing bulk/boundary-value gates, and explicitly add the
    # unrelaxed strong all-node/boundary gate for this new family.
    result['gates']['strong_all_node_rate']=(result['all_node_metric_rate_rms']<1e-8 and result['raw_scri_metric_rate_max']<1e-7)
    (args.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2));raise SystemExit(0 if all(result['gates'].values()) else 1)
