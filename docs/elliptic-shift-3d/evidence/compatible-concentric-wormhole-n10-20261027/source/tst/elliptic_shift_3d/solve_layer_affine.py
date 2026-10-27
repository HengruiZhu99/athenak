#!/usr/bin/env python3
"""Full tensor physical initial gauge or variable-metric MMS with layer updates.

No time evolution, spherical averaging, source subtraction or boundary-rate
projection. The unchanged independent all-node and scri gates remain mandatory.
"""
import argparse,json
from pathlib import Path
import numpy as np
import affine_sbp_probe
from endpoint_sbp import shell_grid
from layer_affine_sbp import LayerAffineGrid


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True)
    p.add_argument('--case',choices=['physical','mms'],default='physical')
    p.add_argument('--geometry',choices=['blended','concentric'],default='blended')
    p.add_argument('--n',type=int,choices=[10,12,14],default=10)
    p.add_argument('--radial',type=float,nargs='+')
    p.add_argument('--max-iterations',type=int,default=1100);p.add_argument('--restart',type=int,default=120)
    args=p.parse_args();args.amr_core=False;args.affine_correction=True;args.cached=True
    if args.radial is None:args.radial=[0.,.4,.55,.7,1.] if args.geometry=='concentric' else [0.,.45,.8,1.]
    if args.geometry=='concentric':
        from concentric_sbp import ConcentricGrid
        factory=ConcentricGrid
    else:factory=shell_grid
    args.output.mkdir(parents=True,exist_ok=False);np.seterr(all='raise')
    if args.case=='physical':import solve_wormhole_sbp as experiment
    else:
        import solve_mapped_sbp as experiment
        args.mortar_seed='zero';args.metric_amplitude=.08;args.coarse_degree=1;args.balanced=False
    affine_sbp_probe.AffineConsistentGrid=LayerAffineGrid;experiment.ShellGrid=factory
    if args.case=='mms':
        experiment.ShellGrid=lambda n,transfer_seed:factory(n,radial=tuple(args.radial),transfer_seed=transfer_seed)
    result=experiment.run(args);result['coefficient_scope']=__doc__
    result['geometry']=args.geometry
    if args.geometry=='concentric':
        result['half_width']=.2;result['physical_radial_knots']=args.radial
        result.pop('radial_partition_s',None)
    if args.case=='physical':
        result['gates']['strong_all_node_rate']=(result['all_node_metric_rate_rms']<1e-8 and result['raw_scri_metric_rate_max']<1e-7)
    (args.output/'assessment.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2));raise SystemExit(0 if all(result['gates'].values()) else 1)
