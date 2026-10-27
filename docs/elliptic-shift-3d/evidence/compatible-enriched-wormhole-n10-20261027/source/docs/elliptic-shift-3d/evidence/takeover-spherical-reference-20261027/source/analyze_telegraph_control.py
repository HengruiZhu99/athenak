#!/usr/bin/env python3
"""Verified quadrature/KO differences relative to an independent spatial pair.

Quadrature differences have the predeclared 10% threshold. KO differences are
reported only: each KO amplitude still requires a separate three-grid study.
"""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_telegraph_time import load,controlled_inputs,differences
from telegraph_analysis_io import physical_lapse_gradient,validate_telegraph_profile



def fraction(numerator,denominator):
    if denominator>0:return numerator/denominator
    return 0. if numerator==0 else None


def compare(base,space,controls,parameter):
    if parameter not in ['quadrature_width','puncture_ko']:raise ValueError('Unsupported numerical control')
    if not controls:raise ValueError('Require at least one control')
    controlled_inputs(base,space,{'n'})
    if space['manifest']['arguments']['n']<=base['manifest']['arguments']['n']:raise ValueError('Require a finer spatial reference')
    values=[]
    for run in controls:
        controlled_inputs(base,run,{parameter})
        value=run['manifest']['arguments'][parameter]
        if value==base['manifest']['arguments'][parameter] or value in values:raise ValueError('Repeated or unchanged control value')
        values.append(value)
    rows=[]
    for t in sorted(base['data']):
        a,b=base['data'][t],space['data'][t]
        for d in [a,b,*[run['data'][t] for run in controls]]:validate_telegraph_profile(d)
        scale=float(np.max(abs(physical_lapse_gradient(a)[a['r']<.3])))
        rows.append(dict(t=t,spatial=differences(a,b,scale),controls=[differences(a,run['data'][t],scale) for run in controls]))
    metrics=[]
    for i,value in enumerate(values):
        entries={}
        for key in rows[0]['spatial']:
            control_peak=max(row['controls'][i][key] for row in rows)
            spatial_peak=max(row['spatial'][key] for row in rows)
            ratio=fraction(control_peak,spatial_peak)
            entries[key]=dict(control_difference=control_peak,spatial_difference=spatial_peak,raw_control_to_space_fraction=ratio,
                below_ten_percent=bool(ratio is not None and ratio<.1) if parameter=='quadrature_width' else None)
        metrics.append(dict(value=value,metrics=entries))
    return dict(parameter=parameter,baseline_value=base['manifest']['arguments'][parameter],history=rows,controls=metrics,
        status=('passed_quadrature_difference_checks' if all(v['below_ten_percent'] for record in metrics for v in record['metrics'].values()) else 'failed_quadrature_difference_checks') if parameter=='quadrature_width' else 'KO_differences_measured_requires_three_grid_assessment',
        goal_status='incomplete',scope='Complete verified trajectories, identical numerical source/library bindings, and only the named input changed. Differences are raw finite-grid comparisons, not certified error bounds. Quadrature uses the predeclared10% threshold. KO comparisons alone cannot establish robustness; separate three-grid gates remain mandatory. No result overrides failed boundary, constraint, timestep or adjacent-parameter checks.')


def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',type=Path,required=True);p.add_argument('--spatial',type=Path,required=True)
    p.add_argument('--controls',type=Path,nargs='+',required=True);p.add_argument('--parameter',choices=['quadrature_width','puncture_ko'],required=True)
    p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    runs=[load(path) for path in [args.baseline,args.spatial,*args.controls]]
    result=compare(runs[0],runs[1],runs[2:],args.parameter)
    result['records']=[{k:v for k,v in run.items() if k!='data'} for run in runs]
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['parameter','controls','status','goal_status']},indent=2))

if __name__=='__main__':main()
