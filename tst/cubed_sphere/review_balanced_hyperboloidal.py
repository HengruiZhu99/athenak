#!/usr/bin/env python3
"""Recount coupled constraints in fixed regions and report temporal sensitivity."""
import argparse
import csv
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_spherical import analyze_case
from review_puncture_radial import combine


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('results',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    launch=json.loads((a.results/'launch.json').read_text());reviews={};metrics={}
    for g in launch['grids']:
        name=g['name'];case=a.results/name;r=json.loads((case/'receipt.json').read_text())
        review=analyze_case(case);review.pop('passed_old_failure_times',None)
        review['initial_profile_unchanged']=r.get('initial_profile_unchanged')
        steps=list(csv.DictReader((case/'cubed_sphere_conformal_steps.csv').open()))
        review['actual_max_dt']=max(float(x['dt']) for x in steps)
        with (case/'cubed_sphere_conformal_regions.csv').open() as f:
            data=[x for x in csv.DictReader(f) if x['time']==r['last_rows'][-1]['time']]
        regions={region:combine([x for x in data if x['region']==region and x['subset']=='all']) for region in ['bulk','shell']}
        regions['global']=combine([x for x in data if x['subset']=='all'])
        regions['layer']=combine([x for x in data if x['subset'] in [f'radial_{i}' for i in range(6,12)]])
        regions['fixed_exterior']=combine([x for x in data if x['subset'] in [f'radial_{i}' for i in range(2,14)]])
        for subset in ['fixed_interior','fixed_interface','fixed_outer_collar']:
            regions[subset]=combine([x for x in data if x['subset']==subset])
        metrics[name]=regions;reviews[name]=review
        if r['outcome']=='complete':
            assert r['reference_unchanged'] and r.get('initial_profile_unchanged',True)
    orders={};monotonic={}
    grids={g['name']:g for g in launch['grids']}
    def final_time(name):
        return float(reviews[name]['last_rows'][-1]['time'])
    if all(n in metrics for n in ['coarse','medium','fine']):
        assert len({final_time(n) for n in ['coarse','medium','fine']})==1, 'spatial endpoints differ'
        assert all(reviews[n].get('outcome')=='complete' for n in ['coarse','medium','fine']), 'incomplete spatial run'
        for region in metrics['fine']:
            orders[region]={};monotonic[region]={}
            for key in ['H_l2','M_conformal_l2','Z_conformal_l2','H_linf']:
                errors=[metrics[n][region][key] for n in ['coarse','medium','fine']]
                monotonic[region][key]=errors[0]>errors[1]>errors[2]
                orders[region][key]=[math.log(errors[i]/errors[i+1])/math.log(grids[b]['bulk']/grids[c]['bulk'])
                    for i,(c,b) in enumerate([('coarse','medium'),('medium','fine')])]
    temporal={};budget={}
    baseline='fine-temporal' if 'fine-temporal' in metrics else 'fine'
    if 'fine-halfstep' in metrics:
        assert final_time(baseline)==final_time('fine-halfstep'), 'temporal endpoints differ'
        assert all(reviews[n].get('outcome')=='complete' for n in [baseline,'fine-halfstep'])
        assert reviews['fine-halfstep']['actual_max_dt'] < reviews[baseline]['actual_max_dt'], 'timestep was not reduced'
        for region in metrics['fine-halfstep']:
            temporal[region]={};budget[region]={}
            for key in ['H_l2','M_conformal_l2','Z_conformal_l2','H_linf']:
                difference=abs(metrics['fine-halfstep'][region][key]-metrics[baseline][region][key])
                temporal[region][key]=difference/max(abs(metrics[baseline][region][key]),1e-30)
                if baseline=='fine':budget[region][key]=difference/max(abs(metrics['medium'][region][key]-metrics['fine'][region][key]),1e-30)
    out={'scope':'Coordinate-volume norms in fixed physical regions. Observed constraint-error slopes, not solution convergence. Active collar and sampling change under refinement. Initial jets are analytic.',
         'cases':reviews,'final_metrics':metrics,'monotonic_decrease':monotonic,'observed_orders_using_bulk_scale':orders,
         'temporal_relative_sensitivity':temporal,'temporal_to_finest_spatial_norm_difference':budget}
    required=['global','bulk','shell','layer','fixed_interior','fixed_interface','fixed_outer_collar']
    keys=['H_l2','M_conformal_l2','Z_conformal_l2']
    out['short_time_screen']={
        'all_required_RMS_decrease':bool(monotonic) and all(monotonic[r][k] for r in required for k in keys),
        'all_required_finest_RMS_orders_above_half':bool(orders) and all(orders[r][k][-1]>.5 for r in required for k in keys),
        'timestep_below_ten_percent_of_spatial_difference':all(budget[r][k]<.1 for r in required for k in keys)
            if all(k in budget.get(r,{}) for r in required for k in keys) else None,
        'scope':'Necessary short-time screen only. Does not accept longer evolution, maxima convergence, fixed-minimum-Omega convergence or a trumpet.'}
    out['acceptance_gaps']={
        'exact_transition_region':'The layer key aggregates radial bins 6..11 (0.375<=r<0.75 at unit reference radius), not the entire transition beginning at r=0.35. Global/shell include the omitted strip; exact-transition regional convergence remains unverified.',
        'regional_M_and_Z_maxima_with_locations':'Not recorded in this CSV schema; requires checkpoint reconstruction or additional diagnostics. No maxima acceptance claim.',
        'closure_difference_refinement':'Requires matched alternate-closure data; not evaluated by this single-matrix review.',
        'roundoff_resolution':'No independently measured noise floor supplied. Small differences cannot be certified as resolved by this review.',
        'complete_constraint_acceptance':False}
    (a.output/'review.json').write_text(json.dumps(out,indent=2,allow_nan=False)+'\n')
    if orders:
        fig,axs=plt.subplots(1,3,figsize=(13,4.6),layout='constrained')
        for region,color in zip(['bulk','shell','layer','fixed_outer_collar'],['#333333','#0072b2','#d55e00','#009e73']):
            for ax,key in zip(axs,['H_l2','M_conformal_l2','Z_conformal_l2']):
                ax.loglog([grids[n]['bulk'] for n in ['coarse','medium','fine']],
                          [metrics[n][region][key] for n in ['coarse','medium','fine']],'o-',label=('layer bins 6–11' if region=='layer' else region),color=color)
                ax.set(xlabel='Bulk resolution per axis (angles and radius refined together)',ylabel=key+' RMS')
                ax.grid(alpha=.2)
        h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncols=4,frameon=False)
        fig.suptitle('Coupled puncture refinement at t = '+reviews['fine']['last_rows'][-1]['time'])
        fig.savefig(a.output/'constraints.png',dpi=180);fig.savefig(a.output/'constraints.pdf')
    print(json.dumps({'metrics':{n:{r:{k:v[k] for k in ['H_l2','M_conformal_l2']} for r,v in d.items()} for n,d in metrics.items()}},indent=2))


if __name__=='__main__':main()
