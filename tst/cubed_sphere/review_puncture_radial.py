#!/usr/bin/env python3
"""Recount radial controls, including a fixed region inside the layer."""
import argparse
import csv
import json
import math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from review_puncture_comparison import spherical


def combine(rows):
    volume=sum(float(r['volume']) for r in rows)
    return {'volume':volume,'cells':sum(int(r['cells']) for r in rows),
            **{k:math.sqrt(sum(float(r['volume'])*float(r[k])**2 for r in rows)/volume)
               for k in ['H_l2','M_conformal_l2','Z_conformal_l2']},
            'H_linf':max(float(r['H_linf']) for r in rows)}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capsule',type=Path);p.add_argument('results',type=Path)
    a=p.parse_args();reviews={};final={};profiles={}
    for name in ['R41','R82','R123']:
        case=a.results/name;review,_=spherical(case)
        r=json.loads((case/'receipt.json').read_text())
        assert r['reached_target'] and r['reference_unchanged'] and r['initial_profile_unchanged']
        steps=list(csv.DictReader((case/'cubed_sphere_conformal_steps.csv').open()))
        review['max_dt']=max(float(r['dt']) for r in steps)
        review['min_dt']=min(float(r['dt']) for r in steps)
        assert review['max_dt']<=.000080000000001
        final_time=r['last_rows'][-1]['time']
        with (case/'cubed_sphere_conformal_regions.csv').open() as f:
            data=[r for r in csv.DictReader(f) if r['time']==final_time]
        layer=[r for r in data if r['subset'] in [f'radial_{i}' for i in range(6,12)]]
        review['fixed_layer_0.375_le_r_lt_0.75']=combine(layer)
        profiles[name]=[combine([r for r in data if r['subset']==f'radial_{i}'])
                        for i in range(16) if any(r['subset']==f'radial_{i}' for r in data)]
        for i,binrow in zip([i for i in range(16) if any(r['subset']==f'radial_{i}' for r in data)],profiles[name]):
            binrow['radius_bin_midpoint']=(i+.5)/16
        reviews[name]=review
        final[name]={'combined':review['combined_coordinate_volume_last'],
            'fixed_layer':review['fixed_layer_0.375_le_r_lt_0.75'],
            'regions':{x['region']:{k:float(x[k]) for k in ['H_l2','H_linf','M_conformal_l2','min_Omega']} for x in r['last_rows']}}
    changes={name:{k:final[name]['fixed_layer'][k]/final['R41']['fixed_layer'][k]-1
                   for k in ['H_l2','H_linf','M_conformal_l2']} for name in ['R82','R123']}
    record={'scope':'Three radial counts; fixed B16/A12, same analytic map/domain/equations. Active minimum Omega decreases. Fixed region 0.375<=r<0.75 is inside the transition layer and excludes the outer boundary collar. Not a coupled convergence proof.',
            'cases':reviews,'final_metrics':final,'fixed_layer_relative_change_from_R41':changes}
    (a.capsule/'puncture-radial-review.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'axes.grid':True,'grid.alpha':.2,'font.size':10})
    fig,axs=plt.subplots(1,2,figsize=(11,4.7),layout='constrained')
    for name,color in zip(profiles,['#333333','#0072b2','#d55e00']):
        for ax,key in zip(axs,['H_l2','M_conformal_l2']):
            ax.semilogy([r['radius_bin_midpoint'] for r in profiles[name]],
                        [r[key] for r in profiles[name]],'o-',color=color,label=name,ms=3)
            ax.axvspan(.35,.75,color='#888888',alpha=.04)
            ax.set(xlabel='Radius (bin midpoint)',ylabel=('H' if key=='H_l2' else 'M')+' RMS in fixed radial bins')
    h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncols=3,frameon=False)
    fig.suptitle('Puncture radial refinement at t = 0.05\nB16/A12 fixed; common maximum timestep 0.00008',fontsize=13)
    fig.savefig(a.capsule/'puncture-radial.png',dpi=180)
    fig.savefig(a.capsule/'puncture-radial.pdf')
    print(json.dumps({'final_metrics':final,'fixed_layer_relative_change_from_R41':changes},indent=2))


if __name__=='__main__':main()
