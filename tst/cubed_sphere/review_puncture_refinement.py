#!/usr/bin/env python3
"""Review independent bulk/angular refinement at a common time and step cap."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from review_puncture_comparison import spherical


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('capsule',type=Path);p.add_argument('results',type=Path)
    a=p.parse_args();reviews={};curves={};final={}
    for name in ['B16-A12','B32-A12','B16-A24','B32-A24']:
        case=a.results/name;review,_=spherical(case)
        receipt=json.loads((case/'receipt.json').read_text())
        assert receipt['reached_target'] and receipt['reference_unchanged'] and receipt['initial_profile_unchanged']
        steps=list(csv.DictReader((case/'cubed_sphere_conformal_steps.csv').open()))
        review['max_dt']=max(float(r['dt']) for r in steps)
        review['min_dt']=min(float(r['dt']) for r in steps)
        assert review['max_dt']<=.000250000000001
        reviews[name]=review
        with (case/'cubed_sphere_conformal.csv').open() as f:
            curves[name]=list(csv.DictReader(f))
        final[name]={r['region']:{k:float(r[k]) for k in ['H_l2','H_linf','M_conformal_l2','min_Omega']}
                     for r in curves[name][-2:]}
    changes={}
    for name in ['B32-A12','B16-A24','B32-A24']:
        changes[name]={r:{k:final[name][r][k]/final['B16-A12'][r][k]-1
                         for k in ['H_l2','H_linf','M_conformal_l2']} for r in ['bulk','shell']}
    record={'scope':'Two-level bulk/angular factorial control; same map, domain, radial grid, equations and fixed initial profile policy. Angular sampling changes minimum Omega. Not a convergence proof.',
            'cases':reviews,'final_metrics':final,'relative_change_from_B16_A12':changes}
    (a.capsule/'puncture-refinement-review.json').write_text(json.dumps(record,indent=2,allow_nan=False)+'\n')
    plt.rcParams.update({'axes.grid':True,'grid.alpha':.2,'font.size':10})
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for name,color in zip(curves,['#333333','#d55e00','#0072b2','#009e73']):
        for row,region in enumerate(['bulk','shell']):
            data=[r for r in curves[name] if r['region']==region and float(r['time'])>0]
            for col,key in enumerate(['H_l2','M_conformal_l2']):
                axs[row,col].semilogy([float(r['time']) for r in data],[float(r[key]) for r in data],label=name,color=color)
                axs[row,col].set(xlabel='Coordinate time',ylabel=region+' '+('H' if col==0 else 'M')+' RMS')
    h,l=axs[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncols=4,frameon=False)
    fig.suptitle('Puncture: independent bulk and shell angular refinement\n41 radial cells per patch; common maximum timestep 0.00025',fontsize=13)
    fig.savefig(a.capsule/'puncture-refinement.png',dpi=180)
    fig.savefig(a.capsule/'puncture-refinement.pdf')
    print(json.dumps({'final_metrics':final,'relative_change_from_B16_A12':changes},indent=2))


if __name__=='__main__':main()
