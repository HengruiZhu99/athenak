#!/usr/bin/env python3
"""Merge verified pulse segments, retaining exact recorded samples and boundaries."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_spherical import rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evidence',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--continuation',type=Path,action='append',default=[])
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    sources=[(directory,json.loads((directory/'chain-review.json').read_text()))
             for directory in [args.evidence,*args.continuation]]
    series={};inputs={}
    colors={'degree2':'#0072b2','degree4':'#d55e00','half-degree2':'#009e73','half-degree4':'#cc79a7'}
    for name in colors:
        merged={}
        for directory,review in sources:
            for segment in review['segments']:
                case=directory/segment['job_id'].split('.')[0]/name
                path=case/'cubed_sphere_conformal.csv'
                if not path.exists():continue
                inputs[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
                for row in rows(path):
                    key=(float(row['time']),int(row['cycle']),row['region'])
                    if key in merged:assert merged[key]==row
                    merged[key]=row
        if not merged:continue
        series[name]=[r for _,r in sorted(merged.items())]
        with (args.output/(name+'-history.csv')).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(series[name][0]));writer.writeheader();writer.writerows(series[name])
    fig,axes=plt.subplots(2,3,figsize=(13,7.2),layout='constrained')
    keys=['H_l2','M_conformal_l2','Z_conformal_l2','min_alpha','min_chi','H_linf']
    labels=['Hamiltonian RMS','Conformal momentum RMS','Conformal Z RMS','Minimum lapse','Minimum chi','Hamiltonian maximum']
    for name in series:
        color=colors[name]
        for region,style in [('bulk','-'),('shell','--')]:
            data=[r for r in series[name] if r['region']==region]
            for ax,key in zip(axes.flat,keys):
                ax.plot([float(r['time']) for r in data],[float(r[key]) for r in data],
                        style,color=color,lw=1.1,label=name+' '+region)
    for ax,key,label in zip(axes.flat,keys,labels):
        ax.set(xlabel='Evolution time',ylabel=label,xlim=(0,2))
        if key not in ['min_alpha','min_chi']:ax.set_yscale('log')
        for time in [.7927734374996852,1.5837303635638276]:ax.axvline(time,color='#777777',lw=1,ls=':')
        ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    end={name:max(float(r['time']) for r in data) for name,data in series.items()}
    fig.suptitle('Matched-collar pulse · original parameters · immutable Minkowski reference')
    fig.supxlabel('Coordinate-volume norms; actual samples only. Dotted lines: archived Cartesian failure times.\n'
                   +'Last verified times: '+', '.join(f'{name} = {time:.6f}' for name,time in end.items())+'. Survival is not convergence.',fontsize=10)
    for suffix in ['png','pdf']:fig.savefig(args.output/('pulse-history.'+suffix),dpi=180)
    (args.output/'plot-inputs.json').write_text(json.dumps({'input_sha256':inputs,'end_times':end,
        'scope':'Exact segment-boundary deduplication, no extrapolation or synthetic samples; Cartesian thresholds only, not Cartesian norm curves.'},indent=2)+'\n')


if __name__=='__main__':main()
