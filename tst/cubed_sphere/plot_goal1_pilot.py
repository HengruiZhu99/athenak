#!/usr/bin/env python3
"""Plot archived native pilot data; no evolution or new error calculation."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
p=argparse.ArgumentParser();p.add_argument('evidence',type=Path);a=p.parse_args();out=a.evidence
review=json.loads((out/'native-pilot-review.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(11,7.5),constrained_layout=True)
for case,ax in [('full',axes[0,0]),('stationary',axes[0,1])]:
    directory=Path(review['cases'][case]['case_directories'][-1])
    rows=list(csv.DictReader((directory/'cubed_sphere_conformal.csv').open()))
    for region,style in [('bulk','--'),('shell','-')]:
        r=[q for q in rows if q['region']==region]
        fields=[('H_l2','#b34c35','H'),('M_conformal_l2','#3479a8','M'),('Z_conformal_l2','#578a43','Z')] if case=='full' else [('max_deviation','#3479a8','field drift'),('H_linf','#b34c35','max |H|')]
        for key,color,label in fields:
            x=[float(q['time']) for q in r];y=[float(q[key]) for q in r]
            if case=='stationary':ax.semilogy(x,np.maximum(y,1e-17),style,color=color,label=f'{region} {label}')
            else:ax.plot(x,y,style,color=color,label=f'{region} {label}')
    ax.set(xlabel='Coordinate time');ax.legend(fontsize=8,ncol=2)
    ax.xaxis.set_major_locator(MaxNLocator(5));ax.ticklabel_format(axis='x',style='sci',scilimits=(-2,2),useMathText=True)
axes[0,0].set(title='Small native pulse: constraint RMS',ylabel='L0 = 1; compact coordinate-volume RMS')
axes[0,0].ticklabel_format(axis='y',style='sci',scilimits=(0,0),useMathText=True)
axes[0,1].set(title='Short stationary native control',ylabel='Raw maximum; L0 = 1')
x=np.arange(2);v=review['time_sensitivity']
axes[1,0].bar(x-.17,[v[k]['maximum_component_regional_field_rms'] for k in ['full-half','half-quarter']],.34,label='field RMS',color='#3479a8')
axes[1,0].bar(x+.17,[v[k]['maximum_component_regional_gradient_rms'] for k in ['full-half','half-quarter']],.34,label='gradient RMS',color='#b34c35')
axes[1,0].set(yscale='log',xticks=x,xticklabels=['full/half','half/quarter'],title='Temporal differences at t = .002',ylabel='Largest component/region RMS');axes[1,0].legend(fontsize=8)
for row in json.loads((out/'pilot-radial-profiles.json').read_text()):
    axes[1,1].semilogy(row['radius'],row['max_component_rms'],'o-',ms=3,label=row['pair'])
axes[1,1].set(title='Temporal state difference by radial bin',xlabel='Compact radius',ylabel='Max component weighted RMS');axes[1,1].legend(fontsize=8)
for ax in axes.ravel():ax.grid(alpha=.2)
fig.suptitle('Goal 1 local CPU pilots — finite cut .98, B8/A8/R48\nDiagnostic evidence; no spatial, crossing-time or exact-scri acceptance',fontsize=12)
fig.savefig(out/'pilot-summary.png',dpi=160);fig.savefig(out/'pilot-summary.pdf')
