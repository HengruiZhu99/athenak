#!/usr/bin/env python3
"""Plot audited GPU agreement and the remaining CPU admission failures."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scalar_evidence import waveform,history

p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();E=a.evidence
if any(a.output.with_suffix(s).exists() for s in ['.png','.pdf']):raise RuntimeError('Existing figures are immutable')
g=json.loads((E/'aurora/minkowski-gpu-v3-review.json').read_text());fig,axes=plt.subplots(2,2,figsize=(11,7));ax=axes.ravel();ns=[12,18,24]
for key,label in [('scaled_max_field_difference','field'),('scaled_max_derivative_difference','spatial derivative')]:ax[0].semilogy(ns,[max(c[key] for c in g['cpu_comparisons'] if c['n']==n) for n in ns],'o-',label=label)
ax[0].axhline(1e-9,color='k',ls='--',label='Declared agreement gate');ax[0].set(title='Original Minkowski: GPU / CPU agreement PASSED',xlabel='Resolution N',ylabel='Maximum scaled point difference');ax[0].legend(fontsize=8)
for field,label in [('psi','field'),('Dpsi','spatial derivative')]:
    values=[float(history(E/f'aurora/minkowski-gpu-v3-collected/result/minkowski-n{n}')[6,'all'][field+'_L2']) for n in ns];ax[1].loglog(ns,values,'o-',label=label)
ax[1].set(title='Original Minkowski: GPU convergence PASSED',xlabel='Resolution N',ylabel='Compact-volume L2 error at t=6');ax[1].legend(fontsize=8)
fixed=json.loads((E/'cpu-v20/fine-review.json').read_text());scaled=json.loads((E/'cpu-v21/scaled-reduction-review.json').read_text())
for report,label in [(fixed,'Fixed damping: failed'),(scaled,'Grid-scaled damping: failed')]:
    row=next(c for c in report['checks'] if c['region']=='all' and c['time']==12 and c['field']=='reduction' and c['norm']=='Linf');ax[2].loglog(report['resolutions'],row['errors'],'o-',label=label)
row=next(c for c in fixed['checks'] if c['region']=='all' and c['time']==12 and c['field']=='reduction' and c['norm']=='Linf');n=np.array(fixed['resolutions']);ax[2].loglog(n,row['errors'][0]*(n/n[0])**-1.5,'k--',label='Minimum order 1.5');ax[2].set(title='New SBP scalar: reduction convergence FAILED',xlabel='Resolution N',ylabel='Maximum |q-D psi| at t=12');ax[2].legend(fontsize=8)
w=waveform(E/'cpu-v20/schwarzschild-n24-normal-sbp-pilot');ref=np.genfromtxt(E/'radial-oracle/smooth/waveform-upwind-n1200.csv',delimiter=',',names=True);z=-.5*np.sqrt(15/(32*np.pi))*np.pi*np.sum((1-(-1+(np.arange(24)+.5)*2/24)**2)**2)*2/24*ref['radial_psi'];ax[3].plot(w['time'],w['imag22'],label='Seven-block N24 pilot');ax[3].plot(ref['time'],z,'--',label='Independent radial solution');ax[3].set(xlim=(25,75),title='Schwarzschild pilot: stable, accuracy FAILED',xlabel='Time / M',ylabel='Im psi_22');ax[3].legend(fontsize=8)
for panel in ax:panel.grid(alpha=.25)
for panel,ticks in [(ax[1],ns),(ax[2],[24,32,40])]:panel.set_xticks(ticks,labels=[str(n) for n in ticks]);panel.minorticks_off()
ax[3].set_ylim(-.0055,.0095)
fig.suptitle('Frozen scalar evidence — original Minkowski validated; black holes remain unadmitted',fontsize=12);fig.tight_layout();fig.savefig(a.output.with_suffix('.png'),dpi=160);fig.savefig(a.output.with_suffix('.pdf'));plt.close(fig)
