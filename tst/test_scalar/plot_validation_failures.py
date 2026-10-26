#!/usr/bin/env python3
"""Plot the current passed accuracy checks and unresolved admission failures."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scalar_evidence import waveform

p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();E=a.evidence
if any(a.output.with_suffix(s).exists() for s in ['.png','.pdf']):raise RuntimeError('Existing figures are immutable')
g=json.loads((E/'aurora/minkowski-undamped-gpu-v4-review.json').read_text());r=json.loads((E/'cpu-schwarzschild-undamped/primary-review.json').read_text());ns=r['resolutions'];fig,axes=plt.subplots(2,2,figsize=(11,7));ax=axes.ravel()
for t in [2,6,12]:ax[0].semilogy(ns,[next(c['scaled_max_field_derivative_difference'] for c in g['comparisons'] if c['n']==n and c['time']==t) for n in ns],'o-',label=f't={t}')
ax[0].axhline(1e-9,color='k',ls='--',label='Declared agreement gate');ax[0].set(title='New Minkowski GPU: CPU agreement FAILED',xlabel='Resolution N',ylabel='Scaled maximum field/derivative difference');ax[0].legend(fontsize=8)
for field,label in [('psi','field'),('v','velocity'),('Dpsi','spatial derivative'),('reduction','reduction')]:
    q=next(c for c in r['field_checks'] if c['time']==100 and c['field']==field and c['norm']=='Linf');v=np.array(q['errors']);ax[1].loglog(ns,v/v[0],'o-',label=label)
ax[1].loglog(ns,(np.array(ns)/ns[0])**-1.5,'k--',label='Minimum order 1.5');ax[1].set(title='Schwarzschild late convergence FAILED',xlabel='Resolution N',ylabel='Maximum error / N24 maximum at t=100');ax[1].legend(fontsize=8)
for field,label,limit in [('omega_real','real frequency',.02),('decay','decay rate',.02)]:ax[2].plot(ns,[100*q['window_spreads'][field] for q in r['QNM']],'o-',label=label)
ax[2].axhline(2,color='k',ls='--',label='Declared window-spread gate');ax[2].set(title='Schwarzschild QNM: finer windows PASSED',xlabel='Resolution N',ylabel='Free-fit window spread (%)');ax[2].legend(fontsize=8)
for d,q in zip(r['runs'],r['independent_waveform']):
    w=waveform(Path(d));ax[3].plot(w['time'],w['imag22'],label=f'N{json.loads((Path(d)/"parameters.json").read_text())["n"]}, error {100*q["relative_L2"]:.3f}%')
ref=np.genfromtxt(E/'radial-oracle/smooth/waveform-upwind-n1200.csv',delimiter=',',names=True);ct=-1+(np.arange(24)+.5)*2/24;z=-.5*np.sqrt(15/(32*np.pi))*np.pi*np.sum((1-ct*ct)**2)*2/24*ref['radial_psi'];ax[3].plot(ref['time'],z,'k--',label='Independent radial solution');ax[3].set(xlim=(25,75),ylim=(-.0055,.0095),title='Schwarzschild waveform: finest accuracy PASSED',xlabel='Time / M',ylabel='Im psi_22');ax[3].legend(fontsize=8)
for panel in ax:panel.grid(alpha=.25);panel.set_xticks(ns) if panel is not ax[3] else None
ax[1].set_xticks(ns,labels=[str(n) for n in ns]);ax[1].minorticks_off();fig.suptitle('Accuracy successes do not override failed agreement or convergence gates',fontsize=12);fig.tight_layout();fig.savefig(a.output.with_suffix('.png'),dpi=160);fig.savefig(a.output.with_suffix('.pdf'));plt.close(fig)
