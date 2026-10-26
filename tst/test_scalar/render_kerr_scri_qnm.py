#!/usr/bin/env python3
"""Render existing exact-scri QNM fits with unobstructed comparison panels."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from localize_schwarzschild_error import COLORS, sha

p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--review',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();r=json.loads(a.review.read_text())
if any(a.output.with_suffix(ext).exists() for ext in ['.png','.pdf','.json']):raise RuntimeError('Existing figure is immutable')
quantities=[('omega_real','Co-rotating frequency',r'$M\omega_{R,+}$',.02),
            ('counterrotating_real','Counter-rotating frequency magnitude',r'$M|\omega_{R,-}|$',.02),
            ('decay','Co-rotating damping rate',r'$M\Gamma_+$',.05),
            ('counterrotating_decay','Counter-rotating damping rate',r'$M\Gamma_-$',.05)]
fig,axes=plt.subplots(2,2,figsize=(12.5,8.8),constrained_layout=True)
for ax,(key,title,label,tol) in zip(axes.flat,quantities):
    target=r['targets'][key]
    ax.axhspan(target*(1-tol),target*(1+tol),color='gray',alpha=.12,label='accuracy guides: 2% frequency / 5% damping')
    ax.axhline(target,color='black',ls='--',lw=1.4,label='independent Leaver target')
    for n,color in zip([12,16,20],COLORS):
        fs=[f for f in r['fits'] if f['n']==n];ax.plot([f['window'][0] for f in fs],[f[key] for f in fs],'o-',color=color,label=f'N={n}',ms=4)
    ax.plot([f['window'][0] for f in r['raw_fits']],[f[key] for f in r['raw_fits']],'s',color='#666666',mfc='none',ms=5,label='N20 raw projection')
    ax.set(xlabel=r'Window start $t_0/M$; end = $t_0/M+30$',ylabel=label,title=title+f'\nExpected {target:.6f}',xticks=r['protocol']['window_starts'])
    ax.grid(alpha=.2)
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='outside lower center',ncol=3,fontsize=9,frameon=False)
fig.suptitle(r'Exact-$\mathscr{I}^{+}$ scalar Kerr QNM fits: $a/M=0.5$, $s=0$, $\ell=2$ fundamentals'+'\nAll 30M windows shown; frequencies and amplitudes fitted freely. Shading is not a confidence interval.',fontsize=12)
for ext in ['.png','.pdf']:fig.savefig(a.output.with_suffix(ext),dpi=170)
plt.close(fig)
a.output.with_suffix('.json').write_text(json.dumps(dict(review=str(a.review.resolve()),review_sha256=sha(a.review),source_sha256=sha(__file__),new_fits=0,new_evolutions=0,output_sha256={ext:sha(a.output.with_suffix(ext)) for ext in ['.png','.pdf']}),indent=2)+'\n')
