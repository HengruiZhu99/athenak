#!/usr/bin/env python3
import argparse,json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);a=p.parse_args();d=pathlib.Path(a.evidence)
x=json.loads((d/'accuracy.json').read_text());fig,ax=plt.subplots(1,2,figsize=(10,4),layout='constrained')
for name,label in [('first','First derivative'),('second','Second derivative'),('mixed','Mixed derivative'),('advection','Biased derivative'),('ko_interpolation','KO halo interpolation')]:
    ns=np.array([r['n'] for r in x['operators']]);err=[r[name] for r in x['operators']];ax[0].loglog(ns,err,'o-',label=label)
ax[0].set(xlabel='Cells per block (N)',ylabel='Maximum absolute error',title='Mixed-level smooth-data stencil audit');ax[0].legend(fontsize=8);ax[0].grid(True,which='both',alpha=.3)
ns=np.array([r['n'] for r in x['transfer']]);err=np.array([r['state_error'] for r in x['transfer']]);ax[1].loglog(ns,err,'o-',label='Native refine/coarsen cycle');ax[1].loglog(ns,err[0]*(ns/ns[0])**-5,'--',label='N⁻⁵ reference');ax[1].set(xlabel='Cells per block (N)',ylabel='Maximum field error',title='AthenaK point-transfer accuracy');ax[1].legend(fontsize=8);ax[1].grid(True,which='both',alpha=.3)
for axis in ax:
    axis.set_xticks([12,16,24],labels=['12','16','24']);axis.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
fig.suptitle('CPU SERIAL validation · consistency tests, not an evolution convergence proof',fontsize=11);fig.savefig(d/'accuracy.png',dpi=180);plt.close(fig)
