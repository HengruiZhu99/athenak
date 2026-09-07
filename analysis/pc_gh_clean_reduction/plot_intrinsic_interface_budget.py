#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();d=json.loads(a.results.read_text());n=np.array([8,16,32]);fig,axes=plt.subplots(1,3,figsize=(13,3.7),layout='constrained')
for ax,key,title in zip(axes[:2],['curl','qcurl'],['Intrinsic curl error','Reconstructed Q-curl error']):
 for mode,label in [('before','Ordinary transfer'),('after','Residual reconstruction')]:
  ax.loglog(n,[r['metrics'][key+'_error_'+mode]['group_RMS'] for r in d['records']],'o-',label=label)
 ax.set(title=title,xlabel='Cells per block edge',ylabel='Full-area RMS error');ax.grid(True,which='both',alpha=.2);ax.set_xticks(n,labels=n)
ax=axes[2]
for mode,label in [('before','Ordinary transfer'),('after','Residual reconstruction')]:
 ax.loglog(n,[r['region_metrics']['level0-near2-block-faces']['metrics']['curl_error_'+mode]['maximum'] for r in d['records']],'o-',label=label)
ax.set(title='Coarse block corner layers',xlabel='Cells per block edge',ylabel='Maximum intrinsic curl error');ax.grid(True,which='both',alpha=.2);ax.set_xticks(n,labels=n)
[ax.xaxis.set_minor_formatter(NullFormatter()) for ax in axes]
axes[0].legend(fontsize=8);fig.suptitle('FD6 static periodic initial transfer: reconstruction increases curl error')
fig.savefig(a.output,dpi=170)
