#!/usr/bin/env python3
"""Preserve the independent before/after endpoint geometry evidence."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
E=Path(__file__).resolve().parents[2]/'docs/validation/test-scalar-20261016'
out=E/'smooth-endpoint-validation-v22'
if out.with_suffix('.png').exists():raise RuntimeError('Existing plot is immutable')
fig,axes=plt.subplots(1,2,figsize=(10.5,4.2),layout='constrained')
for version,label,color,marker in [('cpu-v21','Previous evaluation','#b33b35','x'),('cpu-v22','Stable evaluation','#23764d','o')]:
 rows=json.loads((E/version/'smooth-endpoint-review.json').read_text())['records']
 for ax,M in zip(axes,[0,1]):
  rr=[r for r in rows if r['M']==M and 15.8<r['rho']<16]
  ax.scatter([16-r['rho'] for r in rr],[max(r['max_scaled_error'],1e-18) for r in rr],label=label,color=color,marker=marker,s=32,alpha=.75)
for ax,M in zip(axes,[0,1]):
 ax.axhline(1e-9,ls='--',color='black',lw=1,label='Declared tolerance')
 ax.set(xscale='log',yscale='log',xlabel=r'Distance to layer endpoint $16-\rho$',ylabel='Maximum scaled physical coefficient error',title='Minkowski' if M==0 else 'Trumpet black holes: spins 0, ±0.5, 0.9',ylim=(1e-18,.2));ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Independent 80-digit physical-metric pullback: endpoint roundoff fixed',fontsize=12)
fig.savefig(out.with_suffix('.png'),dpi=180);fig.savefig(out.with_suffix('.pdf'))
