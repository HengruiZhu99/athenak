#!/usr/bin/env python3
"""Plot group convergence and the component-level KO control, preserving failures."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
read=lambda path:json.loads((a.root/path).read_text())
t=read('intrinsic-time-convergence-001/results.json')
s=read('intrinsic-space-convergence-001/results.json')
with_ko=read('intrinsic-space-convergence-001/fd6-signed-differences-components.json')['components']
without=read('intrinsic-space-convergence-ko0-001/components.json')['components']
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for name,label in [('primary_curvature_GH','First 20 fields'),('all_fields','All 50 fields')]:
    axes[0,0].loglog([.001,.0005],t['groups'][name]['difference_L2'],'o-',label=label)
axes[0,0].set(xlabel='Coarser timestep',ylabel='L2 of successive state differences',title='Fixed-grid RK3 convergence (orders ≈ 3)')
for row in s:axes[0,1].loglog([1/16,1/32],row['groups']['all_fields']['difference_L2'],'o-',label=f"FD{row['order']}")
axes[0,1].set(xlabel='Coarser relative spacing (1/N)',ylabel='L2 difference on common coarse points',title='Grouped spatial convergence, KO=0.3')
for values,label in [(with_ko,'KO=0.3'),(without,'KO=0')]:
    axes[1,0].plot(range(50),[v['observed_order'] for v in values],'.-',label=label)
    axes[1,1].plot(range(50),[v['alignment'] for v in values],'.-',label=label)
axes[1,0].axhline(6,color='gray',ls='--',lw=1)
axes[1,0].set(xlabel='Intrinsic state index',ylabel='Observed order',title='FD6 component audit')
axes[1,1].axhline(.99,color='gray',ls='--',lw=1)
axes[1,1].annotate('rho',xy=(1,with_ko[1]['alignment']),xytext=(8,.8),arrowprops={'arrowstyle':'->'})
axes[1,1].set(xlabel='Intrinsic state index',ylabel='Difference-vector alignment',ylim=(.7,1.01),title='Aggregate rates hide poorly aligned components')
for ax in axes.flat:ax.grid(alpha=.2);ax.legend(fontsize=9)
fig.suptitle('Smooth off-constraint PDE fixture, T=0.02 — not Einstein-data qualification',fontsize=13)
fig.savefig(a.output,dpi=150)
