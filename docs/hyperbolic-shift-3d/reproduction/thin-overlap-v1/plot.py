#!/usr/bin/env python3
"""Reproduce preliminary geometry, trace consistency and preflight costs."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
traces=json.loads((root/'shifted-trace-audit.json').read_text())
preflight=json.loads((root/'preflight/results.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
a=axes[0,0];theta=np.linspace(0,2*np.pi,501)
a.plot(.16*np.cos(theta),.16*np.sin(theta),label='spherical inner boundary')
for layers in (1,3):
 h=.36/32;w=.18+(layers-.5)*h
 a.plot(np.array([-w,w,w,-w,-w]),np.array([-w,-w,w,w,-w]),label=f'Cartesian trace: {layers} layer(s)')
a.set(aspect='equal',xlabel='x',ylabel='y',title='Equatorial geometry (n=16)');a.legend(fontsize=8)
a=axes[0,1];n16=[r for r in traces if r['n']==16]
base=json.loads((root/'trace-audit.json').read_text());n16+=[r for r in base if r['n']==16 and r['layers']==3]
a.bar([r['layers'] for r in n16],[r['evolved'] for r in n16],color='#4477aa')
a.set(xlabel='Extra Cartesian evolved layers',ylabel='Evolved points',title='Shell extensions unchanged (n=16)',xticks=[1,2,3]);a.set_ylim(0,175000)
a=axes[1,0];r=sorted([r for r in traces if r['layers']==1],key=lambda x:x['n'])
a.loglog([x['n'] for x in r],[x['oblique_wave_value_jump_max'] for x in r],'o-',label='value mismatch')
a.loglog([x['n'] for x in r],[x['oblique_wave_normal_jump_max'] for x in r],'s-',label='h-scaled normal mismatch')
a.set(xlabel='Cells per block n',ylabel='Maximum trace mismatch',title='Smooth oblique data: consistency only');a.legend(fontsize=8);a.grid(alpha=.3)
a=axes[1,1];cases=[r for r in preflight if r['name'].startswith('minkowski')];x=np.arange(len(cases));bottom=np.zeros(len(cases))
for label,index,color in [('Bulk RHS',0,'#4477aa'),('Ghost interpolation',4,'#66ccee'),('Boundary/checks',2,'#ccbb44'),('Paired penalty',None,'#ee6677')]:
 vals=np.array([r['report']['phase_seconds_by_rank'][0][index] if index is not None else r['report']['interface_seconds_by_rank'][0] for r in cases]);a.bar(x,vals,bottom=bottom,label=label,color=color);bottom+=vals
a.set_xticks(x,[r['name'].replace('minkowski-','') for r in cases],rotation=30)
a.set(ylabel='Seconds for four steps',title='Serial CPU preflight, excludes setup/MPI/diagnostics');a.legend(fontsize=8)
fig.suptitle('Thin-overlap prototype: initial checks, not a stability or GPU speedup claim')
fig.savefig(root/'preliminary.png',dpi=180);fig.savefig(root/'preliminary.pdf')
