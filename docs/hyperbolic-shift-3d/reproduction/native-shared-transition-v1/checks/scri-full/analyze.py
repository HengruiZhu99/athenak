"""Absolute manufactured truncation errors. No time-evolution claims."""
from pathlib import Path
import json, math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
data = {}
for variant in ['baseline', 'first6']:
    paths = sorted(HERE.glob(f'{variant}-n*.json'), key=lambda p: int(p.stem.split('-n')[1]))
    if paths:
        data[variant] = [json.loads(p.read_text()) for p in paths]
summary = {'scope': 'Regular manufactured snapshot tests, not evolved punctures. Global and seam/interior norms retained; seam band is two angular cells and shrinks with N.', 'variants': {}}
for variant, runs in data.items():
    ns = [r['n'] for r in runs]
    cases = {}
    for j in range(3):
        cases[runs[0]['fixtures'][j]['fixture']] = result = {}
        for key in runs[0]['fixtures'][j]['norms']:
            result[key] = {}
            for region in ['all', 'seam', 'interior']:
                result[key][region] = {}
                for norm in ['rms', 'max_component']:
                    values = [r['fixtures'][j]['norms'][key][region][norm] for r in runs]
                    orders = [math.log(a/b)/math.log(m/n) if a*b>0 else None
                              for a,b,n,m in zip(values,values[1:],ns,ns[1:])]
                    result[key][region][norm] = dict(values=values,effective_orders=orders)
    summary['variants'][variant] = dict(resolutions=ns,fixtures=cases)
fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
colors = ['#b55315', '#1769aa', '#228844']
labels = ['Regular radial wave', 'Regular angular wave + B', 'Nonconstant angular metric']
for variant, runs in data.items():
    ns = [r['n'] for r in runs]
    for j in range(3):
        for ax, key in zip(axes, ['surface_full', 'surface_penalty', 'row1_full']):
            values = [r['fixtures'][j]['norms'][key]['all']['rms'] for r in runs]
            ax.loglog(ns,values,('o-' if variant=='baseline' else 's--'),color=colors[j],
                      label=f'{labels[j]} ({variant})')
for ax,title in zip(axes,['Complete surface RHS','Surface regularity penalty','Closest volume row: complete RHS']):
    ax.set(title=title,xlabel='N',ylabel='Absolute error RMS');ax.grid(alpha=.2)
axes[0].legend(fontsize=7)
fig.suptitle('Scri operator consistency: nonconstant geometry reveals amplification of derivative errors',fontsize=12)
fig.tight_layout(rect=(0,.06,1,.92))
fig.text(.5,.015,'Actual SG1 mesh, exchange, sampled numerator, penalties and device kernels. Analytic regularity and jet references checked independently.',ha='center',fontsize=9)
fig.savefig(HERE/'consistency.png',dpi=160)
(HERE/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for variant,s in summary['variants'].items():
    print(variant,s['resolutions'])
    for name,case in s['fixtures'].items():
        print(name, {key:case[key]['all']['rms']['effective_orders'] for key in ['surface_full','normal_native','surface_penalty','row1_full']})
