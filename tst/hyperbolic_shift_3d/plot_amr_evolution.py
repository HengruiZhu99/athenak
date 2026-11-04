#!/usr/bin/env python3
"""Plot preserved bounded histories, including rejected runs without extrapolation."""
import argparse,json,pathlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--evidence',required=True);a=p.parse_args();root=pathlib.Path(a.evidence)
fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
wave=[('restart-v1/wave-full','Wave AMR'),('matrix-v1/wave-uniform','Wave uniform fine')]
boost=[('controls-v1/boost-fixed','Boost fixed mixed'),('controls-v1/boost-rejected','Boost dynamic (rejected)'),('controls-v1/boost-uniform','Boost uniform fine')]
for base,label in wave+boost:
    rows=[json.loads(s) for s in (root/(base+'.history.jsonl')).read_text().splitlines()];t=[r['time']*3 for r in rows]
    if (base,label) in wave:
        axes[0,0].plot(t,[r['H'] for r in rows],'.-',label=label)
        axes[1,0].plot(t,[r['leaves'] for r in rows],'.-',label=label)
    else:
        axes[0,1].plot(t,[r['H'] for r in rows],'.-',label=label)
        accepted=[r for r in rows if r['horizon_found']]
        axes[0,2].plot([r['time']*3 for r in accepted],[(r['mass']-1/3)*1e6 for r in accepted],'.-',label=label)
        axes[1,2].plot(t,[r['tracker'][0] for r in rows],'.-',label=label)
    axes[1,1].plot(t,[max(r['null'],r['ghost_null']) for r in rows],'.-',label=label)
labels=[('Wave constraints','Hamiltonian norm'),('Boost constraints','Hamiltonian norm'),('Accepted horizon masses','Mass − 1/3 (×10⁻⁶)'),('Dynamic topology','Cartesian leaves'),('Boundary closure','Maximum null residual'),('Puncture tracker','x coordinate')]
for ax,(title,ylabel) in zip(axes.flat,labels):
    ax.set(title=title,xlabel='t / M (M = 1/3)',ylabel=ylabel);ax.grid(alpha=.3);ax.legend(fontsize=7)
fig.suptitle('Fresh CPU MPI tests · dt = 0.00025 · only 0.0045M evolved',fontsize=13)
fig.savefig(root/'evolution.png',dpi=180)
