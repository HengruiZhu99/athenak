#!/usr/bin/env python3
"""Plot completed allocation data with explicit unfinished-run scope."""
import argparse,hashlib,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scalar_evidence import waveform,history,snapshot
p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=json.loads(a.review.read_text());E=a.evidence.resolve();limit=r['closed_prefix_time'];base=E/'aurora/cpu-schwarzschild-commensurate-v1-collected/result';runs=[base/'schwarzschild-n24-fixed-cube',base/'schwarzschild-n48-fixed-cube',Path(r['lineage'][0]['run'])];ns=[24,48,96];colors=['#d55e00','#0072b2','#009e73']
if a.output.with_suffix('.png').exists():raise RuntimeError('Existing partial-evidence plot is immutable')
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for n,run,color in zip(ns,runs,colors):
 w=waveform(run);mask=w['time']<=limit;axes[0,0].plot(w['time'][mask],w['imag22'][mask],color=color,lw=1,label=f'N{n}');initial=snapshot(run,0);outside=np.linalg.norm(initial[:,:3],axis=1)>=1.2;scale=np.linalg.norm(initial[outside,6:9],axis=1).max();rows=sorted((t,float(q['reduction_Linf'])/scale) for (t,region),q in history(run).items() if region=='exterior' and t<=limit);axes[1,1].semilogy(*np.array(rows).T,color=color,label=f'N{n}')
ref=np.genfromtxt(E/'radial-oracle/smooth/waveform-upwind-n4800.csv',delimiter=',',names=True);mask=ref['time']<=limit;ct=-1+(np.arange(24)+.5)*2/24;factor=-.5*np.sqrt(15/(32*np.pi))*np.pi*np.sum((1-ct*ct)**2)*2/24;axes[0,0].plot(ref['time'][mask],factor*ref['radial_psi'][mask],'k--',lw=1,label='Independent4800');axes[0,0].set(xlabel='t/M',ylabel='Imaginary scalar mode22',title=f'Closed waveform through {limit:g}M');axes[0,0].legend(fontsize=8)
times=[2,6,12]
for k,label in enumerate(['N24/N48','N48/N96']):axes[0,1].plot(times,[min(c['orders'][k] for c in r['checks'] if c['time']==t) for t in times],'o-',label=label)
axes[0,1].axhline(1.5,color='black',ls='--',lw=1,label='Required1.5');axes[0,1].set(xlabel='t/M',ylabel='Minimum pair order',title='Both norms: field, velocity, gradient, reduction',xticks=times);axes[0,1].legend(fontsize=8)
axes[1,0].bar(times,[100*max(c['fraction'] for c in r['reference_uncertainty'] if c['time']==t) for t in times],width=1.2,color='#009e73');axes[1,0].axhline(10,color='black',ls='--',lw=1,label='Required10%');axes[1,0].set(xlabel='t/M',ylabel='Maximum reference difference / field error (%)',title='2400/4800 uncertainty at actual N96 points',xticks=times);axes[1,0].legend(fontsize=8)
axes[1,1].axhline(.02,color='black',ls='--',lw=1,label='Finest required0.02');axes[1,1].set(xlabel='t/M',ylabel='Reduction maximum / initial gradient peak',title='Every closed half-time report');axes[1,1].legend(fontsize=8)
for ax in axes.flat:ax.grid(alpha=.2)
fig.suptitle(f'Closed-prefix scalar evidence through {limit:g}M; no stage admission\n100M and timestep/inner/outer controls remain outstanding',fontsize=12)
for suffix in ['.png','.pdf']:fig.savefig(a.output.with_suffix(suffix),dpi=170)
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest();a.output.with_suffix('.json').write_text(json.dumps(dict(admission=False,review_sha256=sha(a.review),source_sha256=sha(Path(__file__)),outputs={suffix:sha(a.output.with_suffix(suffix)) for suffix in ['.png','.pdf']},native_trajectories_added=0),indent=2)+'\n')
