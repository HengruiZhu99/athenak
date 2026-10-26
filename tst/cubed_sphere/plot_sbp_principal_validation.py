#!/usr/bin/env python3
import argparse,json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('coarse',type=Path);p.add_argument('fine',type=Path);p.add_argument('output',type=Path);a=p.parse_args();coarse=json.loads(a.coarse.read_text());fine=json.loads(a.fine.read_text());fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for index,(r,s) in enumerate(zip(coarse['records'],fine['records'])):
 runs=r['runs'][:2]+s['runs'];n=[x['n'] for x in runs];label='pulse samples' if index==0 else ('puncture samples' if index==5 else None);color='tab:blue' if index<5 else 'tab:orange'
 for ax,field in zip(axes.flat,['q','v','Dq','energy']):
  values=[x['history'][-1]['energy_ratio'] if field=='energy' else x['history'][-1]['errors'][field]['l2'] for x in runs];ax.loglog(n,values,'o-',color=color,alpha=.4,label=label);ax.set_xlabel('Radial cells');ax.set_ylabel('Energy / initial energy' if field=='energy' else field+' weighted RMS error');ax.grid(alpha=.25)
axes[1,1].axhline(1e-4,color='black',ls='--',label='Declared settling gate');axes[0,0].legend();axes[1,1].legend();fig.suptitle('Frozen principal SBP-SAT model: all nine sampled states\nCoarse failures retained; finer 128/192/256 sequence passes. No native stability claim.',fontsize=12);fig.savefig(a.output.with_suffix('.png'),dpi=160);fig.savefig(a.output.with_suffix('.pdf'));plt.close(fig)
