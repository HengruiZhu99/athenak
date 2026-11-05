#!/usr/bin/env python3
"""Regenerate the rejected derivative-boundary evidence plot."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--evidence',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();a.output.mkdir(parents=True,exist_ok=False)
 fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained');rows=[]
 for name,label in [('chart-reference-n128-1M-v6','Algebraic D (unqualified)'),('chart-derivative-n128-1M-v7','Derivative closure (rejected)')]:
  h=json.loads((a.evidence/name/'history.json').read_text());axes[0].semilogy([x['t_M'] for x in h],[max(x['pole'],1e-16) for x in h],label=label)
 for n in (32,64):
  folder=a.evidence/f'semidiscrete-derivative-n{n}-initial-v8';d=json.loads((folder/'audit.json').read_text());v=np.load(folder/'linearization.npz');amp=np.max(abs(v['leading_mode']),axis=1);amp/=max(amp)
  axes[1].semilogy(v['r'],np.maximum(amp,1e-18),label=f"N={n}, growth={d['max_real_eigenvalue']:.1f}")
 for label in ('pquarter','initial','pdouble'):
  d=json.loads((a.evidence/f'semidiscrete-derivative-n32-{label}-v8'/'audit.json').read_text());rows.append(d)
 axes[2].plot([x['penalty'] for x in rows],[x['max_real_eigenvalue'] for x in rows],'o-');axes[2].axhline(0,color='k',lw=.7)
 axes[0].set(xlabel='Time / M',ylabel='Maximum scri pole residual',title='Clock identity does not ensure regularity')
 axes[1].set(xlabel='Compact radius',ylabel='Normalized leading mode amplitude',xlim=(.7,1),ylim=(1e-10,2),title='Boundary mode grows as resolution increases')
 axes[2].set(xlabel='Penalty coefficient',ylabel='Largest real eigenvalue (coordinate time)',title='Penalty changes do not remove growth')
 for ax in axes:ax.grid(alpha=.25)
 axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
 fig.savefig(a.output/'boundary-rejection.png',dpi=160)
 (a.output/'summary.json').write_text(json.dumps(dict(penalty_sweep=[dict(penalty=x['penalty'],max_real_eigenvalue=x['max_real_eigenvalue'],min_real_eigenvalue=x['min_real_eigenvalue']) for x in rows],disposition='Derivative closure rejected. Shared D PDE and clock/algebraic identities hold, but resolution-growing boundary modes and regularity errors remain.'),indent=2)+'\n')
if __name__=='__main__':main()
