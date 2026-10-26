#!/usr/bin/env python3
"""Render preserved scalar evidence without overwriting previous figures."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scalar_evidence import history,waveform,snapshot

def main():
 p=argparse.ArgumentParser();p.add_argument('--report',type=Path,required=True);p.add_argument('--evidence',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 for suffix in ['.png','.pdf']:
  if a.output.with_suffix(suffix).exists():raise RuntimeError('Existing plot is immutable')
 r=json.loads(a.report.read_text());E=a.evidence;ns=r['resolutions'];runs=[Path(d) for d in r['runs']];fig,ax=plt.subplots(2,2,figsize=(11,7))
 for field,label in [('psi','field'),('v','velocity reconstruction'),('Dpsi','independent spatial derivative')]:
  row=next(c for c in r['checks'] if c['time']==6 and c['region']=='all' and c['field']==field and c['norm']=='L2');ax[0,0].loglog(ns,row['errors'],'o-',label=label)
 ax[0,0].set_xticks(ns,labels=[str(n) for n in ns]);ax[0,0].minorticks_off();ax[0,0].set(title='Angular Minkowski convergence',xlabel='Resolution N',ylabel='Compact-volume L2 error at t=6');ax[0,0].legend(fontsize=8)
 for n,d in zip(ns,runs):
  h=history(d);scale=np.max(np.linalg.norm(snapshot(d,0)[:,6:9],axis=1));rows=sorted((t,float(v['reduction_Linf'])/scale) for (t,region),v in h.items() if region=='all');ax[0,1].semilogy(*np.array(rows).T,label=f'N{n}')
 ax[0,1].axhline(.02,color='k',ls='--',label='Declared maximum .02');ax[0,1].set(title='Scalar reduction constraint',xlabel='Hyperboloidal time',ylabel='Max |q-D psi| / initial derivative peak');ax[0,1].legend(fontsize=8)
 for d,label in [(runs[-1],'exact scri')]+[(Path(c['run']),f'cap sensitivity {100*c["relative_L2"]:.4f}%') for c in r['boundary']]:
  w=waveform(d);ax[1,0].plot(w['time'],w['imag22'],label=label)
 ax[1,0].set(title='Common rho=10 extraction',xlabel='Hyperboloidal time',ylabel='Im psi_22');ax[1,0].legend(fontsize=8)
 for d,label in [(E/'cpu-v19/schwarzschild-n24-normal','Failed earlier radial closure'),(E/'cpu-v20/schwarzschild-n24-normal-sbp-pilot','SBP kappa=2 pilot, stable to 100')]:
  h=history(d);rows=sorted((t,max(float(v['psi_max']),1e-16)) for (t,region),v in h.items() if region=='all');ax[1,1].semilogy(*np.array(rows).T,label=label)
 ax[1,1].set(title='Schwarzschild N24: stability is not admission',xlabel='Time / M',ylabel='Maximum scalar field');ax[1,1].legend(fontsize=8)
 for panel in ax.flat:panel.grid(alpha=.25)
 damping=json.loads((runs[0]/'parameters.json').read_text())['reduction_damping'];fig.suptitle(f'Minkowski kappa={damping:g}: convergence {"PASSED" if r["convergence_passed"] else "FAILED"}; full gates {"PASSED" if r["passed"] else "NOT PASSED"}',fontsize=12)
 fig.tight_layout();fig.savefig(a.output.with_suffix('.png'),dpi=160);fig.savefig(a.output.with_suffix('.pdf'));plt.close(fig)
if __name__=='__main__':main()
