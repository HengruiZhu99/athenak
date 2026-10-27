#!/usr/bin/env python3
"""Assess failed endpoint candidates without weakening fixed acceptance criteria."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
records=json.loads((a.runs/'summary.json').read_text());byname={d['name']:d for d in records}
spatial=[byname[f'pulse-n{n}-cfl05'] for n in [32,64,128]]
summary={'kind':'failed actual-scri diagnostic; no nonlinear stability or physical reproduction acceptance',
 'spatial':[],'endpoint_gate':1e-7,'plot_sha256':{},'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
for d in spatial:
 row=d['last_report'];summary['spatial'].append({'n':int(d['name'].split('-')[1][1:]),'scri_pole_max':d['max_scri_pole'],
   **{k:row[k] for k in ['H_rms','H_bulk_max','H_endpoint_max','M_rms','M_bulk_max','M_endpoint_max','Theta_rms','Z_rms','elliptic_sampled_residual']}})
for key in ['H_rms','H_bulk_max','M_rms','M_bulk_max','scri_pole_max']:
 values=[d[key] for d in summary['spatial']];hs=[1/(n-.5) for n in [32,64,128]]
 summary[key+'_pair_orders']=[float(np.log(values[i]/values[i+1])/np.log(hs[i]/hs[i+1])) for i in range(2)]

def checkpoint(name):
 with np.load(a.runs/name/'checkpoint.npz') as z:return z['q'][:,:7].copy()
q=[checkpoint(name) for name in ['pulse-n64-cfl05','pulse-n64-cfl0.025','pulse-n64-cfl0.0125']]
diff=[float(np.linalg.norm(q[i]-q[i+1])) for i in range(2)]
summary['temporal']={'state_difference_L2':diff,'ratio':diff[0]/diff[1],'observed_order':float(np.log2(diff[0]/diff[1])),
 'interpretation':'RK4 temporal consistency of this failed spatial discretization; not continuum correctness'}
summary['restart']=json.loads((a.runs/'restart-review.json').read_text())
summary['all_endpoint_pole_gates_passed']=all(d['scri_pole_max']<=1e-7 for d in summary['spatial'])
fig,axes=plt.subplots(1,3,figsize=(12,3.6),constrained_layout=True)
for n,color in zip([32,64,128],['#2563eb','#d97706','#15803d']):
 hist=[json.loads(s) for s in (a.runs/f'pulse-n{n}-cfl05'/'history.jsonl').read_text().splitlines()]
 t=[d['t'] for d in hist]
 axes[0].semilogy(t,[max(d['scri_pole'],1e-16) for d in hist],label=f'N={n}',color=color)
 axes[1].semilogy(t,[max(d['H_rms'],1e-16) for d in hist],color=color)
 profiles=sorted((a.runs/f'pulse-n{n}-cfl05').glob('profile-*.npz'))
 with np.load(profiles[-1]) as z:axes[2].semilogy(z['r'],np.maximum(abs(z['diagnostic'][:,1]),1e-16),color=color)
axes[0].axhline(1e-7,color='black',ls='--',lw=1,label='Declared pole limit')
axes[0].set(title='Scri compatibility fails under refinement',xlabel='Hyperboloidal time',ylabel='Max pole numerator');axes[0].legend(fontsize=8)
axes[1].set(title='Hamiltonian constraint',xlabel='Hyperboloidal time',ylabel='Volume-weighted RMS')
axes[2].set(title='Momentum error at t=1',xlabel='Compact radius r',ylabel='Absolute M_r')
for ax in axes:ax.grid(alpha=.2)
fig.suptitle('Experimental elliptic-shift Z4c: failed weak-penalty endpoint candidate',fontsize=12)
for ext in ['png','pdf']:
 f=a.output/('endpoint-refinement-v2.'+ext);fig.savefig(f,dpi=180);summary['plot_sha256'][f.name]=hashlib.sha256(f.read_bytes()).hexdigest()
(a.output/'diagnostic-assessment-v2.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
