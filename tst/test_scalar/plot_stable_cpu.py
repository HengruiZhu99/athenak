#!/usr/bin/env python3
"""Plot the completed corrected CPU admission without replacing old figures."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scalar_evidence import history
E=Path(__file__).resolve().parents[2]/'docs/validation/test-scalar-20261016';R=E/'cpu-v22';d=json.loads((R/'stable-full-review.json').read_text());long=json.loads((R/'long-minkowski-review.json').read_text());out=E/'stable-minkowski-cpu-validation-v22'
if out.with_suffix('.png').exists():raise RuntimeError('Existing plot is immutable')
fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained');ns=np.array(d['resolutions'])
for field,label in [('psi','Scalar'),('Dpsi','Spatial derivative'),('reduction','Reduction')]:
 row=next(c for c in d['checks'] if c['time']==12 and c['region']=='all' and c['norm']=='L2' and c['field']==field);axes[0,0].loglog(ns,row['errors'],'o-',label=label)
axes[0,0].set(title='Three-resolution field and derivative convergence',xlabel='Resolution N',ylabel='Volume L2 error at t=12');axes[0,0].legend(fontsize=8);axes[0,0].grid(alpha=.2)
for field,label in [('psi','Scalar'),('v','Velocity'),('Dpsi','Spatial derivative')]:
 rows=[c for c in d['timestep'] if c['field']==field and c['norm']=='L2'];axes[0,1].semilogy([r['time'] for r in rows],[r['fraction'] for r in rows],'o-',label=label)
axes[0,1].axhline(.1,color='black',ls='--',lw=1,label='Declared limit');axes[0,1].set(title='Half-timestep difference / spatial error gap',xlabel='Time',ylabel='Error fraction',ylim=(1e-10,1));axes[0,1].legend(fontsize=8)
axes[1,0].bar(['0.90','0.95'],[r['relative_L2'] for r in d['boundary']],color=['#356c9c','#32764c']);axes[1,0].axhline(.01,color='black',ls='--',lw=1,label='Declared limit');axes[1,0].set(yscale='log',ylim=(1e-6,.03),title='Finite-cap waveform sensitivity to exact scri',xlabel='Outer compact radius / scri radius',ylabel='Relative waveform L2 difference');axes[1,0].legend(fontsize=8)
h=history(Path(long['run']));rows=sorted((t,r) for (t,region),r in h.items() if region=='all');axes[1,1].semilogy([t for t,r in rows],[float(r['psi_max'])/long['initial_peak'] for t,r in rows],color='#356c9c');axes[1,1].plot([20,100],[.01,.01],'k--',lw=1,label='Late stability limit');axes[1,1].axvline(20,color='gray',lw=1,alpha=.5);axes[1,1].set(title='N12 stability through t=100',xlabel='Time',ylabel='Maximum scalar / initial maximum');axes[1,1].legend(fontsize=8)
fig.suptitle('Stable physical smooth-layer evaluation: all Minkowski CPU gates PASS',fontsize=13);fig.savefig(out.with_suffix('.png'),dpi=180);fig.savefig(out.with_suffix('.pdf'))
