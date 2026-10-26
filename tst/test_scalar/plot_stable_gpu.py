#!/usr/bin/env python3
"""Preserve corrected GPU validation alongside the superseded failure."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scalar_evidence import waveform,history as load_history
from matplotlib.ticker import NullFormatter
E=Path(__file__).resolve().parents[2]/'docs/validation/test-scalar-20261016';A=E/'aurora';out=E/'stable-gpu-validation-v22b'
if out.with_suffix('.png').exists():raise RuntimeError('Existing figure is immutable')
old=json.loads((A/'minkowski-undamped-gpu-v4-review.json').read_text());new=json.loads((A/'minkowski-stable-gpu-v5-review.json').read_text());ns=[24,32,40];fig,axes=plt.subplots(2,2,figsize=(11,8),layout='constrained')
for review,label,color in [(old,'Previous smooth-layer evaluation','#b33b35'),(new,'Stable smooth-layer evaluation','#23764d')]:
 values=[max(c['scaled_max_field_derivative_difference'] for c in review['comparisons'] if c['n']==n) for n in ns];axes[0,0].semilogy(ns,values,'o-',color=color,label=label)
axes[0,0].axhline(1e-9,color='black',ls='--',lw=1,label='Unchanged tolerance');axes[0,0].set(title='CPU/GPU pointwise agreement',xlabel='Resolution N',ylabel='Maximum difference / joint initial scale',xticks=ns);axes[0,0].legend(fontsize=8)
for field,label in [('psi','Scalar'),('v','Velocity'),('Dpsi','Spatial derivative'),('reduction','Reduction')]:
 history=[]
 for n in ns:
  history.append(float(load_history(A/f'minkowski-stable-gpu-v5-collected/result/minkowski-n{n}')[12,'all'][field+'_L2']))
 axes[0,1].loglog(ns,history,'o-',label=label)
axes[0,1].set(title='Direct analytic GPU convergence at t=12',xlabel='Resolution N',ylabel='Volume L2 error');axes[0,1].set_xticks(ns,labels=[str(n) for n in ns]);axes[0,1].xaxis.set_minor_formatter(NullFormatter());axes[0,1].legend(fontsize=8)
values=[json.loads((E/version/'flat-cached-coefficient-review.json').read_text())['max_scaled_error'] for version in ['cpu-v21','cpu-v22']];axes[1,0].bar(['Previous','Stable'],values,color=['#b33b35','#23764d']);axes[1,0].axhline(1e-10,color='black',ls='--',lw=1,label='Declared cache-audit limit');axes[1,0].set(yscale='log',title='Independent check of 1,007,432 cached points',ylabel='Maximum scaled physical coefficient error');axes[1,0].legend(fontsize=8)
w=waveform(E/'cpu-schwarzschild-stable/schwarzschild-n24-normal');ref=np.genfromtxt(E/'radial-oracle/smooth/waveform-upwind-n1200.csv',delimiter=',',names=True);ct=-1+(np.arange(24)+.5)*2/24;factor=-.5*np.sqrt(15/(32*np.pi))*np.pi*np.sum((1-ct*ct)**2)*2/24;axes[1,1].plot(w['time'],w['imag22'],label='Corrected seven-block N24');axes[1,1].plot(ref['time'],factor*ref['radial_psi'],'k--',label='Independent radial solution');axes[1,1].set(xlim=(25,75),ylim=(-.0055,.0095),title='Schwarzschild coarse pilot: full-series error 2.18%',xlabel='Time / M',ylabel='Imaginary scalar mode 22');axes[1,1].legend(fontsize=8)
fig.suptitle('Minkowski GPU admission PASS; black-hole scalar validation continues',fontsize=13);fig.savefig(out.with_suffix('.png'),dpi=180);fig.savefig(out.with_suffix('.pdf'))
