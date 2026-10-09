"""Static scientific plot of complete projected global tangent histories."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
w=Path(__file__).resolve().parent
fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
for g,color in [('production','#205493'),('spatialnorm','#dc6f20')]:
 d=json.loads((w/f'{g}-projected-expm-analysis.json').read_text());f=json.loads((w/f'{g}-projected-krylov-field-analysis.json').read_text());field={x['name']:x['history'] for x in f['histories']};cross=d['outward_reference_crossing_time']
 for item in d['histories']:
  hist=item['history'];t=np.asarray([x['time']/cross for x in hist]);sty='-' if item['name']=='gauge_pulse' else '--';label=f'{g}: '+('smooth gauge' if sty=='-' else 'shell random');c=np.asarray([x['native_H_M_Z_rms'] for x in hist]);c[c==0]=np.nan
  for j,ax in enumerate(axes.ravel()[:3]):ax.semilogy(t,c[:,j],sty,color=color,label=label);ax.set_ylabel(['Hamiltonian RMS','Momentum conformal RMS','Z conformal RMS'][j]);ax.grid(alpha=.2)
  axes[1,1].semilogy(t,[x['configuration_H1_momentum_L2_amplification'] for x in field[item['name']]],sty,color=color,label=label)
for ax in axes[1]:ax.set_xlabel('t / outward reference light-crossing time')
axes[1,1].set_ylabel('configuration H1 + momenta L2 amplification');axes[1,1].grid(alpha=.2);axes[0,0].legend(fontsize=8);fig.suptitle('N16 wide layer: global projected continuous native tangent\nκ=10, symmetric quadratic spherical ghosts, native KO=0.1',fontsize=12)
fig.savefig(w/'global-tangent-histories.png',dpi=180);fig.savefig(w/'global-tangent-histories.svg')
