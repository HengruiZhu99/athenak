#!/usr/bin/env python3
"""Plot accepted puncture evolution and retain unavailable horizon samples as gaps."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=next(p for p in Path(__file__).resolve().parents if (p/'CMakeLists.txt').is_file())
raw=root/'output/hyperbolic-core-shell/aurora'
first=raw/'stability-continuous-positive-puncture-v1/output'
last=raw/'stability-continuous-positive-puncture-to3m-v1/output'
a=np.genfromtxt(first/'puncture.history.csv',delimiter=',',names=True)
b=np.genfromtxt(last/'puncture.history.csv',delimiter=',',names=True)
a=np.concatenate([a,b[b['t']>a['t'][-1]]])
r=json.loads((last/'puncture.report.json').read_text());x=a['t_over_M']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,3,figsize=(12,7),layout='constrained');ax=axes.ravel()
ax[0].plot(x,a['min_chi']);ax[0].set(ylabel='Minimum χ',title='Positive conformal factor')
ax[1].plot(x,a['min_lapse']);ax[1].set(ylabel='Minimum conformal lapse',title='No floor interventions')
ax[2].plot(x,np.abs(3*a['horizon_mass']-1),'o-',ms=3);ax[2].set(ylabel='|M_AH / M − 1|',title='Horizon mass error')
ax[3].semilogy(x,a['H_exterior_L2'],'o-',ms=3,label='Hamiltonian');ax[3].semilogy(x,a['M_exterior_L2'],'o-',ms=3,label='Momentum');ax[3].set(ylabel='Exterior constraint norm',title='Gaps: horizon tolerance not met');ax[3].legend()
ax[4].plot(x,a['horizon_radius'],'o-',ms=3);ax[4].set(ylabel='Coordinate horizon radius',title='Fast-flow horizon')
ax[5].semilogy(x,a['null_surface'],label='Surface');ax[5].semilogy(x,a['null_ghost'],label='Ghost');ax[5].set(ylabel='Maximum null residual to date',title='Algebraic scri closure');ax[5].legend()
for axis in ax:axis.set_xlabel('t / M');axis.set_xlim(0,3);axis.grid(alpha=.2)
fig.suptitle('V11 continuous-core puncture: n16, Δt=0.0005, 12 GPU tiles\nFinite evolution is not a convergence or settled-trumpet claim',fontsize=13)
out=Path(__file__).resolve().parent
fig.savefig(out/'puncture-duration.png',dpi=170);fig.savefig(out/'puncture-duration.pdf')
(out/'duration-analysis.json').write_text(json.dumps(dict(status=r['status'],final_t_over_M=r['t_over_M'],sample_count=len(a),missing_horizon_samples=int(np.count_nonzero(a['horizon_found']==0)),floor_interventions=r['lapse_floor_node_interventions']),indent=2)+'\n')
