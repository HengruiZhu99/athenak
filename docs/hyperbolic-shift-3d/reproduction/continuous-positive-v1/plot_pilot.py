#!/usr/bin/env python3
"""Plot archived V10 failure and V11 bounded pilot; no convergence inference."""
from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=next(p for p in Path(__file__).resolve().parents if (p/'CMakeLists.txt').is_file())
raw=root/'output/hyperbolic-core-shell/aurora'
old=raw/'stability-spherical-overlap-puncture-v1/output'
new=raw/'stability-continuous-positive-puncture-v1/output'
a=np.genfromtxt(old/'puncture.history.csv',delimiter=',',names=True)
b=np.genfromtxt(new/'puncture.history.csv',delimiter=',',names=True)
r=json.loads((new/'puncture.report.json').read_text());oldr=json.loads((old/'puncture.report.json').read_text())
cp=np.memmap(old/'puncture.checkpoint',dtype='<f8',mode='r',offset=136,shape=(189512,31))
last_chi=float(cp[:,0].min())
phase=np.asarray(r['phase_seconds_by_rank']);rank=int(np.argmax(phase.sum(axis=1)))
rhs,comm,boundary,diag,halo=phase[rank];other=r['seconds']-sum(phase[rank]);assert other>=0
seconds=np.array([rhs,comm,boundary,diag,halo,other]);labels=['Bulk RHS','MPI communication','Boundary work','Diagnostics','Bulk halo fill','Other / startup']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
fig,(ax,ay)=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
x=np.r_[a['t_over_M'],oldr['t_over_M']];y=np.r_[a['min_chi'],last_chi]
ax.semilogy(x,y,'o--',color='#777777',label='V10: physical χ, block-local KO')
ax.semilogy(b['t_over_M'],b['min_chi'],'-',lw=2,color='#0072B2',label='V11: logarithmic χ, continuous core')
ax.axvline(oldr['t_over_M'],color='#999999',ls=':',lw=1)
ax.annotate('Next RK stage: χ < 0',xy=(oldr['t_over_M'],last_chi),xytext=(.12,7e-7),arrowprops={'arrowstyle':'->','color':'#666666'},fontsize=9)
ax.set(xlabel='t / M',ylabel='Minimum accepted χ',title='Puncture positivity',xlim=(0,.46));ax.grid(alpha=.2);ax.legend(loc='lower left',fontsize=8)
colors=['#0072B2','#E69F00','#CC79A7','#777777','#009E73','#BBBBBB']
percent=100*seconds/r['seconds'];ay.barh(labels,percent,color=colors);ay.invert_yaxis();ay.set(xlabel='Fraction of native wall time (%)',title=f'Exclusive timing on rank {rank}',xlim=(0,50))
for i,(s,p) in enumerate(zip(seconds,percent)):ay.text(p+.6,i,f'{p:.1f}%  ({s:.1f} s)',va='center',fontsize=9)
fig.suptitle('Bounded V11 pilot: n16, Δt=0.0005, one node / 12 GPU tiles\n300 steps to 0.45M; no lapse-floor interventions; max null error 1.6×10⁻¹⁵',fontsize=12)
out=Path(__file__).resolve().parent
fig.savefig(out/'diagnosis-and-cost.png',dpi=180);fig.savefig(out/'diagnosis-and-cost.pdf')
summary=dict(wall_seconds=r['seconds'],steps=r['steps'],rank=rank,seconds=dict(zip(labels,seconds.tolist())),percent=dict(zip(labels,percent.tolist())),old_last_accepted_min_chi=last_chi,new_final_min_chi=r['min_chi'],note='Pilot numerical criteria qualified by explicit metadata audit. No convergence or 3M claim. Diagnostics every .01; timing includes horizon work.')
(out/'cost-analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
