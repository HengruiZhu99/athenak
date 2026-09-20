"""Compact scientific figure for the independent spherical annulus diagnostic."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).parent
CASES=[('Current: $\\sigma=0.1\\alpha$', 'collocation-baseline-n128-r02.json','#1f77b4'),('$\\sigma=0.1$', 'collocation-sigma0.1-1-rin0.2-rout4-n128.json','#d95f02'),('$\\sigma=0.3$', 'collocation-sigma0.3-1-rin0.2-rout4-n128.json','#2ca02c')]
sweep=json.loads((ROOT/'radial-sweep.json').read_text())
fig,axs=plt.subplots(2,2,figsize=(9.2,6.4),layout='constrained')
for label,name,color in CASES:
 data=json.loads((ROOT/name).read_text());p=data['leading_profiles'][0]['profile'];r=np.array(p['r_M']);th=np.array(p['Theta_real_normalized']);H=np.array(p['H_real_normalized'])
 axs[0,0].plot(r,th,label=label,color=color)
 axs[0,1].plot(r,H/max(abs(H)),label=label,color=color)
 rows=[q for q in sweep if q['rate_per_M']==data['rate_per_M'] and q['lapse_scaled']==data['lapse_scaled'] and q['outer_M']==4]
 paired=json.loads((ROOT/'radial-boundary-comparison.json').read_text())
 rows=sorted([q for q in paired if q['rate_per_M']==data['rate_per_M'] and q['lapse_scaled']==data['lapse_scaled'] and q['polynomial_degree']==128],key=lambda q:q['outer_wave_power'])
 axs[1,0].plot([q['outer_wave_power'] for q in rows],[q['mode']['eigenvalue_real_per_M'] for q in rows],'o-',color=color,label=label)
 rows=sorted([q for q in sweep if q['rate_per_M']==data['rate_per_M'] and q['lapse_scaled']==data['lapse_scaled'] and q['inner_M']==.2],key=lambda q:q['outer_M'])
 axs[1,1].plot([q['outer_M'] for q in rows],[q['leading']['eigenvalue_real_per_M'] for q in rows],'o-',color=color,label=label)
for ax in axs[0]:
 ax.axvline(1,color='.4',ls='--',lw=.9);ax.set_xlabel('$r/M$');ax.set_xlim(.2,4)
axs[0,0].set_ylabel('$\\Theta/\\max|\\Theta|$');axs[0,1].set_ylabel('$H/\\max|H|$');axs[0,0].legend(frameon=False,fontsize=9)
axs[1,0].set_xlabel('Outer radiation coefficient $p$ ($L=4M$)');axs[1,0].set_xticks([0,1]);axs[1,1].set_xlabel('Outer radius $L/M$ ($r_{\\rm in}=0.2M$)')
for ax in axs[1]:ax.set_ylabel('Mode growth rate $\\gamma M$');ax.axhline(0,color='.3',lw=.8)
axs[1,0].set_ylim(-.13,.06);axs[1,1].set_ylim(0,.057)
for ax in axs.flat:ax.grid(alpha=.18)
axs[0,0].set_title('Constraint mode with $p=0$',fontsize=10);axs[0,1].set_title('Constraint mode with $p=0$',fontsize=10)
fig.suptitle('Spherical continuum constraints: boundary sensitivity on an annulus',fontsize=11)
fig.savefig(ROOT/'radial-annulus-modes.png',dpi=190,facecolor='white')
