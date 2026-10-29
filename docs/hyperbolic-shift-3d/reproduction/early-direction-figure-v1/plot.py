"""Render the retained small scientific plotting table; no large inputs needed."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

directory = Path(__file__).resolve().parent
data = json.loads((directory/'data.json').read_text())
rows = data['snapshots'];times=[r['time'] for r in rows]
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(1,3,figsize=(12.8,4.4))
axes[0].semilogy(times,[r['raw_volume_max'] for r in rows],'o-',label='Volume')
axes[0].semilogy(times,[r['raw_surface_max'] for r in rows],'s-',label='Actual scri surface')
axes[0].axhline(1e-8,color='0.5',ls=':',label='Original volume limit')
axes[0].set(xlabel='Coordinate time',ylabel='Maximum stored-field deviation',title='Error grows rapidly')
axes[0].legend(fontsize=9)
for row in rows:
    axes[1].plot(row['radial_centers']+[1.],row['A_signed_projection_on_surface']+[1.],
                 'o-',ms=4,label=f"t={row['time']:.2f}")
axes[1].axhline(0,color='0.6',lw=.8)
axes[1].set(xlim=(.91,1.003),xlabel='Compactified radius',ylabel='A projection / surface A amplitude',
            title='A thin radial boundary pattern')
axes[1].legend(fontsize=9)
axes[2].plot(times,[r['surface_Q_energy_quotient'] for r in rows],'o-',color='#008a6c')
axes[2].set(xlabel='Coordinate time',ylabel='Isolated surface-Q damping quotient',
            title='The angular filter acts weakly',ylim=(0,1.1))
axes[2].text(.03,.11,'Trajectory amplitude growth ≈ 73\n(different measure; not an eigenvalue shift)',
             transform=axes[2].transAxes,fontsize=9)
for ax in axes:
    ax.grid(alpha=.18)
axes[0].set_xticks(times);axes[2].set_xticks(times)
fig.suptitle('V4 Minkowski diagnostic snapshots: n=16, dt=0.0005, angular ε=2',fontsize=14)
fig.text(.5,.02,'All captures fail the original state-error limit. Radial profiles use global Cartesian A components and reference-area weights.\nThe changing direction is not a certified eigenmode; isolated Q response does not establish coupled stability.',
         ha='center',fontsize=9)
fig.tight_layout(rect=(0,.13,1,.91))
fig.savefig(directory/'early-direction.png',dpi=160)
fig.savefig(directory/'early-direction.pdf')
plt.close(fig)
