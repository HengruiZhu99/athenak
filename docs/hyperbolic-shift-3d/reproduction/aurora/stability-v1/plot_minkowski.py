#!/usr/bin/env python3
"""Reproduce the static failure-history figure from the retained CSV and report."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path(__file__).resolve().parent
with (root/'output/minkowski.history.csv').open() as stream:
    rows=list(csv.DictReader(stream))
report=json.loads((root/'output/minkowski.report.json').read_text())
t=[float(r['t']) for r in rows]
def column(name):
    return [float(r[name]) for r in rows]

plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
                     'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(3,1,figsize=(8.2,8.6),sharex=True)
fig.subplots_adjust(left=.13,right=.97,top=.87,bottom=.115,hspace=.24)
fig.suptitle('Minkowski duration test: growing error and rejected geometry',y=.97,fontsize=14)
fig.text(.5,.932,r'Aurora | $n=16$, $\Delta t=0.0005$ | 12 MPI ranks / 12 GPU tiles',ha='center',fontsize=10)

axes[0].plot(t,column('state_deviation'),'o-',color='#1769a6',lw=1.8,ms=4,label='Maximum stored-field deviation')
axes[0].axhline(1e-8,color='#777777',ls=':',lw=1,label=r'Planned limit $10^{-8}$')
axes[0].set_yscale('symlog',linthresh=1e-14,linscale=.7)
axes[0].set_ylim(0,3)
axes[0].set_ylabel('State error')
axes[0].legend(loc='upper left',frameon=False)
axes[0].set_title('Exact equilibrium error grows before the geometry rejection',loc='left')

axes[1].plot(t,column('H_L2'),'o-',color='#1769a6',lw=1.8,ms=4,label=r'$\|H\|_2$')
axes[1].plot(t,column('M_L2'),'s-',color='#d27615',lw=1.8,ms=4,label=r'$\|M\|_2$ (Penrose norm)')
axes[1].axhline(1e-7,color='#777777',ls=':',lw=1,label=r'Planned limit $10^{-7}$')
axes[1].set_yscale('symlog',linthresh=1e-15,linscale=.7)
axes[1].set_ylim(0,.1)
axes[1].set_ylabel('Constraint norm')
axes[1].legend(loc='upper left',frameon=False,ncol=2)
axes[1].set_title('Normalized compactified Cartesian coordinate-volume norms',loc='left')

axes[2].plot(t,[v/1e-15 for v in column('null_surface')],'o-',color='#1769a6',lw=1.8,ms=4,label='Surface')
axes[2].plot(t,[v/1e-15 for v in column('null_ghost')],'s-',color='#d27615',lw=1.8,ms=4,label='Interpolated ghost')
axes[2].set_ylim(0,1.5)
axes[2].set_ylabel(r'Null residual / $10^{-15}$')
axes[2].legend(loc='lower left',frameon=False,ncol=2)
axes[2].set_title('Recorded null closure remains at roundoff while other errors grow',loc='left')
axes[2].set_xlabel('Coordinate time $t$ (no black-hole mass scale for Minkowski)')

last=t[-1]
for ax in axes:
    ax.axvspan(last,report['t'],color='#9e9e9e',alpha=.16,zorder=-1)
    ax.axvline(report['t'],color='#b22222',ls='--',lw=1.2)
    ax.grid(True,alpha=.22,which='major')
    ax.set_xlim(0,.515)
axes[0].text((last+report['t'])/2,.5,'No fresh diagnostics',transform=axes[0].get_xaxis_transform(),
             rotation=90,ha='center',va='center',fontsize=8,color='#555555')
axes[2].text(report['t']+.005,.95,'Geometry rejected\nafter step 979',
             transform=axes[2].get_xaxis_transform(),ha='left',va='top',fontsize=8,
             color='#a02222',rotation=90)
fig.text(.13,.052,'History ends at t = 0.45; final accepted time is t = 0.4895. The saved checkpoint is not admitted.',fontsize=9)
fig.text(.13,.028,'Zero values are retained on the first two symmetric-log axes. These records do not isolate the growing mode.',fontsize=9)
fig.savefig(root/'minkowski-failure.png',dpi=200,facecolor='white')
fig.savefig(root/'minkowski-failure.pdf',facecolor='white',metadata={'Title':'Aurora stability-v1 Minkowski failure','Author':'AthenaK hyperboloidal reproduction'})
