#!/usr/bin/env python3
"""Plot retained puncture history without filling missing horizon measurements."""
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np

root=Path(__file__).resolve().parent
verification=json.loads((root/'verification.json').read_text())
reference=root/verification['original_1d_reference']['history']
assert hashlib.sha256(reference.read_bytes()).hexdigest()==verification['original_1d_reference']['history_sha256']
for artifact in verification['artifacts']:
    if artifact['retained_in_package']:
        assert hashlib.sha256((root/artifact['path']).read_bytes()).hexdigest()==artifact['sha256']

def table(path):
    with path.open() as stream:
        rows=list(csv.DictReader(stream))
    return {key:np.array([float(row[key]) for row in rows]) for key in rows[0]}

one=table(reference)
three=table(root/'output/puncture.history.csv')
report=json.loads((root/'output/puncture.report.json').read_text())
mass=verification['original_1d_reference']['mass']
assert np.allclose(three['t_over_M'],three['t']/mass,rtol=1e-14,atol=0)
valid=three['horizon_found']==1
for key in ['horizon_mass','horizon_radius','H_exterior_L2']:
    assert np.all(np.isfinite(three[key][valid]))
    assert np.all(np.isnan(three[key][~valid]))
last_horizon=three['t_over_M'][valid][-1]
last_diagnostic=three['t_over_M'][-1]
failure=report['t_over_M']
mass_error_1d=np.abs(one['apparent_horizon_mass']-mass)/mass
mass_error_3d=np.abs(three['horizon_mass']-mass)/mass
# No replacement floor is applied on the log axes: every visible valid value
# is positive. NaNs in the 3D horizon-dependent columns break the curves.
assert np.all(mass_error_1d[one['t_over_M']<=1.5]>0)
assert np.all(mass_error_3d[valid]>0)

plt.rcParams.update({'font.size':10,'axes.titlesize':11,'axes.labelsize':10,
                     'legend.fontsize':9,'pdf.fonttype':42,'ps.fonttype':42})
fig,axes=plt.subplots(2,2,figsize=(11.4,8.8),sharex=True)
fig.subplots_adjust(left=.095,right=.97,top=.82,bottom=.19,wspace=.28,hspace=.30)
fig.suptitle('Original hyperbolic driver: early 3D puncture failure',y=.976,fontsize=16)
fig.text(.5,.94,r'Original 1D $N=513$ reference vs 3D $n=16$ per block | $M=1/3$ | 12 MPI ranks / 12 GPU tiles',
         ha='center',fontsize=10)
orange,blue,teal,red='#c47513','#1769a6','#008b8b','#ad2f2f'
missing='#eeeeee'
legend=[Line2D([],[],color=orange,lw=1.8,label='1D original control, N=513'),
        Line2D([],[],color=blue,lw=1.8,marker='o',ms=4,label='3D original control, n=16'),
        Line2D([],[],color=teal,lw=1.6,ls='--',marker='s',ms=3.5,label='3D interpolated-ghost closure'),
        Patch(facecolor=missing,label='Missing 3D measurements'),
        Line2D([],[],color=red,lw=1.2,ls='--',label='3D failure at 1.107 M')]
fig.legend(handles=legend,loc='upper center',bbox_to_anchor=(.5,.916),ncol=3,frameon=False,
           columnspacing=1.7,handlelength=2.7)

panels=[(axes[0,0],one['H_exterior_rms'],three['H_exterior_L2'],
         '(a) Exterior Hamiltonian constraint',r'$\|H\|_{2,\,\mathrm{ext}}$'),
        (axes[0,1],mass_error_1d,mass_error_3d,
         '(b) Apparent-horizon relative mass error',r'$|M_{\rm AH}-M|/M$'),
        (axes[1,0],one['apparent_horizon_r'],three['horizon_radius'],
         '(c) Apparent-horizon coordinate radius',r'$r_{\rm AH}$ (compactified coordinate)')]
for ax,reference_values,values,title,ylabel in panels:
    ax.axvspan(last_horizon,1.2,color=missing,zorder=-2)
    # The full recorded reference is clipped to the view. This includes the
    # segment between its neighboring samples at 1.125 M and 1.5 M.
    ax.plot(one['t_over_M'],reference_values,color=orange,lw=1.8)
    ax.plot(three['t_over_M'],values,'o-',color=blue,lw=1.8,ms=4)
    ax.set_title(title,loc='left')
    ax.set_ylabel(ylabel)
axes[0,0].set_yscale('log')
axes[0,0].set_ylim(1e-6,.12)
axes[0,1].set_yscale('log')
axes[0,1].set_ylim(5e-13,5e-4)
axes[1,0].set_ylim(.0565,.0673)
for ax in [axes[0,0],axes[0,1],axes[1,0]]:
    ax.text(.64,.96,'3D AH-dependent data unavailable',transform=ax.transAxes,
            ha='center',va='top',fontsize=8,color='#555555')

ax=axes[1,1]
ax.axvspan(last_diagnostic,1.2,color=missing,zorder=-2)
ax.plot(one['t_over_M'],one['max_relative_null']/1e-15,color=orange,lw=1.8)
ax.plot(three['t_over_M'],three['null_surface']/1e-15,'o-',color=blue,lw=1.8,ms=4)
ax.plot(three['t_over_M'],three['null_ghost']/1e-15,'s--',color=teal,lw=1.6,ms=3.5)
ax.set_title('(d) Outer-boundary null closure: recorded maxima',loc='left')
ax.set_ylabel(r'Relative null residual / $10^{-15}$')
ax.set_ylim(-.04,1.55)
ax.set_yticks([0,.5,1,1.5])
ax.text(1.073,.45,'No fresh diagnostics',transform=ax.get_xaxis_transform(),rotation=90,
        ha='center',va='center',fontsize=8,color='#555555')
for ax in axes.flat:
    ax.axvline(failure,color=red,ls='--',lw=1.2,zorder=5)
    ax.set_xlim(0,1.2)
    ax.set_xticks([0,.3,.6,.9,1.2])
    ax.grid(True,alpha=.2,which='major')
for ax in axes[1]:
    ax.set_xlabel(r'Time $t/M$')
fig.text(.095,.126,'Last valid 3D horizon: 0.45 M; first rejected finder result: 0.60 M. Missing values are not filled or extrapolated.',fontsize=9)
fig.text(.095,.101,'Geometry rejection: 1.107 M; last diagnostics: 1.05 M. The saved failure checkpoint is not admitted.',fontsize=9)
fig.text(.095,.076,r'$H$ uses roughly comparable coordinate-volume weighting; 3D invariant and 1D radial momentum norms are not overlaid.',fontsize=9)
fig.text(.095,.051,'This failed low-resolution trajectory does not demonstrate a trumpet transition, long-time stability, or convergence.',fontsize=9)
fig.savefig(root/'puncture-vs-1d.png',dpi=200,facecolor='white')
fig.savefig(root/'puncture-vs-1d.pdf',facecolor='white',metadata={
    'Title':'Original 3D puncture failure versus original 1D control',
    'Author':'AthenaK hyperboloidal reproduction'})
