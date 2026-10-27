#!/usr/bin/env python3
"""Plot verified saved-state constraints; permit explicitly labelled partial runs."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from analyze_telegraph_time import numerical_binding
from telegraph_analysis_io import validate_telegraph_profile

p = argparse.ArgumentParser()
p.add_argument('--matrix', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
names = ['H', 'M', 'Theta', 'Z']
records = []
binding = configuration = None
for n in [1025, 2049, 4097]:
    folder = a.matrix / f'n{n}'
    raw = (folder / 'manifest.json').read_bytes()
    m = json.loads(raw)
    if 'checkpoint_parent' in m or m['status'] not in ['running', 'finished', 'failed']:
        raise ValueError('Expect initialized initial-to-time trajectories')
    current = numerical_binding(folder.resolve(), m)
    config = {k:v for k,v in m['arguments'].items() if k not in ['n','output','resume','library','fast_library']}
    if binding is None:
        binding, configuration = current, config
    if current != binding or config != configuration:
        raise ValueError('Uncontrolled numerical sources or inputs')
    (a.output / f'manifest-n{n}-at-capture.json').write_bytes(raw)
    history = {round(x['t'],9):x for x in map(json.loads,(folder/'history.jsonl').read_text().splitlines())}
    rows, receipts = [], []
    cadence = m['arguments']['report_dt']
    for index in range(round(m['time']/cadence)+1):
        path = folder / f'profile-{index+1:04d}.npz'
        content = path.read_bytes()
        with np.load(path, allow_pickle=False) as z:
            t = float(z['t'])
            if abs(t-index*cadence)>1e-11:
                raise ValueError('Missing or mismatched report time')
            d = {k:z[k].copy() for k in ['r','grid_jacobian','q','q_physical','diagnostic','B','beta','gauge_speeds','rays']}
        validate_telegraph_profile(d)
        r = d['r']; w = r*r*d['grid_jacobian']; c = d['diagnostic'][:,:4]
        rms = np.sqrt(np.sum(w[:,None]*c*c,axis=0)/np.sum(w))
        saved = history[round(t,9)]
        np.testing.assert_allclose(rms,[saved[key+'_rms'] for key in names],rtol=2e-12,atol=1e-25)
        row = dict(t=t, **{key+'_rms':float(v) for key,v in zip(names,rms)},
                   H_max=float(np.max(abs(c[:,0]))), H_bulk_max=float(np.max(abs(c[r<.8,0]))))
        np.testing.assert_allclose(row['H_bulk_max'], saved['H_bulk_max'], rtol=2e-12, atol=1e-25)
        rows.append(row)
        receipts.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(content).hexdigest(),t=t))
    records.append(dict(n=n,status_at_capture=m['status'],available_through=m['time'],rows=rows,profiles=receipts))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'mathtext.fontset':'stix','savefig.dpi':180})
styles = {1025:dict(color='#cc6677',ls='--',lw=1.5),2049:dict(color='#228899',ls='-',lw=1.6),4097:dict(color='#332288',ls=':',lw=1.8)}
def draw(ax, key):
    for rec in records:
        t = [row['t'] for row in rec['rows']]
        y = np.array([row[key] for row in rec['rows']])
        label = f"N={rec['n']:,}" + (f" (partial: t≤{rec['available_through']:g})" if rec['status_at_capture']!='finished' else '')
        ax.plot(t,np.ma.masked_less_equal(y,0),label=label,**styles[rec['n']])
        if rec['status_at_capture']!='finished' and y[-1]>0:
            ax.plot(t[-1],y[-1],'o',ms=4,color=styles[rec['n']]['color'])
    ax.set(xlabel='Evolution time t',yscale='log',xlim=(0,10))
    ax.grid(alpha=.2,which='major')

fig,axes = plt.subplots(2,2,figsize=(10.5,7),layout='constrained')
for ax,key,title,symbol in zip(axes.flat,names,['Hamiltonian','Radial momentum','Z4 scalar','Connection constraint'],[r'\mathcal{H}',r'\mathcal{M}_r',r'\Theta',r'Z_r']):
    draw(ax,key+'_rms')
    ax.set(title=title,ylabel=rf'$\|{symbol}\|_r$')
axes[0,0].legend(fontsize=8,loc='best')
fig.suptitle('Constraint norms over time • spherical telegrapher lapse',fontsize=14)
fig.supxlabel(r'Full-domain radial RMS: $\|C\|_r=[\sum_i r_i^2 J_i C_i^2/\sum_i r_i^2 J_i]^{1/2}$;  $J=dr/dx$',fontsize=10)
for ext in ['png','pdf']:
    fig.savefig(a.output/f'constraint-norms.{ext}')
plt.close(fig)

fig,axes = plt.subplots(1,2,figsize=(10.5,4.2),layout='constrained')
for ax,key,title in zip(axes,['H_max','H_bulk_max'],['Entire grid, including endpoints',r'Interior diagnostic: $0\leq r<0.8$']):
    draw(ax,key)
    ax.set(title=title,ylabel=r'$\max_r |\mathcal{H}|$')
axes[1].axhline(1e-3,color='#555555',lw=1,ls='-.',label=r'Acceptance threshold: $10^{-3}$')
axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
fig.suptitle('Hamiltonian pointwise error • complements the radial RMS',fontsize=13)
for ext in ['png','pdf']:
    fig.savefig(a.output/f'hamiltonian-maximum.{ext}')
plt.close(fig)

report = dict(captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),records=records,source_binding=binding,
              norm='Full-grid discrete radial RMS with weights r^2 dr/dx, identical to evolution diagnostics; not a proper-volume norm. No inner mask. Diagnostic columns H,M,Theta,Z are the implemented conformal constraint quantities.',
              verification='Every saved norm and interior Hamiltonian maximum recomputed from profiles and compared to history. All numerical bindings and profile finiteness checked. No smoothing, temporal interpolation or extrapolation. Exact zeros masked only for logarithmic display.',
              scope='Saved report times, not every RK stage or accepted step. Partial runs stop at captured manifest time. These plots do not establish full-time three-grid convergence or robustness.',goal_status='incomplete')
(a.output/'review.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps([dict(n=x['n'],status=x['status_at_capture'],through=x['available_through'],last=x['rows'][-1],sampled_max={key:max(row[key] for row in x['rows']) for key in ['H_rms','H_max','H_bulk_max']}) for x in records],indent=2))
