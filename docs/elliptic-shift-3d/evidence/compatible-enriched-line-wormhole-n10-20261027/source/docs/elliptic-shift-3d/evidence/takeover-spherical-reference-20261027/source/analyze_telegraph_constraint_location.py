#!/usr/bin/env python3
"""Locate saved Hamiltonian errors relative to the apparent horizon, without excision."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from horizon import apparent_horizons
from telegraph_analysis_io import validate_telegraph_profile
from analyze_telegraph_time import numerical_binding

p=argparse.ArgumentParser()
p.add_argument('--capture',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
capture=json.loads(a.capture.read_text());records=[]
for rec in capture['records']:
    manifest=json.loads((a.capture.parent/f"manifest-n{rec['n']}-at-capture.json").read_text())
    folder=Path(rec['profiles'][0]['path']).parent
    if numerical_binding(folder,manifest)!=capture['source_binding']:
        raise ValueError('Changed numerical binding')
    history={round(row['t'],9):row for row in map(json.loads,(folder/'history.jsonl').read_text().splitlines())}
    rows=[]
    for receipt in rec['profiles']:
        path=Path(receipt['path'])
        if hashlib.sha256(path.read_bytes()).hexdigest()!=receipt['sha256']:
            raise ValueError('Changed captured profile')
        with np.load(path,allow_pickle=False) as z:
            t=float(z['t']);d={k:z[k].copy() for k in ['r','grid_jacobian','q','q_physical','diagnostic','B','beta','gauge_speeds','rays']}
        validate_telegraph_profile(d)
        r=d['r'];diag=d['diagnostic'];H=diag[:,0];i=int(np.argmax(abs(H)))
        ah=apparent_horizons(r,diag[:,8],diag[:,11])
        if ah['apparent_horizon_count']!=1:
            raise ValueError('This diagnostic expects one outer apparent horizon')
        saved=history[round(t,9)]
        for key in ['apparent_horizon_r','apparent_horizon_mass']:
            np.testing.assert_allclose(ah[key],saved[key],rtol=2e-13,atol=1e-15)
        rh=ah['apparent_horizon_r'];outside=r>=rh
        contribution=r*r*d['grid_jacobian']*H*H;total=float(sum(contribution))
        rows.append(dict(t=t,H_max=float(abs(H[i])),H_peak_r=float(r[i]),
            H_at_puncture=float(abs(H[0])),H_max_outside_AH=float(max(abs(H[outside]))),
            AH_r=rh,AH_mass=ah['apparent_horizon_mass'],AH_mass_error=float(abs(ah['apparent_horizon_mass']-1/3)),
            H_squared_norm_fraction_inside_AH=float(sum(contribution[~outside])/total) if total>0 else None,
            H_squared_norm_fraction_r_lt_001=float(sum(contribution[r<.01])/total) if total>0 else None))
    records.append(dict(n=rec['n'],status_at_capture=rec['status_at_capture'],available_through=rec['available_through'],rows=rows))

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'mathtext.fontset':'stix','savefig.dpi':180})
fig,axes=plt.subplots(1,3,figsize=(13.5,4.5),layout='constrained')
middle=next(rec for rec in records if rec['n']==2049);rows=middle['rows'];t=[x['t'] for x in rows]
axes[0].semilogy(t,[x['H_peak_r'] for x in rows],color='#aa3377',label=r'Location of $\max|\mathcal{H}|$')
axes[0].semilogy(t,[x['AH_r'] for x in rows],color='#333333',ls='--',label='Apparent horizon')
axes[0].set(title='Error peak moves toward the puncture',ylabel='Compactified radius r')
axes[1].semilogy(t,[x['H_max'] for x in rows],color='#aa3377',label='Full grid (acceptance diagnostic)')
axes[1].semilogy(t,[x['H_max_outside_AH'] for x in rows],color='#228899',label='Outside apparent horizon (supplement)')
axes[1].set(title='Localized error remains counted',ylabel=r'$\max|\mathcal{H}|$')
styles={1025:dict(color='#cc6677',ls='--'),2049:dict(color='#228899',ls='-'),4097:dict(color='#332288',ls=':')}
for rec in records:
    y=np.array([x['AH_mass_error'] for x in rec['rows']]);times=[x['t'] for x in rec['rows']]
    label=f"N={rec['n']:,}"+(' (partial)' if rec['status_at_capture']!='finished' else '')
    axes[2].semilogy(times,np.ma.masked_less_equal(y,0),label=label,**styles[rec['n']])
axes[2].set(title='Independent apparent-horizon mass',ylabel=r'$|M_{\rm AH}-1/3|$')
for ax in axes:
    ax.set(xlabel='Evolution time t',xlim=(0,10));ax.grid(alpha=.2);ax.legend(fontsize=7)
fig.suptitle('Constraint localization • first two panels: N=2,049 • no evolution excision',fontsize=13)
for ext in ['png','pdf']:fig.savefig(a.output/f'constraint-location.{ext}')
plt.close(fig)
result=dict(capture=str(a.capture.resolve()),capture_sha256=hashlib.sha256(a.capture.read_bytes()).hexdigest(),records=records,
    extractor_sha256=hashlib.sha256(Path(__file__).with_name('horizon.py').read_bytes()).hexdigest(),
    method='Rechecked captured profile hashes, numerical sources and finite arrays. Recomputed apparent horizon from expansion sign change and local degree-six interpolation; compared to saved history. Exterior maxima use nodes r>=r_AH. Full-grid maxima and all original acceptance tests retain every node.',
    scope='Diagnostic localization only. Apparent horizon is not an event-horizon determination. Accurate horizon mass or small exterior error cannot override failed full-domain constraints. Mass errors near machine precision do not establish convergence order. Partial finest history is not full-time convergence.',goal_status='incomplete')
(a.output/'review.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps([dict(n=rec['n'],through=rec['available_through'],last=rec['rows'][-1],max_AH_mass_error=max(x['AH_mass_error'] for x in rec['rows'])) for rec in records],indent=2))
