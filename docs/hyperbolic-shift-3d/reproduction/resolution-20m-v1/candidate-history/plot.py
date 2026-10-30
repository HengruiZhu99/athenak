#!/usr/bin/env python3
"""Plot measured exterior histories; validate the diagnostic replay against originals."""
from pathlib import Path
import json
import hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REP = HERE.parents[1]
OLD = REP/'n16-timestep-to3m-v1/n16-original-dt-reproduction-v1/output'

def read(path):
    return np.genfromtxt(path, delimiter=',', names=True)

def unique(a):
    rows = {round(float(r['t_over_M']),10):r for r in a}
    return np.array([rows[t] for t in sorted(rows)],dtype=a.dtype)

early = read(OLD/'puncture-initial.history.csv')
original = {16:unique(np.concatenate([early,read(OLD/'puncture.history.csv')])),
            24:unique(read(HERE.parent/'n24-at3m/puncture.history.csv'))}
data = {16:unique(np.concatenate([early,read(HERE/'n16.history.csv')])),
        24:unique(read(HERE/'n24.history.csv'))}
fields = ['H_L2','M_L2','Mr_L2','Theta_L2','Zr_L2','min_chi','min_lapse','horizon_trial_radius']
verification = {}
for n,d in data.items():
    report=json.loads((HERE/f'n{n}.report.json').read_text())
    assert report['status']=='completed' and report['t_over_M']==3
    assert report['complete_candidate_exterior_enabled']
    assert d[0]['t_over_M']==0 and d[-1]['t_over_M']==3
    for k in ['H_exterior_L2','M_exterior_L2','Mr_exterior_L2','Theta_exterior_L2']:
        assert np.isfinite(d[k]).all(), (n,k,'missing diagnostic')
    diffs={k:0. for k in fields}; matched=0
    for r in d:
        a=original[n];i=np.argmin(abs(a['t_over_M']-r['t_over_M']))
        if abs(a[i]['t_over_M']-r['t_over_M'])>1e-8:continue
        matched+=1
        for k in fields:
            diffs[k]=max(diffs[k],float(abs(r[k]-a[i][k])))
            assert np.isclose(r[k],a[i][k],rtol=1e-7,atol=1e-11),(n,k,r['t_over_M'],r[k],a[i][k])
    verification[n]=dict(matched_samples=matched,max_absolute_differences=diffs)

plt.rcParams.update({'font.size':12,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True)
fig.suptitle('Exterior constraint norms through 3M: N16 versus N24',fontsize=17,y=.98)
fig.text(.5,.934,'Complete horizon candidates used as integration cutoffs, including above-tolerance candidates',
         ha='center',fontsize=10)
for ax,k,title,label in zip(axes.flat,
    ['H_exterior_L2','Mr_exterior_L2','M_exterior_L2','Theta_exterior_L2'],
    ['Hamiltonian','Radial momentum','Conformal invariant momentum',r'Z4 scalar $\Theta$'],
    [r'$\|H\|_{2,\rm ext}$',r'$\|M_r\|_{2,\rm ext}$',r'$\|M\|_{2,\rm ext}$',r'$\|\Theta\|_{2,\rm ext}$']):
    for n,c in [(16,'#b45418'),(24,'#1769aa')]:
        d=data[n];ax.plot(d['t_over_M'],d[k],'-o',color=c,ms=3.5,label=f'N{n}')
    ax.set(title=title,ylabel=label,xlim=(0,3),ylim=(0,None));ax.grid(alpha=.23)
for ax in axes[1]:ax.set_xlabel(r'$t/M$')
axes[0,0].legend()
fig.subplots_adjust(top=.85,bottom=.17,hspace=.30,wspace=.28)
fig.text(.5,.055,'Every marker is a measured diagnostic; no missing-value interpolation.\n'
         'Each run uses its own candidate radius and normalized compactified-coordinate volume.',ha='center',fontsize=10)
for ext in ['png','pdf']:fig.savefig(HERE/f'exterior-constraint-history.{ext}',dpi=180)
sources={str(p.relative_to(REP)):hashlib.sha256(p.read_bytes()).hexdigest()
         for p in [OLD/'puncture-initial.history.csv',OLD/'puncture.history.csv',
                   HERE.parent/'n24-at3m/puncture.history.csv',HERE/'n16.history.csv',HERE/'n24.history.csv']}
(HERE/'history-verification.json').write_text(json.dumps(dict(comparison=verification,sources=sources),indent=2)+'\n')
print(json.dumps(verification,indent=2))
