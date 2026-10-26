#!/usr/bin/env python3
"""Assemble reviewed scalar-localization plots from immutable diagnostics."""
import argparse
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
import numpy as np

from localize_schwarzschild_error import sha, save, COLORS, REGIONS


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();E=args.evidence.resolve();out=args.output.resolve()
    if out.exists():raise RuntimeError('Preserve existing output')
    out.mkdir(parents=True)
    initial=E/'localization-20261017-v1';follow=E/'localization-20261017-followup-v1'
    files=[initial/'review.json',initial/'plot-data.npz',follow/'review.json',follow/'dyadic-scri-early.json']
    r=json.loads(files[0].read_text());a=np.load(files[1]);extra=json.loads(files[2].read_text());dyadic=json.loads(files[3].read_text())
    inputs={str(p):sha(p) for p in files}
    ns=r['resolutions'];plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.2})
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    p=E/'radial-oracle/smooth/waveform-upwind-n4800.csv-profile-12.csv';inputs[str(p)]=sha(p)
    ref=np.genfromtxt(p,delimiter=',',names=True);rho=ref['rho'];P=ref['psi'];grad=np.gradient(P,rho);visible=(rho>=9)&(rho<=16)
    lo,hi=extra['pulse_resolution']['gradient_peak_half_prominence_interval']
    for n,color in zip(ns,COLORS):
        nodes=a[f'ray_nodes_n{n}'];values=a[f'ray_node_psi_t12_n{n}'];mask=(nodes>=9)&(nodes<=16)
        count=extra['pulse_resolution']['shell_radial_nodes_in_interval'][str(n)]
        axes[0,0].plot(nodes[mask],values[mask],'o',ms=3,color=color,label=f'N={n}: {count} nodes in shaded width')
        mask=(a['ray_rho']>=9)&(a['ray_rho']<=16)
        axes[1,0].plot(a['ray_rho'][mask],(a[f'ray_psi_t12_n{n}']-a['ray_reference_t12'])[mask],color=color,label=f'N={n}')
    axes[0,0].plot(rho[visible],P[visible],'k-',lw=1,label='independent 4800')
    axes[0,0].set(ylabel='radial amplitude psi/(nx ny)',title='Outgoing feature in the transition at 12M')
    axes[0,0].axvspan(lo,hi,color='gray',alpha=.15);axes[0,0].legend(fontsize=8)
    axes[1,0].set(xlabel='compact radius rho/M',ylabel='signed radial-amplitude error');axes[1,0].legend(fontsize=8)
    axes[0,1].plot(rho[visible],grad[visible],'k-');axes[0,1].axvspan(lo,hi,color='gray',alpha=.15)
    axes[0,1].set(ylabel='reference radial derivative',title=f'Steep-gradient width: {hi-lo:.2f}M (half prominence)')
    x=np.arange(len(REGIONS))
    for time,color in zip([6,12,100],['#e69f00','#56b4e9','#000000']):
        values=[extra['transient_regional_RMS_orders'][str(time)][reg]['psi'][1] if time!=100 else r['regional_orders'][reg]['psi']['L2'][1] for reg in REGIONS]
        axes[1,1].plot(x,values,'o-',color=color,label=f't={time}M')
    axes[1,1].axhline(1.5,color='black',ls='--');axes[1,1].set(xticks=x,xticklabels=REGIONS,ylabel='field RMS order N48/60',title='Where convergence slows during evolution')
    axes[1,1].tick_params(axis='x',labelrotation=25);axes[1,1].legend(fontsize=8)
    fig.suptitle('Transition resolution along direction (0.8,0.6,0): actual shell radial nodes\nThe steep feature has 7/8/10 nodes on N36/48/60; the N96 spacing would place 17 nodes across it',fontsize=12)
    save(fig,out,'transition-resolution')

    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for time,color in zip([6,12,100],['#e69f00','#56b4e9','#000000']):
        nodes=sorted(set(ns+([24,96] if time!=100 else [])))
        for ax,key,norm,label in [(axes[0,0],'psi','mode_error','radiation (2,2) mode absolute error'),
                                 (axes[0,1],'psi','angular_L2','endpoint angular field RMS error'),
                                 (axes[1,0],'v','mode_error','radiation time-derivative mode error')]:
            vals=[(r['scri'][str(time)][str(n)] if n in ns else dyadic['times'][str(time)]['records'][str(n)])['fields'][key][norm] for n in nodes]
            ax.loglog(nodes,vals,'o-',color=color,label=f't={time}M')
            if time!=100:ax.scatter([96],[vals[-1]],marker='^',color=color,s=70,zorder=3)
            ax.set(xlabel='resolution N',ylabel=label,xticks=[24,36,48,60,96]);ax.set_xticklabels([24,36,48,60,96]);ax.xaxis.set_minor_locator(NullLocator());ax.legend(fontsize=9)
    ax=axes[1,1];ax.axis('off')
    cells=[]
    for time in [6,12,100]:
        old=r['scri_orders'][str(time)]['psi']['mode_error'][1]
        new=dyadic['times'][str(time)]['orders']['psi']['mode_error'][1] if time!=100 else None
        vnew=dyadic['times'][str(time)]['orders']['v']['mode_error'][1] if time!=100 else None
        cells.append([str(time),f'{old:.2f}',f'{new:.2f}' if new is not None else 'pending',f'{vnew:.2f}' if vnew is not None else 'pending'])
    table=ax.table(cellText=cells,colLabels=['t/M','field p48/60','field p48/96','dt field p48/96'],cellLoc='center',loc='upper center',bbox=[0,.55,1,.4]);table.auto_set_font_size(False);table.set_fontsize(9)
    ax.text(0,.40,'Minimum declared order: 1.5\nTriangles: existing N96 endpoint data\n100M N96 comparison: pending\n\nSaved times only; no resolved scri ringdown trace.\nProjection normalized for angular quadrature.\nBoth full angular RMS and mode errors are shown.',transform=ax.transAxes,va='top',fontsize=10,linespacing=1.5)
    fig.suptitle('Direct convergence at exact scri: radiation r Phi = 24 psi\nThe N48/60 plateau improves on N48/96; no N96 late-time result is inferred',fontsize=12)
    save(fig,out,'scri-convergence')

    for name in ['spatial-error-map','regional-convergence','radial-profiles']:
        for ext in ['png','pdf']:
            p=initial/f'{name}.{ext}';inputs[str(p)]=sha(p);shutil.copy2(p,out/p.name)
    for name in ['late-regional-errors.csv','scri-snapshot-convergence.csv','dyadic-scri-early.json']:
        p=follow/name;inputs[str(p)]=sha(p);shutil.copy2(p,out/name)
    protected_file=E/'frozen-baseline-sha256.json';protected=json.loads(protected_file.read_text());inputs[str(protected_file)]=sha(protected_file)
    bad=[name for name,digest in protected.items() if not Path(name).is_file() or sha(name)!=digest]
    if bad:raise RuntimeError('Protected production baseline changed')
    sources=[Path(__file__),Path(__file__).with_name('localize_schwarzschild_error.py'),Path(__file__).with_name('refine_scalar_localization.py'),Path(__file__).with_name('reference_interpolation.py')]
    for p in sources:shutil.copy2(p,out/p.name)
    summary=dict(additional_trajectories=0,protected_baseline_files=len(protected),protected_baseline_mismatches=bad,
        original_failure_reproduced_exactly=all(x['relative_difference']==0 for x in r['original_gate_reproduction']),
        late_regional_field={reg:dict(finest_RMS=r['regional']['60'][reg]['errors']['psi']['L2'],finest_squared_error_fraction=r['regional']['60'][reg]['errors']['psi']['squared_error_fraction'],orders=r['regional_orders'][reg]['psi']['L2']) for reg in REGIONS},
        early_dyadic_scri_orders={t:dyadic['times'][str(t)]['orders'] for t in [6,12]},
        early_dyadic_scri_max_reference_error_fraction=max(dyadic['times'][str(t)]['records']['96']['fields'][key]['reference_mode_difference']/dyadic['times'][str(t)]['records']['96']['fields'][key]['mode_error'] for t in [6,12] for key in ['psi','v']),
        pulse_resolution=extra['pulse_resolution'],
        qualifications=['These results locate the residual error, not its causal origin.',
            'Global volume-RMS convergence can hide endpoint failures; scri evaluated independently here.',
            'Early N48/60 endpoint mode orders fail; N48/96 early endpoint mode and angular checks exceed1.5.',
            'N96 late comparison and full time/boundary controls remain required; no admission change.',
            'Equatorial maps omit rho<0.9 for display. Quantitative exterior norms use all active nodes with rho>=1.2.',
            'The saved scri snapshots do not resolve a time series or support a scri QNM fit.'],
        inputs_sha256=inputs,source_sha256={str(p.resolve()):sha(p) for p in sources},
        outputs_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()})
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(dict(output=str(out),summary=str(out/'summary.json'),protected_files=len(protected),mismatches=bad),indent=2))


if __name__=='__main__':main()
