#!/usr/bin/env python3
"""Verified ray fans and spatial differences, retaining terminal records.

Finite-grid noncrossing is not a continuum shock theorem. This report never
substitutes for the independent timestep or evolving-field acceptance checks.
"""
import argparse,json
from pathlib import Path
import numpy as np
from analyze_telegraph_time import load,controlled_inputs
from telegraph_analysis_io import validate_telegraph_profile


def analyze(runs):
    if not runs:raise ValueError('Require at least one completed trajectory')
    runs=sorted(runs,key=lambda run:run['manifest']['arguments']['n'])
    ns=[run['manifest']['arguments']['n'] for run in runs]
    if len(set(ns))!=len(ns):raise ValueError('Repeated spatial resolution')
    for other in runs[1:]:controlled_inputs(runs[0],other,{'n'})
    histories=[]
    for run in runs:
        data=run['data'];start=run['manifest']['arguments']['ray_start'];seed=data[0.]['rays']
        if start not in data or not np.array_equal(data[start]['rays'],seed):raise ValueError('Unverified ray activation state')
        if seed.shape[0]%2 or not np.array_equal(seed[::2,0],seed[1::2,0]) or np.any(seed[:,1]!=0):raise ValueError('Unmatched characteristic seeds')
        rows=[]
        for t,d in sorted(data.items()):
            validate_telegraph_profile(d)
            if np.any(d['rays'][:,0]<0) or np.any(d['rays'][:,0]>1):raise ValueError('Ray outside terminal boundaries')
            families=[]
            for f in range(2):
                ray=d['rays'][f::2];active=(ray[:,0]>0)&(ray[:,0]<1);gap=np.diff(ray[:,0]);both=active[:-1]&active[1:]
                if np.any(gap<0):raise ValueError('Ray ordering reversed; integration unresolved')
                families.append(dict(active=int(sum(active)),terminal=int(sum(~active)),
                    min_active_logJ=float(np.min(ray[active,1])) if active.any() else None,
                    max_active_logJ=float(np.max(ray[active,1])) if active.any() else None,
                    min_active_neighbor_gap=float(np.min(gap[both])) if both.any() else None,
                    active_coincidences=int(sum(gap[both]==0))))
            rows.append(dict(t=t,families=families))
        histories.append(dict(n=run['manifest']['arguments']['n'],rows=rows))
    pairs=[]
    for coarse,fine in zip(runs,runs[1:]):
        na=coarse['manifest']['arguments']['n'];nb=fine['manifest']['arguments']['n'];ratio=(nb-1)/(na-1)
        if ratio<=1 or not ratio.is_integer():raise ValueError('Noninteger grid refinement')
        rows=[]
        for t,a in sorted(coarse['data'].items()):
            b=fine['data'][t]
            if not np.array_equal(a['r'],b['r'][::int(ratio)]):raise ValueError('Nonmatching physical nodes')
            if a['rays'].shape!=b['rays'].shape:raise ValueError('Unmatched ray packet')
            aa=(a['rays'][:,0]>0)&(a['rays'][:,0]<1);ab=(b['rays'][:,0]>0)&(b['rays'][:,0]<1)
            row=dict(t=t,position=float(np.max(abs(a['rays'][:,0]-b['rays'][:,0]))),logJ=float(np.max(abs(a['rays'][:,1]-b['rays'][:,1]))),exit_status_mismatches=int(sum(aa!=ab)))
            rows.append(row)
        pairs.append(dict(grids=[na,nb],ratio=ratio,history=rows,peak_position=max(row['position'] for row in rows),peak_logJ=max(row['logJ'] for row in rows)))
    orders=[]
    for a,b in zip(pairs,pairs[1:]):
        if a['ratio']!=b['ratio']:raise ValueError('Unequal refinement factors cannot use this order estimate')
        entry=dict(grids=[*a['grids'],b['grids'][-1]])
        for key in ['position','logJ']:
            ea,eb=a['peak_'+key],b['peak_'+key]
            entry[key]=float(np.log(ea/eb)/np.log(a['ratio'])) if min(ea,eb)>0 else None
        orders.append(entry)
    return dict(histories=histories,pairs=pairs,spatial_orders=orders,goal_status='incomplete',
        scope='All rays, including terminal compression, enter spatial differences. Active neighbor gaps distinguish finite-grid coincidences from terminal boundary coincidences. Finite-grid ordering or positive exp(logJ) cannot prove absence of continuum characteristic crossing. Spatial observed orders are diagnostic; independent timestep and evolving-field/boundary/robustness controls remain required.')


def plot(result,runs,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    runs=sorted(runs,key=lambda run:run['manifest']['arguments']['n']);fine=runs[-1];start=fine['manifest']['arguments']['ray_start']
    times=np.array(sorted(t for t in fine['data'] if t>=start));rays=np.array([fine['data'][t]['rays'] for t in times]);nr=rays.shape[1]//2
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    indices=np.unique(np.linspace(0,nr-1,min(9,nr),dtype=int));colors=plt.cm.viridis(np.linspace(.1,.9,len(indices)))
    for seed,color in zip(indices,colors):
        for f,ax in enumerate(axes[0]):
            radius=rays[:,2*seed+f,0];active=(radius>0)&(radius<1)
            ax.plot(times,np.where(active,radius,np.nan),color=color,lw=1)
    axes[0,0].set_yscale('log');axes[0,0].set(title='Ingoing rays: shrinking coordinate radius',ylabel='r')
    axes[0,1].set(title='Outgoing rays: paths end at domain exit',ylabel='r',ylim=(0,1.02))
    for record in result['histories']:
        for f,style in [(0,'-'),(1,'--')]:
            x=[row['t'] for row in record['rows']];y=[row['families'][f]['min_active_logJ'] for row in record['rows']]
            axes[1,0].plot(x,[v if v is not None else np.nan for v in y],style,label=f"N={record['n']}, {'−' if f==0 else '+'}")
    axes[1,0].set(title='Compression of active characteristic families',ylabel=r'$\min\log J$');axes[1,0].legend(fontsize=8)
    for pair in result['pairs']:
        axes[1,1].semilogy([r['t'] for r in pair['history']],[r['logJ'] if r['logJ']>0 else np.nan for r in pair['history']],label=f"N={pair['grids'][0]} vs {pair['grids'][1]}")
    if result['pairs']:axes[1,1].legend(fontsize=8)
    else:axes[1,1].text(.5,.5,'One completed grid\nSpatial comparison pending',ha='center',va='center',transform=axes[1,1].transAxes)
    axes[1,1].set(title='All-ray spatial differences, including exits',ylabel=r'$\max|\Delta\log J|$')
    for ax in axes.flat:ax.set_xlabel('Time');ax.grid(alpha=.2)
    fig.suptitle(f"Stage characteristics • finest completed N={fine['manifest']['arguments']['n']} • no continuum crossing claim")
    for ext in ['png','pdf']:fig.savefig(output/('characteristics.'+ext),dpi=180)
    plt.close(fig)


def main():
    p=argparse.ArgumentParser();p.add_argument('--cases',type=Path,nargs='+',required=True);p.add_argument('--output',type=Path,required=True);args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    runs=[load(path) for path in args.cases];result=analyze(runs);result['records']=[{k:v for k,v in run.items() if k!='data'} for run in runs]
    args.output.mkdir(parents=True);(args.output/'review.json').write_text(json.dumps(result,indent=2)+'\n');plot(result,runs,args.output)
    print(json.dumps(dict(spatial_orders=result['spatial_orders'],last=[record['rows'][-1] for record in result['histories']],goal_status='incomplete'),indent=2))

if __name__=='__main__':main()
