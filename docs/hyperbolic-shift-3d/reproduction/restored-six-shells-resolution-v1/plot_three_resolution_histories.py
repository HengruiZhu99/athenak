"""Three-resolution residual trends from preserved diagnostic histories, not field self-convergence."""
from pathlib import Path
import json, hashlib, io
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--complete',action='store_true',help='Use verified complete histories, preserving the earlier snapshot')
args=parser.parse_args()
if args.complete:
    from completed_three_data import histories_and_sources
    data,sources=histories_and_sources()
    OUT=HERE/'n16-n20-n24-complete-history'; OUT.mkdir(exist_ok=True)
    receipt={'retrieved_utc':'Completed immutable campaign records; see source receipts'}
else:
    OUT=HERE/'n16-n20-n24-history-snapshot'
    prior=json.loads((HERE/'n16-n20/comparison.json').read_text())
    receipt=json.loads((OUT/'snapshot-provenance.json').read_text())
    data={};sources={}
    for n,key,raw in [(16,'N16*','restored-six-shells'),(20,'N20','restored-six-shells-resolution')]:
        seg,=prior['cases'][key]['segments']
        name,record=next((k,v) for k,v in seg['artifacts'].items() if k.endswith('/puncture.history.csv'))
        path=ROOT/'output'/raw/'aurora'/name
        b=path.read_bytes();assert len(b)==record['bytes'] and hashlib.sha256(b).hexdigest()==record['sha256']
        data[n]=np.genfromtxt(io.BytesIO(b),delimiter=',',names=True)
        sources[n]=dict(path=str(path.relative_to(ROOT)),**record)
    name='output/puncture.history.csv';record=receipt['files'][name]
    b=(OUT/'snapshot'/name).read_bytes();assert len(b)==record['bytes'] and hashlib.sha256(b).hexdigest()==record['sha256']
    assert b.endswith(b'\n'), 'Only complete CSV snapshots'
    data[24]=np.genfromtxt(io.BytesIO(b),delimiter=',',names=True);sources[24]=record
for n,d in data.items():
    assert np.all(np.diff(d['t_over_M'])>0)
    for key in ['H_exterior_L2','M_exterior_L2','exterior_cutoff']:
        assert np.all(np.isfinite(d[key]) & (d[key]>0))
# Match actual diagnostic times. Never interpolate the N24 walltime endpoint.
indices=[];mismatch=[]
for i,t in enumerate(data[24]['t_over_M']):
    j=int(np.argmin(abs(data[16]['t_over_M']-t)));k=int(np.argmin(abs(data[20]['t_over_M']-t)))
    delta=max(abs(data[16]['t_over_M'][j]-t),abs(data[20]['t_over_M'][k]-t))
    if delta<1e-9:indices.append((j,k,i));mismatch.append(float(delta))
assert indices
matched={n:data[n][np.array(indices)[:,j]] for j,n in enumerate([16,20,24])}
t=matched[24]['t_over_M'];lnN=np.log(np.array([16,20,24]));x=lnN-lnN.mean()
# Restart can shift the diagnostic clock. Do not visually bridge a long gap
# in the exact-time order estimates; preserve the isolated final endpoint.
gaps=np.flatnonzero(np.diff(t)>.151)+1
plot_t=np.insert(t,gaps,np.nan)
plot_values=lambda a:np.insert(a,gaps,np.nan)
results={};columns=[t];headers=['t_over_M']
for metric,key in [('H','H_exterior_L2'),('M','M_exterior_L2')]:
    E=np.array([matched[n][key] for n in [16,20,24]])
    y=np.log(E);pfit=-np.sum(x[:,None]*y,axis=0)/np.sum(x*x)
    p1620=np.log(E[0]/E[1])/np.log(20/16);p2024=np.log(E[1]/E[2])/np.log(24/20)
    residual=y-(y.mean(axis=0)[None,:]-x[:,None]*pfit[None,:])
    rms=np.sqrt(np.mean(residual**2,axis=0))
    results[metric]=dict(pair16_20=p1620,pair20_24=p2024,fit=pfit,log_fit_rms=rms)
    columns.extend([*E,p1620,p2024,pfit,rms]);headers.extend([f'{metric}{n}' for n in [16,20,24]]+[f'p{metric}_16_20',f'p{metric}_20_24',f'p{metric}_fit',f'{metric}_log_fit_rms'])
for n in [16,20,24]:
    columns.extend([matched[n]['exterior_cutoff'],matched[n]['horizon_found']]);headers.extend([f'cutoff{n}',f'horizon_accepted{n}'])
np.savetxt(OUT/'matched-orders.csv',np.column_stack(columns),delimiter=',',header=','.join(headers),comments='')
summary=dict(retrieved_utc=receipt['retrieved_utc'],sources=sources,matched_rows=len(t),common_last_t_over_M=float(t[-1]),n24_last_t_over_M=float(data[24]['t_over_M'][-1]),max_time_mismatch=max(mismatch),pair_formula='ln(E_coarse/E_fine)/ln(N_fine/N_coarse)',fit_formula='unweighted least-squares ln(E_N)=a-p ln(N), N=16,20,24',interpolation='none',smoothing='none',unmatched_n24_times=[float(v) for i,v in enumerate(data[24]['t_over_M']) if i not in {z[2] for z in indices}],order_plot_gap_threshold_M=.151,last={m:{k:float(v[-1]) for k,v in vals.items()} for m,vals in results.items()},last_norms={n:{m:float(matched[n][key][-1]) for m,key in [('H','H_exterior_L2'),('M','M_exterior_L2')]} for n in data},qualifications=['Each resolution uses its own moving candidate-horizon cutoff; no common fixed-domain historical norms exist.','Candidate norms remain plotted when finder acceptance fails; CSV preserves acceptance flags.','Residuals converge toward zero in the continuum; fitting their log norms is not Richardson self-convergence of numerical fields.','Space and time refine together, same clipped penalty family; old N16 campaign policy failure remains preserved.',('All three histories reach10M; N24 joins two verified healthy segments.' if args.complete else 'N24 snapshot is not a 10M endpoint, and its history collection does not admit a restart checkpoint.'),'Adjacent-pair disagreements expose failure of a single power law; three-point fit alone is not asymptotic proof.'])
(OUT/'orders-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
colors={16:'#238b45',20:'#2166ac',24:'#d95f02'}
fig,axes=plt.subplots(1,2,figsize=(12,4.8))
for ax,key,label in zip(axes,['H_exterior_L2','M_exterior_L2'],['Hamiltonian','Momentum']):
    for n,d in data.items():ax.semilogy(d['t_over_M'],d[key],color=colors[n],lw=1.7,label=f'N{n}')
    ax.axvline(t[-1],ls=':',color='gray',lw=1);ax.set(xlabel=r'$t/M$',ylabel=f'Exterior {label.lower()} RMS',title=label,xlim=(0,10));ax.grid(alpha=.2,which='both');ax.legend()
fig.suptitle('Six-shell broad/smooth formulation: exterior constraints')
fig.tight_layout(rect=(0,.08,1,.95));fig.text(.5,.025,f'Each run’s moving candidate exterior; N24 available to {data[24]["t_over_M"][-1]:.3f}M. Dotted line: last shared sample {t[-1]:.2f}M.',ha='center',fontsize=9)
fig.savefig(OUT/'exterior-histories.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
for ax,m in zip(axes,['H','M']):
    v=results[m]
    ax.plot(plot_t,plot_values(v['pair16_20']),marker='.',ms=3,color='#238b45',lw=1.3,label='N16–N20')
    ax.plot(plot_t,plot_values(v['pair20_24']),marker='.',ms=3,color='#d95f02',lw=1.3,label='N20–N24')
    ax.plot(plot_t,plot_values(v['fit']),marker='.',ms=3,color='#2166ac',lw=2.2,label='Fit N16, N20, N24')
    for p,ls in [(2,':'),(3,'-.'),(4,'--')]:ax.axhline(p,color='gray',ls=ls,lw=.8)
    ax.set(ylabel=f'Effective order $p_{m}(t)$',xlim=(0,float(t[-1])));ax.grid(alpha=.2);ax.legend(fontsize=9,ncol=3,loc='best')
axes[-1].set_xlabel(r'$t/M$')
fig.suptitle('Constraint-residual orders: adjacent pairs and three-resolution log fit')
fig.tight_layout(rect=(0,.085,1,.95));fig.text(.5,.025,'Matched times only; gaps >0.151M shown as breaks. No interpolation/smoothing; moving cuts, joint space/time refinement.\nReference lines: p = 2, 3, 4. These are effective residual orders, not a fixed-domain self-convergence proof.',ha='center',fontsize=9)
fig.savefig(OUT/'orders-timeseries.png',dpi=180);plt.close(fig)
fig,ax=plt.subplots(figsize=(10,3.5))
for n in data:ax.plot(data[n]['t_over_M'],data[n]['exterior_cutoff'],label=f'N{n}',color=colors[n])
ax.set(xlabel=r'$t/M$',ylabel='Candidate exterior cutoff r',xlim=(0,10));ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(OUT/'candidate-cutoffs.png',dpi=180);plt.close(fig)
print(json.dumps({k:summary[k] for k in ['matched_rows','common_last_t_over_M','n24_last_t_over_M','max_time_mismatch','last','last_norms']},indent=2))
