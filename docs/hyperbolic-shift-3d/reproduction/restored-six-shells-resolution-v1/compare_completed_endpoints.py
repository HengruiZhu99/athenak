"""Extend the preserved N16/N20 reconstruction to the admitted N24 endpoint only."""
from pathlib import Path
import json,hashlib,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from completed_three_data import HERE,ROOT,load_cases,histories
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prior=HERE/'n16-n20';out=HERE/'n16-n20-n24-complete-endpoints'
cases,audit=load_cases()
histories.LABELS.update({f'N{n}':f'N{n}'+(' (old policy failure retained)' if n==16 else '') for n in cases})
colors={16:'#238b45',20:'#2166ac',24:'#d95f02'}
histories.COLORS.update({f'N{n}':c for n,c in colors.items()})
summary=histories.render({f'N{n}':c for n,c in cases.items()},out,
 title='Completed six-shell resolution family: N16, N20, N24',
 footer='Own-candidate histories; mass gaps retained. Same clipped penalty scale 0.651, dt proportional to 1/N.\nN16 original policy failure remains recorded; three resolutions do not establish uniform fourth-order convergence.')
summary['cases']['N16']['status']='native_completed_old_penalty_policy_failed'
(out/'comparison.json').write_text(json.dumps(summary,indent=2)+'\n')
cost={}
for n,c in cases.items():
 segments=[dict(seconds=r['seconds'],steps=r['segment_steps'],start=r['start_time'],end=r['t']) for r in c['reports']]
 cost[f'N{n}']=dict(segments=segments,native_seconds=sum(r['seconds'] for r in c['reports']))
(out/'cost.json').write_text(json.dumps(cost,indent=2)+'\n')

raw=ROOT/'output/restored-six-shells-resolution/aurora'
cp=raw/Path(cases[24]['records'][-1]['plan']).parent/'output/puncture.checkpoint'
cp_hash=sha(cp);assert cp_hash==cases[24]['records'][-1]['checkpoint_sha256']
provenance=read(prior/'reconstruction-provenance.json')
binary=Path(provenance['N20_command'][0]);assert sha(binary)==provenance['diagnostic_binary_sha256']
coeff=np.loadtxt(cp.with_name('puncture.fastflow.horizon_shape_0.txt'))[-1];assert len(coeff)==169
bound=float(coeff[0]/np.sqrt(4*np.pi)+sum(np.sqrt((2*l+1)/(4*np.pi))*np.linalg.norm(coeff[l*l:(l+1)**2]) for l in range(1,13)))
assert bound<.1099 and cases[24]['reports'][-1]['horizon_complete_surface']
cmd=[str(binary),'24',str(cp),str(1/3000),'.12',str(out/'n24-radial.json'),provenance['N20_command'][-1]]
record=dict(command=cmd,checkpoint_sha256=cp_hash,binary_sha256=sha(binary),horizon_upper_bound=bound,
 reused={str(n):dict(path=str(prior/f'n{n}-radial.json'),sha256=sha(prior/f'n{n}-radial.json')) for n in [16,20]},
 qualification='Same diagnostic operators and fixed cutoff; N16 native completion under its recorded original policy failure.')
(out/'reconstruction-provenance.json').write_text(json.dumps(record,indent=2)+'\n')
with (out/'n24-radial.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
assert sha(cp)==cp_hash
D={n:read(prior/f'n{n}-radial.json') for n in [16,20]};D[24]=read(out/'n24-radial.json')
for n,d in D.items():
 assert d['n']==n and d['t_over_M']==10 and d['cutoff']==.12
 for k in ['H_L2','M_L2','Mr_L2']:assert abs(d['norms']['total'][k]/cases[n]['reports'][-1][k]-1)<1e-9
ns=np.array([16,20,24]);x=np.log(ns);x-=x.mean()
def orders(E):
 E=np.asarray(E);y=np.log(E);p=-sum(x*y)/sum(x*x)
 return dict(values={str(n):float(v) for n,v in zip(ns,E)},pair16_20=float(np.log(E[0]/E[1])/np.log(20/16)),pair20_24=float(np.log(E[1]/E[2])/np.log(24/20)),fit=float(p),log_fit_rms=float(np.sqrt(np.mean((y-y.mean()+p*x)**2))))
endpoint={k:orders([D[n]['norms']['exterior'][k] for n in ns]) for k in ['H_L2','M_L2','Mr_L2']}
sensitivity={};regions={}
def row(n,c):return next(v for v in D[n]['radial_cumulative'] if v['cutoff']==c)
for cut in [.1099,.12,.13,.16,.3,.6,.7,.775,.9]:
 sensitivity[str(cut)]={k:orders([np.sqrt(row(n,cut)[k]/row(n,cut)['volume']) for n in ns]) for k in ['H2','M2','Mr2']}
for lo,hi in [(.12,.16),(.16,.3),(.3,.6),(.6,.7),(.7,.775),(.775,.9),(.9,1)]:
 regions[f'{lo:g}–{hi:g}']={k:orders([np.sqrt((row(n,lo)[k]-row(n,hi)[k])/(row(n,lo)['volume']-row(n,hi)['volume'])) for n in ns]) for k in ['H2','M2','Mr2']}
result=dict(time_over_M=10,cutoff=.12,endpoint=endpoint,cutoff_sensitivity=sensitivity,regions=regions,horizon_upper_bounds={**provenance['horizon_upper_bounds'],'24':bound},
 qualifications=['Constraint residuals have zero continuum target; these are log-norm effective orders, not field Richardson self-convergence.','Space and time refine together; fixed clipped-penalty family, N16 original policy failure retained.','Fixed coordinate cutoff outside all complete final horizons, with resolution-dependent quadrature.','Unequal refinement ratios handled explicitly; pair disagreements remain visible.'])
(out/'orders.json').write_text(json.dumps(result,indent=2)+'\n')
fig,axes=plt.subplots(1,2,figsize=(12,4.7))
for ax,k,label in zip(axes,['H_L2','M_L2'],['Hamiltonian','Momentum']):
 e=endpoint[k];y=np.array(list(e['values'].values()));ax.loglog(ns,y,'o-',lw=2,label=f"Measured: pair p={e['pair16_20']:.2f}, {e['pair20_24']:.2f}")
 ax.loglog(ns,y[0]*(16/ns)**4,'--',color='gray',label='Fourth-order reference')
 ax.set_xticks(ns,labels=[str(n) for n in ns]);ax.minorticks_off();ticks=[.002,.004,.008] if k=='H_L2' else [.004,.008,.016];ax.set_yticks(ticks,labels=[f'{v:g}' for v in ticks]);ax.set(xlabel='N',ylabel=label+' constraint RMS');ax.grid(alpha=.2);ax.legend(fontsize=9)
fig.suptitle('10M endpoint: common cutoff r > 0.12, same operators');fig.tight_layout(rect=(0,.1,1,.94));fig.text(.5,.02,'Effective residual orders; space and time refined together. N16 original policy failure retained.',ha='center',fontsize=9);fig.savefig(out/'effective-order.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(12,4.7))
for n in ns:
 rows=D[n]['radial_cumulative'];r=np.array([v['cutoff'] for v in rows]);vol=-np.diff([v['volume'] for v in rows]);keep=r[:-1]>=.12
 for ax,k in zip(axes,['H2','M2']):
  sums=-np.diff([v[k] for v in rows]);v=np.full_like(vol,np.nan);ok=vol>1e-14;v[ok]=np.sqrt(np.maximum(sums[ok],0)/vol[ok]);ax.stairs(v[keep],r[np.r_[keep,True]],baseline=None,color=colors[n],label=f'N{n}')
for ax,k in zip(axes,['H','M']):
 ax.axvspan(.7,.775,color='gray',alpha=.12);ax.axvline(.75,color='gray',ls=':');ax.set(xlabel='compact radius r',ylabel=k+' regional RMS',yscale='log',xlim=(.12,1));ax.grid(alpha=.2);ax.legend()
fig.suptitle('10M radial constraints: identical radial bands');fig.tight_layout(rect=(0,.1,1,.94));fig.text(.5,.02,'Shaded: r = 0.7–0.775. Empty quadrature bins omitted. No uniform-order claim.',ha='center',fontsize=9);fig.savefig(out/'radial.png',dpi=180);plt.close(fig)
print(json.dumps(endpoint,indent=2))
