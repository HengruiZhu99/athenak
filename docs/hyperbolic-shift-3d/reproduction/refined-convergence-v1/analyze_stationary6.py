"""Read-only, independent-verification-bound stationary RF2 analysis."""
from pathlib import Path
import json,hashlib,shutil
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
ROOT=next(p for p in HERE.parents if (p/'CMakeLists.txt').exists())
OUT=HERE/'stationary6-terminal';raw=ROOT/'output/refined-convergence/aurora/campaigns/stationary6-n16-v2-segment01/output'
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
verification=json.loads((OUT/'independent-verification.json').read_text());assert sha(raw/'state.checkpoint')==verification['checkpoint']['sha256']
assert verification['original_worker_admitted'] is False
rows=[json.loads(s) for s in (OUT/'state.history.jsonl').read_text().splitlines()];report=json.loads((OUT/'state.report.json').read_text());t=np.array([q['t_over_M'] for q in rows]);diag=ROOT/'output/refined-convergence/endpoint-diagnostic/stationary6-10M.json';q=json.loads(diag.read_text());assert q['analysis_override']=='none' and q['steps']==report['steps'];ints=np.array(q['integrals']);bins=np.array(q['radial_bins']);full=np.sqrt(ints[0,1:]/ints[0,0]);assert np.allclose(full[:2],[report['H'],report['M']],rtol=1e-12,atol=0)
shutil.copy2(diag,OUT/'endpoint.json')
fig,ax=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
for k,field in enumerate(['H','M']):
 a=ax[0,k]
 for label,v in [('full',[r[field] for r in rows]),('candidate enclosure',[r[field+'_exterior'] for r in rows]),('fixed r > 0.3',[r['fixed'][1][field] for r in rows]),('fixed r > 0.6',[r['fixed'][2][field] for r in rows])]:a.semilogy(t,v,label=label)
 a.set(xlabel='t/M',ylabel=field+' constraint RMS');a.legend(fontsize=8);a.grid(alpha=.25)
a=ax[1,0];exp=np.array([r['horizon_trial_expansion_rms'] for r in rows]);a.semilogy(t,exp,label='complete candidate expansion RMS');a.axhline(.001,color='k',ls=':',label='online tolerance');a.axhline(.01,color='grey',ls='--',label='post quality threshold');a.set(xlabel='t/M',ylabel='Horizon expansion RMS');a.legend(fontsize=8);a.grid(alpha=.25)
a=ax[1,1]
for key,label in [('null','surface null'),('ghost_null','ghost null'),('pole_max','pole cumulative maximum'),('clock_max','clock cumulative maximum')]:a.semilogy(t,[r[key] for r in rows],label=label)
a.set(xlabel='t/M',ylabel='Residual (cumulative where labeled)');a.legend(fontsize=8);a.grid(alpha=.25)
fig.suptitle('Stationary RF2 N16, SMR: native 10M; original worker verification failed, independent checks passed',fontsize=11);fig.savefig(OUT/'histories.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
for k,col in enumerate([3,4]):
 a=ax[k];x=(bins[:,0]+bins[:,1])/2;a.semilogy(x,np.where(bins[:,2]>0,bins[:,col],np.nan),label='native integral per Δr=0.01 bin')
 for lo,hi,c,l in [(.24,.30,'C1','Omega transition'),(.36,.65,'C2','gauge transition')]:a.axvspan(lo,hi,color=c,alpha=.12,label=l)
 for x0 in [.21,.29,.38,.5,.68,.875]:a.axvline(x0,color='grey',lw=.6,ls=':')
 a.axvline(.3,color='k',ls='--',label='common cut');a.set(xlim=(.15,1),xlabel='compact r',ylabel=['Hamiltonian squared integral','Momentum squared integral'][k]);a.grid(alpha=.25);a.legend(fontsize=8)
fig.suptitle('10M radial errors: dotted lines mark shell radii; interior puncture norms excluded from interpretation');fig.savefig(OUT/'radial-errors.png',dpi=160);plt.close(fig)
# Native-center profiles keep separately evolved copies, no smoothing/interpolation.
parts=[np.genfromtxt(ROOT/f'output/refined-convergence/endpoint-diagnostic/stationary6-10M.rank{i}.csv',delimiter=',',names=True) for i in range(2)];data=np.concatenate(parts)
fig,ax=plt.subplots(3,2,figsize=(11,10),constrained_layout=True)
for ray,label,color in [(0,'+x shell centers (angular offset)','C0'),(2,'Cartesian near-axis centers','C1'),(1,'shell near face corner','C3')]:
 d=data[(data['ray']==ray)&(data['r']>.15)&(data['r']<.7)];order=np.lexsort((d['r'],d['block']));d=d[order]
 for a,f in zip(ax.flat,['chi','alpha','dr_chi','dr_alpha','H','Mnorm']):
  # Scatter preserves overlap copies instead of drawing a spurious interpolated jump.
  a.scatter(d['r'],d[f],s=8,label=label,color=color);a.set(xlabel='compact r',ylabel=f);a.axvspan(.36,.65,color='C2',alpha=.08);a.grid(alpha=.25)
for a in ax.flat:a.legend(fontsize=7)
fig.suptitle('Unsmoothed 10M native-center fields and constraints; distinct Cartesian/shell sampling');fig.savefig(OUT/'profiles.png',dpi=160);plt.close(fig)
phases={};nested={}
for f in raw.glob('state.rank*.performance.json'):
 d=json.loads(f.read_text())
 for key,v in d['exclusive_phases'].items():phases.setdefault(key,[]).append(v['seconds'])
for f in raw.glob('state.rank*.generations.jsonl'):
 d=json.loads(f.read_text().splitlines()[-1])
 for key,v in d['nested_seconds'].items():nested.setdefault(key,[]).append(v)
metrics=dict(status='native_completed_independently_verified_not_accuracy_success',original_worker_admitted=False,
 full_norms=full.tolist(),fixed_cut_03_norms=np.sqrt(ints[1,1:]/ints[1,0]).tolist(),
 radial_03_06_squared_fractions={name:float(bins[(bins[:,0]>=.3)&(bins[:,1]<=.60000001),col].sum()/bins[bins[:,0]>=.3,col].sum()) for name,col in [('H',3),('M',4)]},
 online_accepted=sum(bool(r['horizon_found']) for r in rows),post_expansion_quality=sum(bool(r['horizon_complete']) and r['horizon_trial_expansion_rms']<=.01 for r in rows),total_rows=len(rows),
 last_online_accepted_t=max(r['t_over_M'] for r in rows if r['horizon_found']),final_mass=rows[-1]['mass'],final_expansion_rms=exp[-1],
 native_seconds=report['seconds'],floor_counts=report['floor_counts'],
 exclusive_phase_rank_seconds={k:dict(min=min(v),median=float(np.median(v)),max=max(v)) for k,v in phases.items()},
 nested_rank_seconds={k:dict(min=min(v),median=float(np.median(v)),max=max(v)) for k,v in nested.items()},
 input_checkpoint_sha256=sha(raw/'state.checkpoint'),analysis_sha256=sha(diag),
 qualification='One stationary resolution; original worker/PBS permission failure retained. Independent identical-byte verifier passed. Common coordinate cut, not invariant physical domain. Cumulative pole/clock are not instantaneous values; post expansion quality does not recover missing mass. Native rank timers are not a controlled scaling experiment; nested timers overlap.')
(OUT/'analysis.json').write_text(json.dumps(metrics,indent=2)+'\n');print(json.dumps(metrics,indent=2))
