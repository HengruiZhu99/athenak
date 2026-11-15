"""Plot read-only final checkpoint diagnostics: python3 plot_checkpoints.py."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parent;D=R/'checkpoints';A={};B={};O={};out={}
manifest=json.loads((D/'manifest.json').read_text())
for name,rec in manifest.items():assert hashlib.sha256((D/name).read_bytes()).hexdigest()==rec['sha256']
results=json.loads((D/'results.json').read_text());assert len(results)==4
assert all(x['native_read_only_pass'] and x['checkpoint_hash_unchanged'] and x['independent_reducer_pass'] for x in results)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
for n in [12,16]:
 for mode in ['full','legacy']:
  p=D/f'N{n}-{mode}-final';a=json.loads((p/'independent/audit.json').read_text());A[n,mode]=a
  assert 'PASS checkpoint diagnostics; zero evolution steps; accepted fields/records bitwise preserved' in (p/'run.log').read_text()
  b=np.array([np.loadtxt(p/f'checkpoint-band-{j}.user.hst') for j in range(8)]);assert np.isfinite(b).all()
  B[n,mode]=np.sqrt(b[:,3:7]/b[:,2,None])*[1/9,1/9,1/3,1/3]
  O[n,mode]=np.loadtxt(p/'checkpoint-full.overlap.hst');assert O[n,mode][8]==0
 x=O[n,'full'];y=O[n,'legacy'];assert abs(x[9]/y[9]-1)<1e-12
 v=np.array(A[n,'full']['fixed_band_sums'])[:,0];w=np.array(A[n,'legacy']['fixed_band_sums'])[:,0];assert np.allclose(v,w,rtol=1e-12,atol=0)
 corner={}
 for mode in ['full','legacy']:
  sh=np.array(A[n,mode]['shell_sums']);corner[mode]=np.sqrt(sh[:,3]/sh[:,2])/3
 out[str(n)]={'full_time_M0':A[n,'full']['time_M0'],'legacy_time_M0':A[n,'legacy']['time_M0'],'time_offset_M0':A[n,'legacy']['time_M0']-A[n,'full']['time_M0'],'fixed_band_full_over_legacy':(B[n,'full']/B[n,'legacy']).tolist(),'full_observer_groups':['chi','metric','curvature','Lambda','gauge','D-Lambda'],'full_observer_full_over_legacy':(x[2:8]/y[2:8]).tolist(),'common_full_observer_volume':float(x[9]),'shell_corner_Theta_full_over_legacy':(corner['full']/corner['legacy']).tolist(),'chi_cumulative':{mode:A[n,mode]['all_evolved_and_scri_cumulative_chi_counts'] for mode in ['full','legacy']},'chi_count_columns':A[n,'full']['chi_count_columns'],'max_radius_historical_chi_floor':{mode:A[n,mode]['max_radius_any_historical_floor'] for mode in ['full','legacy']}}
fig,axs=plt.subplots(2,2,figsize=(12,7),layout='constrained')
labels=[f'{x:g}–{y:g}' for x,y in zip(A[12,'full']['fixed_band_edges'][:-1],A[12,'full']['fixed_band_edges'][1:])]
for j,ax in enumerate(axs.flat):
 for n,c in [(12,'#167db3'),(16,'#aa3e65')]:ax.semilogy(range(8),np.array(out[str(n)]['fixed_band_full_over_legacy'])[:,j],marker='o',c=c,label=f'N{n}')
 ax.axhline(1,c='#596575',ls=':');ax.set(xticks=range(8),xticklabels=labels,title=['Hamiltonian H','Momentum M','Theta','Z'][j],ylabel='Full agreement / legacy RMS',xlabel='Fixed coordinate radial band');ax.tick_params(axis='x',rotation=30);ax.grid(alpha=.2)
axs[0,0].legend();fig.suptitle('Final-checkpoint constraints in identical radial bands\nLegacy endpoints are one timestep later (N12: 0.001408 M₀; N16: 0.001056 M₀)',fontsize=12)
for ext in ['png','pdf']:fig.savefig(R/f'fixed-band-ratios.{ext}',dpi=180)
plt.close(fig)
fig,axs=plt.subplots(1,2,figsize=(12,4.5),layout='constrained')
for n,c,shift in [(12,'#167db3',-.18),(16,'#aa3e65',.18)]:
 vals=out[str(n)]
 axs[0].bar(np.arange(6)+shift,vals['full_observer_full_over_legacy'],width=.34,color=c,label=f'N{n}')
 axs[1].semilogy(range(11),vals['shell_corner_Theta_full_over_legacy'],marker='o',c=c,label=f'N{n}')
axs[0].set(xticks=range(6),xticklabels=out['12']['full_observer_groups'],ylabel='Full agreement / legacy mismatch RMS',title='Same full-overlap observer (κ = 0)');axs[0].tick_params(axis='x',rotation=25)
rad=A[12,'full']['shell_radii'];axs[1].set(xticks=range(11),xticklabels=[f'{x:g}–{y:g}' for x,y in zip(rad[:-1],rad[1:])],ylabel='Full agreement / legacy corner Θ RMS',title='Shell angular corners');axs[1].tick_params(axis='x',rotation=50)
for ax in axs:ax.axhline(1,c='#596575',ls=':');ax.grid(axis='y',alpha=.2);ax.legend()
fig.suptitle('Read-only endpoint comparison near 30 M₀; legacy is one timestep later',fontsize=12)
for ext in ['png','pdf']:fig.savefig(R/f'overlap-and-corners.{ext}',dpi=180)
plt.close(fig)
out['scope']='Same observers and same geometric weights within each resolution; four finite checkpoint reductions pass. One-step endpoint offsets are not exact-time comparisons. Chi counts are historical cumulative records, not current field values.'
(R/'checkpoint-assessment.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
