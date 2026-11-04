"""Preserved BN1 three-resolution histories; preserve separate restart segments."""
from pathlib import Path
import json,hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
OUT=HERE/'n16-n20-n24';OUT.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
hist={};reports={};proof={}
for n,dirs in [(16,['n16-completed']),(20,['n20-segment01','n20-segment02']),(24,['n24-segment01','n24-segment02'])]:
 hist[n]=[];reports[n]=[];proof[n]=[]
 for name in dirs:
  p=HERE/name;w=read(p/'worker-status.json');b=read(p/'binding.json');s=read(p/'submission.json')
  assert w['job_id']==s['job_id'] and sha(p/'plan.json')==b['plan_sha256']==w['plan_sha256']==s['plan_sha256']
  assert w['binaries']==b['binaries'] and w['status'] in ['passed','checkpointed']
  hashes={}
  for f in ['puncture.report.json','puncture.history.csv']:
   a=[v for k,v in w['cases'][0]['artifacts'].items() if k.endswith('/'+f)];assert len(a)==1
   assert sha(p/f)==a[0]['sha256'] and (p/f).stat().st_size==a[0]['bytes'];hashes[f]=a[0]['sha256']
  r=read(p/'puncture.report.json');assert r['checkpoint_schema']=='HYP3D_MPI_BN1' and r['n']==n and r['checkpoint_admitted']
  a=np.genfromtxt(p/'puncture.history.csv',delimiter=',',names=True);hist[n].append(a);reports[n].append(r)
  proof[n].append(dict(job=s['job_id'],source_manifest_sha256=b['source_manifest_sha256'],binaries=b['binaries'],hashes=hashes,native_seconds=w['cases'][0]['elapsed_seconds']))
assert proof[16][0]['binaries']==proof[20][0]['binaries']==proof[20][1]['binaries']
assert proof[16][0]['source_manifest_sha256']==proof[20][0]['source_manifest_sha256']==proof[20][1]['source_manifest_sha256']
for n in [20,24]:
 assert abs(hist[n][0][-1]['t_over_M']-hist[n][1][0]['t_over_M'])<1e-10
 for part in proof[n]:
  assert part['binaries']==proof[16][0]['binaries'] and part['source_manifest_sha256']==proof[16][0]['source_manifest_sha256']
fields=[('H_exterior_L2','Own-candidate exterior H',True),('M_exterior_L2','Own-candidate exterior M',True),('exterior_cutoff','Moving exterior cutoff',False),('horizon_trial_expansion_rms','Candidate expansion RMS',True),('puncture_x','Puncture x',False),('horizon_mass','Recorded relative mass error',True)]
fig,axs=plt.subplots(2,3,figsize=(13,7.2))
for ax,(field,title,log) in zip(axs.flat,fields):
 for n,color in [(16,'#2166ac'),(20,'#d95f02'),(24,'#1b9e77')]:
  for i,a in enumerate(hist[n]):
   y=a[field]
   if field=='horizon_mass':y=np.where(a['horizon_found']==1,abs(3*y-1),np.nan)
   ax.plot(a['t_over_M'],y,'.-',ms=3,color=color,label=f'N{n}' if i==0 else None)
 if log:ax.set_yscale('log')
 for nr in [20,24]:ax.axvline(hist[nr][0][-1]['t_over_M'],color='gray',ls=':',lw=1)
 if field=='horizon_trial_expansion_rms':
  ax.axhline(.01,color='black',ls='--',lw=1);ax.axhline(.001,color='gray',ls='--',lw=1)
 ax.set(title=title,xlabel='t/M',xlim=(0,10));ax.grid(alpha=.2)
axs[0,0].legend();fig.suptitle('BN1 normalized inner shift: N16 / N20 / N24\nMoving cuts; restart segments plotted separately; unsaved masses remain gaps')
fig.tight_layout(rect=(0,0,1,.92));fig.savefig(OUT/'histories.png',dpi=170);fig.savefig(OUT/'histories.pdf');plt.close(fig)
summary={'provenance':proof,'qualification':'Own-candidate moving domains, not fixed-cutoff convergence. Restart seed sidecars absent; finder initialization and sampling differ. Duplicate restart-time row counted only from segment01. Segment counts are summed; boundary counts separate.'}
for n in [16,20,24]:
 parts=hist[n];a=np.concatenate([parts[0]]+[x[1:] for x in parts[1:]])
 accepted=a[a['horizon_found']==1];quality=(a['horizon_complete_surface']==1)&np.isfinite(a['horizon_trial_expansion_rms'])&(a['horizon_trial_expansion_rms']<=.01)&(a['horizon_trial_radius']>0)
 summary[str(n)]=dict(rows=len(a),online_accepted=len(accepted),post_expansion_pass=int(quality.sum()),last_online_mass_time_M=float(accepted[-1]['t_over_M']),final_candidate_rms=float(a[-1]['horizon_trial_expansion_rms']),final_H=float(a[-1]['H_exterior_L2']),final_M=float(a[-1]['M_exterior_L2']),final_cut=float(a[-1]['exterior_cutoff']),native_seconds=sum(x['native_seconds'] for x in proof[n]),counters={k:sum(r[k] for r in reports[n]) for k in ['chi_floor_stage_interventions','chi_floor_acceptance_interventions','surface_chi_floor_stage_interventions','surface_chi_floor_final_interventions','lapse_floor_node_interventions','scri_interpolated_chi_negative_count']},cumulative_max={k:max(r[k] for r in reports[n]) for k in ['max_null_surface','max_null_ghost','max_pole','max_clock']})
for n in [16,20,24]:
 totals={}
 for region in ['volume','surface']:
  for field,name in [(0,'chi'),(1,'lapse')]:
   events=[e for r in reports[n] for e in r['candidate_sign_events'] if (e[0]>=0)==(region=='volume') and e[1]==field]
   totals[region+'_'+name]={name:sum(e[i] for e in events) for name,i in [('negative_stage',5),('negative_final',6),('zero_stage',7),('zero_final',8)]}
 summary[str(n)]['summed_pre_floor_signs']=totals
(OUT/'comparison.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in summary.items() if k!='provenance'},indent=2))

# Full residuals and cumulative boundary maxima: no instantaneous-pole claim.
fig,axs=plt.subplots(2,2,figsize=(10,7))
for ax,field,title in zip(axs.flat,['H_L2','M_L2','pole_max','clock_max'],['Full H','Full M','Segment cumulative pole maximum','Segment cumulative clock maximum']):
 for n,color in [(16,'#2166ac'),(20,'#d95f02'),(24,'#1b9e77')]:
  for i,part in enumerate(hist[n]):
   if field in part.dtype.names:ax.plot(part['t_over_M'],part[field],'.-',ms=3,color=color,label=f'N{n}' if i==0 else None)
 ax.set(yscale='log',xlabel='t/M',title=title);ax.grid(alpha=.2)
axs[0,0].legend();fig.suptitle('Boundary maxima reset at restarts; these are not instantaneous residuals');fig.tight_layout();fig.savefig(OUT/'full-and-boundary.png',dpi=170);fig.savefig(OUT/'full-and-boundary.pdf');plt.close(fig)
