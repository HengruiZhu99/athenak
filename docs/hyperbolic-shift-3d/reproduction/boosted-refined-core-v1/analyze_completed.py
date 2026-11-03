"""Reproduce BC2 terminal preservation and qualified BG1 history comparison."""
from pathlib import Path
import hashlib,json,sys,shutil,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]
RAW=ROOT/'output/boosted-refined-core/aurora'
REL='campaigns/n16-boosted-refined-core-cfl-v2-c16-evolution-segment01/plan.json'
P=RAW/Path(REL).parent; OUT=HERE/'completed';OUT.mkdir(exist_ok=True)
BG=HERE.parent/'boosted-free-gauge-v1/n16-failure'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,v):(OUT/n).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
sys.path.insert(0,str(HERE.parent/'native-shared-transition-v1'))
from frozen_campaign import load
sys.path.insert(0,str(RAW/'launcher-boosted-refined-core-cfl-v2'))
helper=load(RAW/'launcher-boosted-refined-core-cfl-v2/native_transition_campaign.py')
plan,binding,worker,sub,collected=[read(P/n) for n in ['plan.json','binding.json','worker-status.json','submission.json','collected.json']]
report=read(P/'output/puncture.report.json');science=helper.science(RAW)
assert sha(P/'plan.json')==binding['plan_sha256']==worker['plan_sha256']==sub['plan_sha256']
assert worker['job_id']==sub['job_id']==collected['job_id']
assert binding['source_manifest_sha256']==worker['source_manifest_sha256']==science['source_manifest_sha256']
assert binding['binaries']==worker['binaries']=={helper.binary('C16'):science['binaries'][helper.binary('C16')]}
for f,h in binding['inputs'].items():assert sha(RAW/f)==h
for f,h in binding['launcher'].items():assert sha(RAW/'launcher-boosted-refined-core-cfl-v2'/f)==h
assert worker['status']=='passed' and collected['scheduler']['Exit_status']==0
case=worker['cases'][0];assert case['returncode']==0 and not case['timed_out'] and case['storage_error'] is None
for f,v in case['artifacts'].items():assert (RAW/f).stat().st_size==v['bytes'] and sha(RAW/f)==v['sha256']
assert all(helper.control.evaluate_check(report,c)['passed'] for c in plan['commands'][0]['checks'])
cp=RAW/plan['commands'][0]['args'][1];cp=Path(str(cp)+'.checkpoint')
cpverify=helper.inspect_checkpoint(cp,'C16',case['artifacts'][str(cp.relative_to(RAW))]['sha256'],'puncture',10/3)
assert cpverify['time']==10/3 and cpverify['puncture_position']==report['puncture_position']
save('local-verification.json',dict(verified=True,checkpoint=cpverify,artifact_hashes=case['artifacts'],job_id=sub['job_id'],plan_sha256=binding['plan_sha256'],source_manifest_sha256=binding['source_manifest_sha256'],binaries=binding['binaries']))
for n in ['terminal-verification.json','allocation-released.json','worker-status.json','collected.json','plan.json','binding.json','submission.json']:shutil.copy2(P/n,OUT/n)
for n in ['puncture.history.csv','puncture.report.json','puncture.log','puncture.fastflow.horizon_summary_0.txt','puncture.fastflow.horizon_shape_0.txt']:shutil.copy2(P/'output'/n,OUT/n)
bgverification=read(BG/'local-verification.json')
for f,v in bgverification['artifacts'].items():
 q=ROOT/'output/boosted-free-gauge/aurora'/f;assert q.stat().st_size==v['bytes'] and sha(q)==v['sha256']
hist={k:np.genfromtxt(p/'puncture.history.csv',delimiter=',',names=True) for k,p in [('BG1',BG),('BC2',OUT)]}
reports={'BG1':read(BG/'puncture.report.json'),'BC2':report};summaries={}
for k,a in hist.items():
 r=reports[k];accepted=(a['horizon_found']==1)&np.isfinite(a['horizon_mass']);last=a[accepted][-1]
 xyz=np.stack([a[f'puncture_{c}'] for c in 'xyz'],axis=1);signs=r['candidate_sign_events']
 signs_summary={}
 for field,label in [(0,'chi'),(18,'lapse')]:
  for region,select in [('volume',lambda x:x[0]>=0),('surface',lambda x:x[0]<0)]:
   rows=[x for x in signs if x[1]==field and select(x)]
   signs_summary[f'{region}_{label}']={name:sum(x[i] for x in rows) for name,i in [('negative_stage',5),('negative_final',6),('zero_stage',7),('zero_final',8)]}
 summaries[k]=dict(status=r['status'],last_accepted_step_time_M=r['t_over_M'],steps=r['steps'],history_rows=len(a),accepted_horizon_rows=int(accepted.sum()),last_accepted_horizon_time_M=float(last['t_over_M']),last_accepted_mass_error_percent=float(100*(3*last['horizon_mass']-1)),final_horizon_accepted=bool(r['horizon_found']),tracker_final=r['puncture_position'],tracker_x_min=float(xyz[:,0].min()),tracker_x_max=float(xyz[:,0].max()),sampled_min_cube_margin=float((.18-np.max(abs(xyz),axis=1)).min()),sampled_min_support_margin=float((.16-np.max(abs(xyz),axis=1)).min()),history_max_candidate_cutoff=float(np.nanmax(a['exterior_cutoff'])),chi_floor_stage=r['chi_floor_stage_interventions'],chi_floor_final=r['chi_floor_acceptance_interventions'],lapse_floor_nodes=r['lapse_floor_node_interventions'],surface_chi_floor_stage=r['surface_chi_floor_stage_interventions'],surface_chi_floor_final=r['surface_chi_floor_final_interventions'],boundary_interpolated_negative_chi=r['scri_interpolated_chi_negative_count'],boundary_interpolated_zero_chi=r['scri_interpolated_chi_zero_count'],sign_counts=signs_summary,max_null_surface=r['max_null_surface'],max_null_ghost=r['max_null_ghost'],max_pole=r['max_pole'],max_clock=r['max_clock'],max_cfl=r['max_cfl'],native_seconds=r['seconds'],final_recorded={n:float(a[n][-1]) for n in ['H_L2','M_L2','H_exterior_L2','M_exterior_L2','exterior_cutoff']})
# Same-time histories only; domains are own-candidate and differ. No fitted order.
pairs=[]
for x in hist['BG1']:
 idx=np.where(abs(hist['BC2']['t_over_M']-x['t_over_M'])<1e-9)[0]
 if len(idx)==1:
  y=hist['BC2'][idx[0]];pairs.append(dict(t_over_M=float(x['t_over_M']),BG1_H=float(x['H_exterior_L2']),BC2_H=float(y['H_exterior_L2']),BG1_M=float(x['M_exterior_L2']),BC2_M=float(y['M_exterior_L2']),BG1_cutoff=float(x['exterior_cutoff']),BC2_cutoff=float(y['exterior_cutoff'])))
qualification='BC2 refines core space and time together; outer grid and effective boundary penalty fixed. BG1 has no admitted endpoint. Own-candidate histories have different moving domains, not fixed-cutoff error/convergence. BC2 completed10M with substantial floors and no accepted horizon after3.75M; survival is not accurate black-hole evolution or a pure spatial attribution.'
save('comparison.json',dict(cases=summaries,matched_history_rows=pairs,qualification=qualification,inputs={k:sha(p/'puncture.history.csv') for k,p in [('BG1',BG),('BC2',OUT)]}))
colors={'BG1':'#b2182b','BC2':'#2166ac'}
fig,axes=plt.subplots(2,3,figsize=(13,7.5))
for ax,key,title in zip(axes.flat[:4],['H_L2','M_L2','H_exterior_L2','M_exterior_L2'],['Full Hamiltonian','Full momentum','Own-candidate exterior Hamiltonian','Own-candidate exterior momentum']):
 for k,a in hist.items():ax.semilogy(a['t_over_M'],a[key],'.-',ms=3,label=k,color=colors[k])
 ax.set_title(title)
for k,a in hist.items():
 axes[1,1].plot(a['t_over_M'],a['exterior_cutoff'],'.-',ms=3,label=k,color=colors[k])
 for field,style in [('pole_max','-'),('clock_max','--')]:axes[1,2].semilogy(a['t_over_M'],np.maximum(a[field],1e-18),style,color=colors[k],label=k+' '+field)
axes[1,1].set_title('Moving candidate cutoff');axes[1,2].set_title('Cumulative pole / clock maxima');axes[1,2].legend(fontsize=7)
for ax in axes.flat:ax.axvline(4.23,ls=':',c='gray');ax.set_xlabel('t/M');ax.grid(alpha=.2)
axes[0,0].legend();fig.suptitle('Refined core BC2 completes10M; coarse BG1 fails4.23M\nDifferent candidate cuts; no fixed-domain convergence order');fig.tight_layout();fig.savefig(OUT/'histories.png',dpi=170);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(11,7))
for k,a in hist.items():
 accepted=(a['horizon_found']==1)&np.isfinite(a['horizon_mass']);x=a['t_over_M'];color=colors[k]
 axes[0,0].plot(x,a['puncture_x'],'.-',ms=3,label=k,color=color)
 axes[0,1].semilogy(x,np.where(accepted,np.maximum(abs(3*a['horizon_mass']-1),1e-16),np.nan),'o-',ms=3,label=k,color=color)
 axes[1,0].plot(x,.16-np.maximum.reduce([abs(a['puncture_'+c]) for c in 'xyz']),'.-',ms=3,color=color,label=k)
 axes[1,1].semilogy(x,a['horizon_trial_expansion_rms'],'.-',ms=3,color=color,label=k)
axes[1,1].axhline(.001,color='k',ls='--',label='acceptance tolerance');axes[1,0].axhline(0,color='k',lw=.5)
for ax,title in zip(axes.flat,['Puncture x (compact coordinate)','Accepted |M_AH/M - 1|; gaps retained','Conservative support margin .16 - max|x_p|','Candidate expansion RMS']):ax.set(xlabel='t/M',title=title,xlim=(0,10));ax.axvline(4.23,ls=':',c='gray');ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('No accepted BC2 horizon after3.75M; coordinate trajectory is gauge dependent');fig.tight_layout();fig.savefig(OUT/'trajectory-horizon.png',dpi=170);plt.close(fig)
# Independent coordinate cutoff outside all recorded complete candidate cuts.
cutoff=.3;assert cutoff>max(v['history_max_candidate_cutoff'] for v in summaries.values())
suffix='control_broad_smooth_six_shells_boosted_free_gauge_refined_core_matched_cfl';target='hyperbolic_shift_3d_checkpoint_constraints_native_'+suffix
binary=ROOT/'build-boosted-refined-core-v1/tst/hyperbolic_shift_3d'/target
flags=binary.parent/'CMakeFiles'/f'{target}.dir/flags.make'
radial=OUT/'endpoint-radial.json';cmd=[str(binary),'16',str(cp),'.00025',str(cutoff),str(radial),'0,.03,.06,.1,.12,.16,.18,.2,.25,.3,.305,.4,.45,.55,.6,.675,.7,.725,.75,.775,.8,.85,.875,.9,.95,1']
prov=dict(command=cmd,binary_sha256=sha(binary),checkpoint_sha256=sha(cp),flags=flags.read_text(),cutoff_basis='r>.3 exceeds all recorded complete candidate enclosing radii; final horizon is NOT accepted. Diagnostic legacy horizon_acceptance_assumed=true denotes chosen cut only and is NOT horizon evidence.',source_hashes={n:sha(ROOT/n) for n in ['tst/hyperbolic_shift_3d/checkpoint_constraints.cpp','tst/hyperbolic_shift_3d/coupled_mpi.cpp']})
save('endpoint-provenance.json',prov)
if not radial.exists():
 with (OUT/'endpoint-radial.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
assert sha(cp)==prov['checkpoint_sha256'];rd=read(radial)
for k in ['H_L2','M_L2','Mr_L2']:assert abs(rd['norms']['total'][k]/report[k]-1)<1e-9,(k,rd['norms']['total'][k],report[k])
rr=rd['radial_cumulative'];regions=[]
for a,b in zip(rr[:-1],rr[1:]):regions.append(dict(lo=a['cutoff'],hi=b['cutoff'],volume=a['volume']-b['volume'],**{k:a[k]-b[k] for k in ['H2','M2','Mr2']}))
save('radial-regions.json',dict(cutoff=cutoff,exterior=rd['norms']['exterior'],regions=regions,qualification=prov['cutoff_basis']))
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,key in zip(axes,['H2','M2']):
 x=np.array([v['lo'] for v in regions]);width=np.array([v['hi']-v['lo'] for v in regions]);val=np.array([v[key] for v in regions]);ax.bar(x,np.maximum(val,1e-30),width=width,align='edge',edgecolor='white');ax.set(yscale='log',xlabel='compact radius r',ylabel=key+' integral',title='BC2 admitted10M endpoint');ax.axvline(cutoff,color='k',ls='--');ax.grid(alpha=.2)
fig.suptitle('Same native diagnostic operators; no BG1 matched endpoint exists');fig.tight_layout();fig.savefig(OUT/'radial.png',dpi=170);plt.close(fig)
print(json.dumps(dict(summaries=summaries,exterior=rd['norms']['exterior']),indent=2))
