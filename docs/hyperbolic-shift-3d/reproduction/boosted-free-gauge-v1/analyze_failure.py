"""Preserve and localize failed BG1 data without admitting it as a restart/endpoint."""
from pathlib import Path
import hashlib,json,sys,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent; ROOT=HERE.parents[3]
RAW=ROOT/'output/boosted-free-gauge/aurora'
REL='campaigns/n16-boosted-free-gauge-v1-b16-evolution-segment01/plan.json'
p=RAW/Path(REL).parent; out=HERE/'n16-failure';out.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(name,x):(out/name).write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
sys.path.insert(0,str(HERE.parent/'native-shared-transition-v1'))
from frozen_campaign import load
sys.path.insert(0,str(RAW/"launcher-boosted-free-gauge-v1"))
helper=load(RAW/'launcher-boosted-free-gauge-v1/native_transition_campaign.py')
plan,binding,worker,sub,collected=[read(p/n) for n in ['plan.json','binding.json','worker-status.json','submission.json','collected.json']]
r=read(p/'output/puncture.report.json');science=helper.science(RAW)
assert sha(p/'plan.json')==binding['plan_sha256']==worker['plan_sha256']==sub['plan_sha256']
assert worker['job_id']==sub['job_id']==collected['job_id']
assert binding['source_manifest_sha256']==worker['source_manifest_sha256']==science['source_manifest_sha256']
assert binding['binaries']==worker['binaries']=={helper.binary('B16'):science['binaries'][helper.binary('B16')]}
for f,h in binding['inputs'].items():assert sha(RAW/f)==h
for f,h in binding['launcher'].items():assert sha(RAW/'launcher-boosted-free-gauge-v1'/f)==h
assert collected['scheduler']['job_state']=='F' and collected['scheduler']['Exit_status']==1
case=worker['cases'][0];assert case['returncode']==143 and not case['timed_out'] and case['storage_error'] is None
assert worker['status']==r['status']=='failed' and not r['checkpoint_admitted']
assert r['checkpoint_schema']=='HYP3D_MPI_BG1'
checks=[helper.control.evaluate_check(r,c) for c in plan['commands'][0]['checks'] if c['path']!=['status']]
assert all(c['passed'] for c in checks if c['check']['path']!=['checkpoint_admitted'])
artifacts={}
for f,meta in case['artifacts'].items():
 q=RAW/f;assert q.stat().st_size==meta['bytes'] and sha(q)==meta['sha256'];artifacts[f]=meta
save('local-verification.json',dict(artifacts=artifacts,all_artifacts_preserved=True,plan_sha256=binding['plan_sha256'],source_manifest_sha256=binding['source_manifest_sha256'],binary=binding['binaries'],report_checks=checks,disposition='Terminal numerical failure, not admitted. Frozen helper rejects scheduler exit1 as unhealthy. No restart or higher-resolution submission.',scheduler_exit=1,native_launcher_exit=143,storage_error=None,timed_out=False))
for f in ['worker-status.json','collected.json','submission.json','binding.json','plan.json']:
 shutil.copy2(p/f,out/f)
for f in ['puncture.report.json','puncture.history.csv','puncture.log','puncture.fastflow.horizon_summary_0.txt','puncture.fastflow.horizon_shape_0.txt']:
 shutil.copy2(p/'output'/f,out/f)
u=np.fromfile(p/'output/puncture.failed-stage.bin',dtype='<f8').reshape(-1,31);n=r['cells'];assert n==232928 and len(u)==234464
assert np.isfinite(u).all()
g=np.zeros((len(u),3,3))
for f,(i,j) in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):g[:,i,j]=g[:,j,i]=u[:,f+1]+(i==j)
eig=np.linalg.eigvalsh(g);bad=np.where(eig[:n,0]<=0)[0];assert list(bad)==[4094,12046,16638,24590]
# CompositeMesh appends Cartesian blocks x-fast, then cells x-fast. No remapping.
idx=np.arange(32768);b=idx//4096;q=idx%4096
xyz=-.18+.18*np.stack([b%2,(b//2)%2,b//4],axis=1)+(.18/16)*(np.stack([q%16,(q//16)%16,q//256],axis=1)+.5)
rad=np.linalg.norm(xyz,axis=1);alpha=u[:32768,18]+(1+rad**2)/2
tracker=np.array(r['puncture_position']);locations=[]
for i in bad:
 locations.append(dict(index=int(i),xyz=xyz[i].tolist(),r=float(rad[i]),distance_to_last_accepted_tracker=float(np.linalg.norm(xyz[i]-tracker)),metric_eigenvalues=eig[i].tolist(),metric_determinant=float(np.linalg.det(g[i])),chi=float(u[i,0]),conformal_lapse=float(alpha[i]),beta=(u[i,22:25]-xyz[i]).tolist(),D=u[i,28:31].tolist()))
h=np.genfromtxt(p/'output/puncture.history.csv',delimiter=',',names=True)
accepted=(h['horizon_found']==1)&np.isfinite(h['horizon_mass']);last=h[accepted][-1]
signs=r['candidate_sign_events'];negative_stage=sum(v[5] for v in signs if v[1]==0);negative_final=sum(v[6] for v in signs if v[1]==0)
summary=dict(last_accepted_t_over_M=r['t_over_M'],steps=r['steps'],last_diagnostic_t_over_M=float(h['t_over_M'][-1]),error=r['error'],failed_stage_abscissa='Not separately recorded; report time is last accepted step.',capture_finite=True,volume_nonpositive_metric_cells=len(bad),surface_min_metric_eigenvalue=float(eig[n:,0].min()),locations=locations,puncture_position=tracker.tolist(),conservative_support_margin=.16-float(abs(tracker).max()),cube_margin=.18-float(abs(tracker).max()),null_surface_max=r['max_null_surface'],null_ghost_max=r['max_null_ghost'],pole_max=r['max_pole'],clock_max=r['max_clock'],chi_floors_stage=r['chi_floor_stage_interventions'],chi_floors_final=r['chi_floor_acceptance_interventions'],negative_chi_stage=negative_stage,negative_chi_final=negative_final,lapse_floor=r['lapse_floor_node_interventions'],surface_floors_stage=r['surface_chi_floor_stage_interventions'],surface_floors_final=r['surface_chi_floor_final_interventions'],accepted_horizon_rows=int(accepted.sum()),horizon_rows=len(h),last_accepted_horizon=dict(t_over_M=float(last['t_over_M']),mass=float(last['horizon_mass']),relative_mass_error=float(last['horizon_mass']*3-1)),last_history={k:float(h[k][-1]) for k in ['H_L2','M_L2','H_exterior_L2','M_exterior_L2','exterior_cutoff']},qualification='Immediate indefinite metric is localized near the puncture, within the original floor patch. This does not establish the root cause or exclude effects propagated from the outer region. No admitted endpoint, fixed-cutoff reconstruction, resolution convergence or long stability claim. Tracker moving negative x is measured gauge motion, not the reference positive Lorentz-chart velocity.',source_hashes={f:sha(ROOT/f) for f in ['src/mesh/cubed_sphere/composite.hpp','src/mesh/cubed_sphere/composite_difference.hpp','src/z4c/hyperboloidal/hyperbolic_composite_device.hpp']})
save('summary.json',summary)
basepath=ROOT/'output/restored-six-shells/aurora/campaigns/n16-restored-six-shells-v1-a-evolution-segment01/output/puncture.history.csv'
base=np.genfromtxt(basepath,delimiter=',',names=True);base=base[base['t_over_M']<=4.23]
save('plot-inputs.json',dict(boosted_history_sha256=sha(p/'output/puncture.history.csv'),unboosted_context_history_sha256=sha(basepath),failed_capture_sha256=sha(p/'output/puncture.failed-stage.bin'),unboosted_context='Completed six-shell unboosted N16, source812667b2e; context only, distinct initial geometry/gauge, own candidate cuts. No matched-domain subtraction.'))
fig,axes=plt.subplots(2,3,figsize=(13,7.5))
for ax,key,title in zip(axes.flat[:4],['H_L2','M_L2','H_exterior_L2','M_exterior_L2'],['Full H','Full M','Own-candidate exterior H','Own-candidate exterior M']):
 for a,label in [(base,'Unboosted six-shell context'),(h,'Boosted BG1')]:ax.semilogy(a['t_over_M'],a[key],'.-',ms=3,label=label)
 ax.set_title(title)
for k in ['null_surface','null_ghost','pole_max','clock_max']:axes[1,1].semilogy(h['t_over_M'],np.maximum(h[k],1e-18),label=k)
axes[1,1].legend(fontsize=7);axes[1,1].set_title('Cumulative boundary residual maxima')
axes[1,2].plot(h['t_over_M'],h['exterior_cutoff'],label='Candidate enclosing cutoff');axes[1,2].plot(h['t_over_M'],h['horizon_found']*.16,'x',label='Accepted flag ×0.16');axes[1,2].legend(fontsize=7);axes[1,2].set_title('Cutoff and finder acceptance')
for ax in axes.flat:ax.axvline(r['t_over_M'],ls=':',color='r');ax.set(xlabel='t/M',xlim=(0,4.3));ax.grid(alpha=.25)
axes[0,0].legend(fontsize=7);fig.suptitle('BG1 N16: failed at last accepted 4.23M; final recorded diagnostics 4.20M\nMoving candidate cutoffs differ; no fixed-domain convergence or endpoint comparison',fontsize=12);fig.tight_layout();fig.savefig(out/'histories.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(1,3,figsize=(13,4))
for k in ['puncture_x','puncture_y','puncture_z']:axes[0].plot(h['t_over_M'],h[k],'.-',label=k)
axes[0].plot(r['t_over_M'],tracker[0],'rx');axes[0].legend(fontsize=7);axes[0].set_title('Measured puncture trajectory')
err=np.where(accepted,abs(3*h['horizon_mass']-1),np.nan);axes[1].semilogy(h['t_over_M'],err,'o-');axes[1].set_title('Accepted |M_AH/M - 1| (gaps retained)')
axes[2].plot(h['t_over_M'],.16-np.maximum.reduce([abs(h[k]) for k in ['puncture_x','puncture_y','puncture_z']]));axes[2].set_title('Conservative Cartesian support margin')
for ax in axes:ax.set(xlabel='t/M',xlim=(0,4.3));ax.grid(alpha=.25)
fig.tight_layout();fig.savefig(out/'trajectory-mass.png',dpi=160);plt.close(fig)
mask=np.isclose(xyz[:,1],-.005625)&np.isclose(xyz[:,2],-.005625)&(abs(xyz[:,0])<.1);ii=np.where(mask)[0];ii=ii[np.argsort(xyz[ii,0])]
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,val,title in zip(axes,[eig[:32768,0],u[:32768,0],alpha],['Minimum metric eigenvalue','chi','Conformal lapse']):
 ax.plot(xyz[ii,0],val[ii],'.-');ax.axvline(tracker[0],ls='--',label='Last accepted tracker');ax.axvline(-.016875,color='r',ls=':',label='Failed cell');ax.axhline(0,color='k',lw=.5);ax.set(xlabel='x (compact coordinate)',title=title);ax.grid(alpha=.25)
axes[0].legend(fontsize=7);fig.suptitle('Rejected-stage core slice y=z=-0.005625: descriptive capture only, not an admitted endpoint');fig.tight_layout();fig.savefig(out/'failed-core-slice.png',dpi=160);plt.close(fig)
print(json.dumps(summary,indent=2))
