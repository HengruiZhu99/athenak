"""Post-collection DG1 comparison; never grants N24 admission or submits jobs."""
from pathlib import Path
import argparse,hashlib,json,math,subprocess
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--plans',nargs='+',required=True);a=p.parse_args();raw=a.root.resolve();out=HERE/'dg1-comparison';out.mkdir(exist_ok=True)
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
records=[];dg_hist=[];reports=[]
for relative in a.plans:
 planpath=raw/relative;plan=json.loads(planpath.read_text());folder=planpath.parent
 w=json.loads((folder/'worker-status.json').read_text());c=json.loads((folder/'collected.json').read_text())
 assert c['scheduler']['job_state'] in ['F','X'] and w['status'] in ['passed','checkpointed']
 assert digest(planpath)==w['plan_sha256'];assert len(w['cases'])==1
 case=w['cases'][0]
 for name,v in case['artifacts'].items():
  f=raw/name;assert f.stat().st_size==v['bytes'] and digest(f)==v['sha256'],name
 cmd=plan['commands'][0];report=json.loads((raw/cmd['report']).read_text());reports.append(report)
 history=raw/(cmd['args'][1]+'.history.csv');dg_hist.append(np.atleast_1d(np.genfromtxt(history,delimiter=',',names=True)))
 records.append(dict(plan=relative,worker_status=w['status'],artifacts=case['artifacts']))
last=reports[-1];assert last['status']=='completed' and last['t_over_M']==10 and last['checkpoint_admitted'] and last['checkpoint_schema']=='HYP3D_MPI_DG1'
v=json.loads((folder/'checkpoint-verified.json').read_text());assert v['verified'] and v['completed'] and v['t_over_M']==10
cp=raw/v['checkpoint'];assert digest(cp)==v['sha256']
cases={'DG1':(last,cp,'damping_ablation',v['sha256'])}
for label,variant,campaign,target,sha in [('IT1','inner-telegrapher-v1','n16-inner-telegrapher-fresh10m-v1-segment01','inner_telegrapher','37665a7936afc1e404c653a1ea8fc1051a437a31f3bc907dc5cadec76c483d9f'),('DO1','direct-outer-v1','n16-direct-outer-fresh10m-v1-segment01','direct_outer','edc58b0df7ef0391567221d8b95990e82ab05aad6264bc87d7e4a49ff2449c5c')]:
 report=json.loads((HERE.parent/variant/'n16-completed/puncture.report.json').read_text());checkpoint=ROOT/'output/hyperbolic-core-shell'/variant/'aurora/campaigns'/campaign/'output/puncture.checkpoint';assert digest(checkpoint)==sha
 cases[label]=(report,checkpoint,target,sha)
radii=[r['horizon_trial_radius'] for r,_,_,_ in cases.values()];assert all(x is not None and math.isfinite(x) and 0<x<1 for x in radii)
cut=max(.1,math.ceil(max(radii)*10000)/10000+.0001)
bands=sorted(set([0,.03,.06,cut,.1,.13,.16,.18,.2,.25,.3,.35,.4,.45,.5,.55,.6,.65,.675,.7,.725,.75,.775,.8,.85,.9,.95,.98,1]))
radial={};execution=[]
for label,(report,checkpoint,target,sha) in cases.items():
 exe=ROOT/'build-hyperbolic-mpi/tst/hyperbolic_shift_3d'/('hyperbolic_shift_3d_checkpoint_constraints_'+target);dest=out/f'{label}-radial.json'
 command=[str(exe),'16',str(checkpoint),'.0005',str(cut),str(dest),','.join(map(str,bands))]
 with (out/f'{label}-radial.log').open('w') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
 assert digest(checkpoint)==sha;d=json.loads(dest.read_text());assert d['t_over_M']==10 and d['cutoff']==cut
 for key in ['H_L2','M_L2','Mr_L2']:
  assert abs(d['norms']['total'][key]-report[key])<1e-10*(1+abs(report[key])),(label,key)
 radial[label]=d;execution.append(dict(label=label,command=command,binary_sha256=digest(exe),checkpoint_sha256=sha))
histories={'DG1':np.concatenate(dg_hist)}
for label,variant in [('IT1','inner-telegrapher-v1'),('DO1','direct-outer-v1')]:histories[label]=np.atleast_1d(np.genfromtxt(HERE.parent/variant/'n16-completed/puncture.history.csv',delimiter=',',names=True))
fig,axes=plt.subplots(2,3,figsize=(13,7));summary={}
fields=[('H_L2','Full-domain Hamiltonian RMS',1),('M_L2','Full-domain momentum RMS',1),('Theta_L2','Full-domain Theta RMS',1),('H_exterior_L2','Candidate-exterior Hamiltonian RMS',1),('Mr_exterior_L2','Candidate-exterior radial momentum RMS',1),('horizon_areal_radius','Accepted horizon areal radius / M',3)]
for label,color in [('IT1','#777777'),('DO1','#b55315'),('DG1','#1769aa')]:
 d=histories[label];d=d[np.argsort(d['t_over_M'],kind='stable')];_,ids=np.unique(d['t_over_M'],return_index=True);d=d[ids]
 for ax,(key,title,scale) in zip(axes.flat,fields):ax.plot(d['t_over_M'],d[key]*scale,label=label,color=color);ax.set(xlabel='t / M',ylabel=title);ax.grid(alpha=.2)
 late=d[d['t_over_M']>=4-1e-8];summary[label]=dict(common_exterior=radial[label]['norms']['exterior'],late_history_max={key:float(np.nanmax(late[key])) for key in ['H_L2','M_L2','Theta_L2','H_exterior_L2','Mr_exterior_L2']},accepted_horizon_rows=int(np.sum(d['horizon_found'])),history_rows=len(d),final_horizon_accepted=bool(d[-1]['horizon_found']))
axes[0,0].legend();fig.suptitle('N16 damping-only ablation: same new gauge, restored previous damping');fig.tight_layout(rect=(0,.06,1,.96));fig.text(.5,.015,'Histories use each run’s own candidate cutoff; final common-cutoff norms are recorded separately. Missing horizon values remain gaps.',ha='center',fontsize=9);fig.savefig(out/'histories.png',dpi=150)
result=dict(common_cutoff=cut,candidate_radii=radii,summary=summary,execution=execution,segments=records,n24_admitted=False,review_required=True)
(out/'comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(summary,indent=2))
