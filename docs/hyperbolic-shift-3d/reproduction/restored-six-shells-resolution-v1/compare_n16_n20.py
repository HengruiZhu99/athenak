"""Paired native endpoint residual orders; no evolution or policy modification."""
from pathlib import Path
import argparse,sys,json,hashlib,subprocess,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[3]
sys.path.insert(0,str(HERE.parent/'native-shared-transition-v1'));sys.path.insert(0,str(HERE.parent/'restored-six-shells-v1'))
import compare_histories as histories
from frozen_campaign import load
from audit_completed import load_completed
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_text())
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=HERE/'n16-n20');args=parser.parse_args();out=args.output
old=ROOT/'output/restored-six-shells/aurora';new=ROOT/'output/restored-six-shells-resolution/aurora'
p16='campaigns/n16-restored-six-shells-v1-a-evolution-segment01/plan.json';p20='campaigns/n20-restored-six-shells-resolution-v1-r20-evolution-segment01/plan.json'
helper=load(new/'launcher-restored-six-shells-resolution-v1/native_transition_campaign.py')
c16,oldaudit=load_completed();c20=histories.load_native(new,'R20',[p20],helper)
plan=read(new/p20);report=c20['reports'][0];assert all(helper.control.evaluate_check(report,x)['passed'] for x in plan['commands'][0]['checks'])
assert report['status']=='completed' and report['checkpoint_admitted']
assert abs(report['scri_penalty_scale_min']-c16['reports'][0]['scri_penalty_scale_min'])<1e-14
assert oldaudit['binary_sha256']==helper.science(new)['binaries'][helper.binary('R20')]
cases={'N16*':c16,'N20':c20}
histories.LABELS.update({'N16*':'N16 six shells, old policy failure retained','N20':'N20 six shells, matching clipped-penalty policy'})
histories.COLORS.update({'N16*':'#238b45','N20':'#2166ac'})
h=histories.render(cases,out,title='Six-shell broad/smooth formulation: N16 versus N20',footer='Histories use each run’s candidate horizon, not a common fixed cutoff; mass gaps retained.\nSame equations and penalty scale 0.651; dt scales as 1/N. N16* failed its original scale=1 policy; that record is unchanged.')
h['cases']['N16*']['status']='native_completed_old_penalty_policy_failed';h['scope']+=' N16 original acceptance failure retained; common clipped-penalty family explicitly selected for N20.'
(out/'comparison.json').write_text(json.dumps(h,indent=2)+'\n')
# Reuse the independently reconstructed N16 endpoint, with original source and hash.
prior=HERE.parent/'restored-six-shells-v1/comparison';a=read(prior/'six-radial-command.json');shutil.copy2(prior/'six-radial.json',out/'n16-radial.json')
binary=ROOT/'build-restored-six-shells-v1/tst/hyperbolic_shift_3d/hyperbolic_shift_3d_checkpoint_constraints_native_control_broad_smooth_six_shells'
assert sha(binary)==a['binary_sha256'],'Use the same diagnostic binary as the N16 reconstruction'
cp16=old/Path(p16).parent/'output/puncture.checkpoint';cp20=new/Path(p20).parent/'output/puncture.checkpoint'
assert sha(cp16)==a['checkpoint_sha256'];cp20hash=sha(cp20)
assert cp20hash==read(new/Path(p20).parent/'checkpoint-verified.json')['cases'][0]['sha256']
bounds={}
for n,root,p in [(16,old,p16),(20,new,p20)]:
 coeff=np.loadtxt(root/Path(p).parent/'output/puncture.fastflow.horizon_shape_0.txt')[-1];assert len(coeff)==169
 bounds[str(n)]=float(coeff[0]/np.sqrt(4*np.pi)+sum(np.sqrt((2*l+1)/(4*np.pi))*np.linalg.norm(coeff[l*l:(l+1)**2]) for l in range(1,13)))
 assert bounds[str(n)]<.1099
cuts=a['command'][-1];cmd=[str(binary),'20',str(cp20),'.0004','.12',str(out/'n20-radial.json'),cuts]
provenance=dict(N16_reused_path=str((prior/'six-radial.json').relative_to(ROOT)),N16_radial_sha256=sha(prior/'six-radial.json'),N16_original_command=a,N20_command=cmd,N20_checkpoint_sha256=cp20hash,diagnostic_binary_sha256=sha(binary),horizon_upper_bounds=bounds,scope='Same diagnostic executable/operators, native boundary refresh at each dt, matched penalty clipping, different N. No change to original N16 policy failure.')
(out/'reconstruction-provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
with (out/'n20-radial.log').open('x') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
assert sha(cp20)==cp20hash
D={16:read(out/'n16-radial.json'),20:read(out/'n20-radial.json')}
for n,c in [(16,c16),(20,c20)]:
 d=D[n];assert d['n']==n and d['t_over_M']==10 and d['cutoff']==.12
 for key in ['H_L2','M_L2','Mr_L2']:
  assert abs(d['norms']['total'][key]/c['reports'][0][key]-1)<1e-9
order=lambda x,y:float(np.log(x/y)/np.log(20/16))
metrics=['H_L2','M_L2','Mr_L2'];endpoint={}
for key in metrics:
 x,y=[D[n]['norms']['exterior'][key] for n in [16,20]]
 endpoint[key]=dict(N16=x,N20=y,reduction_percent=100*(1-y/x),effective_order=order(x,y))
sensitivity={}
for cutoff in [.1099,.12,.13,.16,.2,.3,.6,.7,.775,.9]:
 rows={n:next(r for r in D[n]['radial_cumulative'] if r['cutoff']==cutoff) for n in D}
 sensitivity[str(cutoff)]={k:dict(N16=float(np.sqrt(rows[16][k]/rows[16]['volume'])),N20=float(np.sqrt(rows[20][k]/rows[20]['volume'])),effective_order=order(np.sqrt(rows[16][k]/rows[16]['volume']),np.sqrt(rows[20][k]/rows[20]['volume']))) for k in ['H2','M2','Mr2']}
regions={}
for lo,hi in [(.12,.16),(.16,.3),(.3,.6),(.6,.7),(.7,.775),(.775,.9),(.9,1),(.6,.9)]:
 vals={}
 for n in D:
  aa=next(r for r in D[n]['radial_cumulative'] if r['cutoff']==lo);bb=next(r for r in D[n]['radial_cumulative'] if r['cutoff']==hi)
  vals[n]={k:aa[k]-bb[k] for k in ['volume','H2','M2','Mr2']}
 regions[f'{lo:g}–{hi:g}']={k:dict(N16_integral=vals[16][k],N20_integral=vals[20][k],N16_RMS=float(np.sqrt(vals[16][k]/vals[16]['volume'])),N20_RMS=float(np.sqrt(vals[20][k]/vals[20]['volume'])),effective_order=order(np.sqrt(vals[16][k]/vals[16]['volume']),np.sqrt(vals[20][k]/vals[20]['volume']))) for k in ['H2','M2','Mr2']}
summary=dict(formula='p_eff=ln(E16/E20)/ln(20/16), E=constraint RMS, continuum constraints zero',cutoff=.12,time_over_M=10,endpoint=endpoint,cutoff_sensitivity=sensitivity,regional_RMS=regions,coordinate_volumes={n:D[n]['norms']['exterior']['volume'] for n in D},horizon_upper_bounds=bounds,qualifications=['Two-point effective constraint-residual order, not a solution self-convergence or asymptotic-order proof.','Space and timestep refined together (dt .0005 to .0004); not a purely spatial order.','Same six-shell equations and explicit clipped-penalty family. N16 old acceptance failure preserved.','Radial/angular/Cartesian resolutions change together; fixed coordinate cutoffs with resolution-dependent quadrature.','Histories use own-candidate cuts; no fixed-cutoff historical order inferred.'])
(out/'orders.json').write_text(json.dumps(summary,indent=2)+'\n')
# Honest two-point log plot; reference slopes anchored independently at N16.
fig,axes=plt.subplots(1,2,figsize=(11.5,4.6))
for ax,key,label in zip(axes,metrics[:2],['Hamiltonian','Momentum']):
 e=endpoint[key];ns=np.array([16,20]);ax.loglog(ns,[e['N16'],e['N20']],'o-',color='#2166ac',lw=2,label=f"Measured pair: p = {e['effective_order']:.2f}")
 for p,ls in [(2,':'),(4,'--')]:ax.loglog(ns,e['N16']*(16/ns)**p,ls,color='gray',label=f'p = {p} reference')
 ax.set_xticks([16,20],labels=['16','20']);ax.minorticks_off();ax.set(xlabel='N (core, angular and per-shell radial)',ylabel=label+' constraint RMS');ax.legend(fontsize=9);ax.grid(alpha=.2,which='both')
fig.suptitle('10M endpoint: common cutoff r > 0.12, same diagnostic operators');fig.tight_layout(rect=(0,.12,1,.94));fig.text(.5,.025,'Two-resolution effective order only; matched penalty scale 0.651 and dt ∝ 1/N.\nN16 completed natively but failed its original penalty policy; no retroactive policy change.',ha='center',fontsize=9);fig.savefig(out/'effective-order.png',dpi=180);plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11.5,4.6))
for n,color in [(16,'#238b45'),(20,'#2166ac')]:
 rows=D[n]['radial_cumulative'];r=np.array([v['cutoff'] for v in rows]);vol=-np.diff([v['volume'] for v in rows]);keep=r[:-1]>=.12
 for ax,key in zip(axes,['H2','M2']):
  sums=-np.diff([v[key] for v in rows]);value=np.full_like(vol,np.nan);ok=vol>1e-14;value[ok]=np.sqrt(np.maximum(sums[ok],0)/vol[ok]);ax.stairs(value[keep],r[np.r_[keep,True]],baseline=None,color=color,label=f'N{n}'+('*' if n==16 else ''))
for ax,key in zip(axes,['H','M']):
 ax.axvspan(.7,.775,color='gray',alpha=.12);ax.axvline(.75,color='gray',ls=':',lw=1);ax.set(xlabel='compactified radius r',ylabel=key+' regional RMS',yscale='log',xlim=(.12,1));ax.grid(alpha=.2);ax.legend()
fig.suptitle('10M radial constraint distribution — same physical radial bands');fig.tight_layout(rect=(0,.12,1,.94));fig.text(.5,.025,'Shaded: 0.7–0.775 around original interface. Empty quadrature bins omitted.\n* N16 original policy failure retained; both runs use the same clipped-penalty family.',ha='center',fontsize=9);fig.savefig(out/'radial.png',dpi=180);plt.close(fig)
print(json.dumps(summary,indent=2))
