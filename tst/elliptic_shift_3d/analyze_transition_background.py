#!/usr/bin/env python3
"""Audit frozen fixed/free-background failures and an independent normalization check."""
import argparse,collections,hashlib,json,os,tempfile
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--original',type=Path,required=True);p.add_argument('--normalization',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(exist_ok=False)
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return [json.loads(line) for line in p.read_text().splitlines()]
oldpath=a.original/'probe.stdout';newpath=a.normalization/'probe.stdout';old=read(oldpath);new=read(newpath);oldplans=[r for r in old if r['phase']=='plan'];plans=[r for r in new if r['phase']=='plan'];assert len(oldplans)==288 and len(plans)==576
causes=collections.Counter();combinations=collections.Counter();coefficient_only=[]
for r in oldplans:
 if r['status']=='ready':continue
 reasons=[]
 if r['rank']!=57:reasons=['rank']
 elif r['status']=='weight_amplification':reasons=['early_weight_guard_unseparated']
 else:
  if r['condition']>1e8:reasons.append('condition')
  if r['reproduction']>1e-10:reasons.append('reproduction')
  if r['value_l1']>16 or r['gradient_l1']>32 or r['hessian_l1']>64:reasons.append('physical_functional_amplification')
  if r['coefficient_l1']>10000:reasons.append('coefficient_amplification')
 assert reasons
 combinations[tuple(reasons)]+=1
 for reason in reasons:causes[reason]+=1
 if reasons==['coefficient_amplification']:coefficient_only.append({k:r[k] for k in ['point','field','donors','coefficient_l1']})
weights=[r for r in new if r['phase']=='normalization_weights'];assert all(r['within_1e_9'] for r in weights)
accepted=collections.Counter(r['normalization'] for r in plans if r['status']=='ready');source={}
for r in new:
 if r['phase']=='free_background':source[(r['point'],r['field'],r['donors'],r['case'],r['normalization'])]=r
paired=[]
for key,row in source.items():
 if key[-1]!=1:continue
 other=source.get((*key[:-1],10))
 if other:
  paired.append({'point':key[0],'field':key[1],'donors':key[2],'case':key[3],'coefficient_covariance_error':abs(row['coefficient']-10*other['coefficient']),'gradient_error_change':abs(row['gradient_error']-other['gradient_error']),'hessian_error_change':abs(row['hessian_error']-other['hessian_error']),'H_error_change':abs(row['H_error']-other['H_error']),'source_error_max_change':abs(row['full_physical_source_error_max']-other['full_physical_source_error_max'])})
summary={'scope':'Standalone 48-query initial-background derivative audit. All original288/cap10000 failures remain frozen; normalized run is independent. No evolution/production or universal stability claim.','original_summary':next(r for r in old if r['phase']=='summary'),'normalization_summary':next(r for r in new if r['phase']=='summary'),'old_failure_overlapping_reasons':dict(causes),'old_failure_disjoint_combinations':[{'reasons':list(k),'count':v} for k,v in combinations.items()],'old_coefficient_only_failures':coefficient_only,'early_weight_guard_note':'36 old plans cleared weights before separate audits, so their physical/coefficient contributions are not reconstructed or relabeled.','ready_under_raw_coefficient_cap':{str(k):v for k,v in accepted.items()},'normalization_physical_weight_comparison':{'plans':len(weights),'predeclared_tolerance':1e-9,'maximum_scaled_functional_change':max(r['scaled_functional_change_max'] for r in weights),'all_within_tolerance':all(r['within_1e_9'] for r in weights)},'matched_accepted_source_covariance':{'pairs':len(paired),'maximum_H_error_change':max(r['H_error_change'] for r in paired),'maximum_full_source_error_change':max(r['source_error_max_change'] for r in paired),'maximum_coefficient_covariance_error':max(r['coefficient_covariance_error'] for r in paired),'rows':paired},'target126':[r for r in new if r.get('point')==126],'physical_rhs_preserved':next(r for r in new if r['phase']=='initial_full_physical_rhs'),'normalization_conclusion':'Raw background-coefficient L1 is not invariant to scaling a nuisance basis column; it cannot alone establish intrinsic stability. Do not relabel old failures or relax physical/rank/condition/reproduction gates.','next_guard_gate':'Use physical functional amplification and measured physical-source/noise sensitivity, dimensionless basis/reproduction diagnostics, and sensitivity to independent background-jet uncertainty. A raw coefficient becomes meaningful only together with an invariant bound on the corresponding background uncertainty. Complete transition geometry coverage and matched nonspherical/profile tests remain required before adoption.','scientific_acceptance':False}
strong=[r for r in new if r.get('point')==126 and r.get('case')=='strong_nonspherical'];summary['matched_strong_nonspherical']=strong
with tempfile.TemporaryDirectory(prefix='background-plot-',dir=out) as cache:
 os.environ['MPLCONFIGDIR']=cache;os.environ['XDG_CACHE_HOME']=cache
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 fig,axes=plt.subplots(2,2,figsize=(12,9),layout='constrained');constant=[r for r in new if r['phase']=='fixed_decomposition' and r['field']=='chi' and r['case']=='constant'];axes[0,0].bar(range(2),[r['hessian_error'] for r in constant]);axes[0,0].set_xticks(range(2),[r['rule'] for r in constant],rotation=15,ha='right');axes[0,0].set_title('Fixed derivative decomposition fails constants');axes[0,0].set_ylabel('stored chi Hessian component error')
 for index,field in enumerate(['chi','alpha']):
  values=[]
  for rule in ['ordinary_FD4','DB_plus_FD_deviation']:
   values.append(next(r['full_physical_source_error_max'] for r in strong if r['phase']=='fixed_decomposition' and r['field']==field and r['rule']==rule))
  values.append(next(r['full_physical_source_error_max'] for r in strong if r['phase']=='free_background' and r['field']==field and r['normalization']==10 and r['donors']==256))
  axes[0,1].bar([x+index*.35 for x in range(3)],values,width=.35,label=field)
 axes[0,1].set_xticks([.175,1.175,2.175],['FD4','fixed decomposition','free fitted basis'],rotation=15,ha='right');axes[0,1].set_title('Matched strong nonspherical physical-source error');axes[0,1].legend()
 axes[1,0].bar([0,1],[accepted[1],accepted[10]]);axes[1,0].set_xticks([0,1],['B','10B']);axes[1,0].set_ylim(0,288);axes[1,0].set_title('Raw coefficient cap changes status for same span');axes[1,0].set_ylabel('ready plans /288')
 rhs=summary['physical_rhs_preserved'];axes[1,1].bar(range(3),[rhs[k] for k in ['chi_rate','Khat_rate','alpha_rate']]);axes[1,1].set_xticks(range(3),['chi_t','Khat_t','alpha_t']);axes[1,1].set_title('Exact initial jets retain nonzero physical rates')
 fig.suptitle('Background derivative hypotheses: local probe, no production adoption\nOriginal failures preserved; physical operator/span unchanged by basis normalization');fig.savefig(out/'background-basis-audit.png',dpi=150);plt.close(fig)
summary['input_sha256']={str(f):sha(f) for f in [oldpath,newpath]};summary['analysis_script_sha256']=sha(Path(__file__));(out/'analysis.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n');print(json.dumps({'old_coefficient_only':len(coefficient_only),'old_failure_combinations':summary['old_failure_disjoint_combinations'],'normalized_ready':dict(accepted),'maximum_physical_weight_change':summary['normalization_physical_weight_comparison']['maximum_scaled_functional_change'],'maximum_matched_source_change':summary['matched_accepted_source_covariance']['maximum_full_source_error_change']}))
