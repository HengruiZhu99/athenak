#!/usr/bin/env python3
"""Compare actual native TT timestep/amplitude controls at common coordinates/time."""
from pathlib import Path
import argparse,json,struct,ast
import numpy as np
p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('evidence',type=Path);a=p.parse_args();protocol=json.loads((a.evidence/'short-controls-protocol-v4.json').read_text());L=protocol['length'];config=[0,1,2,3,4,5,6,18,19,20,21];normal=json.loads((a.evidence/'primary-normalization-v2.json').read_text());amp=normal['full_epsilon']*normal['fixed_common_configuration_unit_scale'];scale=np.full(22,amp/L);scale[config]=amp;ns={'np':np};src=Path(__file__).with_name('review_goal1_clock_pilots.py');tree=ast.parse(src.read_text());exec(compile(ast.Module(body=[f for f in tree.body if isinstance(f,ast.FunctionDef)and f.name in ['norm','reference_weights']],type_ignores=[]),str(src),'exec'),ns)
def load(label):
 case=a.work/label
 with(case/'checkpoint.bin').open('rb')as f:h=struct.unpack('<16s64s22dQdq',f.read(280))
 with(case/'cubed_sphere_native_last_jets.bin').open('rb')as f:n,c=np.fromfile(f,np.uint64,2)
 assert c==300;v=np.memmap(case/'cubed_sphere_native_last_jets.bin',dtype='<f8',mode='r',offset=16,shape=(int(n),300));t=np.memmap(a.work/('analysis-'+label)/'final-target.bin',dtype='<f8',mode='r',offset=16,shape=(int(n),294));return h,v,t
full,x,tf=load('tt-plus-resolved4-v3');weights=ns['reference_weights'](x)['fixed_reference_Penrose'];mask=np.ones(len(x),bool)
def norms(q):
 q=q/scale[None,:,None];return {k:ns['norm'](v,weights,mask)for k,v in {'fields':q[:,:,0],'first':L*q[:,:,1:4],'configuration_second':L*L*q[:,config,4:]}.items()}
uf=x[:,14:].reshape(-1,22,13);err=norms(uf-tf[:,:286].reshape(-1,22,13));controls={}
for label in ['tt-plus-dthalf-v4','tt-plus-dtquarter-v4-segment2','tt-plus-amphalf-v4']:
 h,v,target=load(label);assert h[0]==full[0]and h[1]==full[1]and abs(h[-2]-full[-2])<1e-15;assert np.array_equal(v[:,:4],x[:,:4]);u=v[:,14:].reshape(-1,22,13);oracle=target[:,:286].reshape(-1,22,13)
 if 'amphalf'in label:
  delta=(uf-tf[:,:286].reshape(-1,22,13))-2*(u-oracle);metric=norms(delta);denom=err
 else:metric=norms(uf-u);denom=norms(u-oracle)
 ratios={k:{s:metric[k][s]/denom[k][s]if denom[k][s]else None for s in ['rms','linf']}for k in metric};controls[label]={'time':h[-2],'cycles':h[-1],'full_difference_or_amplitude_model_difference':metric,'control_oracle_error_or_full_oracle_error':denom,'ratios':ratios,'all_below_10percent':all(v is not None and v<.1 for q in ratios.values()for v in q.values())}
hh,vh,th=load('tt-plus-amphalf-v4');hq,vq,tq=load('tt-plus-ampquarter-v5');assert abs(hq[-2]-full[-2])<1e-15 and np.array_equal(vq[:,:4],x[:,:4]);ef=uf-tf[:,:286].reshape(-1,22,13);eh=vh[:,14:].reshape(-1,22,13)-th[:,:286].reshape(-1,22,13);eq=vq[:,14:].reshape(-1,22,13)-tq[:,:286].reshape(-1,22,13);d1=ef-2*eh;d2=eh-2*eq;d1norm=norms(d1);quadratic_defect=norms(d1-4*d2);model_norm=norms(2*d1)
quadratic_ratios={k:{stat:quadratic_defect[k][stat]/d1norm[k][stat] for stat in ['rms','linf']}for k in d1norm};full_model_ratios={k:{stat:model_norm[k][stat]/err[k][stat]for stat in ['rms','linf']}for k in err};quad_pass=all(v<.1 for q in quadratic_ratios.values()for v in q.values());full_model_pass=all(v<.1 for q in full_model_ratios.values()for v in q.values())
hd,vd,td=load('tt-plus-dthalf-v4');hqd,vqd,tqd=load('tt-plus-dtquarter-v4-segment2');halfquarter=norms(vd[:,14:].reshape(-1,22,13)-vqd[:,14:].reshape(-1,22,13));dtorder={k:{stat:float(np.log2(controls['tt-plus-dthalf-v4']['full_difference_or_amplitude_model_difference'][k][stat]/halfquarter[k][stat]))for stat in ['rms','linf']}for k in halfquarter}
report={'scope':'Actual native nonlinear coupled jets, common time/coordinates, fixed physical-length v3 normalization. Linear-target model error is assessed by amplitude scaling; not exact nonlinear vacuum oracle. No expected RK order assumed.','protocol':'short-controls-protocol-v4.json','full_time':full[-2],'full_oracle_error':err,'controls':controls,'temporal_budget_pass':controls['tt-plus-dtquarter-v4-segment2']['all_below_10percent'],'model_budget_pass':full_model_pass and quad_pass,'full_leading_quadratic_model_estimate':model_norm,'full_leading_model_error_fraction':full_model_ratios,'quadratic_scaling_relative_defect':quadratic_ratios,'quadratic_scaling_pass':quad_pass,'empirical_full_half_quarter_dt_difference_order':dtorder,'dt_order_scope':'Short actual method differences, not an asymptotic RK-order proof; near-first-order trend retained.','propagation_admitted':False,'exact_scri_numerical_admitted':False}
(a.evidence/'short-controls-review-v4.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
