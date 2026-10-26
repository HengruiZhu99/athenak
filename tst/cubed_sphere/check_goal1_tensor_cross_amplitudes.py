#!/usr/bin/env python3
"""Native cross-polarization full/half/quarter nonlinear model control."""
from pathlib import Path
import argparse,json,ast,struct
import numpy as np
p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();R=a.root.resolve();W=R/'build-goal1-minkowski-20261018';E=R/'docs/validation/goal1-minkowski-20261018';P=E/'native-tensor-propagation-v1';L=json.loads((P/'bounded-propagation-protocol-v7.json').read_text())['length'];normal=json.loads((E/'native-tensor-pilot-v1/primary-normalization-v2.json').read_text());amp=normal['full_configuration_scale'];config=[0,1,2,3,4,5,6,18,19,20,21];scale=np.full(22,amp/L);scale[config]=amp;src=Path(__file__).with_name('review_goal1_clock_pilots.py');tree=ast.parse(src.read_text());ns={'np':np};exec(compile(ast.Module(body=[f for f in tree.body if isinstance(f,ast.FunctionDef)and f.name in ['norm','reference_weights']],type_ignores=[]),str(src),'exec'),ns)
def read(phase,label):
 case=W/phase/label
 with(case/'checkpoint.bin').open('rb')as f:h=struct.unpack('<16s64s22dQdq',f.read(280))
 with(case/'cubed_sphere_native_last_jets.bin').open('rb')as f:n,c=np.fromfile(f,np.uint64,2)
 assert c==300;x=np.memmap(case/'cubed_sphere_native_last_jets.bin',dtype='<f8',mode='r',offset=16,shape=(int(n),300));t=np.memmap(W/phase/('analysis-'+label)/'final-target.bin',dtype='<f8',mode='r',offset=16,shape=(int(n),294));return h,x,x[:,14:].reshape(-1,22,13)-t[:,:286].reshape(-1,22,13)
h,x,ef=read('native-tensor-pilot-v1','tt-cross-resolved4-v3');hh,xh,eh=read('native-tensor-propagation-v1','tt-cross-amphalf-v7');hq,xq,eq=read('native-tensor-propagation-v1','tt-cross-ampquarter-v7');assert h[-2]==hh[-2]==hq[-2]and np.array_equal(x[:,:4],xh[:,:4])and np.array_equal(x[:,:4],xq[:,:4]);weight=ns['reference_weights'](x)['fixed_reference_Penrose'];mask=np.ones(len(x),bool)
def norms(q):
 q=q/scale[None,:,None];return{k:ns['norm'](z,weight,mask)for k,z in {'fields':q[:,:,0],'first':L*q[:,:,1:4],'configuration_second':L*L*q[:,config,4:]}.items()}
d1=ef-2*eh;d2=eh-2*eq;err=norms(ef);m=norms(2*d1);defect=norms(d1-4*d2);n1=norms(d1);ratios={k:{s:m[k][s]/err[k][s]for s in ['rms','linf']}for k in m};quad={k:{s:defect[k][s]/n1[k][s]for s in ['rms','linf']}for k in m};passed=all(v<.1 for q in ratios.values()for v in q.values())and all(v<.1 for q in quad.values()for v in q.values());j={'scope':'Actual full native coupled cross-polarization common-time amplitude ladder, immutable full root reused; independent linear native target. No nonlinear vacuum exact-solution claim.','time':h[-2],'length':L,'full_oracle_error':err,'full_leading_quadratic_model_estimate':m,'full_model_fraction':ratios,'quadratic_scaling_relative_defect':quad,'short_model_budget_pass':passed,'later_model_contamination_admitted':False};(P/'cross-amplitude-review.json').write_text(json.dumps(j,indent=2)+'\n');print(json.dumps(j,indent=2));assert passed
