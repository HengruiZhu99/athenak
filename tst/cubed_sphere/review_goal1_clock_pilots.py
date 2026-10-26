#!/usr/bin/env python3
"""Quantitative native clock pilot review with declared volume/field scales."""
from pathlib import Path
import argparse,json,csv,struct,hashlib,subprocess,time
import numpy as np
p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('output',type=Path);p.add_argument('cases',nargs='+');a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
config=[0,1,2,3,4,5,6,18,19,20,21];length=.2

def load(case):
 path=a.work/case
 with (path/'checkpoint.bin').open('rb')as f:head=struct.unpack('<16s64s19dQdq',f.read(256))
 assert head[0]==b'ATHENA_CSP_CK03!';time=head[-2];eps=head[20]
 with (path/'cubed_sphere_native_last_jets.bin').open('rb')as f:n,c=np.fromfile(f,np.uint64,2);data=np.fromfile(f,np.float64).reshape(int(n),int(c))
 assert c==300 and np.isfinite(data).all();return path,time,eps,data

def reference_weights(data):
 r=np.sqrt(np.einsum('ij,ij->i',data[:,:3],data[:,:3]));w=np.zeros(len(r));wp=np.zeros(len(r));mid=(r>.05)&(r<.95);s=(r[mid]-.05)/.9;t=(.95-r[mid])/.9;g=-1/s+1/t;e=np.exp(-abs(g));w[mid]=np.where(g<=0,e/(1+e),1/(1+e));wp[mid]=e/(1+e)**2*(1/s**2+1/t**2)/.9;w[r>=.95]=1;omega=1-w*r*r;b=2*r*w;L=1+w*r*r+r**3*wp;alpha=np.sqrt(omega*omega+b*b);conformal=data[:,3]*L/alpha
 return {'mapped_Cartesian':data[:,3],'fixed_reference_Penrose':conformal,'finite_reference_physical':conformal/omega**3}

def norm(z,w,mask):
 z=z[mask];point=np.sum(z*z,axis=tuple(range(1,z.ndim)))if z.ndim>1 else z*z;weights=w[mask];return {'rms':float(np.sqrt(np.einsum('i,i->',weights,point)/weights.sum())),'linf':float(np.sqrt(point.max()))}

report={'scope':'Native coupled clock pilots against source-matched exact solution; no scientific matrix/causal boundary/exact-scri numerical acceptance.','volumes':'All original mapped-Cartesian norms retained; fixed-reference Penrose measure L/A dx³ is the candidate scientific norm. Physical reference proper-volume norm supplied at finite cuts only.','cases':{}}
for label in a.cases:
 path,t,eps,x=load(label);analysis=a.work/('analysis-'+label);analysis.mkdir(exist_ok=True);inp=analysis/'target-input.bin';targetpath=analysis/'target.bin'
 if not targetpath.exists():
  with inp.open('wb')as f:f.write(struct.pack('<ddQ',t,eps,len(x)));x[:,:3].copy().tofile(f)
  cmd=[str((a.work/'clock_export').resolve()),str(inp.resolve()),str(targetpath.resolve())];run=subprocess.run(cmd,capture_output=True,text=True);assert run.returncode==0,run.stderr
 with targetpath.open('rb')as f:n,c=np.fromfile(f,np.uint64,2);target=np.fromfile(f,np.float64).reshape(-1,22,13)
 assert n==len(x)and c==286;fields=x[:,14:].reshape(-1,22,13);amp=4*abs(eps)/.04;scales=np.ones(22)if eps==0 else np.full(22,amp/length)
 if eps:scales[config]=amp
 difference=(fields-target)/scales[None,:,None];weights=reference_weights(x);masks={'all':np.ones(len(x),bool),'bulk':x[:,13]==0,'shell':x[:,13]==1};case={'time':t,'epsilon':eps,'normalization_amplitude':amp,'native_active_nodes':len(x),'native_exact_errors':{},'constraints':{},'raw_field_error_linf':float(np.max(abs(fields[:,:,0]-target[:,:,0]))),'native_snapshot_sha256':hashlib.sha256((path/'cubed_sphere_native_last_jets.bin').read_bytes()).hexdigest(),'exact_target_sha256':hashlib.sha256(targetpath.read_bytes()).hexdigest()}
 for measure,w in weights.items():
  case['native_exact_errors'][measure]={};case['constraints'][measure]={}
  for region,mask in masks.items():
   case['native_exact_errors'][measure][region]={name:norm(z,w,mask)for name,z in [('fields',difference[:,:,0]),('first',length*difference[:,:,1:4]),('configuration_second',length*length*difference[:,config,4:])]}
   constraints={name:norm(x[:,col],w,mask)for name,col in [('H_raw',5),('M_conformal',6),('M_physical',7),('Z_conformal',8),('Z_physical',9),('det_minus_1',10),('A_trace',11),('differential_det_first',12)]};constraints['Theta_physical']=norm(fields[:,17,0],w,mask)
   if eps:constraints['dimensionless']={name:{k:v/(amp/length**2 if name in ['H_raw','M_conformal','M_physical']else amp/length)for k,v in constraints[name].items()}for name in ['H_raw','M_conformal','M_physical','Z_conformal','Z_physical','Theta_physical']}
   case['constraints'][measure][region]=constraints
 primary=case['native_exact_errors']['fixed_reference_Penrose']['all'];cc=case['constraints']['fixed_reference_Penrose']['all'];case['short_screen']={'field_first_budget_pass':all(primary[k]['rms']<=.01and primary[k]['linf']<=.05for k in ['fields','first']),'configuration_second_diagnostic_budget_pass':primary['configuration_second']['rms']<=.01and primary['configuration_second']['linf']<=.05}
 if eps:case['short_screen']['constraint_budget_pass']=all(v['rms']<=.01and v['linf']<=.05for v in cc['dimensionless'].values())
 else:case['short_screen'].update({'stationary_field_pass':case['raw_field_error_linf']<=1e-9,'stationary_raw_constraint_pass':max(cc[name]['linf']for name in ['H_raw','M_conformal','M_physical','Z_conformal','Z_physical','Theta_physical'])<=1e-8})
 report['cases'][label]=case
(a.output/'native-oracle-pilot-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({case:{'time':r['time'],'errors':r['native_exact_errors']['fixed_reference_Penrose']['all'],'screens':r['short_screen']}for case,r in report['cases'].items()},indent=2))
