#!/usr/bin/env python3
"""Quantitative native algebraic-tangent clock pilot review with declared volume/field scales."""
from pathlib import Path
import argparse,json,csv,struct,hashlib,subprocess,time
import numpy as np
p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('output',type=Path);p.add_argument('cases',nargs='+');a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
config=[0,1,2,3,4,5,6,18,19,20,21];length=.2

def load(case):
 path=a.work/case
 with (path/'checkpoint.bin').open('rb')as f:head=struct.unpack('<16s64s23dQdq',f.read(288))
 assert head[0]==b'ATHENA_CSP_CK05!' and head[24]==1 and head[19]==1;time=head[-2];eps=head[20]
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
  oldroot=a.work.parent/'native-global-clock-v1';oldreview=json.loads((Path(__file__).resolve().parents[2]/'docs/validation/goal1-minkowski-20261018/native-global-clock-v1/native-oracle-pilot-review.json').read_text())
  for previous,info in oldreview['cases'].items():
   if info['time']!=t or info['epsilon']!=eps or info['native_active_nodes']!=len(x):continue
   oldsnapshot=oldroot/previous/'cubed_sphere_native_last_jets.bin';oldtarget=oldroot/('analysis-'+previous)/'target.bin'
   with oldsnapshot.open('rb')as f:nn,cc=np.fromfile(f,np.uint64,2);olddata=np.fromfile(f,np.float64).reshape(int(nn),int(cc))
   if not np.array_equal(x[:,:4],olddata[:,:4]):continue
   assert hashlib.sha256(oldtarget.read_bytes()).hexdigest()==info['exact_target_sha256'];targetpath.symlink_to(oldtarget.resolve());(analysis/'oracle-reuse.json').write_text(json.dumps({'source_case':previous,'source_target_sha256':info['exact_target_sha256'],'time':t,'epsilon':eps,'native_coordinates_exactly_equal':True,'new_oracle_evaluation':False},indent=2)+'\n');break
 if not targetpath.exists():
  with inp.open('wb')as f:f.write(struct.pack('<ddQ',t,eps,len(x)));x[:,:3].copy().tofile(f)
  cmd=[str((a.work/'clock_export').resolve()),str(inp.resolve()),str(targetpath.resolve())];run=subprocess.run(cmd,capture_output=True,text=True);assert run.returncode==0,run.stderr
 with targetpath.open('rb')as f:n,c=np.fromfile(f,np.uint64,2);target=np.fromfile(f,np.float64).reshape(-1,22,13)
 assert n==len(x)and c==286;fields=x[:,14:].reshape(-1,22,13);amp=4*abs(eps)/.04;scales=np.ones(22)if eps==0 else np.full(22,amp/length)
 if eps:scales[config]=amp
 difference=(fields-target)/scales[None,:,None];weights=reference_weights(x);bulk=x[:,13]==0;shell=~bulk;receipt=json.loads((path/'receipt.json').read_text());cmd=receipt['command'];Agrid=int(next(v.split('=')[1]for v in cmd if v.startswith('cubed_sphere/nangular=')));Nr=int(next(v.split('=')[1]for v in cmd if v.startswith('cubed_sphere/nr=')));ii=np.arange(shell.sum())%Nr;jj=np.arange(shell.sum())//Nr%Agrid;kk=np.arange(shell.sum())//(Nr*Agrid)%Agrid;ej=(jj<3)|(jj>=Agrid-3);ek=(kk<3)|(kk>=Agrid-3)
 def sm(z):
  out=np.zeros(len(x),bool);out[shell]=z;return out
 core=bulk&(np.max(abs(x[:,:3]),axis=1)<.105);masks={'all':np.ones(len(x),bool),'bulk':bulk,'shell':shell,'bulk_core':core,'bulk_interface':bulk&~core,'cube_shell_four_rows':sm(ii<4),'outer_four_rows':sm(ii>=Nr-4),'seam_corner':sm(ej&ek),'seam_edge':sm(ej^ek)};case={'time':t,'epsilon':eps,'normalization_amplitude':amp,'native_active_nodes':len(x),'native_exact_errors':{},'constraints':{},'raw_field_error_linf':float(np.max(abs(fields[:,:,0]-target[:,:,0]))),'native_snapshot_sha256':hashlib.sha256((path/'cubed_sphere_native_last_jets.bin').read_bytes()).hexdigest(),'exact_target_sha256':hashlib.sha256(targetpath.read_bytes()).hexdigest()}
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
 case['short_screen']['regional_RMS_pass']=all(all(v['rms']<=.01 for v in values.values())for values in case['native_exact_errors']['fixed_reference_Penrose'].values()) if eps else True
 if eps:case['short_screen']['regional_constraint_RMS_pass']=all(v['rms']<=.01 for region in case['constraints']['fixed_reference_Penrose'].values()for v in region['dimensionless'].values())
 report['cases'][label]=case
(a.output/'native-oracle-pilot-review.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({case:{'time':r['time'],'errors':r['native_exact_errors']['fixed_reference_Penrose']['all'],'screens':r['short_screen']}for case,r in report['cases'].items()},indent=2))
