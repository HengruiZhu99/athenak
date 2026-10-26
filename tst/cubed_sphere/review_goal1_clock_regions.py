#!/usr/bin/env python3
"""Saved native clock regional norms and error locations; no trajectory repeats."""
import argparse,json,csv,hashlib,ast,struct
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('output',type=Path);a=p.parse_args();root=Path(__file__).resolve().parents[2];work=a.case.parent
with (a.case/'checkpoint.bin').open('rb')as f:head=struct.unpack('<16s64s19dQdq',f.read(256))
assert head[0]==b'ATHENA_CSP_CK03!';time=head[-2];eps=head[20];assert eps!=0
with (a.case/'cubed_sphere_native_last_jets.bin').open('rb')as f:n,c=np.fromfile(f,np.uint64,2);x=np.fromfile(f,np.float64).reshape(int(n),int(c))
assert c==300;targetpath=work/('analysis-'+a.case.name)/'target.bin'
with targetpath.open('rb')as f:nt,ct=np.fromfile(f,np.uint64,2);target=np.fromfile(f,np.float64).reshape(-1,22,13)
assert n==nt and ct==286
utility=root/'tst/cubed_sphere/review_goal1_clock_pilots.py';tree=ast.parse(utility.read_text());functions=[f for f in tree.body if isinstance(f,ast.FunctionDef)and f.name in ['reference_weights','norm']];ns={'np':np};exec(compile(ast.Module(body=functions,type_ignores=[]),str(utility),'exec'),ns);weights=ns['reference_weights'](x);norm=ns['norm'];bulk=x[:,13]==0;shell=~bulk;B=round(np.count_nonzero(bulk)**(1/3));assert B**3==np.count_nonzero(bulk)
receipt=json.loads((a.case/'receipt.json').read_text());cmd=receipt['command'];value=lambda prefix:float(next(c.split('=')[1]for c in cmd if c.startswith(prefix+'=')));A=int(value('cubed_sphere/nangular'));Nr=int(value('cubed_sphere/nr'));cube=value('mesh/x1max');assert np.count_nonzero(shell)==6*A*A*Nr
r=np.sqrt(np.einsum('ij,ij->i',x[:,:3],x[:,:3]));ray=r[shell].reshape(6,A,A,Nr);direction=(x[shell,:3]/r[shell,None]).reshape(6,A,A,Nr,3);assert np.all(np.diff(ray,axis=3)>0)and np.max(abs(direction-direction[:,:,:,0:1,:]))<1e-12
indices=np.arange(np.count_nonzero(shell));ii=indices%Nr;jj=indices//Nr%A;kk=indices//(Nr*A)%A;edgej=(jj<3)|(jj>=A-3);edgek=(kk<3)|(kk>=A-3)
def shellmask(test):
 mask=np.zeros(len(x),bool);mask[shell]=test;return mask
core=bulk&(np.max(abs(x[:,:3]),axis=1)<.7*cube);masks={'all':np.ones(len(x),bool),'bulk':bulk,'shell':shell,'bulk_core':core,'bulk_interface':bulk&~core,'cube_shell_four_rows':shellmask(ii<4),'outer_four_rows':shellmask(ii>=Nr-4),'seam_corner':shellmask(edgej&edgek),'seam_edge':shellmask(edgej^edgek)}
# Match exact native subsets/cell counts and their original mapped volumes.
rows=list(csv.DictReader((a.case/'cubed_sphere_native_oracle_errors.csv').open()));last=[row for row in rows if abs(float(row['time'])-time)<1e-15]
for row in last:
 key=('bulk'if row['region']=='0'else'shell')if row['subset']=='all'else row['subset'];mask=masks[key];assert np.count_nonzero(mask)==int(row['cells']);assert abs(np.sum(x[mask,3])-float(row['volume']))<1e-12
config=[0,1,2,3,4,5,6,18,19,20,21];length=.2;amp=4*abs(eps)/.04;scale=np.full(22,amp/length);scale[config]=amp;fields=x[:,14:].reshape(-1,22,13);difference=(fields-target)/scale[None,:,None];quantities={'fields':difference[:,:,0],'first':length*difference[:,:,1:4],'configuration_second':length*length*difference[:,config,4:],'H_raw':x[:,5],'M_conformal':x[:,6],'M_physical':x[:,7],'Z_conformal':x[:,8],'Z_physical':x[:,9],'det_minus_one':x[:,10],'A_trace':x[:,11],'differential_det_first':x[:,12],'Theta_physical':fields[:,17,0]};constraints=['H_raw','M_conformal','M_physical','Z_conformal','Z_physical','Theta_physical'];out={'scope':'Full saved native clock field/first/Hessian and raw/normalized constraints, exact native subsets independently reconstructed and checked against native counts/volumes. No extra trajectory.','time':time,'case':a.case.name,'normalization_amplitude':amp,'length':length,'regions':{}}
for name,mask in masks.items():
 entry={'cells':int(np.count_nonzero(mask)),'norms':{},'maxima_locations':{}}
 for measure,w in weights.items():
  norms={k:norm(z,w,mask)for k,z in quantities.items()};norms['dimensionless_constraints']={k:{metric:v/(amp/length**2 if k in ['H_raw','M_conformal','M_physical']else amp/length)for metric,v in norms[k].items()}for k in constraints};entry['norms'][measure]=norms
 for k,z in quantities.items():
  point=np.sum(z*z,axis=tuple(range(1,z.ndim)))if z.ndim>1 else z*z;index=int(np.argmax(np.where(mask,point,-1)));entry['maxima_locations'][k]={'node':index,'xyz':x[index,:3].tolist(),'radius':float(r[index]),'magnitude':float(np.sqrt(point[index]))}
 out['regions'][name]=entry
out['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();out['reused_norm_source_sha256']=hashlib.sha256(utility.read_bytes()).hexdigest();out['snapshot_sha256']=hashlib.sha256((a.case/'cubed_sphere_native_last_jets.bin').read_bytes()).hexdigest();a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'case':a.case.name,'time':time,'regions':len(masks),'max_regional_reference_RMS':{k:max(v['norms']['fixed_reference_Penrose'][k]['rms']for v in out['regions'].values())for k in ['fields','first','configuration_second']},'max_regional_dimensionless_constraint_RMS':max(v['rms']for r in out['regions'].values()for v in r['norms']['fixed_reference_Penrose']['dimensionless_constraints'].values())}))
