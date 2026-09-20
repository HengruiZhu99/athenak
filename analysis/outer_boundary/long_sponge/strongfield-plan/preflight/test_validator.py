"""Read-only positive checks and deliberate mutation of private checkpoint copies."""
from pathlib import Path
import copy,hashlib,json,os,shutil,struct,sys
import numpy as np
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.parent));from check_smr_trumpet_checkpoint import read_header,check_supported,validate
source=P/'zero';audit=P.parent/'mesh-audit.json';zero=validate(source,8,3,True,audit);assert zero['passed']
case=P/'invalid_fine_ghost';case.mkdir(exist_ok=False);record0=zero['files'][0];first=Path(record0['path']);h=read_header(first);ng,n,ns,lo,hi,shape,offset=check_supported(h)
blocks=json.loads(audit.read_text())['cases']['mesh16_r1']['blocks_geometry'];block=next(b for b in blocks if b['dx_M']==.125 and b['min'][0]==-4);gid=block['gid'];owner=next(f for f in zero['files']if f['gid_first']<=gid<=f['gid_last']);rank=owner['rank'];local=gid-owner['gid_first']
for f in zero['files']:
 src=Path(f['path']);dst=case/'rst'/f'rank_{f["rank"]:08d}'/src.name;dst.parent.mkdir(parents=True,exist_ok=True)
 if f['rank']==rank:shutil.copyfile(src,dst)
 else:os.link(src,dst)
mutated=case/'rst'/f'rank_{rank:08d}'/first.name;i,j,k=ng-1,ng+int(n[1])//2,ng+int(n[2])//2;index=((1*shape[0]+k)*shape[1]+j)*shape[2]+i # residual conformal gxx, one fine ghost
address=h['payload_start']+local*h['stride']+offset+8*index
with mutated.open('r+b')as stream:stream.seek(address);original=struct.unpack('<d',stream.read(8))[0];assert original==0;stream.seek(address);stream.write(struct.pack('<d',-2.0))
failed=validate(case,8,3,False,audit);assert not failed['passed'] and failed['all_payload_finite'] and failed['invalid_metric_cells_including_ghosts']==1
sample=failed['invalid_metric_samples'][0];expected=[block['min'][a]+(u-ng+.5)*.125 for a,u in enumerate((i,j,k))];assert sample['gid']==gid and sample['rank']==rank and sample['relative_level']==3 and sample['ghost_depth']==[1,0,0] and sample['xyz_M']==expected
assert sample['raw_full_values']['gxx']==-1 and sample['raw_full_values']['determinant']==-1
(case/'checkpoint-validity.json').write_text(json.dumps(failed,indent=2)+'\n')
# Exercise fail-closed scope guards without editing any source checkpoint.
guards={}
for name,section,key,value in [('AMR','mesh_refinement','refinement','adaptive'),('wrong_background','problem','bh_background','kerr_schild'),('wrong_mass','problem','bh_mass','2'),('wrong_chi_exponent','z4c','chi_psi_power','-2')]:
 bad=copy.deepcopy(h);bad['params'][section][key]=value
 try:check_supported(bad)
 except ValueError as exc:guards[name]={'rejected':True,'reason':str(exc)}
 else:raise AssertionError('Unsupported input accepted: '+name)
# Verify all original cohort files remain byte-for-byte intact after private mutation.
unchanged={str(f['rank']):hashlib.sha256(Path(f['path']).read_bytes()).hexdigest()==f['sha256']for f in zero['files']};assert all(unchanged.values())
result={'positive_zero_passed':True,'level_dependent_geometry_matches_mesh_audit':True,'indefinite_fine_ghost_rejected':True,'mutation':{'rank':rank,'gid':gid,'local_block':local,'relative_level':3,'array_ijk':[i,j,k],'expected_xyz_M':expected,'byte_offset':int(address),'field':'residual gxx','before':0,'after':-2,'full_gxx':-1,'checkpoint':str(mutated)},'reported_invalid_sample':sample,'scope_guards':guards,'source_cohort_unchanged':unchanged,'note':'Private malformed checkpoint is never supplied to AthenaK or used to continue evolution.'};(P/'validator-regression.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
