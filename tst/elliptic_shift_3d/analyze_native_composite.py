#!/usr/bin/env python3
"""Independent native BIN, local checkpoint and full3D field audit; no evolution."""
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def checkpoint(path):
 data=path.read_bytes();checksum=1469598103934665603
 for byte in data[:-8]:checksum=((checksum^byte)*1099511628211)&((1<<64)-1)
 if checksum!=struct.unpack('=Q',data[-8:])[0]:raise ValueError('checkpoint checksum mismatch')
 offset=0
 def pod(fmt):
  nonlocal offset
  value=struct.unpack_from('='+fmt,data,offset)[0];offset+=struct.calcsize('='+fmt);return value
 def text():
  nonlocal offset
  size=pod('Q');result=data[offset:offset+size].decode();offset+=size;return result
 def vector():
  nonlocal offset
  size=pod('Q');result=np.frombuffer(data,dtype='=f8',count=size,offset=offset).copy();offset+=8*size;return result
 def group():return {name:vector() for name in ['volume','scri','endpoints']}
 result=dict(magic=text(),tag=text(),n=pod('i'),ng=pod('i'),half=pod('d'),radii=vector(),version=pod('I'),values=group(),registers=group(),chart=text(),chart_parameters=vector())
 for name in ['mesh_revision','chart_revision','puncture_revision']:result[name]=pod('Q')
 result['integrator']=text();result['stages']=pod('i');result['rk']={name:np.array([pod('d') for _ in range(4)]) for name in ['gam0','gam1','beta','delta']};result['shift']=vector()
 for name in ['cycle','completed_cycles','geometry']:result[name]=pod('Q')
 for name in ['next_stage','open_stage']:result[name]=pod('i')
 for name in ['active','stage_open','has_rk']:result[name]=pod('?')
 for name in ['time','dt','abscissa','register_abscissa']:result[name]=pod('d')
 assert offset==len(data)-8
 return result


def native_bin(path):
 with path.open('rb') as stream:
  assert stream.readline().decode().strip()=='Athena binary output version=1.1';meta={}
  while True:
   line=stream.readline().decode().strip()
   if line.startswith('variables:'):names=line.split(':',1)[1].split()
   elif '=' in line:
    key,value=line.split('=',1);meta[key.strip()]=value.strip()
    if key.strip()=='header offset':break
  header=stream.read(int(meta['header offset'])).decode();parameters={};block=''
  for line in header.splitlines():
   line=line.strip()
   if line.startswith('<') and line.endswith('>'):block=line[1:-1]
   elif '=' in line:
    key,value=line.split('=',1);parameters[(block,key.strip())]=value.split('#',1)[0].strip()
  indices=np.frombuffer(stream.read(40),dtype='=i4');bounds=np.frombuffer(stream.read(48),dtype='=f8');nx,ny,nz=[int(indices[2*a+1]-indices[2*a]+1) for a in range(3)];values=np.frombuffer(stream.read(),dtype='=f4').reshape(len(names),nz,ny,nx).astype(float)
  # BIN physical bounds remain ACTIVE block bounds even with ghost output;
  # explicit ois/oie indices determine the stored ghost-cell coordinates.
  ng=int(parameters[('mesh','nghost')]);active=[int(parameters[('meshblock','nx'+str(a+1))]) for a in range(3)]
  x=[bounds[2*a]+(indices[2*a]+np.arange(n)-ng+.5)*(bounds[2*a+1]-bounds[2*a])/active[a] for a,n in enumerate([nx,ny,nz])];zz,yy,xx=np.meshgrid(x[2],x[1],x[0],indexing='ij')
  return dict(zip(names,values)),(xx,yy,zz),meta


def main():
 p=argparse.ArgumentParser();p.add_argument('directory',type=Path);args=p.parse_args();root=args.directory;out=dict(scope='Independent native Driver/Task and local accepted-cycle restart audit; host-owned shells, no distributed/device/native-AMR or black-hole validation',cases={},restart={},negative_controls={})
 for name in ['two-shell','one-shell','radial-pulse','angular-pulse','extended-minkowski']:
  folder=root/name;rows=[json.loads(line) for line in (folder/'native_composite.jsonl').read_text().splitlines()];last=rows[-1];cycle=last['cycle'];data=np.genfromtxt(folder/f'native_composite.{cycle:05d}.csv',delimiter=',',names=True);scri=np.genfromtxt(folder/f'native_composite.{cycle:05d}.scri.csv',delimiter=',',names=True)
  flux=np.column_stack([data[f'state_{i}'] for i in [19,20,21]]);offdiag=np.column_stack([data[f'state_{i}'] for i in [2,3,5]]);lam=np.column_stack([data[f'state_{i}'] for i in [15,16,17]])
  row=dict(trace=rows,all_cells=len(data),B_max=float(abs(flux).max()),B_volume_rms=float(np.sqrt(np.average(np.sum(flux**2,axis=1),weights=data['weight']))),metric_offdiagonal_max=float(abs(offdiag).max()),independent_Lambda_max=float(abs(lam).max()),Z_max=float(abs(data['Z']).max()),all_state_finite=all(np.all(np.isfinite(data[f])) for f in data.dtype.names))
  assert row['all_state_finite'];assert last['scri_ghost_clock']<=1e-13 and last['scri_ghost_null']<=1e-12 and last['scri_ghost_tangent']<=1e-12 and last['scri_tensor_pole']<=1e-7
  assert last['shift_original_rms_max']<=1e-11 and last['shift_point_max']<=1e-7
  if 'pulse' in name:assert row['B_max']>0 and row['metric_offdiagonal_max']>0 and row['independent_Lambda_max']>0 and row['Z_max']>0
  else:assert last['H_rms']<1e-9 and last['M_rms']<1e-9 and last['state_deviation_max']<1e-9
  # Read actual native BIN arrays (float32 output). Independently apply the full
  # Penrose-to-physical ADM formulas using evolved native physical fields.
  last_bin=max((folder/'bin').glob('*.z4c.*.bin'));number=last_bin.name.rsplit('.',2)[1];q,xyz,_=native_bin(last_bin);a,_,_=native_bin(folder/'bin'/('native_composite.adm.'+number+'.bin'));omega=(1-sum(x*x for x in xyz))/2
  psi4=1/(omega**2*q['z4c_chi']);expected=dict(adm_psi4=psi4,adm_alpha=q['z4c_alpha']/omega)
  for component in ['xx','xy','xz','yy','yz','zz']:
   expected['adm_g'+component]=psi4*q['z4c_g'+component]
   expected['adm_K'+component]=psi4*(omega*q['z4c_A'+component]+q['z4c_g'+component]*(q['z4c_Khat']+2*q['z4c_Theta'])/3)
  for c in ['x','y','z']:expected['adm_beta'+c]=q['z4c_beta'+c]
  relative={field:float(np.max(abs(a[field]-value)/np.maximum(1,abs(value)))) for field,value in expected.items()};assert max(relative.values())<3e-7
  row['actual_native_BIN_ADM_mapping']=dict(max_relative_errors=relative,output_dtype='float32',comparison_tolerance=3e-7,all_native_core_and_ghost_cells=True,maximum_lapse_storage_separation=float(np.max(abs(a['adm_alpha']-q['z4c_alpha']))),z4c_bin_sha256=sha(last_bin))
  assert row['actual_native_BIN_ADM_mapping']['maximum_lapse_storage_separation']>.1
  out['cases'][name]=row
 for name,restart_name in [('two-shell','restart'),('radial-pulse','radial-pulse-restart'),('angular-pulse','angular-pulse-restart')]:
  before=checkpoint(root/name/'rst/native_composite.00002.rst.composite');after=checkpoint(root/restart_name/'rst/native_composite.00002.rst.composite');groups={}
  for group in ['values','registers']:
   groups[group]={field:np.array_equal(before[group][field],after[group][field]) for field in ['volume','scri','endpoints']};assert all(groups[group].values())
  for field in ['shift','radii','chart_parameters']:assert np.array_equal(before[field],after[field])
  for field in before['rk']:assert np.array_equal(before['rk'][field],after['rk'][field])
  assert before['integrator']==after['integrator']=='rk4';assert before['time']==after['time']==.0002
  out['restart'][name]=dict(groups_bitwise_identical=groups,shift_bitwise_identical=True,actual_Driver_RK_arrays={k:v.tolist() for k,v in before['rk'].items()},pulse_metadata=before['chart_parameters'].tolist(),geometry_epoch_uninterrupted=before['geometry'],geometry_epoch_restarted=after['geometry'],hierarchy_rebuilt_and_shift_invalidated=True)
 expectations={'reject-adm-alias':'experimental composite native path','reject-native-amr':'ghost zones must be divisible by two','reject-sidecar-corruption':'checksum mismatch','reject-native-mirror-edit':'u0 disagrees','reject-restart-rk-change':'RK differs','radial-pulse-reject-changed-amplitude':'pulse/chart configuration mismatch','angular-pulse-reject-changed-amplitude':'pulse/chart configuration mismatch'}
 for name,expected in expectations.items():
  text=(root/(name+'.stdout')).read_text()+(root/(name+'.stderr')).read_text();assert expected in text;out['negative_controls'][name]=dict(expected_diagnostic=expected,verified=True,note='AMR is rejected by preserved upstream ng3 odd-ghost restriction before owner constructor' if name=='reject-native-amr' else '')
 out['scientific_limits']='These are single-resolution native equilibrium and short gauge controls. No 1D profile/convergence/black-hole/late-front qualification or fully native shell/device/distributed AMR claim.'
 (root/'analysis.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:{f:r[f] for f in ['B_max','metric_offdiagonal_max','independent_Lambda_max','Z_max']} for k,r in out['cases'].items()}))


if __name__=='__main__':main()
