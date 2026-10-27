#!/usr/bin/env python3
"""Independent static native core wormhole restart, ADM and frozen-Mink audit.
Read-only analysis except its new analysis.json and final manifest annotations.
The original rounded-input ADM proxy failure is retained, not relaxed.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np
from analyze_native_composite import native_bin

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def checksum(data):
 value=1469598103934665603
 for byte in data:value=((value^byte)*1099511628211)&((1<<64)-1)
 return value

def sidecar(path):
 data=path.read_bytes();assert checksum(data[:-8])==struct.unpack('=Q',data[-8:])[0];offset=0
 def pod(fmt):
  nonlocal offset
  value=struct.unpack_from('='+fmt,data,offset)[0];offset+=struct.calcsize('='+fmt);return value
 def text():
  nonlocal offset
  n=pod('Q');v=data[offset:offset+n].decode();offset+=n;return v
 def vector():
  nonlocal offset
  n=pod('Q');v=np.frombuffer(data,dtype='=f8',count=n,offset=offset).copy();offset+=n*8;return v
 def group():return {name:vector() for name in ['volume','scri','endpoints']}
 result={'magic':text(),'tag':text(),'n':pod('i'),'ng':pod('i'),'half':pod('d'),'radii':vector()}
 assert result['magic'] in ['ATHENAK_NATIVE_COMPOSITE_WORMHOLE_V1','ATHENAK_NATIVE_COMPOSITE_WORMHOLE_STATIC_CORE_V1']
 result['core_bulk']=1;result['native_count']=1;result['topology']=[{'native':0,'composite':0,'refinement':0,'lo':[-result['half']]*3,'hi':[result['half']]*3}]
 if result['magic'].endswith('STATIC_CORE_V1'):
  result['core_bulk']=pod('i');result['native_count']=pod('i');count=pod('Q');result['topology']=[]
  for _ in range(count):
   native,composite,refinement=pod('i'),pod('i'),pod('i');bounds=[pod('d') for _ in range(6)]
   result['topology'].append({'native':native,'composite':composite,'refinement':refinement,'lo':bounds[::2],'hi':bounds[1::2]})
  assert result['core_bulk']==2 and result['native_count']==count==8
 for name in ['slice','gauge','initial_receipt']:result[name]=text()
 result.update(positive_mode=pod('i'),taper=pod('d'),tolerance=pod('d'),dt_ceiling=pod('d'),version=pod('I'),values=group(),registers=group(),chart=text(),chart_parameters=vector())
 for name in ['mesh_revision','chart_revision','puncture_revision']:result[name]=pod('Q')
 result['integrator']=text();result['stages']=pod('i');result['rk']={name:np.array([pod('d') for _ in range(4)]) for name in ['gam0','gam1','beta','delta']};result['shift']=vector()
 for name in ['cycle','completed_cycles','geometry']:result[name]=pod('Q')
 for name in ['next_stage','open_stage']:result[name]=pod('i')
 for name in ['active','stage_open','has_rk']:result[name]=pod('?')
 for name in ['time','dt','abscissa','register_abscissa']:result[name]=pod('d')
 assert offset==len(data)-8;return result

def native_bins(path):
 records=[]
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
  ng=int(parameters[('mesh','nghost')]);active=[int(parameters[('meshblock','nx'+str(a+1))]) for a in range(3)]
  while True:
   first=stream.read(40)
   if not first:break
   assert len(first)==40;indices=np.frombuffer(first,dtype='=i4').copy();bounds=np.frombuffer(stream.read(48),dtype='=f8').copy()
   nx,ny,nz=[int(indices[2*a+1]-indices[2*a]+1) for a in range(3)];count=len(names)*nx*ny*nz
   raw=stream.read(count*4);assert len(raw)==count*4;values=np.frombuffer(raw,dtype='=f4').reshape(len(names),nz,ny,nx).astype(float)
   axes=[bounds[2*a]+(indices[2*a]+np.arange(n)-ng+.5)*((bounds[2*a+1]-bounds[2*a])/active[a]) for a,n in enumerate([nx,ny,nz])]
   zz,yy,xx=np.meshgrid(axes[2],axes[1],axes[0],indexing='ij')
   records.append({'values':dict(zip(names,values)),'xyz':(xx,yy,zz),'bounds':bounds,'logical':indices[6:9].tolist(),'relative_level':int(indices[9])})
 return records

def physical_adm_check(folder,prefix='native_wormhole'):
 zbin=max((folder/'bin').glob('*.z4c.*.bin'));number=zbin.name.rsplit('.',2)[1]
 qrecords=native_bins(zbin);arecords=native_bins(folder/'bin'/(prefix+'.adm.'+number+'.bin'))
 rst=folder/'rst'/(prefix+'.'+number+'.rst');layout=sidecar(Path(str(rst)+'.composite'));native_count=layout['native_count']
 assert len(qrecords)==len(arecords)==native_count
 shape=next(iter(qrecords[0]['values'].values())).shape;count=native_count*25*np.prod(shape)
 values=np.frombuffer(rst.read_bytes()[-count*8:],dtype='=f8').reshape((native_count,25,*shape))
 all_matches={};all_relative={};all_quantized={};blocks=[]
 for record,adm_record in zip(qrecords,arecords):
  bounds=record['bounds'];assert np.array_equal(bounds,adm_record['bounds']) and record['relative_level']==0
  candidates=[r for r in layout['topology'] if np.array_equal(bounds[::2],r['lo']) and np.array_equal(bounds[1::2],r['hi'])]
  assert len(candidates)==1;slot=candidates[0]['native'];q32=record['values'];adm=adm_record['values'];xyz=record['xyz'];assert len(q32)==25
  q64=dict(zip(q32,values[slot]));matches={name:np.array_equal(q32[name],v.astype('f4').astype(float)) for name,v in q64.items()};assert all(matches.values())
  xx,yy,zz=xyz;omega=(1-xx*xx-yy*yy-zz*zz)/2;psi4=1/(omega**2*q64['z4c_chi']);expected={'adm_psi4':psi4,'adm_alpha':q64['z4c_alpha']/omega}
  for component in ['xx','xy','xz','yy','yz','zz']:
   expected['adm_g'+component]=psi4*q64['z4c_g'+component]
   expected['adm_K'+component]=psi4*(omega*q64['z4c_A'+component]+q64['z4c_g'+component]*(q64['z4c_Khat']+2*q64['z4c_Theta'])/3)
  for c in ['x','y','z']:expected['adm_beta'+c]=q64['z4c_beta'+c]
  relative={name:float(np.max(abs(adm[name]-v)/np.maximum(1,abs(v)))) for name,v in expected.items()}
  quantized={name:np.array_equal(adm[name],v.astype('f4').astype(float)) for name,v in expected.items()}
  for name,value in matches.items():all_matches[name]=all_matches.get(name,True) and value
  for name,value in relative.items():all_relative[name]=max(all_relative.get(name,0),value)
  for name,value in quantized.items():all_quantized[name]=all_quantized.get(name,True) and value
  blocks.append({'native':slot,'composite':candidates[0]['composite'],'bounds':bounds.tolist(),'maximum_relative':max(relative.values()),'quantized_all_identical':all(quantized.values())})
 return {'reference':'actual fullprecision nativeRST pack/ghosts matched by physical bounds, independently mapped Penrose-to-ADM, then quantized32','native_count':native_count,'blocks':blocks,'native_RST_fields_verified_against_BIN32':all_matches,'relative_output_rounding':all_relative,'original_tolerance':3e-7,'within_original_tolerance':max(all_relative.values())<3e-7,'quantized_outputs_bitwise_identical':all_quantized,'passed':all(all_quantized.values()) and max(all_relative.values())<3e-7,'native_RST_sha256':sha(rst),'z4c_BIN_sha256':sha(zbin),'all_native_core_and_ghost_cells':True}

def native_active_check(folder,prefix='native_wormhole'):
 records=native_bins(folder/'bin'/(prefix+'.z4c.00002.bin'))
 rst=folder/'rst'/(prefix+'.00002.rst');layout=sidecar(Path(str(rst)+'.composite'));n=layout['n'];ng=layout['ng'];padded=n+2*ng
 count=layout['native_count']*25*padded**3
 raw=np.frombuffer(rst.read_bytes()[-count*8:],dtype='=f8').reshape(layout['native_count'],25,padded,padded,padded)
 globalq=np.genfromtxt(folder/(prefix+'.00002.csv'),delimiter=',',names=True);checks=[]
 for mapping in layout['topology']:
  slot=mapping['native'];composite=mapping['composite'];g=globalq[composite*n**3:(composite+1)*n**3]
  record=next(r for r in records if np.array_equal(r['bounds'][::2],mapping['lo']) and np.array_equal(r['bounds'][1::2],mapping['hi']))
  native=dict(zip(record['values'],raw[slot,:,ng:ng+n,ng:ng+n,ng:ng+n].reshape(25,-1)))
  expected={'z4c_chi':g['chi'],'z4c_alpha':g['alpha'],'z4c_Khat':g['state_13']-3,'z4c_Theta':g['state_14']}
  for i,c in enumerate(['xx','xy','xz','yy','yz','zz']):
   expected['z4c_g'+c]=g[f'state_{1+i}']+(1 if c in ['xx','yy','zz'] else 0)
   expected['z4c_A'+c]=g[f'state_{7+i}']
  for i,c in enumerate(['x','y','z']):
   expected['z4c_Gam'+c]=g[f'state_{15+i}'];expected['z4c_beta'+c]=g['beta_'+c];expected['z4c_B'+c]=g[f'state_{19+i}']
  assert set(expected)==set(native)
  identical={f:np.array_equal(native[f],value) for f,value in expected.items()};errors={f:float(np.max(abs(native[f]-value))) for f,value in expected.items()};assert all(identical.values())
  checks.append({'native':slot,'composite':composite,'cells':n**3,'physical_fields':identical,'maximum_error':max(errors.values())})
 return {'reference':'All25 native physical RST active fields versus independently decoded global canonical CSV including Cartesian shift, matched by stored physical topology','blocks':checks,'cells':sum(r['cells'] for r in checks),'all_bitwise_identical':all(r['maximum_error']==0 for r in checks)}

def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--minkowski-baseline',type=Path,default=Path('docs/elliptic-shift-3d/evidence/native-composite-cfl-controls-20261027'));a=p.parse_args();root=a.root
 m=json.loads((root/'manifest.json').read_text());out={'scope':'Actual Serial Driver/Task selected wormhole mechanical qualification; static native Cartesian core plus host-owned shells; scientific gates remain failed','scientific_acceptance':False}
 out['retained_rounded_input_ADM_proxy']=m.get('physical_ADM_all_native_core_and_ghosts');out['actual_ADM']=physical_adm_check(root/'full');assert out['actual_ADM']['passed'];out['native_active_mapping']=native_active_check(root/'full')
 full=sidecar(root/'full/rst/native_wormhole.00002.rst.composite');again=sidecar(root/'restart/rst/native_wormhole.00002.rst.composite')
 groups={g:{f:np.array_equal(full[g][f],again[g][f]) for f in full[g]} for g in ['values','registers']}
 vectors={f:np.array_equal(full[f],again[f]) for f in ['radii','chart_parameters','shift']};rk={f:np.array_equal(full['rk'][f],again['rk'][f]) for f in full['rk']}
 assert all(all(row.values()) for row in groups.values()) and all(vectors.values()) and all(rk.values())
 hidden=root/'initial.hidden-after-fresh-run.dat';assert hidden.exists() and not (root/'initial.dat').exists()
 receipt='fnv1a64:'+format(checksum(hidden.read_bytes()),'x')+':'+str(hidden.stat().st_size)
 assert full['initial_receipt']==again['initial_receipt']==receipt
 assert full['positive_mode']==again['positive_mode']==2 and full['taper']==again['taper']==.24
 assert full['chart']==again['chart']==('wormhole-positive-log22-v1-static-core' if full['core_bulk']>1 else 'wormhole-positive-log22-v1') and full['time']==again['time']==.0002
 assert full['topology']==again['topology'] and full['native_count']==again['native_count']
 header=(root/'full/rst/native_wormhole.00002.rst').read_bytes().split(b'<par_end>',1)[0].decode()
 assert receipt in header
 out['restart']={'groups_bitwise_identical':groups,'vectors_bitwise_identical':vectors,'actual_Driver_RK_bitwise':rk,'actual_Driver_RK_arrays':{k:v.tolist() for k,v in full['rk'].items()},'initial_file_hidden':True,'receipt_persisted_in_native_input_and_sidecar':receipt,'slice':full['slice'],'gauge':full['gauge'],'chart':full['chart'],'chart_parameters':full['chart_parameters'].tolist(),'time':full['time'],'native_core_blocks':full['native_count'],'core_topology':full['topology'],'geometry_epoch_full':full['geometry'],'geometry_epoch_restart':again['geometry']}
 out['matched_actual_Driver_RK']=m['matched_actual_RK_comparison'];assert all(r['bitwise_all_fields'] for r in out['matched_actual_Driver_RK'].values())
 expected={'reject-receipt-change':'slice/gauge/receipt/chart controls mismatch','reject-taper-change':'verified taper .24','reject-chart-change':'positive log mode2','reject-minkowski-reinterpretation':('restart config/tag mismatch' if full['core_bulk']>1 else 'coarse solve exceeds 1024 cells'),'reject-minkowski-valid-geometry':'restart config/tag mismatch'}
 out['negative_controls']={}
 for name,diagnostic in expected.items():
  text=(root/(name+'.stdout')).read_text()+(root/(name+'.stderr')).read_text();assert diagnostic in text;out['negative_controls'][name]={'diagnostic':diagnostic,'passed':True}
 baseline=a.minkowski_baseline;current=root/'minkowski-controls';checks={}
 for name in ['two-shell','one-shell','radial-pulse','angular-pulse','extended-minkowski']:
  for old in sorted((baseline/name).glob('*.csv')):
   new=current/name/old.name;checks[str(old.relative_to(baseline))]={'bitwise_identical':new.exists() and old.read_bytes()==new.read_bytes(),'old_sha256':sha(old),'new_sha256':sha(new) if new.exists() else None}
 out['old_Minkowski_regression']={'compared':len(checks),'all_identical':len(checks)==34 and all(c['bitwise_identical'] for c in checks.values()),'files':checks};assert out['old_Minkowski_regression']['all_identical']
 trace=m['native_trace'];out['execution']={'steps':2,'time':trace[-1]['time'],'cells':trace[-1]['cells'],'native_updates':trace[-1]['native_update_calls'],'all_snapshots_restart_bitwise':all(r['bitwise_identical'] for r in m['accepted_restart'].values())}
 out['scientific_gates']={'acceptance':False,'final_diagnostics':trace[-1],'tensor_pole_target':1e-7,'tensor_pole_passed':trace[-1]['scri_tensor_pole']<=1e-7,'clock_target':1e-13,'null_target':1e-12,'minimum_cell_radius':m['initializer']['minimum_cell_radius'],'throat_radius':m['initializer']['throat_radius'],'throat_unresolved':m['initializer']['minimum_cell_radius']>m['initializer']['throat_radius'],'reason':'Coarse static core pilot with nonanalytic remainder underresolved even after log enrichment; H/M/poles and complete radial-collapse scientific milestone fail. Shell resolution is unchanged in the eight-core control.'}
 if full['core_bulk']>1:
  out['old_onecore_wormhole_regression']=m['default_onecore_wormhole_bitwise_comparison'];assert out['old_onecore_wormhole_regression']['all_identical']
 out['binary_unchanged']=sha(Path('build-shift-3d/src/athena'))==m['binary_sha256'];assert out['binary_unchanged']
 out['source_hashes']={'analyzer':sha(Path(__file__)),'native_owner':sha(Path('src/z4c/hyperboloidal/native_composite_evolution.cpp'))}
 (root/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
 m['analysis_file']='analysis.json';m['mechanical_execution_qualified']=True;m['scientific_acceptance']=False;m['binary_unchanged']=out['binary_unchanged'];(root/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 print(json.dumps({'execution':out['execution'],'fullprecision_ADM_quantized_bitwise':all(out['actual_ADM']['quantized_outputs_bitwise_identical'].values()),'old_Minkowski34bitwise':out['old_Minkowski_regression']['all_identical'],'scientific_acceptance':False}))

if __name__=='__main__':main()
