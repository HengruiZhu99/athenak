#!/usr/bin/env python3
"""Independent one-native-block wormhole restart, ADM and frozen-Mink audit.
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
 assert result['magic']=='ATHENAK_NATIVE_COMPOSITE_WORMHOLE_V1'
 for name in ['slice','gauge','initial_receipt']:result[name]=text()
 result.update(positive_mode=pod('i'),taper=pod('d'),tolerance=pod('d'),dt_ceiling=pod('d'),version=pod('I'),values=group(),registers=group(),chart=text(),chart_parameters=vector())
 for name in ['mesh_revision','chart_revision','puncture_revision']:result[name]=pod('Q')
 result['integrator']=text();result['stages']=pod('i');result['rk']={name:np.array([pod('d') for _ in range(4)]) for name in ['gam0','gam1','beta','delta']};result['shift']=vector()
 for name in ['cycle','completed_cycles','geometry']:result[name]=pod('Q')
 for name in ['next_stage','open_stage']:result[name]=pod('i')
 for name in ['active','stage_open','has_rk']:result[name]=pod('?')
 for name in ['time','dt','abscissa','register_abscissa']:result[name]=pod('d')
 assert offset==len(data)-8;return result

def physical_adm_check(folder,prefix='native_wormhole'):
 zbin=max((folder/'bin').glob('*.z4c.*.bin'));number=zbin.name.rsplit('.',2)[1]
 q32,xyz,_=native_bin(zbin);adm,_,_=native_bin(folder/'bin'/(prefix+'.adm.'+number+'.bin'))
 shape=next(iter(q32.values())).shape;rst=folder/'rst'/(prefix+'.'+number+'.rst')
 assert len(q32)==25
 # The qualified owner permits exactly one native core block. Its actual native
 # u0 double payload is the final25*shape doubles (matter/tracking are disabled).
 # Verify every field's float32 image against BIN before trusting this layout.
 values=np.frombuffer(rst.read_bytes()[-25*np.prod(shape)*8:],dtype='=f8').reshape((25,*shape))
 q64=dict(zip(q32,values));matches={name:np.array_equal(q32[name],v.astype('f4').astype(float)) for name,v in q64.items()};assert all(matches.values())
 omega=(1-sum(x*x for x in xyz))/2;psi4=1/(omega**2*q64['z4c_chi']);expected={'adm_psi4':psi4,'adm_alpha':q64['z4c_alpha']/omega}
 for component in ['xx','xy','xz','yy','yz','zz']:
  expected['adm_g'+component]=psi4*q64['z4c_g'+component]
  expected['adm_K'+component]=psi4*(omega*q64['z4c_A'+component]+q64['z4c_g'+component]*(q64['z4c_Khat']+2*q64['z4c_Theta'])/3)
 for c in ['x','y','z']:expected['adm_beta'+c]=q64['z4c_beta'+c]
 relative={name:float(np.max(abs(adm[name]-v)/np.maximum(1,abs(v)))) for name,v in expected.items()}
 quantized={name:np.array_equal(adm[name],v.astype('f4').astype(float)) for name,v in expected.items()}
 return {'reference':'actual fullprecision nativeRST core+ghosts, independently mapped Penrose-to-ADM, then quantized32','native_RST_fields_verified_against_BIN32':matches,'relative_output_rounding':relative,'original_tolerance':3e-7,'within_original_tolerance':max(relative.values())<3e-7,'quantized_outputs_bitwise_identical':quantized,'passed':all(quantized.values()) and max(relative.values())<3e-7,'native_RST_sha256':sha(rst),'z4c_BIN_sha256':sha(zbin),'all_native_core_and_ghost_cells':True}

def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('--minkowski-baseline',type=Path,default=Path('docs/elliptic-shift-3d/evidence/native-composite-cfl-controls-20261027'));a=p.parse_args();root=a.root
 m=json.loads((root/'manifest.json').read_text());out={'scope':'Actual Serial Driver/Task selected wormhole mechanical qualification; one native Cartesian block plus host-owned shells; scientific gates remain failed','scientific_acceptance':False}
 out['retained_rounded_input_ADM_proxy']=m.get('physical_ADM_all_native_core_and_ghosts');out['actual_ADM']=physical_adm_check(root/'full');assert out['actual_ADM']['passed']
 full=sidecar(root/'full/rst/native_wormhole.00002.rst.composite');again=sidecar(root/'restart/rst/native_wormhole.00002.rst.composite')
 groups={g:{f:np.array_equal(full[g][f],again[g][f]) for f in full[g]} for g in ['values','registers']}
 vectors={f:np.array_equal(full[f],again[f]) for f in ['radii','chart_parameters','shift']};rk={f:np.array_equal(full['rk'][f],again['rk'][f]) for f in full['rk']}
 assert all(all(row.values()) for row in groups.values()) and all(vectors.values()) and all(rk.values())
 hidden=root/'initial.hidden-after-fresh-run.dat';assert hidden.exists() and not (root/'initial.dat').exists()
 receipt='fnv1a64:'+format(checksum(hidden.read_bytes()),'x')+':'+str(hidden.stat().st_size)
 assert full['initial_receipt']==again['initial_receipt']==receipt
 assert full['positive_mode']==again['positive_mode']==2 and full['taper']==again['taper']==.24
 assert full['chart']==again['chart']=='wormhole-positive-log22-v1' and full['time']==again['time']==.0002
 header=(root/'full/rst/native_wormhole.00002.rst').read_bytes().split(b'<par_end>',1)[0].decode()
 assert receipt in header
 out['restart']={'groups_bitwise_identical':groups,'vectors_bitwise_identical':vectors,'actual_Driver_RK_bitwise':rk,'actual_Driver_RK_arrays':{k:v.tolist() for k,v in full['rk'].items()},'initial_file_hidden':True,'receipt_persisted_in_native_input_and_sidecar':receipt,'slice':full['slice'],'gauge':full['gauge'],'chart':full['chart'],'chart_parameters':full['chart_parameters'].tolist(),'time':full['time'],'geometry_epoch_full':full['geometry'],'geometry_epoch_restart':again['geometry']}
 out['matched_actual_Driver_RK']=m['matched_actual_RK_comparison'];assert all(r['bitwise_all_fields'] for r in out['matched_actual_Driver_RK'].values())
 expected={'reject-receipt-change':'slice/gauge/receipt/chart controls mismatch','reject-taper-change':'verified taper .24','reject-chart-change':'positive log mode2','reject-minkowski-reinterpretation':'coarse solve exceeds 1024 cells','reject-minkowski-valid-geometry':'restart config/tag mismatch'}
 out['negative_controls']={}
 for name,diagnostic in expected.items():
  text=(root/(name+'.stdout')).read_text()+(root/(name+'.stderr')).read_text();assert diagnostic in text;out['negative_controls'][name]={'diagnostic':diagnostic,'passed':True}
 baseline=a.minkowski_baseline;current=root/'minkowski-controls';checks={}
 for name in ['two-shell','one-shell','radial-pulse','angular-pulse','extended-minkowski']:
  for old in sorted((baseline/name).glob('*.csv')):
   new=current/name/old.name;checks[str(old.relative_to(baseline))]={'bitwise_identical':new.exists() and old.read_bytes()==new.read_bytes(),'old_sha256':sha(old),'new_sha256':sha(new) if new.exists() else None}
 out['old_Minkowski_regression']={'compared':len(checks),'all_identical':len(checks)==34 and all(c['bitwise_identical'] for c in checks.values()),'files':checks};assert out['old_Minkowski_regression']['all_identical']
 trace=m['native_trace'];out['execution']={'steps':2,'time':trace[-1]['time'],'cells':trace[-1]['cells'],'native_updates':trace[-1]['native_update_calls'],'all_snapshots_restart_bitwise':all(r['bitwise_identical'] for r in m['accepted_restart'].values())}
 out['scientific_gates']={'acceptance':False,'final_diagnostics':trace[-1],'tensor_pole_target':1e-7,'tensor_pole_passed':trace[-1]['scri_tensor_pole']<=1e-7,'clock_target':1e-13,'null_target':1e-12,'minimum_cell_radius':m['initializer']['minimum_cell_radius'],'throat_radius':m['initializer']['throat_radius'],'throat_unresolved':m['initializer']['minimum_cell_radius']>m['initializer']['throat_radius'],'reason':'Coarse h=.08333 one-resolution pilot with nonanalytic remainder underresolved even after log enrichment; H/M/poles and complete radial-collapse scientific milestone fail.'}
 out['binary_unchanged']=sha(Path('build-shift-3d/src/athena'))==m['binary_sha256'];assert out['binary_unchanged']
 out['source_hashes']={'analyzer':sha(Path(__file__)),'native_owner':sha(Path('src/z4c/hyperboloidal/native_composite_evolution.cpp'))}
 (root/'analysis.json').write_text(json.dumps(out,indent=2)+'\n')
 m['analysis_file']='analysis.json';m['mechanical_execution_qualified']=True;m['scientific_acceptance']=False;m['binary_unchanged']=out['binary_unchanged'];(root/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 print(json.dumps({'execution':out['execution'],'fullprecision_ADM_quantized_bitwise':all(out['actual_ADM']['quantized_outputs_bitwise_identical'].values()),'old_Minkowski34bitwise':out['old_Minkowski_regression']['all_identical'],'scientific_acceptance':False}))

if __name__=='__main__':main()
