#!/usr/bin/env python3
"""Independent accepted restart/mirror/ADM checks on actual changing native packs."""
import argparse,json,struct
from pathlib import Path
import numpy as np
from analyze_native_wormhole import checksum,native_bins

def sidecar(path):
 data=path.read_bytes();assert checksum(data[:-8])==struct.unpack('=Q',data[-8:])[0];offset=0
 def pod(fmt):
  nonlocal offset
  value=struct.unpack_from('='+fmt,data,offset)[0];offset+=struct.calcsize('='+fmt);return value
 def text():
  nonlocal offset
  n=pod('Q');value=data[offset:offset+n].decode();offset+=n;return value
 def vector():
  nonlocal offset
  n=pod('Q');v=np.frombuffer(data,dtype='=f8',count=n,offset=offset).copy();offset+=8*n;return v
 def group():return {name:vector() for name in ['volume','scri','endpoints']}
 r={'magic':text(),'tag':text(),'n':pod('i'),'ng':pod('i'),'half':pod('d'),'radii':vector()};assert r['magic']=='ATHENAK_NATIVE_COMPOSITE_ADAPTIVE_REGULAR_V1';r['bulk']=pod('i');r['native_count']=pod('i');count=pod('Q');assert count==r['native_count'];r['topology']=[]
 for i in range(count):
  native,composite,level=pod('i'),pod('i'),pod('i');bounds=[pod('d') for _ in range(6)];r['topology'].append({'native':native,'composite':composite,'refinement':level,'lo':bounds[::2],'hi':bounds[1::2]})
 r.update(version=pod('I'),values=group(),registers=group(),chart=text(),chart_parameters=vector())
 for name in ['mesh_revision','chart_revision','puncture_revision']:r[name]=pod('Q')
 r['integrator']=text();r['stages']=pod('i');r['rk']={name:np.array([pod('d') for _ in range(4)]) for name in ['gam0','gam1','beta','delta']};r['shift']=vector()
 for name in ['cycle','completed_cycles','geometry']:r[name]=pod('Q')
 for name in ['next_stage','open_stage']:r[name]=pod('i')
 for name in ['active','stage_open','has_rk']:r[name]=pod('?')
 for name in ['time','dt','abscissa','register_abscissa']:r[name]=pod('d')
 assert offset==len(data)-8;return r

def compare_restart(a,b):
 same={}
 for k in a:
  if k=='geometry':continue
  if isinstance(a[k],dict):same[k]=all(np.array_equal(a[k][j],b[k][j]) if isinstance(a[k][j],np.ndarray) else a[k][j]==b[k][j] for j in a[k])
  elif isinstance(a[k],np.ndarray):same[k]=np.array_equal(a[k],b[k])
  else:same[k]=a[k]==b[k]
 assert all(same.values()),same;assert b['geometry']==a['geometry']+1
 return {'all_numerical_register_RK_metadata_bitwise':True,'expected_restart_geometry_epoch_increment':1,'equal_fields':same}

def physical(folder,cycle):
 prefix='native_adaptive';number=f'{cycle:05d}';rst=folder/'rst'/f'{prefix}.{number}.rst';layout=sidecar(Path(str(rst)+'.composite'));n,ng=layout['n'],layout['ng'];padded=n+2*ng;count=layout['native_count']*25*padded**3;raw=np.frombuffer(rst.read_bytes()[-8*count:],dtype='=f8').reshape(layout['native_count'],25,padded,padded,padded);qrecords=native_bins(folder/'bin'/f'{prefix}.z4c.{number}.bin');arecords=native_bins(folder/'bin'/f'{prefix}.adm.{number}.bin');globalq=np.genfromtxt(folder/f'{prefix}.{number}.csv',delimiter=',',names=True);relative={};quantized={};active={};blocks=[]
 assert len(qrecords)==len(arecords)==layout['native_count']
 for record,admrecord in zip(qrecords,arecords):
  bounds=record['bounds'];assert np.array_equal(bounds,admrecord['bounds']);m=[b for b in layout['topology'] if np.array_equal(bounds[::2],b['lo']) and np.array_equal(bounds[1::2],b['hi'])];assert len(m)==1;m=m[0];assert record['relative_level']==m['refinement'];q=dict(zip(record['values'],raw[m['native']]));assert all(np.array_equal(record['values'][k],v.astype('f4').astype(float)) for k,v in q.items())
  g=globalq[m['composite']*n**3:(m['composite']+1)*n**3];expected={'z4c_chi':g['chi'],'z4c_alpha':g['alpha'],'z4c_Khat':g['state_13']-3,'z4c_Theta':g['state_14']}
  for i,c in enumerate(['xx','xy','xz','yy','yz','zz']):expected['z4c_g'+c]=g[f'state_{1+i}']+(1 if c in ['xx','yy','zz'] else 0);expected['z4c_A'+c]=g[f'state_{7+i}']
  for i,c in enumerate(['x','y','z']):expected['z4c_Gam'+c]=g[f'state_{15+i}'];expected['z4c_beta'+c]=g['beta_'+c];expected['z4c_B'+c]=g[f'state_{19+i}']
  assert set(expected)==set(q)
  for k,v in expected.items():
   identical=np.array_equal(q[k][ng:ng+n,ng:ng+n,ng:ng+n].reshape(-1),v);active[k]=active.get(k,True) and identical;assert identical,k
  xx,yy,zz=record['xyz'];omega=(1-xx*xx-yy*yy-zz*zz)/2;psi4=1/(omega**2*q['z4c_chi']);oracle={'adm_psi4':psi4,'adm_alpha':q['z4c_alpha']/omega}
  for c in ['xx','xy','xz','yy','yz','zz']:oracle['adm_g'+c]=psi4*q['z4c_g'+c];oracle['adm_K'+c]=psi4*(omega*q['z4c_A'+c]+q['z4c_g'+c]*(q['z4c_Khat']+2*q['z4c_Theta'])/3)
  for c in ['x','y','z']:oracle['adm_beta'+c]=q['z4c_beta'+c]
  for k,v in oracle.items():
   output=admrecord['values'][k];relative[k]=max(relative.get(k,0),float(np.max(abs(output-v)/np.maximum(1,abs(v)))));quantized[k]=quantized.get(k,True) and np.array_equal(output,v.astype('f4').astype(float))
  blocks.append({'native':m['native'],'level':m['refinement'],'active_cells':n**3,'all_ghosts_checked':True})
 assert max(relative.values())<3e-7 and all(quantized.values());return {'native_count':layout['native_count'],'levels':sorted({b['refinement'] for b in layout['topology']}),'blocks':blocks,'all25_active_physical_fields_bitwise':active,'independent_ADM_relative':relative,'quantized_ADM_bitwise':quantized,'unchanged_tolerance':3e-7}

def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();full=a.root/'full';again=a.root/'restart';prefix='native_adaptive';result={'restart':compare_restart(sidecar(full/'rst'/f'{prefix}.00004.rst.composite'),sidecar(again/'rst'/f'{prefix}.00004.rst.composite')),'physical_refined_cycle2':physical(full,2),'physical_coarsened_cycle4':physical(full,4),'transfers':{}}
 for revision,cycle in [(1,1),(2,3)]:
  old=sidecar(full/'rst'/f'{prefix}.{cycle:05d}.rst.composite');new=sidecar(full/f'{prefix}.regrid-{revision}.composite');assert new['time']==old['time'] and new['cycle']==old['cycle'] and new['integrator']==old['integrator'] and new['mesh_revision']==revision;assert new['geometry']==old['geometry']+1;assert new['chart_parameters'][-3]==new['native_count'];assert old['chart_parameters'][-3]==old['native_count'];assert all(np.array_equal(old['rk'][k],new['rk'][k]) for k in old['rk']);assert new['dt']==old['dt']
  for k in new['values']:assert np.array_equal(new['values'][k],new['registers'][k])
  per=old['n']**3;source=old['values']['volume'].reshape(-1,22)[old['native_count']*per:];target=new['values']['volume'].reshape(-1,22)[new['native_count']*per:];assert np.array_equal(source,target);assert np.array_equal(old['values']['scri'],new['values']['scri']);result['transfers'][str(revision)]={'cycle':cycle,'old_native_count':old['native_count'],'new_native_count':new['native_count'],'shell_scri_coefficients_bitwise':True,'accepted_register_reset_exact':True,'time_cycle_RK_preserved':True}
 (a.root/'independent-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'actual_refine_coarsen_analyzed':True,'native_counts':[result[k]['native_count'] for k in ['physical_refined_cycle2','physical_coarsened_cycle4']],'all_native_active_ghost_ADM':True,'complete_numerical_restart':True}))
if __name__=='__main__':main()
