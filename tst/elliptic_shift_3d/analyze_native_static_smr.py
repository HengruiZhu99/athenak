#!/usr/bin/env python3
"""Independent topology, complete restart, native physical mirror and ADM audit."""
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np
from analyze_native_wormhole import native_bins,checksum

def sidecar(path):
 data=path.read_bytes();assert checksum(data[:-8])==struct.unpack('=Q',data[-8:])[0];offset=0
 def pod(fmt):
  nonlocal offset
  v=struct.unpack_from('='+fmt,data,offset)[0];offset+=struct.calcsize('='+fmt);return v
 def text():
  nonlocal offset
  n=pod('Q');v=data[offset:offset+n].decode();offset+=n;return v
 def vector():
  nonlocal offset
  n=pod('Q');v=np.frombuffer(data,dtype='=f8',count=n,offset=offset).copy();offset+=8*n;return v
 def group():return {name:vector() for name in ['volume','scri','endpoints']}
 d={'magic':text(),'tag':text(),'n':pod('i'),'ng':pod('i'),'half':pod('d'),'radii':vector()};assert d['magic'] in ['ATHENAK_NATIVE_COMPOSITE_STATIC_SMR_V1','ATHENAK_NATIVE_COMPOSITE_WORMHOLE_STATIC_SMR_V1'];d['wormhole']='WORMHOLE' in d['magic'];d['bulk']=pod('i');count_offset=offset;d['native_count']=pod('i');count=pod('Q');assert count==d['native_count'];d['topology']=[];d['offsets']={'native_count':count_offset}
 for _ in range(count):
  native=pod('i');composite_offset=offset;composite=pod('i');level_offset=offset;level=pod('i');bound_offset=offset;bounds=[pod('d') for _ in range(6)];d['topology'].append({'native':native,'composite':composite,'refinement':level,'lo':bounds[::2],'hi':bounds[1::2]})
  if _==0:d['offsets'].update(composite=composite_offset,level=level_offset,bounds=bound_offset)
 if d['wormhole']:
  for name in ['slice','gauge','receipt']:d[name]=text()
  d.update(positive_mode=pod('i'),taper=pod('d'),tolerance=pod('d'),dt_ceiling=pod('d'))
 d.update(version=pod('I'),values=group(),registers=group(),chart=text(),chart_parameters=vector())
 for name in ['mesh_revision','chart_revision','puncture_revision']:d[name]=pod('Q')
 d['integrator']=text();d['stages']=pod('i');d['rk']={name:np.array([pod('d') for _ in range(4)]) for name in ['gam0','gam1','beta','delta']};d['shift']=vector()
 for name in ['cycle','completed_cycles','geometry']:d[name]=pod('Q')
 for name in ['next_stage','open_stage']:d[name]=pod('i')
 for name in ['active','stage_open','has_rk']:d[name]=pod('?')
 for name in ['time','dt','abscissa','register_abscissa']:d[name]=pod('d')
 assert offset==len(data)-8;return d

def restart_compare(a,b):
 same={}
 for name in a:
  if name in ['geometry','offsets']:continue
  if isinstance(a[name],dict):same[name]=all(np.array_equal(a[name][k],b[name][k]) if isinstance(a[name][k],np.ndarray) else a[name][k]==b[name][k] for k in a[name])
  elif isinstance(a[name],np.ndarray):same[name]=np.array_equal(a[name],b[name])
  else:same[name]=a[name]==b[name]
 assert all(same.values()),same
 assert b['geometry']==a['geometry']+1
 return {'all_coefficients_registers_RK_shift_metadata_bitwise':True,'equal_fields':same,'geometry_epoch_full':a['geometry'],'geometry_epoch_restart':b['geometry'],'epoch_delta':1,'reason':'Restore explicitly invalidates cached geometry/shift once; no numerical state change','whole_sidecar_bytes_identical':False}

def audit(root):
 prefix='native_smr';full=root/'full';restart=root/'restart';last=full/'rst'/(prefix+'.00002.rst');layout=sidecar(Path(str(last)+'.composite'));again=sidecar(restart/'rst'/(last.name+'.composite'))
 rebuilt=restart_compare(layout,again)
 coefficients={}
 for cycle in range(3):
  snap=sidecar(full/'rst'/f'{prefix}.{cycle:05d}.rst.composite');ref=root/f'reference.{cycle:05d}.shift.bin';raw=ref.read_bytes();count=struct.unpack_from('=Q',raw)[0];shift=np.frombuffer(raw,dtype='=f8',count=count,offset=8)
  coefficients[str(cycle)]={'shift_bitwise':np.array_equal(snap['shift'],shift),'shift_max_error':float(np.max(abs(snap['shift']-shift)))};assert coefficients[str(cycle)]['shift_bitwise']
 n,ng=layout['n'],layout['ng'];padded=n+2*ng;count=layout['native_count']*25*padded**3
 payload=np.frombuffer(last.read_bytes()[-8*count:],dtype='=f8').reshape(layout['native_count'],25,padded,padded,padded)
 qs=native_bins(full/'bin'/(prefix+'.z4c.00002.bin'));ads=native_bins(full/'bin'/(prefix+'.adm.00002.bin'));globalq=np.genfromtxt(full/(prefix+'.00002.csv'),delimiter=',',names=True);blocks=[];overall_relative={};all_quantized={};active_identical={}
 assert len(qs)==len(ads)==layout['native_count']
 for qrecord,arecord in zip(qs,ads):
  bounds=qrecord['bounds'];assert np.array_equal(bounds,arecord['bounds']);mapping=[r for r in layout['topology'] if np.array_equal(bounds[::2],r['lo']) and np.array_equal(bounds[1::2],r['hi'])];assert len(mapping)==1;mapping=mapping[0];assert qrecord['relative_level']==mapping['refinement'];slot=mapping['native'];q64=dict(zip(qrecord['values'],payload[slot]));assert all(np.array_equal(qrecord['values'][name],v.astype('f4').astype(float)) for name,v in q64.items())
  q={name:v[ng:ng+n,ng:ng+n,ng:ng+n].reshape(-1) for name,v in q64.items()};g=globalq[mapping['composite']*n**3:(mapping['composite']+1)*n**3];expected={'z4c_chi':g['chi'],'z4c_alpha':g['alpha'],'z4c_Khat':g['state_13']-3,'z4c_Theta':g['state_14']}
  for i,c in enumerate(['xx','xy','xz','yy','yz','zz']):expected['z4c_g'+c]=g[f'state_{1+i}']+(1 if c in ['xx','yy','zz'] else 0);expected['z4c_A'+c]=g[f'state_{7+i}']
  for i,c in enumerate(['x','y','z']):expected['z4c_Gam'+c]=g[f'state_{15+i}'];expected['z4c_beta'+c]=g['beta_'+c];expected['z4c_B'+c]=g[f'state_{19+i}']
  assert set(expected)==set(q);matches={k:np.array_equal(q[k],v) for k,v in expected.items()};assert all(matches.values()),{k:float(np.max(abs(q[k]-expected[k]))) for k,v in matches.items() if not v}
  for k,v in matches.items():active_identical[k]=active_identical.get(k,True) and v
  xx,yy,zz=qrecord['xyz'];omega=(1-xx*xx-yy*yy-zz*zz)/2;psi4=1/(omega**2*q64['z4c_chi']);physical={'adm_psi4':psi4,'adm_alpha':q64['z4c_alpha']/omega}
  for c in ['xx','xy','xz','yy','yz','zz']:
   physical['adm_g'+c]=psi4*q64['z4c_g'+c];physical['adm_K'+c]=psi4*(omega*q64['z4c_A'+c]+q64['z4c_g'+c]*(q64['z4c_Khat']+2*q64['z4c_Theta'])/3)
  for c in ['x','y','z']:physical['adm_beta'+c]=q64['z4c_beta'+c]
  adm=arecord['values'];relative={name:float(np.max(abs(adm[name]-v)/np.maximum(1,abs(v)))) for name,v in physical.items()};quantized={name:np.array_equal(adm[name],v.astype('f4').astype(float)) for name,v in physical.items()}
  for name,v in relative.items():overall_relative[name]=max(overall_relative.get(name,0),v)
  for name,v in quantized.items():all_quantized[name]=all_quantized.get(name,True) and v
  blocks.append({'native':slot,'composite':mapping['composite'],'refinement':mapping['refinement'],'active_cells':n**3,'ghosts_in_ADM_audit':True,'maximum_ADM_relative_error':max(relative.values()),'quantized_all_bitwise':all(quantized.values())})
 assert max(overall_relative.values())<3e-7 and all(all_quantized.values())
 return {'scope':'Actual mixed-level native topology, fullprecision nativeRST active physical mirror + independently converted ADM including all ghost cells','restart_payload':rebuilt,'matched_reference_shift':coefficients,'native_count':layout['native_count'],'levels':sorted(set(r['refinement'] for r in layout['topology'])),'native_active_fields_bitwise':active_identical,'blocks':blocks,'ADM_relative_errors':overall_relative,'ADM_quantized_bitwise':all_quantized,'original_ADM_tolerance':3e-7,'scientific_acceptance':False}

def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);a=p.parse_args();result=audit(a.root);(a.root/'independent-analysis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'native_count':result['native_count'],'levels':result['levels'],'restart_payload_bitwise':True,'all_ADM_ghosts':True,'maximum_ADM_rounding':max(result['ADM_relative_errors'].values())}))
if __name__=='__main__':main()
