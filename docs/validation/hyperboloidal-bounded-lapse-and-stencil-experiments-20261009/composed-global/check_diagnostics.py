from pathlib import Path
import json,struct,subprocess,hashlib
import numpy as np
w=Path(__file__).resolve().parent;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();vectors=dict(np.load(w/'spatialnorm-validation-vectors.npz'));meta=json.loads((w/'spatialnorm-cache0.0001-metadata.json').read_text());N=meta['points'];coords=np.asarray(meta['xyz_omega_volume_ginv_chi']);oldroot=w.parents[1]/'boundary/full-tensor-propagator/full22-v2';oldmeta=json.loads((oldroot/'spatialnorm-cache0.0001-metadata.json').read_text());oldcoords=np.asarray(oldmeta['xyz_omega_volume_ginv_chi']);assert coords.shape==oldcoords.shape
G=np.zeros((N,3,3));ix=[0,0,0,1,1,2];iy=[0,1,2,1,2,2]
for k,(i,j) in enumerate(zip(ix,iy)):G[:,i,j]=G[:,j,i]=coords[:,5]*coords[:,6+k]
def norms(c):return np.sqrt([np.mean(c[:,0]**2),np.mean(np.einsum('pi,pij,pj->p',c[:,1:4],G,c[:,1:4])),np.mean(np.einsum('pi,pij,pj->p',c[:,4:7],G,c[:,4:7]))])
ferr=(w/'diagnostic-gate.stderr').open('w');p=subprocess.Popen([str(w/'constraint-server')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=ferr);assert json.loads(p.stdout.readline())['points']==N
nerr=(w/'diagnostic-native-gate.stderr').open('w');q=subprocess.Popen([str(w/'diagnostic-composed')],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=nerr);native_meta=json.loads(q.stdout.readline());assert native_meta['points']==N
def read(proc,n):
 out=bytearray()
 while len(out)<8*n:
  z=proc.stdout.read(8*n-len(out))
  if not z:raise RuntimeError('diagnostic server terminated')
  out.extend(z)
 return np.frombuffer(out,dtype=np.float64).copy()
def C(v,mode,eps=1e-5):
 p.stdin.write(mode.encode()+struct.pack('dd',eps,0)+v.tobytes());p.stdin.flush();return read(p,7*N).reshape(N,7)
rows=[]
for name in ['gauge_pulse','smooth_geometry','shell_random','white_random']:
 v=vectors[name].copy();v/=np.linalg.norm(v);d=C(v,'d');c=C(v,'c');hn=norms(d);sweep=[]
 for eps in [1e-3,1e-4,1e-5,1e-6]:
  q.stdin.write(struct.pack('d',eps)+v.tobytes());q.stdin.flush();actual=read(q,6).reshape(2,3).mean(axis=0);error=float(np.linalg.norm(actual-hn)/max(np.linalg.norm(hn),1e-300));sweep.append({'epsilon':eps,'actual_composed_native_H_M_Z':actual.tolist(),'relative_error':error})
 rows.append({'name':name,'matching_composed_H_M_Z':hn.tolist(),'standard_H_M_Z_on_same_ng4_state':norms(c).tolist(),'M_Z_callback_max_absolute_difference':float(abs(d[:,1:]-c[:,1:]).max()),'native_Diagnose_projector_sweep':sweep})
for proc in [p,q]:proc.stdin.close();proc.wait()
ferr.close();nerr.close();mx=max(r['M_Z_callback_max_absolute_difference'] for r in rows);best=max(min(x['relative_error'] for x in r['native_Diagnose_projector_sweep']) for r in rows if r['name']!='gauge_pulse');out={'coordinate_mapping_metadata_max_error':float(abs(coords-oldcoords).max()),'coordinate_xyz_max_error':float(abs(coords[:,:3]-oldcoords[:,:3]).max()),'reference_native_H_M_Z':native_meta['reference_H_M_Z'],'rows':rows,'source_hashes':{str(x):sha(x) for x in [w/'constraint_server.cpp',w/'diagnostic_constraint_norms.cpp',Path(__file__)]},'server_returncodes':[p.returncode,q.returncode],'maximum_M_Z_difference':mx,'worst_best_nonzero_native_norm_relative_error':best,'thresholds':{'M_Z_exact_equal':0,'native_norm_best_relative':2e-7,'coordinate_xyz':1e-13},'all_pass':mx==0 and best<2e-7 and abs(coords[:,:3]-oldcoords[:,:3]).max()<1e-13 and p.returncode==q.returncode==0};(w/'diagnostic-gate.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2));assert out['all_pass']
