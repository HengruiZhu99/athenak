"""Matched pointwise gauge source C_h J_h q using original native constraint callback."""
from pathlib import Path
import hashlib,json,struct,subprocess,time
import numpy as np
from scipy.sparse import load_npz
w=Path(__file__).resolve().parent;old=w.parent/'full-tensor-propagator';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();result={'scope':'native instantaneous discrete Bianchi defect at fixed pointwise gauge seed, original C0 spatialnorm, not continuum constraint growth','runs':[]}
for n,folder,seeds in [(16,old/'full22-v2',old),(20,w/'full22',w/'native20')]:
 j=load_npz(folder/'spatialnorm-projected-J20.npz');q=np.load(seeds/'spatialnorm-validation-vectors.npz')['gauge_pulse'];jq=j@q;err=(w/f'N{n}-Hdot-native.stderr').open('w');cmd=[str(old/'server-spatialnorm'),str(n),str(2.2 if n==16 else json.loads((w/'independent-grid-enumeration.json').read_text())['rows'][-1]['span']),'0.0001'];begin=time.monotonic();p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err);meta=json.loads(p.stdout.readline());N=meta['points'];assert j.shape==(N*20,N*20) and len(q)==N*20;coords=np.asarray(meta['xyz_omega_volume_ginv_chi']);G=np.zeros((N,3,3));ti=[0,0,0,1,1,2];tj=[0,1,2,1,2,2]
 for k,(i,l) in enumerate(zip(ti,tj)):G[:,i,l]=G[:,l,i]=coords[:,5]*coords[:,6+k]
 rows=[];arrays=[]
 for eps in [1e-5,3e-5]:
  p.stdin.write(b'd'+struct.pack('d',eps)+np.asarray(jq,dtype=np.float64).tobytes());p.stdin.flush();b=bytearray()
  while len(b)<N*7*8:
   t=p.stdout.read(N*7*8-len(b));assert t;b.extend(t)
  c=np.frombuffer(b,dtype=np.float64).reshape(N,7).copy();arrays.append(c);parts=np.column_stack([c[:,0]**2,np.einsum('pi,pij,pj->p',c[:,1:4],G,c[:,1:4]),np.einsum('pi,pij,pj->p',c[:,4:7],G,c[:,4:7])]);r=np.linalg.norm(coords[:,:3],axis=1)
  rows.append({'eps':eps,'Hdot_Mdot_Zdot_rms':np.sqrt(np.mean(parts,axis=0)).tolist(),'outer_r09_squared_fractions':[float(np.sum(parts[r>=.9,k])/np.sum(parts[:,k])) for k in range(3)]})
 p.stdin.close();p.wait();err.close();assert p.returncode==0
 result['runs'].append({'N':n,'command':cmd,'seconds':time.monotonic()-begin,'matrix_sha256':sha(folder/'spatialnorm-projected-J20.npz'),'seed_sha256':sha(seeds/'spatialnorm-validation-vectors.npz'),'native20_executable_sha256':sha(old/'server-spatialnorm'),'seed_original_Euclidean_norm':float(np.linalg.norm(q)),'native_derivative_rows':rows,'epsilon_relative_l2':float(np.linalg.norm(arrays[0]-arrays[1])/np.linalg.norm(arrays[0]))})
(w/'initial-Hdot-comparison.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
