"""Full22 native stage tangent validation; no generator eigensolve."""
from pathlib import Path
import argparse,hashlib,json,struct,subprocess,time
import numpy as np
from scipy.sparse import csr_matrix,save_npz
w=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('gauge',choices=['production','spatialnorm']);p.add_argument('--local-eps',type=float,default=1e-4);args=p.parse_args();g=args.gauge
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();prefix=w/f'{g}-cache{args.local_eps}';stderr=Path(str(prefix)+'-validation.stderr');err=stderr.open('w');cmd=['/Users/hz0693/research/hyperboloidal/build-layer-research/boundary/full-tensor-propagator/full22-v2/server-spatialnorm','20','2.2',str(args.local_eps),str(prefix)];t0=time.monotonic();proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
line=proc.stdout.readline()
if not line:raise RuntimeError(stderr.read_text())
meta=json.loads(line);N=meta['points'];d20=N*20;d22=N*22
def apply(v,mode='j',eps=1e-5,dt=0):
 v=np.asarray(v,dtype=np.float64);proc.stdin.write(mode.encode()+struct.pack('dd',eps,dt)+v.tobytes());proc.stdin.flush();no=d20 if mode in ['p','s'] else N*7 if mode=='d' else N*2 if mode=='e' else d22;b=bytearray()
 while len(b)<no*8:
  chunk=proc.stdout.read(no*8-len(b))
  if not chunk:raise RuntimeError('server terminated: '+stderr.read_text())
  b.extend(chunk)
 return np.frombuffer(b,dtype=np.float64).copy()
A=csr_matrix((np.fromfile(str(prefix)+'-data.bin',dtype='<f8'),np.fromfile(str(prefix)+'-indices.bin',dtype='<i4'),np.fromfile(str(prefix)+'-indptr.bin',dtype='<u8')),shape=(d22,d22));A.sort_indices();save_npz(str(prefix)+'-J22.npz',A)
L=np.fromfile(str(prefix)+'-lift.bin',dtype='<f8').reshape(N,22,20);P=np.fromfile(str(prefix)+'-restrict.bin',dtype='<f8').reshape(N,20,22)
def lift(v):return np.einsum('pij,pj->pi',L,np.asarray(v).reshape(N,20)).ravel()
def project(v):return np.einsum('pij,pj->pi',P,np.asarray(v).reshape(N,22)).ravel()
def J20(v):return project(A@lift(v))
def step22(v,dt):
 x=lift(v);k1=A@x;k2=A@k1;k3=A@k2;return project(x+dt*k1+dt*dt*k2/2+dt**3*k3/6)
def step20(v,dt):
 k1=J20(v);k2=J20(k1);k3=J20(k2);return v+dt*k1+dt*dt*k2/2+dt**3*k3/6
vectors=dict(np.load(w.parent/'native20'/f'{g}-validation-vectors.npz'));rng=np.random.default_rng(899);raws={'raw_white22':rng.normal(size=d22),'lifted_gauge20':lift(vectors['gauge_pulse']),'lifted_shell20':lift(vectors['shell_random'])};res={'command':cmd,'metadata':meta,'source_sha256':sha(w/'full22_server.cpp'),'base_header_sha256':sha(w/'projected_base.hpp'),'driver_sha256':sha(Path(__file__)),'executable_sha256':sha(Path('/Users/hz0693/research/hyperboloidal/build-layer-research/boundary/full-tensor-propagator/full22-v2/server-spatialnorm')),'matrix_shape':list(A.shape),'matrix_nnz':int(A.nnz),'matrix_file_bytes':sum(Path(str(prefix)+f'-{s}.bin').stat().st_size for s in ['data','indices','indptr']),'lift_project_identity_max':float(abs(np.einsum('pij,pjk->pik',P,L)-np.eye(20)).max()),'raw_Jv':[],'one_step':[]}
for name,v in raws.items():
 fast=apply(v);sparse=A@v;row={'name':name,'sparse_vs_cached_relative_l2':float(np.linalg.norm(sparse-fast)/np.linalg.norm(fast)),'sparse_vs_cached_linf':float(abs(sparse-fast).max()),'native_eps_sweep':[]}
 for eps in [1e-3,1e-4,1e-5,1e-6]:
  f=apply(v,'f',eps);row['native_eps_sweep'].append({'eps':eps,'relative_l2':float(np.linalg.norm(f-sparse)/np.linalg.norm(sparse)),'linf':float(abs(f-sparse).max())})
 res['raw_Jv'].append(row);print(g,'raw',row,flush=True)
v=raws['raw_white22'];t=time.monotonic()
for _ in range(100):A@v
res['sparse_matvec_seconds']=(time.monotonic()-t)/100
res['native_pole_dt']=native_dt=.03*meta['min_omega']
for name in ['gauge_pulse','smooth_geometry','shell_random','white_random']:
 v=vectors[name];jl=A@lift(v);normal=jl-lift(project(jl));row={'name':name,'J22_lift_norm':float(np.linalg.norm(jl)),'normal_J22_lift_norm':float(np.linalg.norm(normal)),'normal_fraction':float(np.linalg.norm(normal)/np.linalg.norm(jl)),'dt_comparison':[]}
 for factor in [4,2,1,.5,.25]:
  dt=native_dt*factor;s=step22(v,dt);p20=step20(v,dt);item={'dt':dt,'factor_native':factor,'native22_vs_projected20_relative_input_l2':float(np.linalg.norm(s-p20)/np.linalg.norm(v)),'native22_vs_projected20_relative_increment_l2':float(np.linalg.norm(s-p20)/np.linalg.norm(s-v)),'native_step_amplitude_sweep':[]}
  if factor in [2,1,.5]:
   for eps in [1e-3,1e-4,1e-5]:
    actual=apply(v,'s',eps,dt);item['native_step_amplitude_sweep'].append({'eps':eps,'relative_input_l2_vs_J22_step':float(np.linalg.norm(actual-s)/np.linalg.norm(v)),'relative_increment_l2_vs_J22_step':float(np.linalg.norm(actual-s)/np.linalg.norm(s-v)),'linf':float(abs(actual-s).max())})
  row['dt_comparison'].append(item)
 res['one_step'].append(row);print(g,'step',row,flush=True)
res['total_seconds']=time.monotonic()-t0;proc.stdin.close();proc.wait();err.close();res['server_exit_status']=proc.returncode;res['stderr_sha256']=sha(stderr);Path(str(prefix)+'-validation.json').write_text(json.dumps(res,indent=2)+'\n');Path(str(prefix)+'-metadata.json').write_text(json.dumps(meta,indent=2)+'\n');print('done',g,res['total_seconds'],res['sparse_matvec_seconds'],flush=True)
