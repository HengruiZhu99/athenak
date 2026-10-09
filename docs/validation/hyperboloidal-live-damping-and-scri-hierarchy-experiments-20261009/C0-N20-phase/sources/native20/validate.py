"""Validate cached actual-kernel/global-stencil tangent against native Prepared/RHS."""
from pathlib import Path
import argparse,hashlib,json,struct,subprocess,time
import numpy as np
from scipy.ndimage import gaussian_filter
w=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('gauge',choices=['production','spatialnorm']);p.add_argument('--local-eps',type=float,default=1e-4);args=p.parse_args()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();stderr_path=w/f'{args.gauge}-cache{args.local_eps}-validation.stderr';err=stderr_path.open('w');cmd=['/Users/hz0693/research/hyperboloidal/build-layer-research/boundary/full-tensor-propagator/server-spatialnorm','20','2.1967074064860954',str(args.local_eps)];begin=time.monotonic();proc=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err)
line=proc.stdout.readline()
if not line:raise RuntimeError(stderr_path.read_text())
meta=json.loads(line);N=meta['points'];dim=meta['dimension'];coords=np.array(meta['xyz_omega_volume_ginv_chi']);xyz=coords[:,:3];r=np.sqrt(np.sum(xyz*xyz,axis=1));rng=np.random.default_rng(690)
def apply(v,mode='c',eps=1e-5):
 v=np.asarray(v,dtype=np.float64);proc.stdin.write(mode.encode()+struct.pack('d',eps)+v.tobytes());proc.stdin.flush();n=N*7 if mode=='d' else dim;count=n*8;b=bytearray()
 while len(b)<count:
  chunk=proc.stdout.read(count-len(b))
  if not chunk:raise RuntimeError('server ended: '+stderr_path.read_text())
  b.extend(chunk)
 return np.frombuffer(b,dtype=np.float64).copy()
def perturbations():
 v=np.zeros((N,20));s=r*r;shape=(1-s)**4*np.exp(-4*s);x,y,z=xyz.T;v[:,16]=.1*shape*(1+.2*x+.3*y*z);v[:,17:]=.02*shape[:,None]*np.column_stack([1+.3*y*z,.2*x,.1*x*y]);result={'gauge_pulse':v.ravel()}
 v=np.zeros((N,20));v[:,0]=np.exp(-((r-.5)/.18)**2)*(1+.2*x-.3*y*z)*(1-r*r)**4;v[:,6]=.3*v[:,0];v[:,12:15]=v[:,0,None]*xyz;result['smooth_geometry']=v.ravel()
 v=rng.normal(size=(N,20));result['white_random']=v.ravel()
 shell=np.exp(-((r-.88)/.065)**2)*(1-r*r)**4;v=rng.normal(size=(N,20))*shell[:,None];result['shell_random']=v.ravel();return result
result={'command':cmd,'metadata':meta,'source_sha256':sha(w/'tangent_server.cpp'),'old_jv_source_sha256':sha(w/'old-jv-source.cpp'),'executable_sha256':sha(Path('/Users/hz0693/research/hyperboloidal/build-layer-research/boundary/full-tensor-propagator/server-spatialnorm')),'driver_sha256':sha(Path(__file__)),'build_command':json.loads((w/f'build-{args.gauge}.json').read_text()),'vectors':[]}
vectors=perturbations()
for name,v in vectors.items():
 fast=apply(v);row={'name':name,'fast_norm':float(np.linalg.norm(fast)),'native_epsilon_sweep':[]};t=time.monotonic()
 for eps in [1e-3,3e-4,1e-4,3e-5,1e-5,3e-6]:
  actual=apply(v,'f',eps);row['native_epsilon_sweep'].append({'eps_normalized_linf':eps,'relative_l2_vs_cache':float(np.linalg.norm(actual-fast)/np.linalg.norm(fast)),'linf_vs_cache':float(abs(actual-fast).max())})
 row['native_sweep_seconds']=time.monotonic()-t;t=time.monotonic()
 for i in range(10):apply(v)
 row['fast_seconds_per_call']=(time.monotonic()-t)/10
 c=apply(v,'d',1e-5).reshape(N,7);d=apply(fast,'d',1e-5).reshape(N,7);row['initial_constraints_component_rms']=np.sqrt(np.mean(c*c,axis=0)).tolist();row['constraint_tangent_component_rms']=np.sqrt(np.mean(d*d,axis=0)).tolist();result['vectors'].append(row);print(args.gauge,name,row,flush=True)
v=rng.normal(size=dim);u=rng.normal(size=dim);Lv=apply(v);Lu=apply(u);Lsum=apply(v+u);result['cache_additivity_relative_l2']=float(np.linalg.norm(Lsum-Lv-Lu)/np.linalg.norm(Lsum));result['total_seconds']=time.monotonic()-begin
proc.stdin.close();proc.wait();err.close();result['server_exit_status']=proc.returncode;result['stderr_sha256']=sha(stderr_path);output=w/f'{args.gauge}-cache{args.local_eps}-validation.json';output.write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(w/f'{args.gauge}-validation-vectors.npz',**vectors);print('done',output,flush=True)
