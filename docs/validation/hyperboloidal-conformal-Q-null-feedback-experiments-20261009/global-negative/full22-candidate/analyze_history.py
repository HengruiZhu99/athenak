"""Native diagnostics and localization of saved global projected tangent histories."""
from pathlib import Path
import argparse,json,struct,subprocess,time
import numpy as np
from scipy.sparse import load_npz
w=Path(__file__).resolve().parent;p=argparse.ArgumentParser();p.add_argument('gauge',choices=['production','spatialnorm']);p.add_argument('--method',choices=['canonical','krylov'],default='krylov');p.add_argument('--stop',type=float,default=2.);a=p.parse_args();g=a.gauge;tag=('projected-expm' if a.method=='canonical' else 'projected-krylov')+f'-t{a.stop}';file=w/(f'{g}-projected-expm-t{a.stop}.npz' if a.method=='canonical' else f'{g}-projected-krylov-m50-80-h0.1-t{a.stop}.npz');data=np.load(file);values=data['values'];times=data['times'];names=data['names'].tolist();meta=json.loads((w/f'{g}-cache0.0001-metadata.json').read_text());N=meta['points'];coords=np.asarray(meta['xyz_omega_volume_ginv_chi']);xyz=coords[:,:3];r=np.linalg.norm(xyz,axis=1);volume=coords[:,4];spacing=meta['spacing'];weights=spacing**3*volume;G=np.zeros((N,3,3));ix=[0,0,0,1,1,2];iy=[0,1,2,1,2,2]
for k,(i,j) in enumerate(zip(ix,iy)):G[:,i,j]=G[:,j,i]=coords[:,5]*coords[:,6+k]
L=np.fromfile(w/f'{g}-cache0.0001-lift.bin',dtype='<f8').reshape(N,22,20);J=load_npz(w/f'{g}-projected-J20.npz');stderr=w/f'{g}-history-t{a.stop}-native-diagnostics.stderr';err=stderr.open('w');proc=subprocess.Popen([str(w.parent/f'server-{g}'),'16','2.2','0.0001'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err);native_meta=json.loads(proc.stdout.readline());assert native_meta['points']==N;assert np.allclose(native_meta['xyz_omega_volume_ginv_chi'],coords,rtol=0,atol=0)
def apply(v,mode,eps=1e-5):
 proc.stdin.write(mode.encode()+struct.pack('d',eps)+np.asarray(v,dtype=np.float64).tobytes());proc.stdin.flush();n=N*(7 if mode=='d' else 2);b=bytearray()
 while len(b)<8*n:
  t=proc.stdout.read(8*n-len(b))
  if not t:raise RuntimeError(stderr.read_text())
  b.extend(t)
 return np.frombuffer(b,dtype=np.float64).copy().reshape(N,-1)
def fraction(q,inside):
 total=np.sum(q);return float(np.sum(q[inside])/total) if total>0 else 0.
def constraint_parts(c):return np.column_stack([c[:,0]**2,np.einsum('pi,pij,pj->p',c[:,1:4],G,c[:,1:4]),np.einsum('pi,pij,pj->p',c[:,4:7],G,c[:,4:7])])
res={'gauge':g,'method':a.method,'semantics':'continuous projected semidiscrete global generator; exact native final-only RK3 one-step separately validated','outward_reference_crossing_time':.7457643839234269,'stop_in_reference_crossings':float(times[-1]/.7457643839234269),'component_norm_definition':'sum of squares of 22 native stored upper-tensor/scalar/vector perturbation components; gradient contracted with reference conformal spatial inverse; Cartesian quadrature h^3 sqrt(det gamma_bar). This is a reference weighted component H1 norm, not invariant tensor or proved symmetrizer energy.','native_constraint_norm_definition':'RMS H and covector M,Z contracted with reference gamma_bar inverse, matching linearized native diagnostics; no volume weight for these RMS norms','radial_bins':[0,.2,.4,.6,.8,.9,.95,1],'histories':[],'ritz_from_history':[]};Csave=np.empty((len(times),N,7,len(names)));Ns=np.empty((len(times),2,len(names)));t0=time.monotonic()
for k,name in enumerate(names):
 hist=[]
 for it,t in enumerate(times):
  v=values[it,:,k];n=apply(v,'e');c=apply(v,'d');Csave[it,:,:,k]=c;Ns[it,:,k]=np.sqrt(np.sum(weights[:,None]*n,axis=0));parts=constraint_parts(c);full=np.einsum('pij,pj->pi',L,v.reshape(N,20));q=np.sum(full*full,axis=1);weighted=q*weights
  row={'time':float(t),'euclidean_component_l2':float(np.linalg.norm(v)),'max_abs_22component':float(abs(full).max()),'reference_volume_component_l2':float(Ns[it,0,k]),'reference_volume_component_H1':float(np.sqrt(Ns[it,0,k]**2+Ns[it,1,k]**2)),'native_H_M_Z_rms':np.sqrt(np.mean(parts,axis=0)).tolist(),'outer_r09_squared_component_fraction':fraction(weighted,r>=.9),'outer_r09_squared_constraints_fraction':[fraction(parts[:,j],r>=.9) for j in range(3)],'peak_component_radius':float(r[np.argmax(q)]),'peak_H_M_Z_radius':[float(r[np.argmax(parts[:,j])]) for j in range(3)]};hist.append(row)
 hist0=hist[0]
 for row in hist:
  for key in ['reference_volume_component_l2','reference_volume_component_H1']:row[key+'_amplification']=row[key]/hist0[key]
  row['euclidean_component_amplification']=row['euclidean_component_l2']/hist0['euclidean_component_l2']
 # Centered native constraint derivative accuracy on initial, mid, final vectors.
 check=[]
 for it in [0,len(times)//2,len(times)-1]:
  c=Csave[it,:,:,k];other=apply(values[it,:,k],'d',3e-5);den=np.linalg.norm(c);check.append({'time':float(times[it]),'eps1e-5_vs3e-5_relative_constraints_l2':float(np.linalg.norm(c-other)/den) if den else float(np.linalg.norm(c-other))})
 res['histories'].append({'name':name,'history':hist,'constraint_amplitude_convergence':check});print(g,name,'initial',hist[0],'final',hist[-1],flush=True)
# No generator eigensolve or reduced Ritz extraction in this candidate gate.
res['diagnostic_seconds']=time.monotonic()-t0;proc.stdin.close();proc.wait();err.close();res['diagnostic_server_exit']=proc.returncode;np.savez_compressed(w/f'{g}-{tag}-native-diagnostics.npz',times=times,constraints=Csave,norm_squared_integral_roots=Ns);(w/f'{g}-{tag}-analysis.json').write_text(json.dumps(res,indent=2)+'\n');print('done',g,'seconds',res['diagnostic_seconds'],'ritz',res['ritz_from_history'][:2],flush=True)
