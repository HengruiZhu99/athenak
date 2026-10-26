#!/usr/bin/env python3
"""All-coordinate independent MP oracle/readback of one saved native trajectory."""
from pathlib import Path
import argparse,json,struct,hashlib,importlib.util,time,math,subprocess
import numpy as np
import mpmath as m
from goal1_global_clock_oracle import Clock
p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True);start=time.time();m.mp.dps=80
root=Path(__file__).resolve().parents[2];phase=root/'docs/validation/goal1-minkowski-20261018/native-global-clock-v1';work=root/'build-goal1-minkowski-20261018/native-global-clock-v1';heightroot=root/'docs/validation/hyperboloidal-reference-wave-map-flat-IVP-values-20261009';cache=json.loads((phase/'reused-height-cache.json').read_text());modulepath=heightroot/'values-source/flat_ivp_values.py';spec=importlib.util.spec_from_file_location('reused_height_only',modulepath);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);recipe=json.loads((heightroot/'values-source/values-recipe.json').read_text());height=module.Layer(recipe,128)
with (a.case/'checkpoint.bin').open('rb') as stream:head=struct.unpack('<16s64s19dQdq',stream.read(256))
assert head[0]==b'ATHENA_CSP_CK03!';t=m.mpf(head[-2]);eps=m.mpf(head[20]);assert eps!=0 # signature index18 after2 header entries
with (a.case/'cubed_sphere_native_last_jets.bin').open('rb') as stream:n,c=np.fromfile(stream,np.uint64,2);x=np.fromfile(stream,np.float64).reshape(int(n),int(c))
assert c==300 and np.isfinite(x).all();native=x[:,14:].reshape(-1,22,13);exact=np.zeros_like(native);groups={}
for i,row in enumerate(x):
 square=sum(m.mpf(float(z))**2 for z in row[:3]);groups.setdefault(square,[]).append(i)
radial_cache=[]
for group,(square,ids) in enumerate(groups.items()):
 r=m.sqrt(square);uref=-r if r<=m.mpf('.05') else height.defect(r);local=dict(cache);local['uref_values']={r:uref};clock=Clock(r,local)
 f=clock.radial(r,t,eps);dr=[m.diff(lambda z,k=k:clock.radial(z,t,eps)[k],r)for k in range(10)];dd=[m.diff(lambda z,k=k:clock.radial(z,t,eps)[k],r,2)if k in [0,1,4,6,7]else m.mpf(0)for k in range(10)];radial_cache.append({'r':m.nstr(r,80),'uref':m.nstr(uref,80),'native_radial':[m.nstr(v,65)for v in f],'radial_first':[m.nstr(v,65)for v in dr],'radial_second_config_only':[m.nstr(v,65)for v in dd],'nodes':len(ids)})
 # Projection uses the independent exact radial jets; FP projection roundoff is
 # included in the oracle uncertainty comparison, with no native evolution reused.
 f=np.array([float(v)for v in f]);dr=np.array([float(v)for v in dr]);dd=np.array([float(v)for v in dd]);rr=float(r)
 for index in ids:
  n=x[index,:3]/rr;dn=(np.eye(3)-np.outer(n,n))/rr;ddn=np.zeros((3,3,3))
  for d in range(3):
   for e in range(3):
    for j in range(3):ddn[d,e,j]=(-(d==j)*n[e]-(e==j)*n[d]-(d==e)*n[j]+3*n[d]*n[e]*n[j])/(rr*rr)
  for field,k in [(0,0),(18,1),(7,2),(17,3)]:
   exact[index,field,0]=f[k];exact[index,field,1:4]=dr[k]*n;exact[index,field,4:]=(dd[k]*np.outer(n,n)+dr[k]*dn).reshape(9)
  for startfield,k in [(19,4),(14,5)]:
   for j in range(3):
    exact[index,startfield+j,0]=f[k]*n[j];exact[index,startfield+j,1:4]=dr[k]*n*n[j]+f[k]*dn[:,j]
    if k==4:
     H=np.zeros((3,3))
     for d in range(3):
      for e in range(3):H[d,e]=dd[k]*n[d]*n[e]*n[j]+dr[k]*(dn[d,e]*n[j]+n[d]*dn[e,j]+n[e]*dn[d,j])+f[k]*ddn[d,e,j]
     exact[index,startfield+j,4:]=H.reshape(9)
  for firstfield,kr,kt,second in [(1,6,7,True),(8,8,9,False)]:
   for ij,(i,j)in enumerate([(0,0),(0,1),(0,2),(1,1),(1,2),(2,2)]):
    field=firstfield+ij;exact[index,field,0]=f[kt]*(i==j)+(f[kr]-f[kt])*n[i]*n[j]
    for d in range(3):exact[index,field,1+d]=dr[kt]*n[d]*(i==j)+(dr[kr]-dr[kt])*n[d]*n[i]*n[j]+(f[kr]-f[kt])*(dn[d,i]*n[j]+n[i]*dn[d,j])
    if second:
     H=np.zeros((3,3))
     for d in range(3):
      for e in range(3):H[d,e]=(dd[kt]*n[d]*n[e]+dr[kt]*dn[d,e])*(i==j)+((dd[kr]-dd[kt])*n[d]*n[e]+(dr[kr]-dr[kt])*dn[d,e])*n[i]*n[j]+(dr[kr]-dr[kt])*(n[d]*(dn[e,i]*n[j]+n[i]*dn[e,j])+n[e]*(dn[d,i]*n[j]+n[i]*dn[d,j]))+(f[kr]-f[kt])*(ddn[d,e,i]*n[j]+dn[d,i]*dn[e,j]+dn[e,i]*dn[d,j]+n[i]*ddn[d,e,j])
     exact[index,field,4:]=H.reshape(9)
 if group%40==0:print('independent radial group',group+1,'/',len(groups),flush=True)
(a.output/'independent-radial-targets.json').write_text(json.dumps({'time':str(t),'epsilon':str(eps),'precision_digits':80,'height_source_sha256':hashlib.sha256(modulepath.read_bytes()).hexdigest(),'height_uncertainty':cache['declared_height_uncertainty'],'rows':radial_cache},indent=2)+'\n')
with (a.output/'independent-targets.bin').open('wb') as stream:np.array([len(x),286],np.uint64).tofile(stream);exact.reshape(len(x),286).tofile(stream)
input=a.output/'oracle-export-input.bin';output=a.output/'runtime-targets.bin'
with input.open('wb') as stream:stream.write(struct.pack('<ddQ',float(t),float(eps),len(x)));x[:,:3].copy().tofile(stream)
cmd=[str(work/'clock_export'),str(input.resolve()),str(output.resolve())];run=subprocess.run(cmd,cwd=root,capture_output=True,text=True);assert run.returncode==0,run.stderr
with output.open('rb')as stream:count,cols=np.fromfile(stream,np.uint64,2);runtime=np.fromfile(stream,np.float64).reshape(-1,22,13)
assert count==len(x)and cols==286
amp=float(4*abs(eps)/m.mpf('.04'));length=.2;config=[0,1,2,3,4,5,6,18,19,20,21];scales=np.full(22,amp/length);scales[config]=amp
masks={'all':np.ones(len(x),bool),'bulk':x[:,13]==0,'shell':x[:,13]==1}
def norms(delta):
 d=delta/scales[None,:,None];out={}
 for region,mask in masks.items():
  weights=x[mask,3];out[region]={}
  for name,z in [('native_fields',d[:,:,0]),('native_first',length*d[:,:,1:4]),('native_config_second',length*length*d[:,config,4:])]:
   zz=z[mask];point=np.sum(zz*zz,axis=tuple(range(1,zz.ndim)));out[region][name]={'rms':float(np.sqrt(np.einsum('i,i->',weights,point)/weights.sum())),'linf':float(np.sqrt(point.max()))}
 return out
uncertainty=norms(runtime-exact);errors=norms(native-exact);ratios={region:{kind:{metric:uncertainty[region][kind][metric]/max(errors[region][kind][metric],1e-300)for metric in ['rms','linf']}for kind in errors[region]}for region in errors}
review={'scope':'All active coordinates of a saved full native coupled trajectory; independent80digit radial jets and independent Cartesian projection, no numerical evolution','case':str(a.case.resolve()),'time':float(t),'epsilon':float(eps),'active_nodes':len(x),'unique_exact_radius_groups':len(groups),'native_errors_against_independent_MP':errors,'runtime_oracle_uncertainty_against_independent_MP':uncertainty,'oracle_uncertainty_fraction_of_native_error':ratios,'oracle_uncertainty_below_10percent':all(v<.1for r in ratios.values()for k in r.values()for v in k.values()),'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'snapshot_sha256':hashlib.sha256((a.case/'cubed_sphere_native_last_jets.bin').read_bytes()).hexdigest(),'runtime_export_command':cmd,'seconds':time.time()-start}
(a.output/'review.json').write_text(json.dumps(review,indent=2)+'\n');print(json.dumps({k:v for k,v in review.items()if k not in ['native_errors_against_independent_MP','runtime_oracle_uncertainty_against_independent_MP','oracle_uncertainty_fraction_of_native_error']},indent=2),flush=True)
