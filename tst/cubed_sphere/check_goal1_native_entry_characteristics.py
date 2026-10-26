#!/usr/bin/env python3
"""Actual native frozen symbols on saved angular/intentional-constraint trajectories; no new evolution."""
from pathlib import Path
import argparse,json,hashlib,subprocess,time,struct
import numpy as np
p=argparse.ArgumentParser();p.add_argument('case',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=False,parents=True)
R=Path(__file__).resolve().parents[2];binary=R/'build-goal1-minkowski-20261018/bin/principal_symbol'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(n,j):(a.output/n).write_text(json.dumps(j,indent=2)+'\n')
put('protocol.json',{'scope':'Full actual raw native22 highest-order extraction, live determinant/Atrace tangent20, constraint8 and gauge/physical quotient checks. Pseudodifferential representation for analysis only, zero added evolved fields.','case':str(a.case),'selection':'max absolute H/M/Z, six extrema, six Cartesian-shell interface directions and minimum-radius bulk node; seven Cartesian/oblique/radial covectors per unique node.','budgets':'Reuse existing 1e-11 algebraic,1e-10 constraint/gauge and1e-12 polynomial gates without modifying reviewer.','reference':'unchanged S1,a.5,layer.05,.95 RWM','not_admitted':'General exact-scri characteristic hierarchy, lower sources, native discrete full-method stability.'})
snapshot=a.case/'cubed_sphere_native_last_jets.bin'
with snapshot.open('rb')as f:n,c=np.fromfile(f,np.uint64,2);data=np.fromfile(f,np.float64).reshape(int(n),int(c))
assert c==300 and np.isfinite(data).all()
with (a.case/'checkpoint.bin').open('rb')as f:
 magic=f.read(16);count={b'ATHENA_CSP_CK05!':23,b'ATHENA_CSP_CK06!':25}[magic];f.seek(0);fmt=f'<16s64s{count}dQdq';head=struct.unpack(fmt,f.read(struct.calcsize(fmt)))
assert head[0]in [b'ATHENA_CSP_CK05!',b'ATHENA_CSP_CK06!']
xyz=data[:,:3];radius=np.sqrt(np.einsum('ij,ij->i',xyz,xyz));fields=data[:,14:].reshape(-1,22,13)[:,:,0]
g=np.empty((len(fields),3,3));g[:,0,0]=fields[:,1];g[:,0,1]=g[:,1,0]=fields[:,2];g[:,0,2]=g[:,2,0]=fields[:,3];g[:,1,1]=fields[:,4];g[:,1,2]=g[:,2,1]=fields[:,5];g[:,2,2]=fields[:,6];eigen=np.linalg.eigvalsh(g);physical_validity={'chi_min':float(fields[:,0].min()),'lapse_min':float(fields[:,18].min()),'metric_min_eigenvalue':float(eigen.min()),'metric_max_eigenvalue':float(eigen.max()),'max_abs_det_minus_one':float(abs(data[:,10]).max()),'max_abs_A_trace':float(abs(data[:,11]).max()),'cells':len(fields),'all_finite':bool(np.isfinite(data).all()),'exact_scri_numerical_admitted':False};assert physical_validity['chi_min']>0 and physical_validity['lapse_min']>0 and physical_validity['metric_min_eigenvalue']>0;put('physical-validity.json',physical_validity)
ds=np.array([[1.,0,0],[0,1,0],[0,0,1],[.8,-.48,.36],[1,1,1],[-1,2,-3]]);ds/=np.linalg.norm(ds,axis=1)[:,None]
indices=[int(np.argmax(abs(data[:,j])))for j in [5,6,8]]+[int(np.argmax(abs(fields[:,17])))]
for axis in range(3):
 for sign in [-1,1]:
  d=np.zeros(3);d[axis]=sign;dot=np.einsum('ij,j->i',xyz,d);indices.append(int(np.argmax(dot)));mask=(data[:,13]==1)&(radius<.32);indices.append(int(np.argmax(np.where(mask,dot,-100))))
indices.append(int(np.argmin(radius)));indices=list(dict.fromkeys(indices));lines=[];meta=[]
for i in indices:
 for d in [*ds,xyz[i]/radius[i]]:
  words=[0,.5,.05,.95,*xyz[i],*fields[i],0,0,0,*d];assert len(words)==35;lines.append(' '.join(format(float(v),'.17g')for v in words));meta.append({'node':i,'xyz':xyz[i].tolist(),'covector':d.tolist()})
samples=a.output/'samples.txt';samples.write_text('\n'.join(lines)+'\n');cmd=[str(binary),str(samples.resolve()),'--logical-normal'];start=time.time();run=subprocess.run(cmd,capture_output=True,text=True);(a.output/'raw-symbol.txt').write_text(run.stdout);(a.output/'raw-symbol.stderr').write_text(run.stderr);assert run.returncode==0,run.stderr
reviewer=R/'tst/cubed_sphere/review_radial_principal_symbol.py';cmd2=[str(R/'build-goal1-minkowski-20261018/python/bin/python'),str(reviewer),str((a.output/'raw-symbol.txt').resolve()),str((a.output/'review.json').resolve()),'--evolved'];run2=subprocess.run(cmd2,capture_output=True,text=True);(a.output/'review.log').write_text(run2.stdout+run2.stderr)
put('run.json',{'extraction_command':cmd,'review_command':cmd2,'extraction_exit':run.returncode,'review_exit':run2.returncode,'seconds':time.time()-start,'snapshot_sha256':sha(snapshot),'checkpoint_sha256':sha(a.case/'checkpoint.bin'),'symbol_binary_sha256':sha(binary),'reviewer_sha256':sha(reviewer),'source_sha256':sha(Path(__file__)),'time':head[-2],'metadata':meta});assert run2.returncode==0,run2.stderr
records=json.loads((a.output/'review.json').read_text())['records'];keys=['algebraic_subspace_leak','constraint_kernel_leak','gauge_kernel_leak','two_branch_polynomial_relative_residual','symmetrizer_condition','max_imaginary'];summary={'samples':len(records),'native_time':head[-2],'maxima':{k:max(x[k]for x in records)for k in keys},'analysis_only':True,'native_evolved_added_fields':0,'passed_declared_finite_characteristic_gates':True,'general_exact_scri_characteristic_gate_open':True};put('summary.json',summary);print(json.dumps(summary))
