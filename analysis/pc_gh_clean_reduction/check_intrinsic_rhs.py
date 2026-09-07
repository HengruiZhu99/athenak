#!/usr/bin/env python3
"""Every-row source/jet and complete symbol comparison to the pinned candidate."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import time
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--binary',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
spec=importlib.util.spec_from_file_location('pinned_candidate',a.reference);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
rng=np.random.default_rng(607091);inputs=[];expected=[];states=[]
eps=1e-30
for n in range(24):
 u=rng.uniform(-.4,.4,50);z=[.05,.1,.100001,.2,.3,.499999,.5,.8][n%8]
 alpha=[.7,1.,1.3][n//8];u[0]=np.sqrt(z/alpha);u[1]=alpha/u[0]
 du=rng.uniform(-.3,.3,(3,50));rate=[0,.1,1.,2][n%4];eta=[0,2,3][n%3];kappa=[0,1,2][n%3]
 kw=dict(rate=rate,eta=eta,kappa=kappa);f,ell=ref.config_sources(u,eta)
 values=[ref.rhs(u,du,**kw),f,ell.ravel()]
 for k in range(3):
  ff,ee=ref.config_sources(u.astype(complex)+1j*eps*du[k],eta)
  values.extend([ff.imag/eps,(ee.imag/eps).ravel()])
 actual_z=u[0]*u[1]*u[0]*u[0]
 values.append([ref.switch(actual_z),ref.switch(actual_z+1j*eps).imag/eps])
 inputs.append(np.r_[u,du.ravel(),rate,eta,kappa]);expected.append(np.concatenate(values));states.append((u,kw))
print('Prepared 24 nonlinear source/jet cases',flush=True)
matrices=[]
for sample in range(8):
 u,kw=states[sample];normal=rng.normal(size=3);normal/=np.linalg.norm(normal)
 start=len(inputs)
 for column in range(-1,50):
  du=np.zeros((3,50))
  if column>=0:du[:,column]=normal
  inputs.append(np.r_[u,du.ravel(),kw['rate'],kw['eta'],kw['kappa']]);expected.append(np.r_[ref.rhs(u,du,**kw),np.zeros(78)])
 matrices.append(dict(sample=sample,start=start,normal=normal.tolist()))
 print('Prepared principal matrix',sample,flush=True)
np.savetxt(a.output/'input.txt',inputs,header=str(len(inputs)),comments='',fmt='%.17g')
np.savetxt(a.output/'expected.txt',expected,fmt='%.17g')
command=[str(a.binary.resolve()),str((a.output/'input.txt').resolve()),str((a.output/'output.txt').resolve()),str((a.output/'kokkos.txt').resolve())]
start=time.time()
with (a.output/'run.log').open('w') as log:run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=180)
assert run.returncode==0
out=np.loadtxt(a.output/'output.txt');expected=np.array(expected)
errors=abs(out[:24]-expected[:24])/(1+abs(expected[:24]));maxerr=float(errors.max())
row_errors=np.max(errors[:,:50],axis=0).tolist();symbol_errors=[]
for m in matrices:
 first=m['start'];compiled=(out[first+1:first+51,:50]-out[first,:50]).T
 oracle=(expected[first+1:first+51,:50]-expected[first,:50]).T
 error=abs(compiled-oracle)/(1+abs(oracle));worst=np.unravel_index(np.argmax(error),error.shape)
 symbol_errors.append(dict(sample=m['sample'],normal=m['normal'],max_error=float(error.max()),worst_row=int(worst[0]),worst_column=int(worst[1])))
status='PASS' if np.isfinite(out).all() and maxerr<=2e-11 and max(x['max_error'] for x in symbol_errors)<=2e-11 else 'FAIL'
summary=dict(status=status,nonlinear_cases=24,symbols=8,compiled_points=len(inputs),tolerance=2e-11,max_source_jet_error=maxerr,rhs_row_errors=row_errors,principal_matrices=symbol_errors,wall_seconds=time.time()-start,command=command,binary_sha256=hashlib.sha256(a.binary.read_bytes()).hexdigest(),input_sha256=hashlib.sha256((a.output/'input.txt').read_bytes()).hexdigest(),reference_path=str(a.reference),reference_sha256=hashlib.sha256(a.reference.read_bytes()).hexdigest(),scope='complete point kernel and derivative matrices; no evolution or characteristic theorem')
(a.output/'results.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));assert status=='PASS'
