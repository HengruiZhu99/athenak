#!/usr/bin/env python3
"""Compiled intrinsic primary rows against physical Ricci/Hessian/Codazzi jets."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
sys.dont_write_bytecode=True
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--binary',type=Path,required=True);p.add_argument('--reference-dir',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
sys.path.insert(0,str(a.reference_dir));source=a.reference_dir/'check_geometric_equivalence.py'
spec=importlib.util.spec_from_file_location('physical_reference',source);ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
rng=np.random.default_rng(20260906);inputs=[];expected=[];advection=[]
for sample in range(128):
 u,du=ref.sample(rng);inputs.append(np.r_[u,du.ravel(),1,2,1]);expected.append(ref.physical_oracle(u,du));advection.append(u[7:10]@du)
np.savetxt(a.output/'input.txt',inputs,header='128',comments='',fmt='%.17g');np.savetxt(a.output/'expected.txt',expected,fmt='%.17g')
cmd=[str(a.binary.resolve()),str((a.output/'input.txt').resolve()),str((a.output/'output.txt').resolve()),str((a.output/'kokkos.txt').resolve())];start=time.time();subprocess.run(cmd,check=True,timeout=180)
rhs=np.loadtxt(a.output/'output.txt')[:,:50]-np.array(advection);actual=rhs[:,[10,19,11,12,13,14,15,16,17,18]];expected=np.array(expected);error=abs(actual-expected)/(1+abs(actual)+abs(expected));worst=np.unravel_index(np.argmax(error),error.shape)
summary={'status':'PASS' if np.isfinite(actual).all() and error.max()<=2e-12 else 'FAIL','samples':128,'tolerance':2e-12,'max_normalized_error':float(error.max()),'worst_index':list(map(int,worst)),'row_errors':np.max(error,axis=0).tolist(),'rows':['K','C','Ahat_xx','Ahat_xy','Ahat_xz','Ahat_yy','Ahat_yz','Zx','Zy','Zz'],'nonzero_GH_constraints':True,'wall_seconds':time.time()-start,'command':cmd,'binary_sha256':hashlib.sha256(a.binary.read_bytes()).hexdigest(),'input_sha256':hashlib.sha256((a.output/'input.txt').read_bytes()).hexdigest(),'reference_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'independent physical-metric oracle on reduction, not evolution'}
(a.output/'results.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));assert summary['status']=='PASS'
