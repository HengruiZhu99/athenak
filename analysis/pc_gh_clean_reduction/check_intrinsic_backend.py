#!/usr/bin/env python3
"""Verify CUDA point/RHS matrices against frozen reference and identical CPU input."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cpu',type=Path,required=True);p.add_argument('--cuda',type=Path,required=True);p.add_argument('--cuda-output',default='output.txt');a=p.parse_args()
manifest=json.loads((a.cuda/'run-manifest.json').read_text());assert manifest['input_sha256']==hashlib.sha256((a.cpu/'input.txt').read_bytes()).hexdigest()
cpu=np.loadtxt(a.cpu/'output.txt');gpu=np.loadtxt(a.cuda/a.cuda_output);reference=np.loadtxt(a.cpu/'expected.txt');meta=json.loads((a.cpu/'results.json').read_text())
assert cpu.shape==gpu.shape==reference.shape==(432,128)
backend=float(np.max(abs(cpu-gpu)/(1+abs(cpu))));jet=float(np.max(abs(gpu[:24]-reference[:24])/(1+abs(reference[:24]))));symbols=[]
for n in range(8):
 start=24+51*n;x=(gpu[start+1:start+51,:50]-gpu[start,:50]).T;y=(reference[start+1:start+51,:50]-reference[start,:50]).T
 symbols.append(float(np.max(abs(x-y)/(1+abs(y)))))
summary={'status':'PASS' if np.isfinite(gpu).all() and max(backend,jet,*symbols)<=2e-11 else 'FAIL','tolerance':2e-11,'input_sha256':manifest['input_sha256'],'cpu_binary_sha256':meta['binary_sha256'],'cuda_binary_sha256':manifest['binary_sha256'],'backend_max_error':backend,'source_jet_max_error':jet,'principal_matrix_max_errors':symbols,'scope':'432 compiled points, all 50 rows and eight full matrices; no evolution','wall_seconds':manifest['wall_seconds']}
(a.cuda/'results.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2));assert summary['status']=='PASS'
