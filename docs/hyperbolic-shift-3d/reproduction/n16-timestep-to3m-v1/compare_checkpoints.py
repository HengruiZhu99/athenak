#!/usr/bin/env python3
"""Compare complete states and RK compensation, excluding only checkpoint magic."""
from pathlib import Path
import argparse,hashlib,json,struct
import numpy as np
layout=struct.Struct('<16s4i3Q2d')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def load(p):
    with p.open('rb') as f:header=f.read(152)
    fields=layout.unpack(header[:72]);values=np.fromfile(p,dtype='<f8',offset=152)
    assert len(values)==2*(31*fields[5]+fields[6])
    return header,fields,values
p=argparse.ArgumentParser();p.add_argument('original',type=Path);p.add_argument('optimized',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
oh,of,ov=load(a.original);nh,nf,nv=load(a.optimized)
assert of[0].rstrip(b'\0')==b'HYP3D_MPI_V11' and nf[0].rstrip(b'\0')==b'HYP3D_MPI_DT1'
assert oh[16:]==nh[16:],'Configuration, time or step count differs'
assert np.isfinite(ov).all() and np.isfinite(nv).all()
volume=31*of[5];surface=of[6]
# Checkpoint write order: state, compensation, surface, surface compensation.
parts={};start=0
for name,count in [('volume',volume),('volume_compensation',volume),('surface',surface),('surface_compensation',surface)]:
    x=ov[start:start+count];y=nv[start:start+count];start+=count
    parts[name]=dict(bitwise_equal=x.tobytes()==y.tobytes(),max_absolute_difference=float(np.max(np.abs(x-y))),max_scaled_difference=float(np.max(np.abs(x-y)/(1+np.abs(x)))))
result=dict(original_path=str(a.original.resolve()),original_sha256=sha(a.original),optimized_path=str(a.optimized.resolve()),optimized_sha256=sha(a.optimized),configuration_equal_except_magic=True,t_over_M=3*of[-2],steps=of[-3],bitwise_equal=ov.tobytes()==nv.tobytes(),parts=parts)
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
