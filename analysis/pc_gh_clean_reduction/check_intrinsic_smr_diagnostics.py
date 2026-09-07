#!/usr/bin/env python3
"""Independent leaf-stencil verification of 78 refined reduction/curl CSV rows."""
import argparse,csv,json,sys
from pathlib import Path
sys.dont_write_bytecode=True
import numpy as np
from intrinsic_restart import read_restart
from analyze_intrinsic_interface_budget import block_budget
p=argparse.ArgumentParser();p.add_argument('--runs',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();records=[]
for folder in sorted(a.runs.glob('n*-*')):
 if not folder.is_dir():continue
 d=read_restart(sorted(folder.glob('rst/*.rst'))[-1],allow_refinement=True);ng,nx,ny,nz=d['mb'][:4];squares=np.zeros(78);maximum=np.zeros(78);volume=0
 for m,loc in enumerate(d['locations']):
  h=d['domain'][6:9]/2.**(loc[3]-d['root_level']);u=d['state'][m];q,_=block_budget(u,u,np.zeros((3,*u.shape)),h,ng,(nz,ny,nx),6);v=np.concatenate([q[k] for k in ['reduction','curl','qcurl']]);dv=np.prod(h[:2]);squares+=np.sum(v*v,axis=(1,2,3))*dv;maximum=np.maximum(maximum,np.max(abs(v),axis=(1,2,3)));volume+=nx*ny*nz*dv
 with (folder/f'intrinsic-diagnostics-c{d["cycle"]}-s3.csv').open() as f:rows=list(csv.DictReader(f))[11:]
 rms=np.sqrt(squares/volume);observed=np.array([float(r['RMS']) for r in rows]);observed_max=np.array([float(r['maximum']) for r in rows]);error=float(max(np.max(abs(rms-observed)/(1+abs(rms))),np.max(abs(maximum-observed_max)/(1+abs(maximum)))))
 assert abs(volume-1.3)<1e-12 and error<2e-12
 records.append(dict(case=folder.name,components=78,maximum_normalized_error=error,status='PASS',area=volume,csv_volume=float(rows[0]['volume'])))
with a.output.open('x') as f:json.dump(records,f,indent=2);f.write('\n')
print('Verified',len(records),'runs; maximum discrepancy',max(r['maximum_normalized_error'] for r in records))
