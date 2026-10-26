#!/usr/bin/env python3
"""Cross-closure reference identity for candidate Cell layout (two trailing ints).
Full record hash includes numerical closure metadata. Report that separately
from immutable Minkowski values and the preceding geometry/reference records.
"""
import argparse,hashlib,json,struct
from pathlib import Path
import numpy as np
from compare_closure_field_norms import read

def digest(case):
 data=read(case);p=case/'cubed_sphere_reference_before.bin';r=data['receipt'];expected=r['reference_payloads'][p.name]['sha256']
 assert hashlib.sha256(p.read_bytes()).hexdigest()==expected
 counts=[z.shape[0]*(z.shape[2]-8)*(z.shape[3]-8)*(z.shape[4]-8) for z in data['packs']]
 hashes=[]
 with p.open('rb') as f:
  magic=f.readline();assert magic==b'CUBED_SPHERE_REFERENCE_V1\n'
  leftover=p.stat().st_size-len(magic)-96-sum(z.size*8 for z in data['packs']);assert leftover%sum(counts)==0
  size=leftover//sum(counts)
  for region,z in enumerate(data['packs']):
   count=struct.unpack('<Q',f.read(8))[0];shape=struct.unpack('<5Q',f.read(40));assert count==counts[region] and shape==z.shape
   cells=np.frombuffer(f.read(count*size),dtype='u1').reshape(count,size)
   metadata=np.frombuffer(cells[:,-8:].copy().tobytes(),dtype='<i4').reshape(count,2)
   mode=int(data['signature'][12])>=100
   assert np.all(metadata[:,0]>=-1) and np.all(metadata[:,0]<=2)
   assert np.count_nonzero(metadata[:,0]>=0)==(6*(shape[2]-8)*(shape[3]-8)*3 if region and mode else 0)
   before=hashlib.sha256(cells[:,:-8].copy().tobytes()).hexdigest()
   values=hashlib.sha256(f.read(z.size*8)).hexdigest()
   hashes.append(dict(region=region,shape=shape,cell_bytes=size,geometry_reference_rhs_quadrature_sha256=before,minkowski_values_sha256=values,closure_metadata_sha256=hashlib.sha256(cells[:,-8:].copy().tobytes()).hexdigest()))
  assert f.tell()==p.stat().st_size
 return dict(case=str(case),full_reference_sha256=expected,packs=hashes)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('first',type=Path);p.add_argument('second',type=Path);p.add_argument('output',type=Path);a=p.parse_args();first=digest(a.first);second=digest(a.second)
 for x,y in zip(first['packs'],second['packs']):
  for key in ['shape','cell_bytes','geometry_reference_rhs_quadrature_sha256','minkowski_values_sha256']:assert x[key]==y[key],key
 a.output.write_text(json.dumps(dict(physical_reference_identical=True,first=first,second=second,scope='Same candidate ABI/grid. Full reference serialization also contains two closure-control integers; these are excluded only from the cross-closure identity, not from within-run immutability checks.'),indent=2)+'\n');print('physical reference identical across closures')
