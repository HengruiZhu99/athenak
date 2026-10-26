#!/usr/bin/env python3
"""Value-only high-precision physical pullback check of frozen lapse/shift jets."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import mpmath as mp
import audit_geometry as geometry

def main():
 p=argparse.ArgumentParser();p.add_argument('geometry',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--smooth-layer',action='store_true');a=p.parse_args()
 if a.output.exists():raise RuntimeError('Existing audit is immutable')
 geometry.SMOOTH_LAYER=a.smooth_layer;records=[]
 for row in csv.DictReader(a.geometry.open()):
  y=np.array([float(row[k]) for k in ['x','y','z']]);rho=np.linalg.norm(y);M=float(row['mass']);spin=float(row['spin'])
  if not any(abs(rho-r)<1e-9 for r in [.1,2,12,23.999]):continue
  h=mp.mpf(str(rho))*mp.mpf('1e-6');grad=np.zeros((3,4))
  for i in range(3):
   values=[]
   for k in [-2,-1,1,2]:
    shifted=[mp.mpf(str(v)) for v in y];shifted[i]+=k*h;G=geometry.conformal_mp(shifted,M,M*spin)**-1
    values.append([1/mp.sqrt(-G[0,0])]+[-G[0,j+1]/G[0,0] for j in range(3)])
   for j in range(4):grad[i,j]=float((values[0][j]-8*values[1][j]+8*values[2][j]-values[3][j])/(12*h))
  target=np.array([[float(row[f'alpha{i}'])]+[float(row[f'beta{i}{j}']) for j in range(3)] for i in range(3)])
  error=float(np.max(abs(grad-target))/(1+np.max(abs(target))));records.append(dict(mass=M,spin=spin,y=y.tolist(),relative_error=error))
 maximum=max(r['relative_error'] for r in records);result=dict(probes=len(records),max_relative_error=maximum,tolerance=1e-9,passed=maximum<1e-9,records=records,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
