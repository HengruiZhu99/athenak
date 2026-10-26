#!/usr/bin/env python3
"""Independent high precision physical pullback near the layer endpoints."""
import argparse,csv,hashlib,json
from pathlib import Path
import mpmath as mp
import numpy as np
import audit_geometry as geometry

def main():
 p=argparse.ArgumentParser();p.add_argument('geometry',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise RuntimeError('Existing endpoint review is immutable')
 mp.mp.dps=80;geometry.SMOOTH_LAYER=True;records=[]
 for row in csv.DictReader(a.geometry.open()):
  M=mp.mpf(row['mass']);spin=mp.mpf(row['spin']);y=[mp.mpf(row[k]) for k in ['x','y','z']];rho=mp.sqrt(mp.fdot(y,y));g=geometry.conformal_mp(y,M,M*spin);iv=g**-1;density=mp.sqrt(-mp.det(g));h=mp.mpf('1e-5');B=mp.matrix(4,1);da=mp.matrix(3,1);db=mp.matrix(3)
  for k in range(3):
   values=[]
   for offset in [-2,-1,1,2]:
    shifted=list(y);shifted[k]+=offset*h;cov=geometry.conformal_mp(shifted,M,M*spin);inv=cov**-1
    values.append((mp.sqrt(-mp.det(cov))*inv,1/mp.sqrt(-inv[0,0]),mp.matrix([-inv[0,j+1]/inv[0,0] for j in range(3)])))
   derivative=lambda index:(values[0][index]-8*values[1][index]+8*values[2][index]-values[3][index])/(12*h)
   div=derivative(0)
   for j in range(4):B[j]+=div[k+1,j]/density
   da[k]=derivative(1);be=derivative(2)
   for j in range(3):db[k,j]=be[j]
  reference=np.array([float(iv[i,j]) for i in range(4) for j in range(4)]+[float(v) for v in B]+[float(da[i]) for i in range(3)]+[float(db[i,j]) for i in range(3) for j in range(3)])
  actual=np.array([float(row[f'g{i}{j}']) for i in range(4) for j in range(4)]+[float(row[f'b{i}']) for i in range(4)]+[float(row[f'alpha{i}']) for i in range(3)]+[float(row[f'beta{i}{j}']) for i in range(3) for j in range(3)])
  delta=abs(reference-actual)/(1+abs(reference));records.append(dict(M=float(M),spin=float(spin),rho=float(rho),max_scaled_error=float(delta.max()),maximum_column=int(delta.argmax()),passed=bool(delta.max()<1e-9)))
 result=dict(passed=bool(records) and all(r['passed'] for r in records),probes=len(records),maximum=max(r['max_scaled_error'] for r in records),tolerance=1e-9,method='80-digit value-only exact physical four-metric pullback, fourth-order differentiation with h=1e-5; physical inverse, contracted divergence, lapse and shift gradients. No reference curvature is substituted.',records=records,geometry_sha256=hashlib.sha256(a.geometry.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
