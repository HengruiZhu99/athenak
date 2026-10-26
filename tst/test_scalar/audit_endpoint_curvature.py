#!/usr/bin/env python3
"""Measure actual conformal physical curvature near the problematic endpoint."""
import argparse,csv,hashlib,json
from pathlib import Path
import mpmath as mp
import numpy as np
import audit_geometry as geometry

def curvature(y,M,a,h):
 cov=geometry.conformal_mp(y,M,a);g=np.array(cov.tolist(),dtype=float);iv=np.linalg.inv(g);dg=np.zeros((4,4,4));dd=np.zeros((4,4,4,4))
 def shifted(i,si,j=None,sj=0):
  z=list(y);z[i]+=si*h
  if j is not None:z[j]+=sj*h
  return geometry.conformal_mp(z,M,a)
 for i in range(3):
  gp=shifted(i,1);gm=shifted(i,-1);dg[i+1]=np.array(((gp-gm)/(2*h)).tolist(),dtype=float);dd[i+1,i+1]=np.array(((gp-2*cov+gm)/(h*h)).tolist(),dtype=float)
  for j in range(i):dd[i+1,j+1]=dd[j+1,i+1]=np.array(((shifted(i,1,j,1)-shifted(i,1,j,-1)-shifted(i,-1,j,1)+shifted(i,-1,j,-1))/(4*h*h)).tolist(),dtype=float)
 connection=np.zeros((4,4,4));derivative=np.zeros((4,4,4,4))
 for u in range(4):
  for v in range(4):
   for w in range(4):
    for z in range(4):connection[u,v,w]+=.5*iv[u,z]*(dg[v,z,w]+dg[w,z,v]-dg[z,v,w])
 for k in range(4):
  div=-iv@dg[k]@iv
  for u in range(4):
   for v in range(4):
    for w in range(4):
     for z in range(4):derivative[k,u,v,w]+=.5*(div[u,z]*(dg[v,z,w]+dg[w,z,v]-dg[z,v,w])+iv[u,z]*(dd[k,v,z,w]+dd[k,w,z,v]-dd[k,z,v,w]))
 ricci=np.zeros((4,4))
 for i in range(4):
  for j in range(4):
   for k in range(4):
    ricci[i,j]+=derivative[k,k,i,j]-derivative[j,k,i,k]
    for l in range(4):ricci[i,j]+=connection[k,k,l]*connection[l,i,j]-connection[k,j,l]*connection[l,i,k]
 return float(np.sum(iv*ricci))/6

def main():
 p=argparse.ArgumentParser();p.add_argument('geometry',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise RuntimeError('Existing curvature review is immutable')
 mp.mp.dps=80;geometry.SMOOTH_LAYER=True;records=[]
 for row in csv.DictReader(a.geometry.open()):
  y=[mp.mpf(row[k]) for k in ['x','y','z']];rho=mp.sqrt(mp.fdot(y,y))
  if abs(float(rho)-15.999872713026816)>1e-9:continue
  M=mp.mpf(row['mass']);spin=mp.mpf(row['spin']);target=float(row['potential']);steps=[mp.mpf('0.0004'),mp.mpf('0.0002'),mp.mpf('0.0001')];values=[curvature(y,M,M*spin,h) for h in steps];extrapolated=(4*values[-1]-values[-2])/3;error=abs(extrapolated-target)/(1+abs(target))
  records.append(dict(M=float(M),spin=float(spin),rho=float(rho),physical_potential=target,curvature_over_six=values,extrapolated=extrapolated,scaled_error=error,passed=error<1e-9 and abs(values[-1]-values[-2])<1e-8))
 result=dict(passed=bool(records) and all(r['passed'] for r in records),probes=len(records),maximum=max(r['scaled_error'] for r in records),records=records,method='Actual conformal physical four-metric from value-only 80-digit pullback; connection and Ricci contractions with three finite-difference steps and Richardson extrapolation. No reference curvature.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
