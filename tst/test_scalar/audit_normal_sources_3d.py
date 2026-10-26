#!/usr/bin/env python3
"""Check all stationary scalar source components using a value-only 3+1 metric."""
import argparse,csv,hashlib,json
from pathlib import Path
import mpmath as mp
import numpy as np
import audit_geometry as geometry

def main():
 p=argparse.ArgumentParser();p.add_argument('geometry',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 if a.output.exists():raise RuntimeError('Existing audit is immutable')
 mp.mp.dps=60;geometry.SMOOTH_LAYER=True;records=[]
 for row in csv.DictReader(a.geometry.open()):
  M=float(row['mass']);spin=float(row['spin']);y=[mp.mpf(row[k]) for k in ['x','y','z']];rho=mp.sqrt(mp.fdot(y,y))
  if M!=1 or spin not in [0,.5,-.5,.9] or not any(abs(float(rho)-r)<1e-9 for r in [.1,1,2,12,23]):continue
  cov=geometry.conformal_mp(y,M,M*spin);iv=cov**-1;alpha=1/mp.sqrt(-iv[0,0]);beta=mp.matrix([-iv[0,i+1]/iv[0,0] for i in range(3)]);gamma=mp.matrix([[cov[i+1,j+1] for j in range(3)] for i in range(3)]);inverse=gamma**-1
  h=rho*mp.mpf('1e-6');da=mp.matrix(3,1);db=mp.matrix(3);dg=[]
  for k in range(3):
   vals=[]
   for offset in [-2,-1,1,2]:
    shifted=list(y);shifted[k]+=offset*h;g=geometry.conformal_mp(shifted,M,M*spin);i=g**-1
    vals.append((1/mp.sqrt(-i[0,0]),mp.matrix([-i[0,j+1]/i[0,0] for j in range(3)]),mp.matrix([[g[j+1,l+1] for l in range(3)] for j in range(3)])))
   derivative=lambda index:(vals[0][index]-8*vals[1][index]+8*vals[2][index]-vals[3][index])/(12*h)
   da[k]=derivative(0);b=derivative(1)
   for j in range(3):db[k,j]=b[j]
   dg.append(derivative(2))
  # K_ij=(L_beta gamma)_ij/(2 alpha), since d_t gamma=0.
  K=mp.mpf(0);connection=mp.matrix(3,1)
  for i in range(3):
   for j in range(3):
    K+=inverse[i,j]*sum(beta[k]*dg[k][i,j]+gamma[i,k]*db[j,k]+gamma[j,k]*db[i,k] for k in range(3))/(2*alpha)
    for aidx in range(3):connection[aidx]+=sum(inverse[aidx,l]*inverse[i,j]*(dg[i][j,l]+dg[j][i,l]-dg[l][i,j])/2 for l in range(3))
  expected=np.array([float(alpha*K)]+[float(alpha*connection[j]-sum(inverse[i,j]*da[i] for i in range(3))) for j in range(3)])
  g=np.array([[float(row[f'g{i}{j}']) for j in range(4)] for i in range(4)]);B=np.array([float(row[f'b{i}']) for i in range(4)]);ap=np.array([float(row[f'alpha{i}']) for i in range(3)]);bp=np.array([[float(row[f'beta{i}{j}']) for j in range(3)] for i in range(3)]);al=1/np.sqrt(-g[0,0]);be=-g[0,1:]/g[0,0];actual=np.r_[al*al*B[0]+be@ap/al,-al*(B[0]*be+B[1:])-be@bp/al];error=float(np.max(abs(actual-expected))/(1+np.max(abs(expected))))
  records.append(dict(spin=spin,rho=float(rho),y=[float(v) for v in y],relative_error=error))
 maximum=max(r['relative_error'] for r in records);result=dict(probes=len(records),max_relative_error=maximum,tolerance=1e-9,passed=maximum<1e-9,records=records,method='Independent value-only covariant physical four-metric pullback; spatial metric derivatives determine stationary K and all three contracted connections. Compare with the cached-B Cartesian normal scalar source algebra, including azimuthal Kerr terms.',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
