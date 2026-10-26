#!/usr/bin/env python3
"""Compare Cartesian stationary normal-scalar sources to an independent radial chart."""
import argparse,csv,json,hashlib
from pathlib import Path
import numpy as np
import mpmath as mp
import normal_coefficients as normal
import radial_coefficients as radial

def main():
 p=argparse.ArgumentParser();p.add_argument('geometry',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--smooth-layer',action='store_true');a=p.parse_args()
 if a.output.exists():raise RuntimeError('Existing source audit is immutable')
 radial.SMOOTH_LAYER=a.smooth_layer;mp.mp.dps=60;records=[]
 for r in csv.DictReader(a.geometry.open()):
  if float(r['mass'])!=1 or float(r['spin'])!=0:continue
  x=np.array([float(r[k]) for k in ['x','y','z']]);rho=np.linalg.norm(x)
  if rho>=24-1e-9:continue
  n=x/rho;g=np.array([float(r[f'g{i}{j}']) for i in range(4) for j in range(4)]).reshape(4,4);B=np.array([float(r[f'b{i}']) for i in range(4)]);ap=np.array([float(r[f'alpha{i}']) for i in range(3)]);bp=np.array([[float(r[f'beta{i}{j}']) for j in range(3)] for i in range(3)]);alpha=1/np.sqrt(-g[0,0]);beta=-g[0,1:]/g[0,0];gamma=g[1:,1:]-np.outer(g[0,1:],g[0,1:])/g[0,0]
  sourcePi=np.dot(beta,ap)/alpha+alpha*alpha*B[0];sourceQ=-np.einsum('i,ij->j',beta,bp)/alpha-alpha*B[0]*beta-alpha*B[1:];primaryRadial=np.dot(n,sourceQ)-alpha*(np.trace(gamma)-n@gamma@n)/rho
  rr=mp.mpf(str(rho));al,b,c,J,V,L=normal.normal(rr);ar=normal.derivative(lambda z:normal.normal(z)[0],rr);aK=normal.derivative(lambda z:normal.normal(z)[3]*normal.normal(z)[1],rr)/J;Gamma=-normal.derivative(lambda z:normal.normal(z)[3]*normal.normal(z)[2],rr)/J;expected=np.array([float(aK),float(al*Gamma-c*ar)]);error=float(np.max(abs(np.array([sourcePi,primaryRadial])-expected))/(1+np.max(abs(expected))));records.append(dict(rho=float(rho),relative_error=error))
 maximum=max(r['relative_error'] for r in records);result=dict(probes=len(records),max_relative_error=maximum,tolerance=1e-9,passed=maximum<1e-9,records=records,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Cartesian normal scalar source transformed to radial form vs independent high-precision covariant two-metric solver; includes the Cartesian radial-vector angular derivative term.')
 a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='records'},indent=2))
if __name__=='__main__':main()
