#!/usr/bin/env python3
"""Locate an existing late scalar field error; no evolution or admission."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scalar_evidence import snapshot
from reference_interpolation import lagrange6

p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise RuntimeError('Existing spatial diagnostic is immutable')
r=json.loads(a.review.read_text());profile=np.genfromtxt(r['oracle']['waveform']+'-profile-100.csv',delimiter=',',names=True);records=[]
for run,n in zip(map(Path,r['runs']),r['resolutions']):
 data=snapshot(run,100);rho=np.linalg.norm(data[:,:3],axis=1);angular=data[:,0]*data[:,1]/rho**2;error=data[:,4]-angular*lagrange6(profile['rho'],profile['psi'],rho);C=data[:,6:9]-data[:,9:12];mask=rho>=1.2;weight=data[:,3];total=float(np.sum(weight[mask]*error[mask]**2));bins=[]
 for lo,hi in zip([1.2,3,6,8,10,12,16,20,23],[3,6,8,10,12,16,20,23,24.00000001]):
  m=(rho>=lo)&(rho<hi);indices=np.flatnonzero(m);i=indices[np.argmax(abs(error[m]))];energy=float(np.sum(weight[m]*error[m]**2));projection=float(np.sum(weight[m]*angular[m]*error[m])/np.sum(weight[m]*angular[m]**2));bins.append(dict(rho=[lo,hi],squared_L2_fraction=energy/total,max_error=float(abs(error[i])),maximum_xyz=data[i,:3].tolist(),spherical_l2_radial_bias=projection,C_volume_L2=float(np.sqrt(np.sum(weight[m]*np.sum(C[m]**2,axis=1))/np.sum(weight[m]))),C_Linf=float(np.linalg.norm(C[m],axis=1).max())))
 records.append(dict(n=n,bins=bins))
a.output.write_text(json.dumps(dict(records=records,primary_review_sha256=hashlib.sha256(a.review.read_bytes()).hexdigest(),source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Read-only existing t100 exterior error decomposition using the same1200-point reference and validated local Lagrange6 values. Diagnose spatial location before any additional3D run; not admission.'),indent=2)+'\n');print(json.dumps([(row['n'],[(b['rho'],b['squared_L2_fraction']) for b in row['bins'] if b['squared_L2_fraction']>.05]) for row in records],indent=2))
