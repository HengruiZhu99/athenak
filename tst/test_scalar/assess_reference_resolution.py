#!/usr/bin/env python3
"""Measure oracle uncertainty at actual finest3D points before reusing data."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from scalar_evidence import snapshot
from reference_interpolation import lagrange6
from validate_schwarzschild import norms

def truth(data,profile):
 xyz=data[:,:3];rho=np.linalg.norm(xyz,axis=1);n=xyz/rho[:,None];Y=n[:,0]*n[:,1];P=lagrange6(profile['rho'],profile['psi'],rho);V=lagrange6(profile['rho'],profile['v'],rho);Pr=lagrange6(profile['rho'],profile['Dpsi'],rho);angular=-2*Y[:,None]*n;angular[:,0]+=n[:,1];angular[:,1]+=n[:,0]
 return dict(psi=Y*P,v=Y*V,Dpsi=angular*P[:,None]/rho[:,None]+Y[:,None]*n*Pr[:,None])
p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);p.add_argument('--refined-waveform',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise RuntimeError('Existing reference assessment is immutable')
r=json.loads(a.review.read_text());ns=r['resolutions'];runs=[Path(d) for d in r['runs']];old=Path(r['oracle']['waveform']);checks=[];uncertainty=[];records=[];inputs=[a.review,a.refined_waveform]
for t in [2,6,12,100]:
 oldfile=Path(str(old)+f'-profile-{t}.csv');newfile=Path(str(a.refined_waveform)+f'-profile-{t}.csv');inputs.extend([oldfile,newfile]);coarse=np.genfromtxt(oldfile,delimiter=',',names=True);fine=np.genfromtxt(newfile,delimiter=',',names=True);errors=[]
 for run in runs:
  all_data=snapshot(run,t);data=all_data[np.linalg.norm(all_data[:,:3],axis=1)>=1.2];exact=truth(data,fine);e={field:norms(data[:,cols]-exact[field],data[:,3]) for field,cols in [('psi',4),('v',5),('Dpsi',slice(9,12))]};e['reduction']=norms(data[:,6:9]-data[:,9:12],data[:,3]);errors.append(e)
 records.append(dict(time=t,errors=errors))
 for field in ['psi','v','Dpsi','reduction']:
  for norm in ['L2','Linf']:
   values=[e[field][norm] for e in errors];orders=[math.log(values[k]/values[k+1])/math.log(ns[k+1]/ns[k]) for k in [0,1]];checks.append(dict(time=t,field=field,norm=norm,errors=values,orders=orders,passed=values[0]>values[1]>values[2] and min(orders)>=1.5))
 previous=truth(data,coarse)
 for field in ['psi','v','Dpsi']:
  delta=norms(exact[field]-previous[field],data[:,3])
  for norm in ['L2','Linf']:
   fraction=delta[norm]/errors[-1][field][norm];uncertainty.append(dict(time=t,field=field,norm=norm,reference_difference=delta[norm],finest_primary_error=errors[-1][field][norm],fraction=fraction,passed=fraction<=.1))
x=np.genfromtxt(old,delimiter=',',names=True);y=np.genfromtxt(a.refined_waveform,delimiter=',',names=True)
if not np.array_equal(x['time'],y['time']):raise RuntimeError('Independent reference waveform times differ')
wave_difference=float(np.linalg.norm(x['radial_psi']-y['radial_psi'])/np.linalg.norm(y['radial_psi']));result=dict(admission=False,field_checks_passed=all(c['passed'] for c in checks),reference_uncertainty_passed=all(c['passed'] for c in uncertainty),waveform_difference=wave_difference,waveform_difference_passed=wave_difference<=.001,checks=checks,uncertainty=uncertainty,records=records,input_sha256={str(f.resolve()):hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs},source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),interpolator_sha256=hashlib.sha256(Path(__file__).with_name('reference_interpolation.py').read_bytes()).hexdigest(),scope='Same completed3D data, higher-order reference sampling and newly resolved independent radial solution. Require reference difference<=10% of actual finest3D error in both norms and all four times. Original scientific thresholds remain unchanged; controls and full source/provenance admission remain outstanding.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(field_checks_passed=result['field_checks_passed'],reference_uncertainty_passed=result['reference_uncertainty_passed'],failed_fields=[c for c in checks if not c['passed']],failed_uncertainty=[c for c in uncertainty if not c['passed']],waveform_difference=wave_difference),indent=2))
