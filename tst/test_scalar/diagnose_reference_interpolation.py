#!/usr/bin/env python3
"""Read-only PCHIP versus high-order oracle sampling; never issues admission."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
import validate_schwarzschild as validation
from scalar_evidence import snapshot
from reference_interpolation import lagrange6

p=argparse.ArgumentParser();p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise RuntimeError('Existing diagnostic is immutable')
r=json.loads(a.review.read_text());runs=[Path(d) for d in r['runs']];ns=r['resolutions'];oracle=Path(r['oracle']['waveform']);original=validation.PchipInterpolator;instrument=[];checks=[];locations=[]
query=np.linspace(1.2,24,24001);scaled=(query-12)/12
poly=lambda z:1+.2*z-.3*z*z+.15*z**3-.1*z**4+.07*z**5
x=np.linspace(.04,24,1200);values=np.c_[poly((x-12)/12),2*poly((x-12)/12)+1j*poly((x-12)/12)];target=np.c_[poly(scaled),2*poly(scaled)+1j*poly(scaled)];polynomial_error=float(np.max(abs(lagrange6(x,values,query)-target)));instrument.append(dict(test='degree5 real/complex columns',error=polynomial_error,tolerance=2e-14,passed=polynomial_error<=2e-14))
function=lambda x:np.exp(-((x-6.137)/1.2)**2)*np.cos(.671*x);errors=[]
for n in [600,900,1200]:
 x=np.linspace(.04,24,n);d=lagrange6(x,function(x),query)-function(query);errors.append(dict(n=n,L2=float(np.sqrt(np.mean(d*d))),Linf=float(np.max(abs(d)))))
orders={key:[math.log(errors[k][key]/errors[k+1][key])/math.log(errors[k+1]['n']/errors[k]['n']) for k in [0,1]] for key in ['L2','Linf']};instrument.append(dict(test='smooth non-QNM analytic function',errors=errors,orders=orders,minimum_required_order=5,passed=all(min(o)>=5 for o in orders.values())))
for t in [2,6,12,100]:
 profile=np.genfromtxt(str(oracle)+f'-profile-{t}.csv',delimiter=',',names=True);arrays=[snapshot(d,t) for d in runs]
 for method in ['PCHIP','Lagrange6']:
  validation.PchipInterpolator=original if method=='PCHIP' else lambda x,y:lambda q:lagrange6(x,y,q)
  e=[validation.compare(data,profile) for data in arrays]
  for field in ['psi','v','Dpsi','reduction']:
   for norm in ['L2','Linf']:
    values=[row[field][norm] for row in e];orders=[math.log(values[k]/values[k+1])/math.log(ns[k+1]/ns[k]) for k in [0,1]];checks.append(dict(time=t,method=method,field=field,norm=norm,errors=values,orders=orders,passed=values[0]>values[1]>values[2] and min(orders)>=1.5))
  if t==2:
   data=arrays[-1];rho=np.linalg.norm(data[:,:3],axis=1);mask=rho>=1.2;data=data[mask];rho=rho[mask];angular=data[:,0]*data[:,1]/rho**2
   for field,col,key in [('psi',4,'psi'),('v',5,'v')]:
    exact=angular*validation.PchipInterpolator(profile['rho'],profile[key])(rho);error=data[:,col]-exact;i=np.argmax(abs(error));j=np.clip(np.searchsorted(profile['rho'],rho[i])-1,0,len(profile)-2);locations.append(dict(method=method,field=field,xyz=data[i,:3].tolist(),rho=float(rho[i]),error=float(error[i]),primary=float(data[i,col]),sampled_reference=float(exact[i]),reference_bracketing_nodes=[float(profile['rho'][j]),float(profile['rho'][j+1])],reference_bracketing_values=[float(profile[key][j]),float(profile[key][j+1])]))
validation.PchipInterpolator=original
result=dict(admission=False,instrument=instrument,instrument_passed=all(row['passed'] for row in instrument),checks=checks,early_maximum_locations=locations,primary_review_sha256=hashlib.sha256(a.review.read_bytes()).hexdigest(),oracle_files=r['oracle']['files'],source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),interpolator_sha256=hashlib.sha256(Path(__file__).with_name('reference_interpolation.py').read_bytes()).hexdigest(),scope='Existing frozen3D trajectories and same independent1200-point profiles only. Additional oracle-resolution uncertainty check remains mandatory; unchanged1.5 criteria. No evolution or stage admission.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(instrument_passed=result['instrument_passed'],failed_Lagrange6_checks=[c for c in checks if c['method']=='Lagrange6' and not c['passed']],locations=locations),indent=2))
