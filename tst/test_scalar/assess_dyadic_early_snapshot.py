#!/usr/bin/env python3
"""Diagnose a verified closed N96 snapshot; never admit an active run."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from reference_interpolation import lagrange6
from validate_schwarzschild import compare,norms
from scalar_evidence import snapshot
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def truth(data,profile):
 rho=np.linalg.norm(data[:,:3],axis=1);n=data[:,:3]/rho[:,None];Y=n[:,0]*n[:,1];P=lagrange6(profile['rho'],profile['psi'],rho);V=lagrange6(profile['rho'],profile['v'],rho);Pr=lagrange6(profile['rho'],profile['Dpsi'],rho);angular=-2*Y[:,None]*n;angular[:,0]+=n[:,1];angular[:,1]+=n[:,0]
 return dict(psi=Y*P,v=Y*V,Dpsi=angular*P[:,None]/rho[:,None]+Y[:,None]*n*Pr[:,None])
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--reference-n',type=int,default=2400);p.add_argument('--previous-n',type=int,default=1200);a=p.parse_args();E=a.evidence.resolve();P=E/'cpu-schwarzschild-dyadic/closed-t2-prefix';output=E/'cpu-schwarzschild-dyadic'/('early-t2-review.json' if a.reference_n==2400 else f'early-t2-reference{a.reference_n}-review.json')
if output.exists():raise RuntimeError('Existing early diagnostic is immutable')
receipt=json.loads((P/'prefix-collection-receipt.json').read_text());plan=json.loads((E/'aurora/cpu-schwarzschild-dyadic-n96-v1/cpu-plan.json').read_text());inv=json.loads((P/'invocation.json').read_text())
if not receipt['passed'] or inv['binary']!=plan['binary_sha256'] or any(sha(P/name)!=digest for name,digest in receipt['local_files'].items()):raise RuntimeError('Closed native snapshot/source binding failed')
O=E/'radial-oracle/smooth';files=[O/f'waveform-upwind-n{n}.csv-profile-2.csv' for n in [a.previous_n,a.reference_n]];profiles=[np.genfromtxt(f,delimiter=',',names=True) for f in files];base=E/'aurora/cpu-schwarzschild-commensurate-v1-collected/result';runs=[base/'schwarzschild-n24-fixed-cube',base/'schwarzschild-n48-fixed-cube',P];errors=[]
for run in runs:
 data=snapshot(run,2);data=data[np.linalg.norm(data[:,:3],axis=1)>=1.2];errors.append(compare(data,profiles[1],sampler='lagrange6'))
checks=[]
for field in ['psi','v','Dpsi','reduction']:
 for norm in ['L2','Linf']:
  e=[r[field][norm] for r in errors];orders=[math.log(e[k]/e[k+1])/math.log(2) for k in [0,1]];checks.append(dict(field=field,norm=norm,errors=e,orders=orders,passed=e[0]>e[1]>e[2] and min(orders)>=1.5))
exact=[truth(data,profile) for profile in profiles];uncertainty=[]
for field in ['psi','v','Dpsi']:
 d=norms(exact[0][field]-exact[1][field],data[:,3])
 for norm in ['L2','Linf']:
  fraction=d[norm]/errors[-1][field][norm];uncertainty.append(dict(field=field,norm=norm,reference_difference=d[norm],primary_error=errors[-1][field][norm],fraction=fraction,passed=fraction<=.1))
result=dict(admission=False,native_run_complete=False,time=2,resolutions=[24,48,96],reference_resolutions=[a.previous_n,a.reference_n],checks=checks,reference_uncertainty=uncertainty,field_checks_passed=all(c['passed'] for c in checks),reference_uncertainty_passed=all(c['passed'] for c in uncertainty),input_sha256={str(f):sha(f) for f in [*files,P/'prefix-collection-receipt.json']},interpolator_sha256=sha(Path(__file__).with_name('reference_interpolation.py')),source_sha256=sha(Path(__file__)),scope='Verified closed t2 snapshot only. No duplicate evolution and no end-of-run or stage admission. Use the unchanged10% reference-uncertainty criterion to decide whether only the independent oracle must be refined while N96 continues.')
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(field_checks_passed=result['field_checks_passed'],reference_uncertainty_passed=result['reference_uncertainty_passed'],max_uncertainty_fraction=max(c['fraction'] for c in uncertainty),failed=[c for c in checks+uncertainty if not c['passed']]),indent=2))
