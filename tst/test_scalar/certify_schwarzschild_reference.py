#!/usr/bin/env python3
"""Bind the resolved independent radial oracle to its controls and sampler."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from reference_interpolation import lagrange6
from scalar_evidence import snapshot
from validate_schwarzschild import compare,norms
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--primary-review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--reference-n',type=int,default=2400);p.add_argument('--previous-n',type=int,default=1200);a=p.parse_args();E=a.evidence.resolve();O=E/'radial-oracle/smooth';primary=json.loads(a.primary_review.read_text());fine_label=f'n{a.reference_n}';previous_label=f'n{a.previous_n}';wave=O/f'waveform-upwind-{fine_label}.csv'
if a.output.exists():raise RuntimeError('Existing oracle certificate is immutable')
inputs=[a.primary_review];checks=[];uncertainty=[];profiles=[]
for name in ['normal-reference-review.json','fine-profile-review.json',f'qnm-{fine_label}-review.json','checkpoint-smoke-n1200-v2-review.json']:
 file=O/name;inputs.append(file);checks.append(dict(name=name,passed=json.loads(file.read_text())['passed']))
instrument=E/'reference-interpolation-diagnostic-v1.json';inputs.append(instrument);instrument_review=json.loads(instrument.read_text());checks.append(dict(name='six-point interpolation instrument',passed=instrument_review['instrument_passed'] and instrument_review['interpolator_sha256']==sha(Path(__file__).with_name('reference_interpolation.py'))))
receiptfile=Path(str(wave)+'.receipt.json');receipt=json.loads(receiptfile.read_text());inputs.append(receiptfile)
inputs += [O/name for name in receipt['files']]
inputs += [Path(receipt['command'][0]),Path(receipt['command'][1]),E/'radial-oracle/radial_upwind-checkpoint-v2.cpp']
checks.append(dict(name='resolved oracle source/binary/table/payload integrity',passed=receipt['returncode']==0 and receipt['complete'] and all(sha(O/name)==digest for name,digest in receipt['files'].items()) and sha(inputs[-3])==receipt['binary_sha256'] and sha(inputs[-2])==receipt['table_sha256'] and sha(inputs[-1])==receipt['source_sha256']))
previous_wave=O/f'waveform-upwind-{previous_label}.csv';inputs.append(previous_wave);old_wave=np.genfromtxt(previous_wave,delimiter=',',names=True);new_wave=np.genfromtxt(wave,delimiter=',',names=True)
if not np.array_equal(old_wave['time'],new_wave['time']):raise RuntimeError('Refined independent waveform times differ')
wave_difference=float(np.linalg.norm(old_wave['radial_psi']-new_wave['radial_psi'])/np.linalg.norm(new_wave['radial_psi']));checks.append(dict(name='new independent waveform resolution',relative_difference=wave_difference,passed=wave_difference<=.001))
for t in [2,6,12,100]:
 files={label:O/(f'waveform-upwind-{label}.csv-profile-{t}.csv') for label in ['n900','n900-inner02','n900-halfdt','n1200',previous_label,fine_label]};inputs.extend(files.values());data={label:np.genfromtxt(file,delimiter=',',names=True) for label,file in files.items()};query=data[fine_label]['rho'];mask=query>=1.2;query=query[mask]
 for label,base in [('n900','n1200'),('n900-inner02','n900'),('n900-halfdt','n900'),(previous_label,fine_label)]:
  for field in ['psi','v','Dpsi']:
   x=lagrange6(data[label]['rho'],data[label][field],query);y=lagrange6(data[base]['rho'],data[base][field],query);delta=norms(x-y,np.ones(len(query)));scale=norms(y,np.ones(len(query)))
   for norm in ['L2','Linf']:
    fraction=delta[norm]/max(scale[norm],1e-30);profiles.append(dict(time=t,comparison=[label,base],field=field,norm=norm,relative_difference=fraction,passed=fraction<=.001))
 native=snapshot(Path(primary['runs'][-1]),t);native=native[np.linalg.norm(native[:,:3],axis=1)>=1.2];rho=np.linalg.norm(native[:,:3],axis=1);n=native[:,:3]/rho[:,None];Y=n[:,0]*n[:,1];angular=-2*Y[:,None]*n;angular[:,0]+=n[:,1];angular[:,1]+=n[:,0];truth=[]
 for label in [previous_label,fine_label]:
  ref=data[label];P=lagrange6(ref['rho'],ref['psi'],rho);V=lagrange6(ref['rho'],ref['v'],rho);Pr=lagrange6(ref['rho'],ref['Dpsi'],rho);truth.append(dict(psi=Y*P,v=Y*V,Dpsi=angular*P[:,None]/rho[:,None]+Y[:,None]*n*Pr[:,None]))
 errors=compare(native,data[fine_label],sampler='lagrange6')
 for field in ['psi','v','Dpsi']:
  delta=norms(truth[0][field]-truth[1][field],native[:,3])
  for norm in ['L2','Linf']:
   fraction=delta[norm]/errors[field][norm];uncertainty.append(dict(time=t,field=field,norm=norm,reference_difference=delta[norm],finest_primary_error=errors[field][norm],fraction=fraction,passed=fraction<=.1))
result=dict(kind='independent_schwarzschild_oracle',passed=all(c['passed'] for c in checks+profiles+uncertainty),waveform=str(wave),normal_coefficients=str(O/f'normal-coefficients-{fine_label}.txt'),reference_resolutions=[a.previous_n,a.reference_n],reference_sampler='lagrange6',interpolator_sha256=sha(Path(__file__).with_name('reference_interpolation.py')),primary_review_sha256=sha(a.primary_review),checks=checks,profile_controls=profiles,uncertainty=uncertainty,input_sha256={str(file.resolve()):sha(file) for file in inputs},source_sha256=sha(Path(__file__)),scope='Resolved independent scalar radial solution, existing waveform/inner/timestep controls plus profile controls, free spin-zero QNM/window checks, output-only checkpoint verification, verified six-point interpolation, and reference uncertainty<=10% of actual finest3D errors. This certificate does not admit the primary3D stage.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],failed=[c for c in checks+profiles+uncertainty if not c['passed']],max_reference_uncertainty=max(c['fraction'] for c in uncertainty)),indent=2))
if not result['passed']:raise SystemExit(1)
