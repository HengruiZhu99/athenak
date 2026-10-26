#!/usr/bin/env python3
"""Bound independent Kerr oracle uncertainty at actual finest3D points."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from validate_kerr import reconstruct,compare
from validate_schwarzschild import norms
from scalar_evidence import snapshot
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
p=argparse.ArgumentParser();p.add_argument('--primary-review',type=Path,required=True);p.add_argument('--reference-review',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise RuntimeError('Existing uncertainty certificate is immutable')
primary=json.loads(a.primary_review.read_text());reference=json.loads(a.reference_review.read_text());waves=reference['profile_resolution_waveforms'];run=Path(primary['runs'][-1]);inputs=[a.primary_review,a.reference_review];checks=[]
if primary['stage']!='kerr' or not reference['passed'] or len(waves)!=2 or waves[-1]!=primary['oracle']['waveform'] or primary['reference_sampler']!=reference['reference_sampler'] or primary['interpolator_sha256']!=reference['interpolator_sha256']:raise RuntimeError('Kerr primary/reference sampler or waveform binding failed')
for t in [2,6,12,100]:
 native=snapshot(run,t);native=native[np.linalg.norm(native[:,:3],axis=1)>=1.2];truth=[]
 for wave in waves:
  file=Path(wave+f'-profile-{t}.csv');inputs.append(file);profile=np.genfromtxt(file,delimiter=',',names=True);d,exact=reconstruct(native,profile,'lagrange6');truth.append(exact)
 errors=compare(native,profile,'lagrange6')
 for field,columns in [('psi',0),('v',1),('Dpsi',slice(2,5))]:
  delta=norms(truth[0][:,columns]-truth[1][:,columns],native[:,3])
  for norm in ['L2','Linf']:
   fraction=delta[norm]/errors[field][norm];checks.append(dict(time=t,field=field,norm=norm,reference_difference=delta[norm],finest_primary_error=errors[field][norm],fraction=fraction,passed=fraction<=.1))
result=dict(stage='kerr-reference-uncertainty',passed=all(c['passed'] for c in checks),admission=False,primary_review_sha256=sha(a.primary_review),reference_review_sha256=sha(a.reference_review),waveforms=waves,checks=checks,input_sha256={str(file.resolve()):sha(file) for file in inputs},interpolator_sha256=primary['interpolator_sha256'],reconstruction_source_sha256=sha(Path(__file__).with_name('validate_kerr.py')),source_sha256=sha(Path(__file__)),scope='Both norms for field, coordinate velocity and independently reconstructed Cartesian gradients at2/6/12/100 on actual finest3D points. Require independent coarse/fine oracle difference<=10% of each measured primary error. Preserve failed certificates and refine the oracle without duplicating3D trajectories.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(passed=result['passed'],maximum_fraction=max(c['fraction'] for c in checks),failed=[c for c in checks if not c['passed']]),indent=2))
if not result['passed']:raise SystemExit(1)
