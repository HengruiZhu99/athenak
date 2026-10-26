#!/usr/bin/env python3
"""Validate the independent complex Kerr scalar after Schwarzschild admission."""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path
import numpy as np
from numpy.polynomial import legendre as leg
from scipy.interpolate import PchipInterpolator
from reference_interpolation import lagrange6

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())
def sampled(profile,radius,z):
 rg=np.unique(profile['rho']);zg=np.unique(profile['z']);nr,na=len(rg),len(zg);inverse=np.linalg.inv(leg.legvander(zg,na-1));angular=leg.legvander(z,na-1)@inverse;derivative=np.column_stack([leg.legval(z,leg.legder(np.eye(na)[:,k])) for k in range(na)])@inverse;result={}
 for key,re,im in [('W','W_real','W_imag'),('v','v_real','v_imag'),('Dr','Dr_real','Dr_imag')]:
  values=(profile[re]+1j*profile[im]).reshape(nr,na);v=lagrange6(rg,values,radius);result[key]=np.sum(v*angular,axis=1)
  if key=='W':result['Dz']=np.sum(v*derivative,axis=1)
 return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();E=a.evidence.resolve();R=E/'kerr-oracle';scripts=Path(__file__).resolve().parent;previous=E/'aurora/schwarzschild-stable-gpu-v1-review.json';output=R/'stable-reference-review.json'
 if output.exists():raise RuntimeError('Existing independent Kerr review is immutable')
 while not previous.exists():time.sleep(15)
 if not read(previous)['passed'] or not read(E/'schwarzschild-stable-sbp-cpu-admission.json')['passed']:raise RuntimeError('Schwarzschild CPU/GPU gates required before Kerr evolution')
 instrument=read(E/'kerr-reference-sampling-instrument-v2.json')
 if not instrument['passed'] or instrument['interpolator_sha256']!=sha(scripts/'reference_interpolation.py') or any(instrument['validator_sha256'][name]!=sha(scripts/name) for name in ['validate_kerr.py','assess_ringdown_rhs.py']):raise RuntimeError('Independent Kerr reference sampling instrument/source mismatch')
 exe=R/'kerr_oracle-checkpoint-v2-clang';source=R/'kerr_oracle-checkpoint-v2.cpp'
 preflight=read(R/'spin0-checkpoint-smoke-v2-clang-review.json')
 if sha(source)!=sha(scripts/'kerr_oracle.cpp') or not preflight['passed'] or sha(exe)!=preflight['binary_sha256'] or sha(source)!=preflight['source_sha256']:raise RuntimeError('Preserved independent oracle source/binary/checkpoint preflight mismatch')
 cases=[('n600-a8','coefficients-spin05-n600-a8.txt',None),('n900-a12','coefficients-spin05-n900-a12.txt',None),('n900-a12-inner02','coefficients-spin05-n900-a12-inner02.txt',None),('n900-a12-halfdt','coefficients-spin05-n900-a12.txt','.5')];waves=[];receipts=[]
 for label,table_name,factor in cases:
  table=R/table_name;meta=read(table.with_suffix('.json'))
  if sha(table)!=meta['table_sha256'] or not meta['smooth_physical_layer'] or meta['spin']!=.5:raise RuntimeError('Independent physical coefficient table mismatch')
  wave=R/('waveform-spin05-stable-'+label+'.csv');inv=Path(str(wave)+'.invocation.json');receipt=Path(str(wave)+'.receipt.json');command=[str(exe),str(table),str(wave),'100']+([factor] if factor else []);identity=dict(command=command,binary_sha256=sha(exe),table_sha256=sha(table),source_sha256=sha(source))
  if wave.exists() or inv.exists():raise RuntimeError('Existing Kerr attempt must be inspected; no duplicate evolution')
  inv.write_text(json.dumps(identity,indent=2)+'\n');start=time.time();print('INDEPENDENT KERR',label,flush=True)
  with Path(str(wave)+'.log').open('w') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
  files={str(p.resolve()):sha(p) for p in R.glob(wave.name+'*') if p.is_file()};record=dict(**identity,returncode=r.returncode,elapsed_seconds=time.time()-start,files=files);receipt.write_text(json.dumps(record,indent=2)+'\n')
  if r.returncode:raise RuntimeError('Independent Kerr attempt failed; checkpoint/output preserved')
  checkpoint=Path(str(wave)+'-checkpoint.c128');cm=read(Path(str(wave)+'-checkpoint.json'));values=np.fromfile(checkpoint,dtype=np.complex128)
  if cm['time']!=100 or len(values)!=3*cm['nr']*cm['na'] or not np.isfinite(values).all():raise RuntimeError('Independent full complex checkpoint is incomplete')
  waves.append(wave);receipts.append(receipt)
 baseline=np.genfromtxt(waves[1],delimiter=',',names=True);z=baseline['real22']+1j*baseline['imag22'];wave_checks=[]
 for wave,label in zip([waves[0],waves[2],waves[3]],['resolution','inner','halfdt']):
  w=np.genfromtxt(wave,delimiter=',',names=True)
  if not np.array_equal(w['time'],baseline['time']):raise RuntimeError('Independent control times differ')
  error=float(np.linalg.norm(w['real22']+1j*w['imag22']-z)/np.linalg.norm(z));wave_checks.append(dict(control=label,relative_L2=error,passed=error<=.001))
 profile_checks=[]
 for t in [2,6,12,35,65,100]:
  coarse=np.genfromtxt(str(waves[0])+f'-profile-{t}.csv',delimiter=',',names=True);fine=np.genfromtxt(str(waves[1])+f'-profile-{t}.csv',delimiter=',',names=True);mask=fine['rho']>=1.2;radius=fine['rho'][mask];z=fine['z'][mask];x=sampled(coarse,radius,z);y=sampled(fine,radius,z)
  for field in x:
   error=x[field]-y[field];l2=float(np.linalg.norm(error)/np.linalg.norm(y[field]));linf=float(np.max(abs(error))/np.max(abs(y[field])));profile_checks.append(dict(time=t,field=field,relative_L2=l2,relative_Linf=linf,passed=l2<=.001 and linf<=.001))
 qnm=R/'stable-qnm-review.json';subprocess.run([sys.executable,str(scripts/'fit_ringdown.py'),str(waves[1]),'--spin','.5','--expected',str(E/'spin-zero-qnm-both-branches.json'),'--output',str(qnm)],check=True)
 result=dict(stage='kerr-independent',passed=all(c['passed'] for c in wave_checks+profile_checks) and read(qnm)['passed'],waveform=str(waves[1].resolve()),wave_checks=wave_checks,profile_checks=profile_checks,QNM_review=str(qnm),QNM_review_sha256=sha(qnm),previous_GPU_review_sha256=sha(previous),receipts={str(p.resolve()):sha(p) for p in receipts},binary_sha256=sha(exe),source_sha256=sha(source),validator_sha256=sha(Path(__file__)),scope='Independent regular complex m2 scalar on exact physical Kerr trumpet; resolution, inner-boundary, timestep, full complex checkpoints, field/radial/angular derivative profiles and free two-branch spin-zero QNMs. No primary provider/RHS or reference curvature.')
 result['reference_sampler']='lagrange6';result['interpolator_sha256']=sha(scripts/'reference_interpolation.py')
 result['sampling_instrument_sha256']=sha(E/'kerr-reference-sampling-instrument-v2.json');result['sampling_instrument']=str(E/'kerr-reference-sampling-instrument-v2.json');result['profile_resolution_waveforms']=[str(waves[0].resolve()),str(waves[1].resolve())]
 output.write_text(json.dumps(result,indent=2)+'\n');print('INDEPENDENT KERR REVIEW',result['passed'],flush=True)
 if not result['passed']:raise SystemExit(1)
if __name__=='__main__':main()
