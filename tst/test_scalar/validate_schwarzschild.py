#!/usr/bin/env python3
"""Common-exterior scalar/derivative errors against independent radial profiles."""
import argparse,csv,json,math,hashlib
from pathlib import Path
import numpy as np
from scipy.interpolate import PchipInterpolator
from fit_ringdown import fit
from scalar_evidence import snapshot as load,waveform,history
from reference_interpolation import lagrange6

def norms(x,w):
 if x.ndim>1:x=np.linalg.norm(x,axis=1)
 return {'L2':float(np.sqrt(np.sum(w*x*x)/np.sum(w))),'Linf':float(np.max(abs(x)))}
def compare(data,profile,corrected=False,coefficients=None,sampler='pchip'):
 xyz=data[:,:3];rho=np.linalg.norm(xyz,axis=1);n=xyz/rho[:,None];mask=rho>=1.2;w=data[mask,3];n=n[mask];rho=rho[mask];a=data[mask]
 interpolate=lambda values:lagrange6(profile['rho'],values,rho) if sampler=='lagrange6' else PchipInterpolator(profile['rho'],values)(rho)
 P=interpolate(profile['psi']);Vp=interpolate(profile['v']);Pr=interpolate(profile['Dpsi'])
 angular=n[:,0]*n[:,1];gangular=-2*angular[:,None]*n;gangular[:,0]+=n[:,1];gangular[:,1]+=n[:,0]
 exactpsi=angular*P;exactv=angular*Vp;exactgrad=gangular*P[:,None]/rho[:,None]+angular[:,None]*n*Pr[:,None]
 v=a[:,5].copy()
 if corrected:
  beta=PchipInterpolator(coefficients[:,0],coefficients[:,1])(rho)
  v-=beta*np.sum(n*(a[:,6:9]-a[:,9:12]),axis=1)
 return dict(psi=norms(a[:,4]-exactpsi,w),v=norms(v-exactv,w),Dpsi=norms(a[:,9:12]-exactgrad,w),q=norms(a[:,6:9]-exactgrad,w),reduction=norms(a[:,6:9]-a[:,9:12],w))
def main():
 p=argparse.ArgumentParser();p.add_argument('--runs',nargs=3,type=Path,required=True);p.add_argument('--oracle',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--corrected-reduction',action='store_true');p.add_argument('--oracle-waveform',default='waveform-v3-n1200.csv');p.add_argument('--normal-coefficients',type=Path);p.add_argument('--expected',type=Path);p.add_argument('--reference-sampler',choices=['pchip','lagrange6'],default='pchip');a=p.parse_args();records=[];checks=[];oracle_files=[]
 if a.output.exists():raise RuntimeError('Existing review is immutable; select a new output')
 if a.normal_coefficients:
  table=np.loadtxt(a.normal_coefficients,skiprows=1);coeff=table[:,[0,2]]
 else:
  table=np.loadtxt(a.oracle/'coefficients-n1200.txt',skiprows=1);coeff=np.c_[table[:,0],-table[:,2]/table[:,1]]
 ns=[json.loads((d/'parameters.json').read_text())['n'] for d in a.runs]
 complete=all((d/'complete.txt').exists() for d in a.runs)
 for t in [2,6,12,100]:
  profilefile=a.oracle/f'{a.oracle_waveform}-profile-{t}.csv'
  if not profilefile.exists():continue
  try:arrays=[load(d,t) for d in a.runs]
  except FileNotFoundError:continue
  profile=np.genfromtxt(profilefile,delimiter=',',names=True)
  oracle_files.append(profilefile)
  errors=[compare(data,profile,a.corrected_reduction,coeff,a.reference_sampler) for data in arrays]
  records.append(dict(time=t,errors=errors))
  for fld in ['psi','v','Dpsi','reduction']:
   for nm in ['L2','Linf']:
    values=[e[fld][nm] for e in errors];orders=[math.log(values[k]/values[k+1])/math.log(ns[k+1]/ns[k]) for k in [0,1]];checks.append(dict(time=t,field=fld,norm=nm,errors=values,orders=orders,passed=values[0]>values[1]>values[2] and min(orders)>=1.5))
 fits=[];expected_path=a.expected or next(parent/'spin-zero-qnm.json' for parent in a.oracle.parents if (parent/'spin-zero-qnm.json').exists());expected=next(m for m in json.loads(expected_path.read_text())['modes'] if m['a_over_M']==0 and m['m']==2)
 for directory in a.runs:
  wave=waveform(directory);values=wave['imag22'].astype(complex);local=[]
  for lo,hi in [[35,65],[40,70],[45,75]]:
   mask=(wave['time']>=lo)&(wave['time']<=hi)
   if np.sum(mask)<40:continue
   f=fit(wave['time'][mask],values[mask]);f['window']=[lo,hi];f['real_error']=abs(f['omega_real']-expected['Momega_real'])/expected['Momega_real'];f['damping_error']=abs(f['decay']+expected['Momega_imag'])/(-expected['Momega_imag']);f['passed']=f['success'] and f['real_error']<=.02 and f['damping_error']<=.05;local.append(f)
  spreads={key:(max(r[key] for r in local)-min(r[key] for r in local))/np.mean([r[key] for r in local]) for key in ['omega_real','decay']} if len(local)==3 else {}
  fits.append(dict(run=str(directory),fits=local,window_spreads=spreads,passed=len(local)==3 and all(x['passed'] for x in local) and all(x<=.02 for x in spreads.values())))
 waveforms=[waveform(d) for d in a.runs];ref=np.genfromtxt(a.oracle/a.oracle_waveform,delimiter=',',names=True);N=np.sqrt(15/(32*np.pi));ct=-1+(np.arange(24)+.5)*2/24;angularfactor=-.5*N*np.pi*np.sum((1-ct*ct)**2)*2/24;wave_errors=[]
 for d,q in zip(a.runs,waveforms):
  if not np.array_equal(q['time'],ref['time'][:len(q)]):raise RuntimeError('Independent waveform times differ')
  length=min(len(q),len(ref));rr=angularfactor*ref['radial_psi'][:length];z=q['real22'][:length]+1j*q['imag22'][:length];e=float(np.linalg.norm(z-1j*rr)/np.linalg.norm(rr));wave_errors.append(dict(run=str(d),time=float(q['time'][length-1]),relative_L2=e,passed=e<=.02))
 reductions=[]
 for d in a.runs:
  init=load(d,0);mask=np.linalg.norm(init[:,:3],axis=1)>=1.2;scale=float(np.max(np.linalg.norm(init[mask,6:9],axis=1)));maxc=0
  for t in [2,6,12,100]:
   try:dat=load(d,t)
   except FileNotFoundError:continue
   mask=np.linalg.norm(dat[:,:3],axis=1)>=1.2;maxc=max(maxc,float(np.max(np.linalg.norm(dat[mask,6:9]-dat[mask,9:12],axis=1))))
  exterior=[float(r['reduction_Linf']) for key,r in history(d).items() if key[1]=='exterior']
  if exterior:maxc=max(maxc,max(exterior))
  reductions.append(dict(run=str(d),max_reduction=maxc,initial_scale=scale,normalized_max=maxc/scale,all_half_time_reports_checked=bool(exterior),passed=maxc/scale<=.02))
 oracle_files += [a.oracle/a.oracle_waveform,expected_path,a.normal_coefficients or a.oracle/'coefficients-n1200.txt'];oracle=dict(waveform=str((a.oracle/a.oracle_waveform).resolve()),normal_coefficients=str((a.normal_coefficients or a.oracle/'coefficients-n1200.txt').resolve()),files={str(path.resolve()):hashlib.sha256(path.read_bytes()).hexdigest() for path in oracle_files})
 result=dict(stage='schwarzschild',runs=[str(d) for d in a.runs],resolutions=ns,complete=complete,passed=False,field_checks=checks,records=records,QNM=fits,independent_waveform=wave_errors,reduction=reductions,oracle=oracle,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scope='Numerical field/QNM checks only. Full admission additionally requires time, inner and outer controls and resolved independent oracle.')
 result['reference_sampler']=a.reference_sampler;result['interpolator_sha256']=hashlib.sha256(Path(__file__).with_name('reference_interpolation.py').read_bytes()).hexdigest()
 result['field_checks_passed']=complete and len(checks)==32 and all(c['passed'] for c in checks);result['waveform_passed']=complete and all(r['passed'] for r in wave_errors);result['finest_accuracy_passed']=complete and wave_errors[-1]['passed'] and fits[-1]['passed'] and reductions[-1]['passed'];a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
 print(json.dumps(dict(complete=complete,field_checks_passed=result['field_checks_passed'],waveform=wave_errors,QNM_pass=[f['passed'] for f in fits],reduction=reductions),indent=2))
if __name__=='__main__':main()
