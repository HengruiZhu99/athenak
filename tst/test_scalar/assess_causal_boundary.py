#!/usr/bin/env python3
"""Report late returning-cap sensitivity and independent scalar accuracy."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from fit_ringdown import fit
from scalar_evidence import waveform,snapshot
from validate_schwarzschild import compare as schwarzschild_compare
from validate_kerr import compare as kerr_compare

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--primary-review',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise RuntimeError('Existing causal diagnostic is immutable')
primary=json.loads(a.primary_review.read_text());params=json.loads((a.run/'parameters.json').read_text());nominal=Path(primary['runs'][-1]);stage=primary['stage'];oracle=Path(primary['oracle']['waveform']);kerr=stage=='kerr'
if params['cap']!=.6 or params['n']!=primary['resolutions'][-1] or 'time=100 ' not in (a.run/'complete.txt').read_text():raise RuntimeError('Completed declared finest causal cap required')
fields=[];inputs=[a.primary_review,a.run/'parameters.json',oracle]
for t in [2,6,12,100]:
 profile=Path(str(oracle)+f'-profile-{t}.csv');inputs.append(profile);data=np.genfromtxt(profile,delimiter=',',names=True);errors=(kerr_compare if kerr else schwarzschild_compare)(snapshot(a.run,t),data,sampler=primary.get('reference_sampler','pchip'));fields.append(dict(time=t,errors=errors))
wave=waveform(a.run);ref=waveform(nominal);independent=np.genfromtxt(oracle,delimiter=',',names=True)
if not np.array_equal(wave['time'],ref['time']) or not np.array_equal(wave['time'],independent['time']):raise RuntimeError('Causal comparison times differ')
z=wave['real22']+1j*wave['imag22'];base=ref['real22']+1j*ref['imag22']
if kerr:truth=independent['real22']+1j*independent['imag22']
else:
 ct=-1+(np.arange(24)+.5)*2/24;factor=-.5*np.sqrt(15/(32*np.pi))*np.pi*np.sum((1-ct*ct)**2)*2/24;truth=1j*factor*independent['radial_psi']
sensitivities=[]
for lo,hi in [(0,100),(50,100)]:
 mask=(wave['time']>=lo)&(wave['time']<=hi);value=float(np.linalg.norm(z[mask]-base[mask])/np.linalg.norm(base[mask]));sensitivities.append(dict(window=[lo,hi],relative_L2=value,one_percent_indicator_passed=value<=.01))
target=next(Path(name) for name in primary['oracle']['files'] if Path(name).name==('spin-zero-qnm-both-branches.json' if kerr else 'spin-zero-qnm.json'));inputs.append(target);modes=json.loads(target.read_text())['modes'];co=next(m for m in modes if m['a_over_M']==params['a_over_M'] and m['m']==2);counter=next((m for m in modes if m['a_over_M']==params['a_over_M'] and m['m']==-2),None);fits=[]
for lo,hi in [(35,65),(40,70),(45,75)]:
 mask=(wave['time']>=lo)&(wave['time']<=hi);f=fit(wave['time'][mask],z[mask],kerr);errors=[abs(f['omega_real']-co['Momega_real'])/co['Momega_real'],abs(f['decay']+co['Momega_imag'])/(-co['Momega_imag'])]
 if kerr:errors.extend([abs(f['counterrotating_real']-counter['Momega_real'])/counter['Momega_real'],abs(f['counterrotating_decay']+counter['Momega_imag'])/(-counter['Momega_imag'])])
 fits.append(dict(window=[lo,hi],fit=f,relative_errors=errors,passed=f['success'] and all(e<=b for e,b in zip(errors,[.02,.05,.02,.05]))))
keys=['omega_real','decay']+(['counterrotating_real','counterrotating_decay'] if kerr else []);spreads={key:(max(f['fit'][key] for f in fits)-min(f['fit'][key] for f in fits))/np.mean([f['fit'][key] for f in fits]) for key in keys};accuracy=float(np.linalg.norm(z-truth)/np.linalg.norm(truth))
result=dict(stage=stage,reported=True,run=str(a.run.resolve()),primary_review_sha256=sha(a.primary_review),fields=fields,waveform_sensitivity=sensitivities,independent_waveform_relative_L2=accuracy,independent_waveform_passed=accuracy<=.02,QNM=fits,window_spreads=spreads,QNM_passed=all(f['passed'] for f in fits) and all(s<=.02 for s in spreads.values()),input_sha256={str(f.resolve()):sha(f) for f in inputs},source_sha256=sha(Path(__file__)),scope='Additional cap.6 causal diagnostic. Same full-series1% sensitivity indicator plus late50-100 sensitivity, independent field/derivative errors and free QNM fits. Failures restrict boundary conclusions and are preserved; original .9/.95 controls remain separately required.')
result['reference_sampler']=primary.get('reference_sampler','pchip')
a.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');print(json.dumps({k:result[k] for k in ['reported','waveform_sensitivity','independent_waveform_relative_L2','QNM_passed']},indent=2))
