#!/usr/bin/env python3
"""Screen one predeclared scalar-only damping diagnostic; never admit a stage."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
from scalar_evidence import snapshot,history,waveform
from fit_ringdown import fit
import validate_schwarzschild as validation
from reference_interpolation import lagrange6

def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);a=p.parse_args();E=a.evidence.resolve();collected=E/'aurora/cpu-schwarzschild-kappa2-diagnostic-v1-collected';prepared=E/'aurora/cpu-schwarzschild-kappa2-diagnostic-v1';output=E/'cpu-schwarzschild-kappa2-diagnostic-review.json'
if output.exists():raise RuntimeError('Existing diagnostic screen is immutable')
while not (collected/'collection-receipt.json').exists():time.sleep(15)
collection=read(collected/'collection-receipt.json');manifest=read(collected/'remote-manifest.json');plan=read(E/'scalar-auxiliary-damping-diagnostic-plan.json');run=collected/'result/schwarzschild-n36-kappa2-diagnostic';old=E/'aurora/cpu-schwarzschild-commensurate-v1-collected/result/schwarzschild-n36-fixed-cube'
if not collection['passed'] or manifest['scheduler']['Jobs'][collection['job']]['Exit_status']!=0 or manifest['source_files']!=read(prepared/'portable-source-manifest.json') or not (run/'complete.txt').exists():raise RuntimeError('Diagnostic source/collection/completion check failed; preserve and inspect any native checkpoint')
g=read(run/'frozen-integrity.json');previous=read(old/'frozen-integrity.json');keys=['physical_before','reference_before','physical_derivatives_before','reference_derivatives_before'];same_geometry=g['unchanged'] and previous['unchanged'] and all(g[k]==previous[k] for k in keys);params=read(run/'parameters.json');target={**read(old/'parameters.json'),'reduction_damping':2};same_settings=params==target;same_initial=sha(run/'fields-t0.000.f64')==sha(old/'fields-t0.000.f64');validation.PchipInterpolator=lambda x,y:lambda q:lagrange6(x,y,q);oracle=E/'radial-oracle/smooth/waveform-upwind-n2400.csv';records=[]
for t in [2,6,12,100]:
 profile=np.genfromtxt(str(oracle)+f'-profile-{t}.csv',delimiter=',',names=True);records.append(dict(time=t,undamped=validation.compare(snapshot(old,t),profile),damped=validation.compare(snapshot(run,t),profile)))
wave=waveform(run);reference=np.genfromtxt(oracle,delimiter=',',names=True)
if not np.array_equal(wave['time'],reference['time']):raise RuntimeError('Independent waveform times differ')
ct=-1+(np.arange(24)+.5)*2/24;factor=-.5*np.sqrt(15/(32*np.pi))*np.pi*np.sum((1-ct*ct)**2)*2/24;truth=1j*factor*reference['radial_psi'];values=wave['real22']+1j*wave['imag22'];wave_error=float(np.linalg.norm(values-truth)/np.linalg.norm(truth));mode=next(m for m in read(E/'spin-zero-qnm.json')['modes'] if m['a_over_M']==0 and m['m']==2);fits=[]
for lo,hi in [(35,65),(40,70),(45,75)]:
 mask=(wave['time']>=lo)&(wave['time']<=hi);f=fit(wave['time'][mask],wave['imag22'][mask].astype(complex));f.update(window=[lo,hi],real_error=abs(f['omega_real']-mode['Momega_real'])/mode['Momega_real'],decay_error=abs(f['decay']+mode['Momega_imag'])/(-mode['Momega_imag']));f['passed']=f['success'] and f['real_error']<=.02 and f['decay_error']<=.05;fits.append(f)
spreads={key:(max(f[key] for f in fits)-min(f[key] for f in fits))/np.mean([f[key] for f in fits]) for key in ['omega_real','decay']};init=snapshot(run,0);mask=np.linalg.norm(init[:,:3],axis=1)>=1.2;scale=float(np.linalg.norm(init[mask,6:9],axis=1).max());maximum=max(float(r['reduction_Linf']) for (t,region),r in history(run).items() if region=='exterior');normalized=maximum/scale;late=next(r for r in records if r['time']==100);screen=plan['promotion_screen'];improved=late['damped']['psi']['L2']<=screen['late_psi_L2_max'];qnm_pass=all(f['passed'] for f in fits) and all(s<=.02 for s in spreads.values());passed=same_geometry and same_settings and same_initial and improved and wave_error<=.02 and qnm_pass and normalized<=.02
result=dict(admission=False,promotion_screen_passed=passed,run=str(run),comparison=str(old),same_geometry=same_geometry,same_scalar_settings_except_damping=same_settings,initial_fields_bitwise_identical=same_initial,frozen=g,records=records,independent_waveform_relative_L2=wave_error,QNM=fits,window_spreads=spreads,QNM_passed=qnm_pass,normalized_reduction_max=normalized,late_field_improvement_passed=improved,late_field_ratio=late['damped']['psi']['L2']/late['undamped']['psi']['L2'],plan_sha256=sha(E/'scalar-auxiliary-damping-diagnostic-plan.json'),collection_sha256=sha(collected/'collection-receipt.json'),source_sha256=sha(Path(__file__)),scope='One diagnostic of existing scalar auxiliary feedback only, on exactly identical frozen physical/reference geometry and initial fields. Not a three-resolution or CPU/GPU stage admission. Preserve all undamped failures and require complete subsequent gates for any candidate.')
output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:result[k] for k in ['promotion_screen_passed','same_geometry','initial_fields_bitwise_identical','late_field_ratio','independent_waveform_relative_L2','QNM_passed','normalized_reduction_max']},indent=2))
