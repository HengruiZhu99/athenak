#!/usr/bin/env python3
"""Assess completed allocation prefixes without admitting an unfinished run."""
import argparse,hashlib,json,math
from pathlib import Path
import numpy as np
from scalar_evidence import snapshot,waveform,history,parent
from validate_schwarzschild import compare,norms
from reference_interpolation import lagrange6
def sha(p):
 with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text())
def exact(data,profile):
 rho=np.linalg.norm(data[:,:3],axis=1);n=data[:,:3]/rho[:,None];Y=n[:,0]*n[:,1];P=lagrange6(profile['rho'],profile['psi'],rho);V=lagrange6(profile['rho'],profile['v'],rho);Pr=lagrange6(profile['rho'],profile['Dpsi'],rho);angular=-2*Y[:,None]*n;angular[:,0]+=n[:,1];angular[:,1]+=n[:,0]
 return dict(psi=Y*P,v=Y*V,Dpsi=angular*P[:,None]/rho[:,None]+Y[:,None]*n*Pr[:,None])
p=argparse.ArgumentParser();p.add_argument('--evidence',type=Path,required=True);p.add_argument('--fine-run',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();E=a.evidence.resolve();run=a.fine_run.resolve()
if a.output.exists():raise RuntimeError('Existing closed-prefix assessment is immutable')
if not (run/'checkpointed.txt').exists() or (run/'complete.txt').exists():raise RuntimeError('Expected a closed wall-checkpoint prefix, not a completed run')
lineage=[];current=run;keys=['physical_before','reference_before','physical_derivatives_before','reference_derivatives_before'];reference_geometry=read(run/'frozen-integrity.json');binary=None
while current is not None:
 restart=(current/'restart-invocation.json').exists();native=read(current/('restart-receipt.json' if restart else 'receipt.json'));inv=read(current/('restart-invocation.json' if restart else 'invocation.json'));digest=inv['binary_sha256'] if restart else inv['binary'];binary=binary or digest;g=read(current/'frozen-integrity.json');collection=read(current.parent.parent/'collection-receipt.json')
 if native['returncode']!=0 or not collection['passed'] or not g['unchanged'] or digest!=binary or any(g[key]!=reference_geometry[key] for key in keys) or sha(current/'checkpoint.f64')!=native['files']['checkpoint.f64']:raise RuntimeError('Closed native lineage/source/frozen integrity failed')
 lineage.append(dict(run=str(current),collection_sha256=sha(current.parent.parent/'collection-receipt.json'),checkpoint_sha256=sha(current/'checkpoint.f64')));current=parent(current)
base=E/'aurora/cpu-schwarzschild-commensurate-v1-collected/result';runs=[base/'schwarzschild-n24-fixed-cube',base/'schwarzschild-n48-fixed-cube',run];O=E/'radial-oracle/smooth';checks=[];uncertainty=[];records=[];inputs=[]
for t in [2,6,12]:
 files=[O/f'waveform-upwind-n{n}.csv-profile-{t}.csv' for n in [2400,4800]];inputs.extend(files);profiles=[np.genfromtxt(file,delimiter=',',names=True) for file in files];errors=[]
 for d in runs:
  data=snapshot(d,t);data=data[np.linalg.norm(data[:,:3],axis=1)>=1.2];errors.append(compare(data,profiles[1],sampler='lagrange6'))
 records.append(dict(time=t,errors=errors))
 for field in ['psi','v','Dpsi','reduction']:
  for norm in ['L2','Linf']:
   values=[e[field][norm] for e in errors];orders=[math.log(values[k]/values[k+1])/math.log(2) for k in [0,1]];checks.append(dict(time=t,field=field,norm=norm,errors=values,orders=orders,passed=values[0]>values[1]>values[2] and min(orders)>=1.5))
 truth=[exact(data,profile) for profile in profiles]
 for field in ['psi','v','Dpsi']:
  delta=norms(truth[0][field]-truth[1][field],data[:,3])
  for norm in ['L2','Linf']:
   fraction=delta[norm]/errors[-1][field][norm];uncertainty.append(dict(time=t,field=field,norm=norm,fraction=fraction,passed=fraction<=.1))
wave=waveform(run);ref_file=O/'waveform-upwind-n4800.csv';inputs.append(ref_file);ref=np.genfromtxt(ref_file,delimiter=',',names=True)[:len(wave)]
if not np.array_equal(wave['time'],ref['time']):raise RuntimeError('Closed waveform/reference times differ')
ct=-1+(np.arange(24)+.5)*2/24;factor=-.5*np.sqrt(15/(32*np.pi))*np.pi*np.sum((1-ct*ct)**2)*2/24;truth=1j*factor*ref['radial_psi'];z=wave['real22']+1j*wave['imag22'];wave_error=float(np.linalg.norm(z-truth)/np.linalg.norm(truth));initial=snapshot(run,0);mask=np.linalg.norm(initial[:,:3],axis=1)>=1.2;scale=float(np.linalg.norm(initial[mask,6:9],axis=1).max());rows=[float(row['reduction_Linf']) for (t,region),row in history(run).items() if region=='exterior'];normalized=max(rows)/scale
result=dict(admission=False,native_run_complete=False,closed_prefix_time=float(wave['time'][-1]),resolutions=[24,48,96],records=records,checks=checks,field_checks_passed=all(c['passed'] for c in checks),reference_uncertainty=uncertainty,reference_uncertainty_passed=all(c['passed'] for c in uncertainty),partial_independent_waveform_relative_L2=wave_error,partial_waveform_two_percent_indicator_passed=wave_error<=.02,normalized_reduction_max=normalized,partial_reduction_indicator_passed=normalized<=.02,lineage=lineage,all_four_frozen_fingerprints=reference_geometry,binary_sha256=binary,input_sha256={str(file):sha(file) for file in inputs},source_sha256=sha(Path(__file__)),scope='Closed allocation data only; same source/executable and exact checkpoint lineage, unchanged physical/reference metrics and derivatives. Early field/derivative/reduction convergence and oracle uncertainty at2/6/12, waveform and every half-time reduction report through the closed prefix. All100M and timestep/inner/outer gates remain required; this assessment admits no stage.')
a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(closed_prefix_time=result['closed_prefix_time'],field_checks_passed=result['field_checks_passed'],minimum_order=min(min(c['orders']) for c in checks),reference_uncertainty_passed=result['reference_uncertainty_passed'],maximum_uncertainty_fraction=max(c['fraction'] for c in uncertainty),partial_waveform_error=wave_error,normalized_reduction_max=normalized,failed=[c for c in checks+uncertainty if not c['passed']]),indent=2))
