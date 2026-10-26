#!/usr/bin/env python3
"""Measure native gauge waveform phase at fixed interface nodes against MP clock."""
import argparse,json,struct,hashlib,importlib.util
from pathlib import Path
import numpy as np
import mpmath as m
from goal1_global_clock_oracle import Clock
p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('evidence',type=Path);a=p.parse_args();root=Path(__file__).resolve().parents[2];m.mp.dps=80;cache=json.loads((a.evidence/'reused-height-cache.json').read_text());heightroot=root/'docs/validation/hyperboloidal-reference-wave-map-flat-IVP-values-20261009/values-source';spec=importlib.util.spec_from_file_location('height_only',heightroot/'flat_ivp_values.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);height=module.Layer(json.loads((heightroot/'values-recipe.json').read_text()),128)
records=[];sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
for receipt in a.work.glob('clock-*/receipt.json'):
 j=json.loads(receipt.read_text())
 if j.get('exit')!=0:continue
 case=receipt.parent
 with (case/'checkpoint.bin').open('rb')as f:head=struct.unpack('<16s64s19dQdq',f.read(256))
 if head[20]!=5e-6:continue
 with (case/'cubed_sphere_native_last_jets.bin').open('rb')as f:n,c=np.fromfile(f,np.uint64,2)
 if n!=26568:continue
 records.append((head[-2],case))
records.sort();assert len(records)>5 and records[-1][0]>=.3;assert len({t for t,_ in records})==len(records)
with (records[0][1]/'cubed_sphere_native_last_jets.bin').open('rb')as f:n,c=np.fromfile(f,np.uint64,2);initial=np.fromfile(f,np.float64).reshape(int(n),int(c))
points=[]
for region in [0,1]:
 for kind,target in [('face',[.15,0,0]),('corner',[.15,.15,.15])]:
  distance=np.einsum('ij,ij->i',initial[:,:3]-target,initial[:,:3]-target);index=int(np.argmin(np.where(initial[:,13]==region,distance,np.inf)));points.append((('bulk'if region==0 else'shell')+'_'+kind,index))
samples={name:[]for name,_ in points};hashes={}
for t,case in records:
 f=case/'cubed_sphere_native_last_jets.bin'
 with f.open('rb')as stream:n,c=np.fromfile(stream,np.uint64,2);x=np.fromfile(stream,np.float64).reshape(int(n),int(c))
 assert np.array_equal(x[:,:3],initial[:,:3]);fields=x[:,14:].reshape(-1,22,13)[:,:,0];hashes[case.name]=sha(f)
 for name,index in points:
  n=x[index,:3]/np.linalg.norm(x[index,:3]);beta=fields[index,19:22];radial=float(np.dot(beta,n));tangent=float(np.linalg.norm(beta-radial*n));samples[name].append([t,float(fields[index,18]),radial,tangent])
results=[];eps=m.mpf(float(5e-6));amp=4*abs(eps)/m.mpf('.04')
for name,index in points:
 xyz=initial[index,:3];r=m.sqrt(sum(m.mpf(float(z))**2for z in xyz));u=height.defect(r);local=dict(cache);local['uref_values']={r:u};clock=Clock(r,local);o,L,A,b=clock.reference(r);ref=[A,-b*A/L];data=samples[name];times=[m.mpf(float(row[0]))for row in data];weights=[(times[min(i+1,len(times)-1)]-times[max(0,i-1)])/2for i in range(len(times))]
 def oracle(t):
  state=clock.core(r,t,eps);return [state['q']*L/m.sqrt(state['G'])-ref[0],-state['q']*state['v']*L/state['G']-ref[1]]
 native=[[m.mpf(row[1])-ref[0],m.mpf(row[2])-ref[1]]for row in data];phase=m.mpf(0);gain=m.mpf(1)
 for _ in range(6):
  gram=m.matrix(2);rhs=m.matrix(2,1)
  for i,t in enumerate(times):
   pred=oracle(t+phase);derivative=[m.diff(lambda tt,k=k:oracle(tt)[k],t+phase)for k in range(2)]
   for k in range(2):
    jac=[gain*derivative[k],pred[k]];residual=native[i][k]-gain*pred[k]
    for aa in range(2):
     rhs[aa]+=weights[i]*jac[aa]*residual
     for bb in range(2):gram[aa,bb]+=weights[i]*jac[aa]*jac[bb]
  step=m.lu_solve(gram,rhs);phase+=step[0];gain+=step[1]
  assert abs(phase)<m.mpf('.01')and abs(gain-1)<m.mpf('.1')
 exact=[oracle(t)for t in times];errors=[m.sqrt(sum((native[i][k]-exact[i][k])**2for k in range(2)))/amp for i in range(len(times))];rms=m.sqrt(sum(w*e*e for w,e in zip(weights,errors))/sum(weights));fitted=[m.sqrt(sum((native[i][k]-gain*oracle(times[i]+phase)[k])**2for k in range(2)))/amp for i in range(len(times))];fittedrms=m.sqrt(sum(w*e*e for w,e in zip(weights,fitted))/sum(weights));radius=r/o;marker=-u+eps*(1-m.exp(-(2*radius/m.mpf('.2'))**2))/radius;native_marker=marker-phase
 result={'name':name,'node':index,'xyz':xyz.tolist(),'compact_radius':float(r),'observed_window':[float(times[0]),float(times[-1])],'oracle_U0_time':float(marker),'native_fitted_U0_time':float(native_marker),'fitted_native_time_advance':float(phase),'fitted_amplitude_gain':float(gain),'normalized_alpha_radial_beta_error':{'time_weighted_rms':float(rms),'linf':float(max(errors))},'normalized_phase_gain_fit_residual_rms':float(fittedrms),'normalized_beta_tangent_max':float(max(row[3]for row in data)/float(amp)),'pilot_phase_1percent_gate':abs(phase)<marker*m.mpf('.01'),'pilot_field_budget_pass':rms<.01 and max(errors)<.05,'samples':[]}
 for i,row in enumerate(data):result['samples'].append({'time':row[0],'native_alpha_perturbation':float(native[i][0]),'oracle_alpha_perturbation':float(exact[i][0]),'native_radial_beta_perturbation':float(native[i][1]),'oracle_radial_beta_perturbation':float(exact[i][1]),'normalized_gauge_error':float(errors[i])})
 results.append(result)
review={'scope':'Measured saved native lapse/radial-shift waveforms at four fixed nodes adjacent to face/corner interface on both sides; phase/gain fit against independent80digit global RWM clock. Potential-centre fit is a diagnostic, not an independent characteristic, absorption or two-crossing scientific acceptance.','native_saved_times':len(records),'precision_digits':80,'results':results,'all_pilot_phase_and_field_gates_pass':all(x['pilot_phase_1percent_gate']and x['pilot_field_budget_pass']for x in results),'source_sha256':sha(Path(__file__)),'snapshots_sha256':hashes,'no_repeated_evolution':True};(a.evidence/'measured-interface-waveforms.json').write_text(json.dumps(review,indent=2)+'\n');print(json.dumps({k:v for k,v in review.items()if k not in ['results','snapshots_sha256']},indent=2));print(json.dumps([{k:v for k,v in x.items()if k!='samples'}for x in results],indent=2))
