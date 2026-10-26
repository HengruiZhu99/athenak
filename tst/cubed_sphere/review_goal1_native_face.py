#!/usr/bin/env python3
"""Review passive native jets; no trajectory, evolved-gradient state or reductions."""
from pathlib import Path
import argparse,json,hashlib
import numpy as np
p=argparse.ArgumentParser();p.add_argument('work',type=Path);p.add_argument('output',type=Path);a=p.parse_args();a.output.mkdir(exist_ok=True,parents=True)
L=.35;amp=.001
# Native field order follows actual adapter; physical P/Theta/A/Lambda remain native.
config=[0,1,2,3,4,5,6,18,19,20,21];scale=np.full(22,amp/L);scale[config]=amp

def read(case,initial=False):
 f=a.work/case/('cubed_sphere_native_initial_jets.bin' if initial else 'cubed_sphere_native_last_jets.bin')
 with f.open('rb') as stream:n,c=np.fromfile(stream,np.uint64,2);arr=np.fromfile(stream,np.float64).reshape(int(n),int(c))
 assert int(c)==300 and np.isfinite(arr).all();return arr

def masks(x):
 radius=np.linalg.norm(x[:,:3],axis=1);extent=np.max(abs(x[:,:3]),axis=1);bulk=x[:,13]==0;shell=~bulk;nr=24 if len(x)==34560 else 36;assert len(x) in [34560,116640];s=np.zeros(len(x));s[shell]=(extent[shell]-.25)/(.7*extent[shell]/radius[shell]-.25)
 return {'all':np.ones(len(x),bool),'bulk':bulk,'bulk-core':bulk&(extent<.18),'bulk-interface':bulk&(extent>=.18),'shell':shell,'cube-shell':shell&(s<4/nr),'outer-four-rows':shell&(s>1-4/nr),'fixed-outer-collar':shell&(radius>.6)}

def norm(z,x,m):
 w=x[m,3];v=z[m];return {'rms':float(np.sqrt(np.einsum('i,i->',w,np.sum(v*v,axis=tuple(range(1,v.ndim))) if v.ndim>1 else v*v)/w.sum())),'linf':float(np.max(np.sqrt(np.sum(v*v,axis=tuple(range(1,v.ndim)))) if v.ndim>1 else abs(v)))}

def differences(x,y):
 assert x.shape==y.shape and np.array_equal(x[:,:5],y[:,:5]);d=(x[:,14:]-y[:,14:]).reshape(-1,22,13);d/=scale[None,:,None];out={}
 for region,m in masks(x).items():out[region]={'native_fields':norm(d[:,:,0],x,m),'native_first':norm(L*d[:,:,1:4],x,m),'native_configuration_second':norm(L*L*d[:,config,4:],x,m)}
 return out

def constraints(x):
 out={}
 for region,m in masks(x).items():
  out[region]={name:norm(x[:,col],x,m) for name,col in [('H_raw',5),('M_conformal',6),('M_physical',7),('Z_conformal',8),('Z_physical',9),('det_minus_1',10),('A_trace',11),('gInv_Dg_norm',12)]};out[region]['Theta_physical']=norm(x[:,14:].reshape(-1,22,13)[:,17,0],x,m)
 return out

def initial_pulse_error(case):
 # Exact pulse derivatives independently evaluated with closed polynomial/Gaussian jets.
 x=read(case,True);ref=read('stationary4-B24-A12',True) if x.shape[0]==34560 else read('stationary4-B36-A18-R36',True)
 assert np.array_equal(x[:,:5],ref[:,:5]);pos=x[:,:3];r2=np.einsum('ij,ij->i',pos,pos);pulse=(1-r2)**4*np.exp(-r2/L**2);pz=pulse*(-4/(1-r2)-1/L**2);pzz=pulse*(12/(1-r2)**2+8/(L*L*(1-r2))+1/L**4)
 g=np.zeros((len(x),4));dg=np.zeros((len(x),4,3));ddg=np.zeros((len(x),4,3,3));X,Y,Z=pos.T
 g[:,0]=1+.2*X+.3*Y*Z;dg[:,0,0]=.2;dg[:,0,1]=.3*Z;dg[:,0,2]=.3*Y;ddg[:,0,1,2]=ddg[:,0,2,1]=.3
 g[:,1]=.5*(1+.3*Y*Z);dg[:,1,1]=.15*Z;dg[:,1,2]=.15*Y;ddg[:,1,1,2]=ddg[:,1,2,1]=.15
 g[:,2]=.1*X;dg[:,2,0]=.1;g[:,3]=.05*X*Y;dg[:,3,0]=.05*Y;dg[:,3,1]=.05*X;ddg[:,3,0,1]=ddg[:,3,1,0]=.05
 exact=np.zeros((len(x),4,13));exact[:,:,0]=amp*pulse[:,None]*g
 exact[:,:,1:4]=amp*(2*pz[:,None,None]*pos[:,None,:]*g[:,:,None]+pulse[:,None,None]*dg)
 h=amp*((2*pz[:,None,None,None]*np.eye(3)[None,None,:,:]+4*pzz[:,None,None,None]*pos[:,None,:,None]*pos[:,None,None,:])*g[:,:,None,None]+2*pz[:,None,None,None]*(pos[:,None,:,None]*dg[:,:,None,:]+pos[:,None,None,:]*dg[:,:,:,None])+pulse[:,None,None,None]*ddg)
 exact[:,:,4:]=h.reshape(-1,4,9);observed=(x[:,14:]-ref[:,14:]).reshape(-1,22,13)[:,18:22,:];d=(observed-exact)/amp
 return {region:{'value':norm(d[:,:,0],x,m),'first':norm(L*d[:,:,1:4],x,m),'second':norm(L*L*d[:,:,4:],x,m),'mixed':norm(L*L*d[:,:,[5,6,7,9,10,11]],x,m)} for region,m in masks(x).items()}

cases=['stationary4-B24-A12','stationary20-B24-A12','pulse4-B24-A12','pulse20-B24-A12','pulse-half-complete','pulse-quarter-complete','pulse-noSAT-control','pulse-degree5-practical-full','pulse-degree5-practical-half','pulse-degree5-practical-quarter','pulse-degree5-conservative-complete','pulse-degree5-growth-complete','stationary-degree5-growth-complete'];out={'scope':'Full native coupled finite-cut entry and timestep/boundary diagnostics. No independent time-dependent waveform oracle, no propagation/long-growth/exact-scri acceptance.','added_evolved_fields':0,'normalizations':{'fields':'config/amp; P,A,Theta,Lambda/(amp/L)','first':'each native field normalization * L','configuration_second':'configuration field normalization * L^2','amp':amp,'L':L,'constraints':'raw physical and conformal norms retained; no Omega-weighted-only pass'},'cases':{},'timestep':{},'native_initial_operator':{}}
for case in cases:
 x=read(case);out['cases'][case]={'constraints':constraints(x),'state_sha256':hashlib.sha256((a.work/case/'cubed_sphere_native_last_jets.bin').read_bytes()).hexdigest()}
out['stationary_field_drift_linf']=float(np.max(abs((read('stationary20-B24-A12')[:,14:]-read('stationary4-B24-A12',True)[:,14:]).reshape(-1,22,13)[:,:,0])))
out['timestep']['full_half']=differences(read('pulse20-B24-A12'),read('pulse-half-complete'));out['timestep']['half_quarter']=differences(read('pulse-half-complete'),read('pulse-quarter-complete'));out['boundary_on_off_self_difference']=differences(read('pulse20-B24-A12'),read('pulse-noSAT-control'));out['native_initial_operator']['B24-A12-R24']=initial_pulse_error('pulse4-B24-A12')
for case in ['pulse4-B36-A18-R36','stationary4-B36-A18-R36','pulse4-degree5','stationary4-degree5']:
 if (a.work/case/'cubed_sphere_native_last_jets.bin').exists():out['cases'][case]={'constraints':constraints(read(case))}
if 'stationary4-B36-A18-R36' in out['cases'] and 'pulse4-B36-A18-R36' in out['cases']:out['native_initial_operator']['B36-A18-R36']=initial_pulse_error('pulse4-B36-A18-R36')
if 'pulse4-degree5' in out['cases']:out['native_initial_operator']['B24-A12-R24-degree5']=initial_pulse_error('pulse4-degree5')
out['growth_screen']={}
for case in ['pulse-degree5-growth-complete','stationary-degree5-growth-complete']:
 x=read(case);cc=constraints(x)['all'];gates={}
 if case.startswith('stationary'):
  ref=read('stationary-degree5-growth-segment1',True);drift=float(np.max(abs((x[:,14:]-ref[:,14:]).reshape(-1,22,13)[:,:,0])));gates['field_drift_linf']=drift;gates['fields_pass']=drift<1e-9;gates['constraints_pass']=max(cc[k]['linf'] for k in ['H_raw','M_conformal','M_physical','Z_conformal','Z_physical','Theta_physical'])<1e-8
 else:
  normalized={k:{kind:value/(amp/L**2 if k in ['H_raw','M_conformal','M_physical'] else amp/L) for kind,value in cc[k].items()} for k in ['H_raw','M_conformal','M_physical','Z_conformal','Z_physical','Theta_physical']};gates['normalized_constraints']=normalized;gates['constraints_pass']=all(v['rms']<.01 and v['linf']<.05 for v in normalized.values())
 out['growth_screen'][case]=gates
out['practical_timestep']={label:differences(read(label),read('pulse-degree5-conservative-complete')) for label in ['pulse-degree5-practical-full','pulse-degree5-practical-half','pulse-degree5-practical-quarter']}
out['timestep']['resolved_RK_order_claimed']=False
out['timestep']['limitation']='Differences are at accumulated roundoff/derivative amplification floor; no temporal order assertion. Half control includes retained 6.5e-19 final RK step.'
(a.output/'native-pilot-review.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({'stationary_drift':out['stationary_field_drift_linf'],'dt_full_half':out['timestep']['full_half']['all'],'dt_half_quarter':out['timestep']['half_quarter']['all'],'boundary_difference':out['boundary_on_off_self_difference']['all'],'initial_operator':out['native_initial_operator']['B24-A12-R24']['all'],'pulse_constraints':out['cases']['pulse20-B24-A12']['constraints']['all']},indent=2))
