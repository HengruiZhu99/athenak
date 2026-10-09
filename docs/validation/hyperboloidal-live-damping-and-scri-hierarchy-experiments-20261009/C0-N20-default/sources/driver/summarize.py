"""Matched fixed pointwise gauge pulse, preserving independently sampled spherical grids."""
from pathlib import Path
import hashlib,json
import numpy as np
w=Path(__file__).resolve().parent;old=w.parent/'full-tensor-propagator';long=w.parent/'full-tensor-C0-long-window-20261009';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();result={'scope':'two-grid original C0 spatialnorm projected continuous exploratory action, fixed pointwise .1/.02 gauge shape, no continuum/physical stability or convergence-order claim','pulse_formula':'s=r²,f=(1−s)^4 exp(−4s); deltaalpha=.1 f(1+.2x+.3yz); deltabeta=.02 f(1+.3yz,.2x,.1xy)','C0_runtime':'27c19d20696ea6dd4704032c51dfd026218f64f2','norms_are_not_proved_physical_energies':True,'original_N16_snapshots_unchanged':True,'grids':{},'comparisons':{}}
for N,v,seeds in [(16,old/'full22-v2',old),(20,w/'full22',w/'native20')]:
 meta=json.loads((v/'spatialnorm-cache0.0001-metadata.json').read_text());coords=np.array(meta['xyz_omega_volume_ginv_chi']);r=np.linalg.norm(coords[:,:3],axis=1);q=np.load(seeds/'spatialnorm-validation-vectors.npz')['gauge_pulse'];scale=float(np.linalg.norm(q));x,y,z=coords[:,:3].T;s=x*x+y*y+z*z;f=(1-s)**4*np.exp(-4*s);expected=np.zeros((len(x),20));expected[:,16]=.1*f*(1+.2*x+.3*y*z);expected[:,17:]=.02*f[:,None]*np.column_stack([1+.3*y*z,.2*x,.1*x*y]);err=float(abs(q-expected.ravel()).max());assert err<1e-16
 record={'N':N,'span':2.2,'h':meta['spacing'],'points':meta['points'],'Omega_min':meta['min_omega'],'Omega_min_over_h2':meta['min_omega']/meta['spacing']**2,'native_nominal_pole03_dt':.03*meta['min_omega'],'outermost_r':float(r.max()),'pointwise_seed_Euclidean_norm':scale,'pointwise_formula_linf_error':err,'seed_sha256':sha(seeds/'spatialnorm-validation-vectors.npz'),'matrix_sha256':sha(v/'spatialnorm-projected-J20.npz'),'times':{}}
 for stop in [2.,6.]:
  if N==16 and stop==2.:
   af=v/'spatialnorm-projected-expm-analysis.json';ff=v/'spatialnorm-projected-krylov-field-analysis.json'
  elif N==16:
   af=long/'spatialnorm-projected-krylov-t6.0-analysis.json';ff=long/'spatialnorm-projected-krylov-t6.0-field-analysis.json'
  else:
   af=v/f'spatialnorm-projected-krylov-t{stop}-analysis.json';ff=v/f'spatialnorm-projected-krylov-t{stop}-field-analysis.json'
  a=json.loads(af.read_text())['histories'][0];fields=json.loads(ff.read_text())['histories'][0];assert a['name']==fields['name']=='gauge_pulse';h=a['history'][-1];g=fields['history'][-1]
  pointwise=[{**r,'native_H_M_Z_rms':[a*scale for a in r['native_H_M_Z_rms']]} for r in a['history']]
  row={'time':stop,'H_M_Z_pointwise':np.array(h['native_H_M_Z_rms']).__mul__(scale).tolist(),'free20_Euclidean_pointwise':h['euclidean_component_l2']*scale,'component_units_norm_pointwise':g['configuration_H1_momentum_L2']*scale,'component_units_amplification':g['configuration_H1_momentum_L2_amplification'],'groups_pointwise':{name:{key:val*scale for key,val in d.items()} for name,d in g['groups'].items()},'outer_squared_H_M_Z_fractions':h['outer_r09_squared_constraints_fraction'],'peak_H_M_Z_radius':h['peak_H_M_Z_radius'],'analysis_sha256':sha(af),'fields_sha256':sha(ff),'sampled_constraint_peaks':[{'value':max(t['native_H_M_Z_rms'][k] for t in pointwise),'time':max(pointwise,key=lambda t:t['native_H_M_Z_rms'][k])['time']} for k in range(3)]}
  record['times'][str(stop)]=row
 result['grids'][str(N)]=record
for t in ['2.0','6.0']:
 a=result['grids']['16']['times'][t];b=result['grids']['20']['times'][t];result['comparisons'][t]={'N20_over_N16_H_M_Z':[x/y for x,y in zip(b['H_M_Z_pointwise'],a['H_M_Z_pointwise'])],'N20_over_N16_component_units_norm':b['component_units_norm_pointwise']/a['component_units_norm_pointwise'],'N20_over_N16_component_units_amplification':b['component_units_amplification']/a['component_units_amplification']}
result['initial_native_C_hJ_h_pointwise']=json.loads((w/'initial-Hdot-comparison.json').read_text());(w/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['comparisons'],indent=2));print('summarySHA',sha(w/'summary.json'))
