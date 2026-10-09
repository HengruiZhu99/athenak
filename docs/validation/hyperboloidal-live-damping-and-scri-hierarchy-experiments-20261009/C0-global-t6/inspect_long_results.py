"""Saved-state prefix check, peaks and diagnostic summaries; no re-evolution."""
from pathlib import Path
import hashlib,json
import numpy as np
w=Path(__file__).resolve().parent;old=w.parent/'full-tensor-propagator/full22-v2'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
r={'scope':'exploratory original C0 projected-continuous N16 t6, no independent long canonical/native run, no energy or continuum stability theorem','gauges':{}}
for g in ['production','spatialnorm']:
 f=w/f'{g}-projected-krylov-m50-80-h0.1-t6.0.npz';c=old/f'{g}-projected-expm-t2.0.npz'
 a=np.load(f);b=np.load(c);assert np.array_equal(a['names'],b['names']);assert np.allclose(a['times'][:81],b['times'],rtol=0,atol=2e-15)
 values=a['values'];bv=b['values'];prefix=[]
 for i in range(81):prefix.append((np.linalg.norm(values[i]-bv[i],axis=0)/np.linalg.norm(bv[i],axis=0)).tolist())
 prefixmax=max(x for row in prefix for x in row);assert prefixmax<1e-10
 run=json.loads((w/f'{g}-projected-krylov-m50-80-h0.1-t6.0.json').read_text())
 h=json.loads((w/f'{g}-projected-krylov-t6.0-analysis.json').read_text());fh=json.loads((w/f'{g}-projected-krylov-t6.0-field-analysis.json').read_text())
 assert run['status']=='completed' and h['diagnostic_server_exit']==fh['server_exit']==0 and np.isfinite(values).all()
 out=[]
 for col,(hh,hf) in enumerate(zip(h['histories'],fh['histories'])):
  assert hh['name']==hf['name'];hist=hh['history'];fields=hf['history'];table=[]
  for t in [0,2,3,4,5,6]:
   z=min(hist,key=lambda q:abs(q['time']-t));u=min(fields,key=lambda q:abs(q['time']-t));assert abs(z['time']-t)<1e-13
   table.append({'time':t,'H_M_Z':z['native_H_M_Z_rms'],'free20_Euclidean_amplification':z['euclidean_component_amplification'],'configuration_H1_momentum_L2_amplification':u['configuration_H1_momentum_L2_amplification'],'outer_squared_H_M_Z_fractions':z['outer_r09_squared_constraints_fraction']})
  peaks={}
  for j,key in enumerate(['H','M','Z']):
   peak=max(hist,key=lambda q:q['native_H_M_Z_rms'][j]);last=hist[-1];prev=hist[-2]
   peaks[key]={'value':peak['native_H_M_Z_rms'][j],'time':peak['time'],'t6':last['native_H_M_Z_rms'][j],'final_step_difference':last['native_H_M_Z_rms'][j]-prev['native_H_M_Z_rms'][j]}
  peak=max(fields,key=lambda q:q['configuration_H1_momentum_L2_amplification']);peaks['field_units_amplification']={'value':peak['configuration_H1_momentum_L2_amplification'],'time':peak['time'],'t6':fields[-1]['configuration_H1_momentum_L2_amplification'],'final_step_difference':fields[-1]['configuration_H1_momentum_L2_amplification']-fields[-2]['configuration_H1_momentum_L2_amplification']}
  steps=run['columns'][col]['steps'];defs=[s['accepted_curve_defect'] for s in steps]+[x for s in steps for x in s['output_curve_defects']]
  out.append({'seed':hh['name'],'sampled_times':table,'sampled_peaks':peaks,'local_coarse_fine_max':max(s['coarse_fine_max_relative_difference'] for s in steps),'max_actual_curve_defect_relative_state_per_time':max(d['relative_state_l2_per_time'] for d in defs),'max_actual_curve_defect_relative_rhs':max(d['relative_rhs_l2'] for d in defs),'max_constraint_epsilon_check_relative_error':max(z['eps1e-5_vs3e-5_relative_constraints_l2'] for z in hh['constraint_amplitude_convergence'])})
 r['gauges'][g]={'prefix_t0_through2_max_relative_state_error_vs_saved_canonical':prefixmax,'prefix_relative_errors_81times_2seeds':prefix,'saved_canonical_sha256':sha(c),'new_long_states_sha256':sha(f),'run_sha256':sha(w/f'{g}-projected-krylov-m50-80-h0.1-t6.0.json'),'seconds':run['seconds'],'matvecs':run['matvecs'],'direct_residual_matvecs':run['residual_matvecs'],'all_vectors_finite':True,'guard_hit':False,'seeds':out}
(w/'summary.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
for g,x in r['gauges'].items():
 print(g,'prefix error',x['prefix_t0_through2_max_relative_state_error_vs_saved_canonical'],'seconds',x['seconds'])
 for z in x['seeds']:print(z['seed'],json.dumps({k:v for k,v in z.items() if k not in ['sampled_times']},indent=2))
print('summary_sha256',sha(w/'summary.json'))
